"""Liquiditeitsfilters op contracten, in vaste volgorde, met een reden per afwijzing.

Volgorde (detailontwerp 6.1): `stale_quote`, `no_iv`, `open_interest`, `volume`, `no_bid`,
`spread`; zonder IV gaat een quote-reden vóór `no_iv` (zie `reject_reason`). De eerste reden die geldt, telt. De spreadregel is "spread ≤ max(absolute
drempel; fractie van mid)": een contract valt pas af als het beide overschrijdt.

De telling per reden gaat naar het journal en de metrics, zodat te zien is of een strategie
niets doet omdat de markt niets biedt of omdat een filter te streng is.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum

from ...schemas.options import FilterConfig
from ..contracts import Quote
from ..rates import Carry
from .builder import Chain, ChainBuilder, PricedOption, RawOption

__all__ = ["FilterResult", "FilterSettings", "Reject", "filter_chain", "filter_settings",
           "build_filtered", "filtered", "quote_reason", "reject_reason"]


class Reject(StrEnum):
    STALE_QUOTE = "stale_quote"
    NO_IV = "no_iv"
    OPEN_INTEREST = "open_interest"
    VOLUME = "volume"
    NO_BID = "no_bid"
    SPREAD = "spread"


@dataclass(frozen=True, slots=True)
class FilterSettings:
    min_open_interest: int = 500
    min_volume: int = 100
    max_spread_abs: float = 0.10
    max_spread_frac_mid: float = 0.10


def filter_settings(cfg: FilterConfig) -> FilterSettings:
    return FilterSettings(cfg.min_open_interest, cfg.min_volume, cfg.max_spread_abs,
                          cfg.max_spread_frac_mid)


@dataclass(frozen=True, slots=True)
class FilterResult:
    kept: tuple[PricedOption, ...]
    rejected: Counter[Reject]


def quote_reason(q: Quote, cfg: FilterSettings) -> Reject | None:
    """De filters die alleen de quote nodig hebben (geen IV): vóór het prijzen te toetsen."""
    if q.open_interest < cfg.min_open_interest:
        return Reject.OPEN_INTEREST
    if q.volume < cfg.min_volume:
        return Reject.VOLUME
    if q.bid is None or q.ask is None or q.bid <= 0:
        return Reject.NO_BID
    spread, mid = q.ask - q.bid, 0.5 * (q.ask + q.bid)
    if spread > max(cfg.max_spread_abs, cfg.max_spread_frac_mid * mid):
        return Reject.SPREAD
    return None


def reject_reason(o: PricedOption, cfg: FilterSettings) -> Reject | None:
    """De eerste reden die geldt. Zonder IV telt een quote-reden vóór `no_iv`: zo'n contract is
    met `skip` niet geprijsd, en zou anders ten onrechte als `no_iv` meetellen."""
    if o.stale:
        return Reject.STALE_QUOTE
    reason = quote_reason(o.quote, cfg)
    if o.iv is None:
        return reason or Reject.NO_IV
    return reason


def filter_chain(chain: Chain, cfg: FilterSettings) -> FilterResult:
    """De contracten die alle filters doorstaan, plus de telling van de afwijzingen."""
    kept: list[PricedOption] = []
    rejected: Counter[Reject] = Counter()
    for o in chain.options:
        reason = reject_reason(o, cfg)
        if reason is None:
            kept.append(o)
        else:
            rejected[reason] += 1
    return FilterResult(tuple(kept), rejected)


def filtered(chain: Chain, cfg: FilterSettings) -> tuple[Chain, FilterResult]:
    """De keten met alleen de doorgelaten contracten (voor het oppervlak en de strategieën)."""
    res = filter_chain(chain, cfg)
    return replace(chain, options=res.kept), res


def build_filtered(builder: ChainBuilder, cfg: FilterSettings, underlying: str, spot: float,
                   raw: Sequence[RawOption], asof: datetime,
                   carry: Carry) -> tuple[Chain, FilterResult]:
    """Prijs en filter één keten; contracten die op hun quote al afvallen, krijgen geen IV-solve.
    """
    def skip(q: Quote) -> bool:
        return quote_reason(q, cfg) is not None
    return filtered(builder.build(underlying, spot, raw, asof, carry, skip=skip), cfg)
