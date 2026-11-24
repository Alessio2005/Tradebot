"""Ruwe quotes plus spot → een `Chain` met IV (mid, bid, ask) en Greeks per contract.

Per contract (detailontwerp 6.1):

1. quote ouder dan `max_quote_age_seconds` → gemarkeerd als `stale`;
2. IV op de mid (en met `spread_iv` ook op bid en ask), met de engine voor dit gebruik
   (`engine_for`: scan = BAW, beslissing = binomiaal, FD bij een discreet dividend);
3. Greeks alleen als de IV op de mid `OK` is. Er bestaat geen pad dat een IV invult.

De builder filtert niet: dat doet `chain.filters`, met een reden per afgewezen contract.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from ...utils.failfast import DataContractError, require
from ..contracts import OptionContract, Quote, Right
from ..pricing.engines import Method, Use, engine_for
from ..pricing.greeks import greeks
from ..pricing.implied_vol import IVResult, IVStatus, implied_vol
from ..pricing.settings import PricerSettings
from ..pricing.vanilla import OptionInputs
from ..rates import Carry
from ..units import GreeksPerShare

__all__ = ["Chain", "ChainBuilder", "PricedOption", "RawOption"]


@dataclass(frozen=True, slots=True)
class RawOption:
    contract: OptionContract
    quote: Quote


@dataclass(frozen=True, slots=True)
class PricedOption:
    contract: OptionContract
    quote: Quote
    iv_mid: IVResult
    iv_bid: IVResult
    iv_ask: IVResult
    greeks: GreeksPerShare | None
    method: Method
    stale: bool

    @property
    def iv(self) -> float | None:
        return self.iv_mid.value if self.iv_mid.status is IVStatus.OK else None

    @property
    def delta(self) -> float | None:
        return self.greeks.delta if self.greeks is not None else None


@dataclass(frozen=True, slots=True)
class Chain:
    underlying: str
    spot: float
    asof: datetime
    carry: Carry
    options: tuple[PricedOption, ...]

    def expiries(self) -> tuple[date, ...]:
        return tuple(sorted({o.contract.expiry for o in self.options}))

    def by_expiry(self, expiry: date, right: Right | None = None) -> tuple[PricedOption, ...]:
        """Contracten van één expiratie, op strike gesorteerd (optioneel één kant)."""
        rows = [o for o in self.options if o.contract.expiry == expiry
                and (right is None or o.contract.right is right)]
        return tuple(sorted(rows, key=lambda o: (o.contract.strike, o.contract.right.value)))

    def find(self, expiry: date, strike: float, right: Right) -> PricedOption | None:
        for o in self.options:
            c = o.contract
            if c.expiry == expiry and c.right is right and abs(c.strike - strike) < 1e-9:
                return o
        return None


class ChainBuilder:
    def __init__(self, settings: PricerSettings, max_quote_age_seconds: float,
                 use: Use = Use.SCAN, spread_iv: bool = False) -> None:
        self.settings = settings
        self.max_quote_age_seconds = max_quote_age_seconds
        self.use = use
        #: IV ook op bid en ask (de IV-spread). Kost twee extra solves per contract; de scan
        #: heeft hem niet nodig, rapportage wel.
        self.spread_iv = spread_iv

    def price_one(self, raw: RawOption, spot: float, asof: datetime,
                  carry: Carry) -> PricedOption:
        c, q = raw.contract, raw.quote
        inp = OptionInputs(spot, c.strike, c.right, c.expiry, asof, carry)
        has_div = bool(carry.dividends_in(asof.date(), c.expiry))
        method = engine_for(self.use, has_div, self.settings.engines)
        def solve(p: float | None) -> IVResult:
            return implied_vol(inp, p, method, self.settings.engines, self.settings.iv)

        mid = solve(q.mid)
        skipped = IVResult(IVStatus.NO_QUOTE)
        bid, ask = (solve(q.bid), solve(q.ask)) if self.spread_iv else (skipped, skipped)
        g = None
        if mid.status is IVStatus.OK and mid.value is not None:
            g = greeks(inp, mid.value, mid.method or method, self.settings.engines,
                       self.settings.bumps)
        stale = (asof - q.ts).total_seconds() > self.max_quote_age_seconds
        return PricedOption(c, q, mid, bid, ask, g, method, stale)

    def build(self, underlying: str, spot: float, raw: Sequence[RawOption], asof: datetime,
              carry: Carry) -> Chain:
        require(all(r.contract.underlying == underlying for r in raw),
                "een keten bevat één onderliggende", DataContractError, underlying=underlying)
        priced = tuple(self.price_one(r, spot, asof, carry) for r in raw
                       if not OptionInputs(spot, r.contract.strike, r.contract.right,
                                           r.contract.expiry, asof, carry).expired)
        return Chain(underlying, spot, asof, carry, priced)
