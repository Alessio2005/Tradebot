"""De delta-hedger: hoeveel aandelen per onderliggende (`options.hedging`). Pure functies.

Regel, per onderliggende `u` in `universe.share_eligible`, over de open posities waarvan de
strategie in `hedging.strategies` staat:

```text
delta_u = Σ_posities ( Σ_optie-legs delta × 100 × qty  +  Σ_aandelen-legs qty )
|delta_u| ≤ band_shares               → geen order
order   = −round(delta_u)              (hele aandelen)
plafond : |S_u + order| × spot ≤ max_notional_frac × kapitaal
          (S_u = de aandelen-legs nu); het plafond verkleint de order, keert hem nooit om:
          zou hij omkeren, dan geen order
```

De order hoort bij één positie: die met de grootste eigen delta in de richting van `delta_u`
(bij gelijkheid het kleinste position-id). Haar aandelen-leg draagt de hedge; sluit ze, dan gaan
haar aandelen terug naar nul en hedget de volgende ronde de rest opnieuw.

Zonder keten of met een optie-leg zonder Greeks is de delta onbekend: geen order.

De limiet van een aandelenorder (`stock_limit`) ligt op spot ± de halve spread
(`paper.stock_spread_bps`), op de cent afgerond van ons af: kopen naar boven, verkopen naar
beneden. Daar vult de paper-broker; tegen IBKR is het een limiet net door de spot.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..schemas.options import OptionsConfig
from .chain.builder import Chain
from .positions import MULTIPLIER, PositionStatus, StrategyPosition

__all__ = ["HedgeOrder", "HedgeRules", "hedge_orders", "hedge_rules", "position_delta",
           "stock_limit"]


@dataclass(frozen=True, slots=True)
class HedgeRules:
    strategies: frozenset[str]
    eligible: frozenset[str]
    band_shares: float
    max_notional_frac: float


def hedge_rules(cfg: OptionsConfig) -> HedgeRules:
    h = cfg.hedging
    return HedgeRules(frozenset(h.strategies), frozenset(cfg.universe.share_eligible),
                      h.band_shares, h.max_notional_frac)


@dataclass(frozen=True, slots=True)
class HedgeOrder:
    underlying: str
    position_id: str
    strategy: str
    qty: int                  # + kopen, − verkopen
    delta_shares: float       # netto delta vóór de order
    capped: bool              # verkleind door het plafond


def position_delta(pos: StrategyPosition, chain: Chain) -> float | None:
    """Delta in aandelen van de optie-legs plus de aandelen-legs; None zonder Greeks."""
    total = float(pos.shares)
    for leg in pos.option_legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)  # type: ignore[union-attr]
        if o is None or o.greeks is None:
            return None
        total += o.greeks.delta * MULTIPLIER * leg.qty
    return total


def hedge_orders(positions: Sequence[StrategyPosition], chains: Mapping[str, Chain],
                 capital: float, rules: HedgeRules) -> list[HedgeOrder]:
    """De aandelenorders die de netto delta per onderliggende binnen de band brengen."""
    groups: dict[str, list[StrategyPosition]] = {}
    for p in positions:
        if (p.status is not PositionStatus.CLOSED and p.strategy in rules.strategies
                and p.underlying in rules.eligible):
            groups.setdefault(p.underlying, []).append(p)
    out = []
    for und, group in sorted(groups.items()):
        chain = chains.get(und)
        if chain is None:
            continue
        own = {p.position_id: position_delta(p, chain) for p in group}
        if any(d is None for d in own.values()):
            continue
        delta = sum(d for d in own.values() if d is not None)
        if abs(delta) <= rules.band_shares:
            continue
        want = -round(delta)
        held = sum(p.shares for p in group)
        cap = math.floor(rules.max_notional_frac * capital / chain.spot)
        target = max(-cap, min(cap, held + want))
        qty = target - held
        if qty == 0 or (qty > 0) != (want > 0):
            continue
        sign = 1.0 if delta > 0 else -1.0
        # `max` geeft bij gelijkheid de eerste: op id gesorteerd wint dus het kleinste id.
        owner = max(sorted(group, key=lambda p: p.position_id),
                    key=lambda p: sign * (own[p.position_id] or 0.0))
        out.append(HedgeOrder(und, owner.position_id, owner.strategy, qty, delta, qty != want))
    return out


def stock_limit(spot: float, qty: int, spread_bps: float) -> float:
    """Limiet per aandeel: spot ± de halve spread, op de cent van ons af afgerond."""
    half = spot * spread_bps / 2e4
    if qty > 0:
        return math.ceil((spot + half) * 100 - 1e-9) / 100
    return math.floor((spot - half) * 100 + 1e-9) / 100
