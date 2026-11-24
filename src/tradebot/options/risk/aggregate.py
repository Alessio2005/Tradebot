"""Greeks van het boek per onderliggende en totaal (detailontwerp 8.3).

```text
leg_greeks      = greeks_per_share × multiplier × qty      (teken via qty)
beta_delta_usd  = Σ_onderliggende  delta_shares × spot_u × beta_u
gamma_loss_5pct = −Σ_onderliggende 0,5 × gamma_shares × (0,05 × spot_u × beta_u)²
vega_usd        = Σ vega_usd         (zonder vol-beta in v1)
theta_usd       = Σ theta_usd
```

Aandelen tellen mee met delta 1 per aandeel. `gamma_loss_5pct` is positief bij short gamma (het
verlies bij een beweging van 5 % van SPY, beta-geschaald) en negatief bij long gamma. De
delta-grens is in dollars (`beta_delta_usd`), zodat er geen SPY-koers nodig is: het is dezelfde
maat als "SPY-aandelen × SPY-koers".
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ...utils.failfast import DataContractError, require
from ..chain.builder import Chain
from ..positions import MULTIPLIER, PositionStatus, StrategyPosition
from ..units import DeltaShares, PositionGreeks

__all__ = ["BookGreeks", "UnderlyingGreeks", "aggregate"]


@dataclass(frozen=True, slots=True)
class UnderlyingGreeks:
    underlying: str
    greeks: PositionGreeks
    beta_delta_usd: float
    gamma_loss_5pct: float


@dataclass(frozen=True, slots=True)
class BookGreeks:
    per_underlying: tuple[UnderlyingGreeks, ...]
    beta_delta_usd: float
    gamma_loss_5pct: float
    vega_usd: float
    theta_usd: float
    complete: bool          # False: een leg had geen Greeks (de poort weigert dan entries)


def _leg_greeks(pos: StrategyPosition, chain: Chain | None) -> tuple[PositionGreeks, bool]:
    total, ok = PositionGreeks.zero(), True
    for leg in pos.option_legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right) if chain is not None else None  # type: ignore[union-attr]
        if o is None or o.greeks is None:
            ok = False
            continue
        total = total + o.greeks.scale(leg.qty, MULTIPLIER)
    return total, ok


def aggregate(positions: tuple[StrategyPosition, ...], shares: Mapping[str, int],
              chains: Mapping[str, Chain], beta: Mapping[str, float]) -> BookGreeks:
    names = sorted({p.underlying for p in positions if p.status is not PositionStatus.CLOSED}
                   | {k for k, v in shares.items() if v})
    rows, complete = [], True
    for und in names:
        require(und in beta, "geen beta voor onderliggende", DataContractError, underlying=und)
        chain = chains.get(und)
        g = PositionGreeks.zero()
        for p in positions:
            if p.underlying == und and p.status is not PositionStatus.CLOSED:
                pg, ok = _leg_greeks(p, chain)
                g, complete = g + pg, complete and ok
        n = shares.get(und, 0)
        if n:
            g = replace(g, delta=g.delta + DeltaShares(float(n)))
        if chain is None:
            complete = False
            rows.append(UnderlyingGreeks(und, g, 0.0, 0.0))
            continue
        s, b = chain.spot, beta[und]
        rows.append(UnderlyingGreeks(und, g, g.delta.value * s * b,
                                     -0.5 * g.gamma.value * (0.05 * s * b) ** 2))
    return BookGreeks(tuple(rows), sum(r.beta_delta_usd for r in rows),
                      sum(r.gamma_loss_5pct for r in rows),
                      sum(r.greeks.vega.value for r in rows),
                      sum(r.greeks.theta.value for r in rows), complete)
