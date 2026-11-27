"""Gamma-verlies van één positie over één dag (D2).

```text
σ_1d       = RVForecast.at_horizon(1) × √(1/252)          (geannualiseerde RV, naar één dag)
gamma_loss = ½ × |Γ_pos| × (S × σ_1d)²                    (USD, alleen bij short gamma, Γ_pos < 0)
```

Γ_pos is de gamma van de optie-legs in aandelen per dollar (zoals in `risk/aggregate.py`). Bij
long gamma of ontbrekende Greeks is het verlies 0: er valt dan niets te beheren.
"""
from __future__ import annotations

import math

from ..chain.builder import Chain
from ..positions import StrategyPosition
from ..vol_regime import RVForecast
from .aggregate import _leg_greeks

__all__ = ["gamma_loss_1d", "sigma_1d"]


def sigma_1d(rv: RVForecast) -> float:
    """De verwachte dagbeweging (fractie van spot): σ(1 dag) geannualiseerd × √(1/252)."""
    return rv.at_horizon(1) * math.sqrt(1.0 / 252.0)


def gamma_loss_1d(pos: StrategyPosition, chain: Chain | None, sigma: float) -> float:
    """½·|Γ_pos|·(S·σ_1d)² in USD voor een short-gamma positie; anders (of zonder Greeks) 0."""
    if chain is None:
        return 0.0
    greeks, complete = _leg_greeks(pos, chain)
    gamma = greeks.gamma.value
    if not complete or gamma >= 0.0:
        return 0.0
    return 0.5 * abs(gamma) * (chain.spot * sigma) ** 2
