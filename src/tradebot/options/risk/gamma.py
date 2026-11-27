"""Gamma-verlies van één positie over één dag (D2).

```text
σ_1d       = RVForecast.at_horizon(1) × √(1/252)          (geannualiseerde RV, naar één dag)
gamma_loss = ½ × |Γ_pos| × (S × σ_1d)²                    (USD, alleen bij short gamma, Γ_pos < 0)
```

Γ_pos is de gamma van de optie-legs in aandelen per dollar (zoals in `risk/aggregate.py`). Bij
long gamma is het verlies 0: er valt dan niets te beheren. Zonder keten of met een leg zonder
Greeks is het onbekend (`None`); de engine meldt dat (`gamma_unknown`) in plaats van het stil
over te slaan.
"""
from __future__ import annotations

import math

from ..chain.builder import Chain
from ..positions import StrategyPosition
from ..vol_regime import RVForecast
from .aggregate import _leg_greeks

__all__ = ["gamma_loss_1d", "sigma_1d"]


def sigma_1d(rv: RVForecast) -> float:
    """De verwachte dagbeweging (fractie van spot): σ(1 dag) geannualiseerd × √(1/252).

    `at_horizon(1)` is één kalenderdag; die geannualiseerde σ gaat met √(1/252) naar één
    handelsdag, zoals het plan voorschrijft."""
    return rv.at_horizon(1) * math.sqrt(1.0 / 252.0)


def gamma_loss_1d(pos: StrategyPosition, chain: Chain | None, sigma: float) -> float | None:
    """½·|Γ_pos|·(S·σ_1d)² in USD voor een short-gamma positie; 0 bij long gamma; `None`
    zonder keten of met een leg zonder Greeks."""
    if chain is None:
        return None
    greeks, complete = _leg_greeks(pos, chain)
    gamma = greeks.gamma.value
    if not complete:
        return None
    if gamma >= 0.0:
        return 0.0
    return 0.5 * abs(gamma) * (chain.spot * sigma) ** 2
