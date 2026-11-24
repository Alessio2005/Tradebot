"""Buying power, als Reg-T-schatting per positie (detailontwerp 8.6).

| Strategie | Schatting |
|---|---|
| Credit spread, iron condor | `max_loss`, dus `(breedte × 100 − credit) × aantal` (condor: de bredere kant) |
| Debit vertical, calendar, long straddle | `max_loss`, dus het debit |
| Cash-secured put | `strike × 100 × aantal − premie` |
| Covered call | 0 bovenop de aandelen |
| Short straddle/strangle | naakt: per kant 20 % van spot × 100 × aantal, plus de premie |

In paper en live vergelijkt de adapter dit met IBKR's `whatIfOrder` (fase 7); in replay geldt
alleen de schatting.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from ..chain.builder import Chain
from ..positions import MULTIPLIER, PositionStatus, StrategyPosition

__all__ = ["NAKED_FRAC", "buying_power", "position_buying_power"]

NAKED_FRAC = 0.20


def position_buying_power(p: StrategyPosition, chains: Mapping[str, Chain]) -> float:
    if p.strategy == "covered_call" or (p.strategy == "wheel" and p.phase == "cc"):
        return 0.0
    if p.strategy == "cash_secured_put" or (p.strategy == "wheel" and p.phase == "csp"):
        return max(p.max_loss - max(p.entry_net, 0.0), 0.0)
    if math.isinf(p.max_loss):
        chain = chains.get(p.underlying)
        spot = chain.spot if chain is not None else max(
            (leg.contract.strike for leg in p.option_legs), default=0.0)  # type: ignore[union-attr]
        return NAKED_FRAC * spot * MULTIPLIER * p.units + max(p.entry_net, 0.0)
    return p.max_loss


def buying_power(positions: Sequence[StrategyPosition], chains: Mapping[str, Chain]) -> float:
    return sum(position_buying_power(p, chains) for p in positions
               if p.status is not PositionStatus.CLOSED)
