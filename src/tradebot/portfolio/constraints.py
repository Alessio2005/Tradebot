# src/tradebot/portfolio/constraints.py
"""Portfolio constraint helpers — leverage, concentration, turnover caps.

All functions are pure utilities that operate on weight arrays.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["PortfolioConstraints", "apply_constraints", "compute_turnover"]


@dataclass
class PortfolioConstraints:
    """Container for portfolio construction constraints.

    Attributes
    ----------
    max_weight :
        Maximum weight per asset (concentration cap).
    min_weight :
        Minimum non-zero weight per asset (avoids tiny positions).
    max_leverage :
        Maximum sum of absolute weights (1.0 = long-only, fully invested).
    max_turnover :
        Maximum one-way turnover relative to current portfolio (0.3 = 30%).
    """

    max_weight: float = 0.40
    min_weight: float = 0.02
    max_leverage: float = 1.0
    max_turnover: float = 1.0


def apply_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    current_weights: pd.Series | None = None,
) -> pd.Series:
    """Apply portfolio constraints to a raw weight series.

    Parameters
    ----------
    weights :
        Raw target weights (should sum to ~1.0, long-only).
    constraints :
        Constraint specification.
    current_weights :
        Current portfolio weights for turnover limiting.  If None,
        turnover constraint is skipped.

    Returns
    -------
    pd.Series of constrained weights.
    """
    w = weights.copy().fillna(0.0)
    assets = w.index

    # Iteratively clip to max_weight and renormalise until convergence.
    # Single clip + renorm can push weights back above the cap.
    for _ in range(20):
        w = w.clip(lower=0.0, upper=constraints.max_weight)
        w[w < constraints.min_weight] = 0.0
        total = w.sum()
        if total < 1e-9:
            w = pd.Series(1.0 / len(assets), index=assets)
            break
        w = w / total
        if w.max() <= constraints.max_weight + 1e-9:
            break

    # Final hard clip to eliminate floating-point overshoot at boundary.
    w = w.clip(upper=constraints.max_weight)
    total = w.sum()
    if total > 1e-9:
        w = w / total

    # Leverage cap (applied after normalisation)
    total = w.sum()
    if total > constraints.max_leverage:
        w = w * constraints.max_leverage / total

    # 5. Turnover cap (blend towards current if needed)
    if current_weights is not None and constraints.max_turnover < 1.0:
        cur = current_weights.reindex(assets).fillna(0.0)
        turnover = float((w - cur).abs().sum()) / 2.0
        if turnover > constraints.max_turnover:
            # Blend: w_new = α*w_target + (1-α)*w_current
            alpha = constraints.max_turnover / (turnover + 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            w = alpha * w + (1 - alpha) * cur
            w = w.clip(lower=0.0)
            s = w.sum()
            if s > 1e-9:
                w /= s

    return w.rename("constrained_weight")


def compute_turnover(
    new_weights: pd.Series,
    old_weights: pd.Series,
) -> float:
    """One-way turnover as fraction of total portfolio value.

    Turnover = 0.5 * sum(|w_new - w_old|)
    """
    combined_idx = new_weights.index.union(old_weights.index)
    new = new_weights.reindex(combined_idx).fillna(0.0)
    old = old_weights.reindex(combined_idx).fillna(0.0)
    return float((new - old).abs().sum() / 2.0)
