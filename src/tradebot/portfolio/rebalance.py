# src/tradebot/portfolio/rebalance.py
"""Turnover-cost-aware rebalancing logic.

Determines whether and how much to rebalance based on:
  - Drift from target (band-based trigger)
  - One-way transaction costs
  - Minimum net-benefit threshold (rebalance only if TCA-benefit > cost)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["RebalanceConfig", "RebalanceDecision", "should_rebalance"]


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing decisions.

    Attributes
    ----------
    band_width :
        Absolute weight deviation from target that triggers rebalancing
        (e.g. 0.05 = 5 percentage points).
    one_way_cost_bps :
        Estimated one-way transaction cost in basis points.
    min_net_benefit :
        Minimum expected net benefit (return improvement minus cost) to
        trigger a rebalance.  Set to 0 to always rebalance on band breach.
    """

    band_width: float = 0.05
    one_way_cost_bps: float = 5.0
    min_net_benefit: float = 0.0


@dataclass
class RebalanceDecision:
    """Output of ``should_rebalance``."""

    rebalance: bool
    turnover: float
    estimated_cost: float
    max_drift: float
    reason: str


def should_rebalance(
    current_weights: pd.Series,
    target_weights: pd.Series,
    cfg: RebalanceConfig | None = None,
) -> RebalanceDecision:
    """Decide whether to rebalance from current to target weights.

    Parameters
    ----------
    current_weights :
        Current portfolio weights.
    target_weights :
        Target (optimised) portfolio weights.
    cfg :
        Rebalance configuration.

    Returns
    -------
    RebalanceDecision with rebalance flag and diagnostics.
    """
    if cfg is None:
        cfg = RebalanceConfig()

    all_assets = current_weights.index.union(target_weights.index)
    cur = current_weights.reindex(all_assets).fillna(0.0)
    tgt = target_weights.reindex(all_assets).fillna(0.0)

    diff = (tgt - cur).abs()
    max_drift = float(diff.max())
    turnover  = float(diff.sum() / 2.0)
    cost_bps  = cfg.one_way_cost_bps * turnover
    cost_frac = cost_bps / 10_000.0

    if max_drift < cfg.band_width:
        return RebalanceDecision(
            rebalance=False,
            turnover=turnover,
            estimated_cost=cost_frac,
            max_drift=max_drift,
            reason=f"drift {max_drift:.3f} < band {cfg.band_width:.3f}",
        )

    # When min_net_benefit == 0, always rebalance on band breach.
    if cfg.min_net_benefit <= 0.0:
        return RebalanceDecision(
            rebalance=True,
            turnover=turnover,
            estimated_cost=cost_frac,
            max_drift=max_drift,
            reason=f"drift {max_drift:.3f} >= band {cfg.band_width:.3f}",
        )

    # Approximate benefit: turnover × expected alpha improvement proxy.
    # A positive min_net_benefit guards against churning for tiny expected gains.
    benefit = turnover * 0.01  # 1% expected alpha per unit of rebalancing (crude estimate)
    net_benefit = benefit - cost_frac

    if net_benefit < cfg.min_net_benefit:
        return RebalanceDecision(
            rebalance=False,
            turnover=turnover,
            estimated_cost=cost_frac,
            max_drift=max_drift,
            reason=f"net_benefit {net_benefit:.4f} < threshold {cfg.min_net_benefit:.4f}",
        )

    return RebalanceDecision(
        rebalance=True,
        turnover=turnover,
        estimated_cost=cost_frac,
        max_drift=max_drift,
        reason=f"drift {max_drift:.3f} >= band {cfg.band_width:.3f}, net_benefit={net_benefit:.4f}",
    )
