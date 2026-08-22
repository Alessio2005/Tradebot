# src/tradebot/portfolio/__init__.py
"""Portfolio construction v2 — HRP, Black-Litterman, ERC, MVO."""
from __future__ import annotations

from .black_litterman import BLViews, black_litterman_weights
from .constraints import PortfolioConstraints, apply_constraints, compute_turnover
from .hrp import HRPOptimizer, hrp_weights
from .markowitz import min_variance_weights, mvo_weights
from .optimizer import OptimisationMethod, optimize
from .rebalance import RebalanceConfig, RebalanceDecision, should_rebalance
from .risk_parity import erc_weights, risk_contributions

__all__ = [
    # HRP
    "hrp_weights",
    "HRPOptimizer",
    # Black-Litterman
    "BLViews",
    "black_litterman_weights",
    # ERC
    "erc_weights",
    "risk_contributions",
    # MVO / MinVar
    "mvo_weights",
    "min_variance_weights",
    # Unified API
    "optimize",
    "OptimisationMethod",
    # Constraints
    "PortfolioConstraints",
    "apply_constraints",
    "compute_turnover",
    # Rebalance
    "RebalanceConfig",
    "RebalanceDecision",
    "should_rebalance",
]
