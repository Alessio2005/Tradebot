# src/tradebot/portfolio/optimizer.py
"""Unified portfolio optimisation API.

Single entry-point ``optimize()`` that dispatches to HRP, BL, ERC or MVO
based on the ``method`` parameter.

Design: thin dispatcher + constraint application.  Each sub-module owns
its own algorithm; this module owns only the routing and post-processing.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import pandas as pd

from .black_litterman import BLViews, black_litterman_weights
from .constraints import PortfolioConstraints, apply_constraints
from .hrp import hrp_weights
from .markowitz import min_variance_weights, mvo_weights
from .risk_parity import erc_weights

logger = logging.getLogger(__name__)

__all__ = ["optimize", "OptimisationMethod"]

OptimisationMethod = Literal["hrp", "bl", "erc", "mvo", "minvar"]


def optimize(
    returns: pd.DataFrame,
    method: OptimisationMethod = "hrp",
    constraints: Optional[PortfolioConstraints] = None,
    current_weights: Optional[pd.Series] = None,
    views: Optional[BLViews] = None,
    expected_returns: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute portfolio weights using the requested method.

    Parameters
    ----------
    returns :
        Asset returns DataFrame (rows=time, cols=assets).
        Minimum 5 rows; recommend 60+ for reliable estimation.
    method :
        Optimisation method.  One of:
        - ``"hrp"``    : Hierarchical Risk Parity (default)
        - ``"bl"``     : Black-Litterman (requires ``views``)
        - ``"erc"``    : Equal Risk Contribution
        - ``"mvo"``    : Mean-Variance (requires ``expected_returns``)
        - ``"minvar"`` : Global Minimum Variance
    constraints :
        Portfolio constraints.  Defaults to long-only, 40% max weight.
    current_weights :
        Current portfolio weights for turnover limiting.
    views :
        BLViews (required for ``method="bl"``).
    expected_returns :
        Expected returns Series (optional for ``method="mvo"``).

    Returns
    -------
    pd.Series of constrained portfolio weights summing to 1.
    """
    if constraints is None:
        constraints = PortfolioConstraints()

    if len(returns) < 5:
        logger.warning("Insufficient returns for optimisation (n=%d) — equal weight.", len(returns))
        n = len(returns.columns)
        raw = pd.Series(1.0 / n, index=returns.columns)
        return apply_constraints(raw, constraints, current_weights)

    if method == "hrp":
        raw = hrp_weights(returns)
    elif method == "bl":
        raw = black_litterman_weights(returns, views=views)
    elif method == "erc":
        raw = erc_weights(returns)
    elif method == "mvo":
        raw = mvo_weights(returns, expected_returns=expected_returns)
    elif method == "minvar":
        raw = min_variance_weights(returns)
    else:
        raise ValueError(f"Unknown optimisation method: {method!r}")

    return apply_constraints(raw, constraints, current_weights)
