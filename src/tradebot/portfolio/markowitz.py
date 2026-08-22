# src/tradebot/portfolio/markowitz.py
"""Mean-Variance Optimisation (Markowitz) with Ledoit-Wolf covariance.

Fallback method when HRP/BL cannot produce valid weights.
Uses long-only constrained MVO via scipy.optimize.

Reference: Markowitz (1952); shrinkage via Ledoit & Wolf (2004).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

__all__ = ["mvo_weights", "min_variance_weights"]


def _cov_lw(returns: pd.DataFrame) -> np.ndarray:
    # Drop NaN rows rather than filling with 0; fillna(0) pulls off-diagonal
    # covariances toward zero, creating phantom diversification.
    clean = returns.dropna(how="any")
    if len(clean) < 5:
        clean = returns.fillna(returns.mean())
    lw = LedoitWolf()
    lw.fit(clean.values)
    return lw.covariance_


def min_variance_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.5,
) -> pd.Series:
    """Global minimum-variance portfolio (long-only).

    Parameters
    ----------
    returns :
        Returns DataFrame (rows=time, cols=assets).
    max_weight :
        Maximum weight per asset (concentration cap).
    """
    assets = returns.columns.tolist()
    n = len(assets)
    cov = _cov_lw(returns)

    w0 = np.ones(n) / n

    def portfolio_var(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_weight)] * n

    result = minimize(
        portfolio_var,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if not result.success:
        logger.warning("MVO min-var solve failed: %s — using equal-weight.", result.message)
        return pd.Series(w0, index=assets)

    w = np.maximum(result.x, 0.0)
    w /= w.sum()
    return pd.Series(w, index=assets, name="minvar_weight")


def mvo_weights(
    returns: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_aversion: float = 2.5,
    max_weight: float = 0.5,
    prev_weights: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
) -> pd.Series:
    """Mean-variance efficient portfolio.

    If ``expected_returns`` is None, uses historical mean returns.

    Parameters
    ----------
    prev_weights :
        Previous portfolio weights (shape n). When supplied together with
        ``turnover_penalty > 0``, adds a penalty  ``0.5 * λ_to * ||w - w_prev||²``
        to the objective so the optimizer accounts for transaction costs.
        Prevents high-turnover corner solutions in daily rebalancing.
    turnover_penalty :
        Transaction-cost coefficient λ_to (default 0 = disabled).
        Typical value: 10–50× the round-trip fee in return units
        (e.g., 5 bp fee × 40 ≈ 0.02).
    """
    assets = returns.columns.tolist()
    n = len(assets)
    cov = _cov_lw(returns)

    if expected_returns is None:
        mu = returns.fillna(0.0).mean().values
    else:
        mu = expected_returns.reindex(assets).fillna(0.0).values

    w0 = np.ones(n) / n
    w_prev = (
        np.asarray(prev_weights, dtype=np.float64)
        if prev_weights is not None and turnover_penalty > 0.0
        else None
    )

    def neg_utility(w: np.ndarray) -> float:
        utility = float(w @ mu) - 0.5 * risk_aversion * float(w @ cov @ w)
        if w_prev is not None:
            delta = w - w_prev
            utility -= 0.5 * turnover_penalty * float(delta @ delta)
        return -utility

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_weight)] * n

    result = minimize(
        neg_utility,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if not result.success:
        logger.warning("MVO solve failed: %s — using min-var fallback.", result.message)
        return min_variance_weights(returns, max_weight=max_weight)

    w = np.maximum(result.x, 0.0)
    w /= w.sum()
    return pd.Series(w, index=assets, name="mvo_weight")
