# src/tradebot/portfolio/risk_parity.py
"""Equal Risk Contribution (ERC) / Risk Parity portfolio.

Each asset contributes equally to total portfolio variance.

Reference: Maillard, Roncalli & Teïletche (2010) "The Properties of
Equally Weighted Risk Contribution Portfolios".
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

__all__ = ["erc_weights", "risk_contributions"]


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute marginal risk contribution of each asset.

    RC_i = w_i * (Σw)_i / (w'Σw)
    """
    sigma_w = cov @ weights
    port_var = float(weights @ sigma_w)
    if port_var < 1e-12:
        return np.ones(len(weights)) / len(weights)
    return weights * sigma_w / port_var


def erc_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.5,
    tol: float = 1e-8,
) -> pd.Series:
    """Equal Risk Contribution portfolio weights.

    Parameters
    ----------
    returns :
        Returns DataFrame (rows=time, cols=assets).
    max_weight :
        Per-asset maximum weight.
    tol :
        Optimiser tolerance.

    Returns
    -------
    pd.Series of weights summing to 1.
    """
    assets = returns.columns.tolist()
    n = len(assets)

    # Drop NaN rows rather than filling with 0 — fillna(0) decorrelates
    # assets artificially during data gaps (phantom diversification).
    clean = returns.dropna(how="any")
    if len(clean) < 5:
        clean = returns.fillna(returns.mean())
    lw = LedoitWolf()
    lw.fit(clean.values)
    cov = lw.covariance_

    w0 = np.ones(n) / n
    target_rc = 1.0 / n  # equal contribution

    def objective(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        return float(np.sum((rc - target_rc) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, max_weight)] * n

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1000},
    )

    if not result.success:
        logger.warning("ERC solve failed: %s — inverse-vol fallback.", result.message)
        vol = np.sqrt(np.diag(cov)) + 1e-9
        w = 1.0 / vol
        w /= w.sum()
        return pd.Series(w, index=assets, name="erc_weight")

    w = np.maximum(result.x, 0.0)
    w /= w.sum()
    return pd.Series(w, index=assets, name="erc_weight")
