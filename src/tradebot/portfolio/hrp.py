# src/tradebot/portfolio/hrp.py
"""Hierarchical Risk Parity (HRP) portfolio construction.

Algorithm: López de Prado (2018) AFML §16.
  1. Covariance matrix (Ledoit-Wolf shrinkage).
  2. Correlation → distance matrix D = sqrt(0.5 * (1 - ρ)).
  3. Hierarchical clustering (Ward linkage on D).
  4. Quasi-diagonalisation (leaf order from linkage).
  5. Recursive bisection: allocate per cluster on inverse variance.

Advantages over MVO:
  - No matrix inversion (numerically stable under crypto correlations).
  - Robust to non-stationarity.
  - Less sensitive to estimation noise in expected returns.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

__all__ = ["hrp_weights", "HRPOptimizer"]


# =============================================================================
# Core HRP algorithm
# =============================================================================

def _cov_ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    # Drop rows where ANY asset has NaN rather than filling with 0.
    # fillna(0.0) artificially decorrelates assets during data gaps
    # (missing bar → zero covariance with all other assets), which
    # produces phantom diversification and incorrect HRP allocations.
    clean = returns.dropna(how="any")
    if len(clean) < 5:
        clean = returns.fillna(returns.mean())
    lw = LedoitWolf()
    lw.fit(clean.values)
    return lw.covariance_


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-9, 1e-9, std)
    return cov / np.outer(std, std)


def _distance_matrix(corr: np.ndarray) -> np.ndarray:
    """Correlation → distance: D = sqrt(0.5 * (1 - ρ))."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diagonalise(link: np.ndarray, n_assets: int) -> list[int]:
    """Extract leaf order from a scipy linkage matrix."""
    return list(hierarchy.leaves_list(link))


def _recursive_bisect(
    cov: np.ndarray,
    sorted_idx: list[int],
    weights: np.ndarray,
) -> None:
    """In-place recursive bisection allocation (modifies ``weights``)."""
    if len(sorted_idx) == 1:
        return

    mid = len(sorted_idx) // 2
    left  = sorted_idx[:mid]
    right = sorted_idx[mid:]

    # Inverse-variance contribution of each cluster
    var_left  = _cluster_var(cov, left)
    var_right = _cluster_var(cov, right)

    total = var_left + var_right + 1e-12
    alpha_left  = var_right / total    # left gets share proportional to right var
    alpha_right = var_left  / total

    weights[left]  *= alpha_left
    weights[right] *= alpha_right

    _recursive_bisect(cov, left, weights)
    _recursive_bisect(cov, right, weights)


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    """Minimum-variance portfolio variance within a cluster."""
    sub_cov = cov[np.ix_(idx, idx)]
    inv_var = 1.0 / (np.diag(sub_cov) + 1e-12)
    w = inv_var / inv_var.sum()
    return float(w @ sub_cov @ w)


def hrp_weights(
    returns: pd.DataFrame,
    linkage_method: str = "ward",
) -> pd.Series:
    """Compute HRP weights from a returns DataFrame.

    Parameters
    ----------
    returns :
        DataFrame of asset returns (rows = time, cols = assets).
        At least 5 rows required; more is better (min 30 recommended).
    linkage_method :
        Linkage method for scipy.cluster.hierarchy (default: ``"ward"``).

    Returns
    -------
    pd.Series of weights indexed by asset names.  Weights sum to 1.
    """
    assets = returns.columns.tolist()
    n = len(assets)

    if n == 1:
        return pd.Series([1.0], index=assets)

    cov  = _cov_ledoit_wolf(returns)
    corr = _corr_from_cov(cov)
    dist = _distance_matrix(corr)

    condensed = squareform(dist, checks=False)
    link = hierarchy.linkage(condensed, method=linkage_method)
    sorted_idx = _quasi_diagonalise(link, n)

    weights = np.ones(n) / n
    _recursive_bisect(cov, sorted_idx, weights)

    weights = weights / weights.sum()
    return pd.Series(weights, index=assets, name="hrp_weight")


class HRPOptimizer:
    """Stateful HRP optimizer that can be updated incrementally."""

    def __init__(self, linkage_method: str = "ward") -> None:
        self.linkage_method = linkage_method
        self._last_weights: pd.Series | None = None

    def optimize(self, returns: pd.DataFrame) -> pd.Series:
        """Compute and cache HRP weights."""
        self._last_weights = hrp_weights(returns, self.linkage_method)
        return self._last_weights

    @property
    def weights(self) -> pd.Series | None:
        return self._last_weights
