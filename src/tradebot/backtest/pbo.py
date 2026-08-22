"""Probability of Backtest Overfitting (PBO) — Bailey & López de Prado 2014.

Combinatorially Symmetric Cross-Validation (CSCV) method.
Wave 17 statistical rigour.
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


def compute_pbo(
    returns_matrix: np.ndarray,
    n_subsets: int = 16,
    metric_fn=None,
) -> dict[str, float]:
    """Compute Probability of Backtest Overfitting via CSCV.

    Args:
        returns_matrix : shape (T, S) — T time periods × S strategy configurations.
        n_subsets      : number of CSCV subsets (must be even, ≥ 4, default 16).
        metric_fn      : callable(returns_1d) → float. Default: Sharpe ratio.

    Returns:
        dict with keys:
            'pbo'          : float [0,1] — probability of overfitting
            'logit_pbo'    : float — logit transform of PBO
            'n_combinations': int — number of IS/OOS sub-period pairs used
    """
    if metric_fn is None:
        metric_fn = _sharpe_ratio

    T, S = returns_matrix.shape
    if n_subsets % 2 != 0 or n_subsets < 4:
        raise ValueError(f"n_subsets must be even and >= 4, got {n_subsets}")
    if T < n_subsets:
        raise ValueError(f"Too few time periods ({T}) for {n_subsets} subsets")

    # CHIEF AUDIT-FIX (Sim-to-Reality #10):
    #   PBO is the empirical CDF of S strategy logit-ranks at 0; when S is
    #   small the rank distribution is coarsely discretized and the resulting
    #   PBO scalar is bimodal noise.  Bailey & LdP recommend S ≥ 50 for
    #   stable estimates.  We surface this so callers do not interpret a
    #   single 0.30-0.70 PBO with S=10 as a real signal.
    if S < 50:
        logger.warning(
            "compute_pbo: S=%d strategies is below the Bailey-LdP stability "
            "floor (S>=50). At small S the logit-lambda distribution is "
            "bimodal noise — a single PBO value flips ±0.40 when one "
            "strategy is added or removed.  Treat the result as indicative, "
            "not statistically conclusive.",
            S,
        )

    # Split T into n_subsets equal blocks
    blocks = _split_into_blocks(returns_matrix, n_subsets)

    logit_lambdas: list[float] = []

    # Enumerate all C(n_subsets, n_subsets//2) IS/OOS combinations
    n_half = n_subsets // 2
    for is_indices in itertools.combinations(range(n_subsets), n_half):
        oos_indices = tuple(i for i in range(n_subsets) if i not in is_indices)

        is_returns  = np.concatenate([blocks[i] for i in is_indices], axis=0)
        oos_returns = np.concatenate([blocks[i] for i in oos_indices], axis=0)

        # IS performance for each strategy
        is_perfs  = np.array([metric_fn(is_returns[:, s]) for s in range(S)])
        oos_perfs = np.array([metric_fn(oos_returns[:, s]) for s in range(S)])

        # Best IS strategy
        best_is_idx = int(np.argmax(is_perfs))

        # Rank of best IS strategy in OOS
        oos_rank = int(np.sum(oos_perfs < oos_perfs[best_is_idx])) / max(S - 1, 1)

        # Logit lambda (relative rank)
        lambda_c = oos_rank
        if lambda_c <= 0.0:
            lambda_c = 1e-6
        elif lambda_c >= 1.0:
            lambda_c = 1.0 - 1e-6
        logit_lambdas.append(math.log(lambda_c / (1.0 - lambda_c)))

    pbo = float(np.mean(np.array(logit_lambdas) < 0.0))
    avg_logit = float(np.mean(logit_lambdas))

    logger.info(
        "PBO computed: pbo=%.3f logit_avg=%.3f n_combinations=%d",
        pbo, avg_logit, len(logit_lambdas),
    )
    return {
        "pbo": pbo,
        "logit_pbo": avg_logit,
        "n_combinations": len(logit_lambdas),
    }


def _sharpe_ratio(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Annualized Sharpe ratio for 1D returns array."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std <= 0.0:
        return 0.0
    return float(mean / std * math.sqrt(annualization))


def _split_into_blocks(matrix: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    """Split time axis into n_blocks roughly equal blocks."""
    T = matrix.shape[0]
    indices = np.array_split(np.arange(T), n_blocks)
    return [matrix[idx, :] for idx in indices]
