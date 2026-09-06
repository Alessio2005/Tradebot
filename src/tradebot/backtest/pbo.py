"""Probability of Backtest Overfitting (PBO) — Bailey & López de Prado 2014.

Combinatorially Symmetric Cross-Validation (CSCV) method.
Wave 17 statistical rigour.
"""
from __future__ import annotations

import itertools
import logging
import math

import numpy as np

from ..schemas.config import backtest_config
from .metrics import sharpe_ratio

logger = logging.getLogger(__name__)


def _default_metric(returns: np.ndarray) -> float:
    """De default-metriek van CSCV: de ENE Sharpe-implementatie, op de ENE annualisatie.

    Tot fase 10 stap 4A stond hier een eigen ``_sharpe_ratio(returns,
    annualization=252.0)`` — handelsdagen. `docs/MEASUREMENT_CONTRACT.md` §10.1
    telde die als de derde annualisatie in omloop (naast de 8760 van
    `metrics.py` en de 365 van `conf/`) en merkte hem aan als tegelijk een
    annualisatiedefect en een R-3-schending: één statistische grootheid, twee
    implementaties.

    De waarde komt uit `conf/backtest/default.yaml` en wordt hier niet herhaald.

    **De PBO-uitkomst verandert hier NIET door.** CSCV rangschikt strategieën
    binnen elke IS/OOS-splitsing en werkt met de logit van die RANG; een
    positieve monotone herschaling (`sqrt(a)` in plaats van `sqrt(b)`) laat elke
    rang ongemoeid. Dat is precies waarom het defect zo lang kon blijven staan,
    en waarom het repareren ervan geen enkel gemeten getal verschuift.
    `tests/unit/test_annualisation_contract.py` legt die invariantie vast.
    """
    return sharpe_ratio(returns, bars_per_year=backtest_config().bars_per_year)


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
        metric_fn = _default_metric

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


def _split_into_blocks(matrix: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    """Split time axis into n_blocks roughly equal blocks."""
    T = matrix.shape[0]
    indices = np.array_split(np.arange(T), n_blocks)
    return [matrix[idx, :] for idx in indices]
