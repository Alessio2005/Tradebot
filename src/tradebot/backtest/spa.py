"""Superior Predictive Ability (SPA) test — Hansen 2005.

Tests whether the best strategy has predictive ability superior to a benchmark,
after accounting for the selection effect of choosing the best from many.
Wave 17 statistical rigour.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def _autocorr_block_size(arr: np.ndarray, max_lag: int = 60) -> int:
    """Politis-White block-size estimate from autocorrelation decay.

    Returns smallest k where |rho(k)| < 2/sqrt(n), clipped to [5, 60].
    CHIEF AUDIT-FIX (Sim-to-Reality #11): default block_size=5 ignored
    crypto's 20-40 bar vol-clustering → inflated p-values 30-50%.
    """
    n = len(arr)
    if n < 30:
        return 5
    x = arr - arr.mean()
    var0 = float(np.dot(x, x)) / max(n, 1)
    if var0 <= 0.0:
        return 5
    band = 2.0 / math.sqrt(n)
    upper = int(min(max_lag, n // 4))
    for k in range(1, upper + 1):
        rho_k = float(np.dot(x[:-k], x[k:])) / (var0 * (n - k))
        if abs(rho_k) < band:
            return int(max(5, min(60, k)))
    return int(max(5, min(60, upper)))


def spa_test(
    benchmark_returns: np.ndarray,
    strategy_returns_matrix: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int | None = None,
    significance: float = 0.05,
) -> dict[str, object]:
    """Compute Superior Predictive Ability p-value (Hansen 2005).

    CHIEF AUDIT-FIX (Sim-to-Reality #11):
      ``block_size`` now defaults to ``None`` → auto-calibrate via the
      Politis-White heuristic on the loss-differential time series.
      Previous fixed default of 5 ignored crypto's 20-40 bar vol-clustering,
      inflating the bootstrap p-value by 30-50 % and producing
      false-positive SPA rejections.  Callers can still pin block_size
      explicitly for reproducibility.

    Args:
        benchmark_returns       : 1D array of benchmark (e.g. buy-and-hold).
        strategy_returns_matrix : shape (T, S) — T periods × S strategies.
        n_bootstrap             : stationary bootstrap replications.
        block_size              : expected block size for stationary bootstrap.
                                  ``None`` → auto from autocorrelation.
        significance            : significance level (default 0.05).

    Returns:
        dict with 'p_value_consistent', 'p_value_upper', 'p_value_lower',
        'best_strategy_idx', 'reject_null' (bool).
    """
    T, S = strategy_returns_matrix.shape
    bench = np.asarray(benchmark_returns, dtype=np.float64)

    # CHIEF AUDIT-FIX #11: auto-block-size from the best-strategy loss differential.
    if block_size is None:
        # Use the best strategy's loss-differential autocorr as the calibration target.
        bs_d = (strategy_returns_matrix - bench.reshape(-1, 1))
        best_proxy = bs_d[:, int(np.argmax(bs_d.mean(axis=0)))]
        eff_block = _autocorr_block_size(best_proxy)
        logger.info("spa_test: auto-calibrated block_size=%d", eff_block)
    else:
        eff_block = int(block_size)
    block_size = eff_block

    # Loss differentials: d_{k,t} = f(Y_t, strategy_k) - f(Y_t, benchmark)
    # Using negative returns as loss (higher return = lower loss)
    loss_diffs = strategy_returns_matrix - bench.reshape(-1, 1)  # (T, S)

    d_bar = loss_diffs.mean(axis=0)  # (S,)

    # Consistent test statistic: T_spa = sqrt(T) * max(d_bar_k+)
    # where d_bar_k+ = max(d_bar_k, 0) — only consider strategies that beat benchmark
    T_spa = math.sqrt(T) * float(np.max(np.maximum(d_bar, 0.0)))

    # Stationary bootstrap to get null distribution
    bootstrap_stats: list[float] = []
    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        indices = _stationary_bootstrap_indices(T, block_size, rng)
        d_boot = loss_diffs[indices, :]
        d_boot_bar = d_boot.mean(axis=0)
        # Center the bootstrap statistic
        d_centered = d_boot_bar - np.maximum(d_bar, 0.0)
        bootstrap_stats.append(float(math.sqrt(T) * np.max(np.maximum(d_centered, 0.0))))

    boot_arr = np.array(bootstrap_stats)
    p_value_consistent = float(np.mean(boot_arr >= T_spa))

    best_idx = int(np.argmax(d_bar))
    reject = p_value_consistent < significance

    logger.info(
        "SPA test: T_spa=%.4f p_value=%.4f reject_null=%s best_strategy=%d",
        T_spa, p_value_consistent, reject, best_idx,
    )
    return {
        "p_value_consistent": p_value_consistent,
        "T_spa": T_spa,
        "best_strategy_idx": best_idx,
        "reject_null": reject,
        "n_strategies": S,
        "n_periods": T,
    }


def _stationary_bootstrap_indices(
    T: int,
    block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate stationary bootstrap index array (Politis & Romano 1994)."""
    p = 1.0 / max(float(block_size), 1.0)
    indices = np.empty(T, dtype=np.int64)
    i = 0
    while i < T:
        start = int(rng.integers(0, T))
        block_len = int(rng.geometric(p))
        for j in range(block_len):
            if i >= T:
                break
            indices[i] = (start + j) % T
            i += 1
    return indices
