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

    # Stationary bootstrap to get the null distribution.
    #
    # HANSEN'S THREE p-VALUES -- CORRECTED IN PHASE 7/8 STAGE B-2.
    # ---------------------------------------------------------------------
    # Hansen (2005) section 3.2 defines three p-values that differ ONLY in the
    # recentering function g applied to the bootstrap means:
    #
    #     lower       g_l(d_bar) = max(d_bar, 0)
    #     consistent  g_c(d_bar) = d_bar * 1{d_bar >= -A_k}
    #     upper       g_u(d_bar) = d_bar
    #
    # They are ordered p_lower <= p_consistent <= p_upper. The lower variant
    # treats every model that loses to the benchmark as infinitely bad, so it
    # contributes nothing to the bootstrap maximum -- that makes it the most
    # LIBERAL of the three.
    #
    # DEFECT FOUND AND FIXED HERE: this function recentered with
    # `np.maximum(d_bar, 0.0)` -- Hansen's LOWER variant -- and returned it
    # under the key `p_value_consistent`. Every SPA verdict in this platform
    # was therefore computed with the most permissive of the three estimators
    # while the reports named the recommended one. The docstring additionally
    # promised `p_value_lower` and `p_value_upper` keys that were never
    # returned, so no caller could have noticed by reading the output.
    #
    # All three are now computed from the SAME bootstrap draws.
    bootstrap_means = np.empty((n_bootstrap, S), dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    for b in range(n_bootstrap):
        indices = _stationary_bootstrap_indices(T, block_size, rng)
        bootstrap_means[b, :] = loss_diffs[indices, :].mean(axis=0)

    # Long-run variance of sqrt(T)*d_bar_k, estimated from the same bootstrap.
    omega_sq = T * bootstrap_means.var(axis=0, ddof=1)          # (S,)

    # Hansen's threshold for the consistent variant. The sqrt(2 log log T) rate
    # is what makes the estimator consistent: it shrinks slowly enough to keep
    # near-benchmark models in the comparison set, and fast enough to drop the
    # ones that are genuinely hopeless.
    log_log_t = math.log(math.log(T)) if T > math.e else 1.0
    a_k = np.sqrt(np.maximum(omega_sq, 0.0) / T * 2.0 * max(log_log_t, 0.0))

    g_lower = np.maximum(d_bar, 0.0)
    g_consistent = np.where(d_bar >= -a_k, d_bar, 0.0)
    g_upper = d_bar

    def _p_value(g: np.ndarray) -> float:
        centered = bootstrap_means - g.reshape(1, -1)            # (B, S)
        stats_b = math.sqrt(T) * np.max(np.maximum(centered, 0.0), axis=1)
        return float(np.mean(stats_b >= T_spa))

    p_value_lower = _p_value(g_lower)
    p_value_consistent = _p_value(g_consistent)
    p_value_upper = _p_value(g_upper)

    best_idx = int(np.argmax(d_bar))
    reject = p_value_consistent < significance

    logger.info(
        "SPA test: T_spa=%.4f p_lower=%.4f p_consistent=%.4f p_upper=%.4f "
        "reject_null=%s best_strategy=%d",
        T_spa, p_value_lower, p_value_consistent, p_value_upper, reject, best_idx,
    )
    return {
        "p_value_lower": p_value_lower,
        "p_value_consistent": p_value_consistent,
        "p_value_upper": p_value_upper,
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
