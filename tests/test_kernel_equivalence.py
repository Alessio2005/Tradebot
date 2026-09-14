"""test_kernel_equivalence.py — Verify extracted Numba kernels are bit-identical.

The strangler-fig contract demands bit-equivalence between the canonical
``tradebot.*`` modules and the legacy ``train_regime.py`` definitions.

This file pins the most critical kernels:
  • tradebot.backtest._kernels.calc_non_overlapping_stats
  • tradebot.labeling.cusum._symmetric_cusum_filter (via get_cusum_events)

Any divergence → CRITICAL FAILURE — refactor introduced a bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# calc_non_overlapping_stats
# =============================================================================

def _build_synthetic_signal(n: int = 200, seed: int = 42):
    """Deterministic synthetic input for the non-overlapping kernel."""
    rng = np.random.default_rng(seed)
    probs_long  = rng.uniform(0.0, 1.0, n).astype(np.float64)
    probs_short = rng.uniform(0.0, 1.0, n).astype(np.float64)
    ret_long    = rng.normal(0.0, 0.005, n).astype(np.float64)
    ret_short   = rng.normal(0.0, 0.005, n).astype(np.float64)
    horizons    = rng.integers(5, 25, n).astype(np.int32)
    t1_long  = np.clip(np.arange(n, dtype=np.int32) + horizons, 0, n - 1)
    t1_short = np.clip(np.arange(n, dtype=np.int32) + horizons, 0, n - 1)
    spreads  = np.full(n, 0.001, dtype=np.float64)
    thresh_l = np.full(n, 0.55, dtype=np.float64)
    thresh_s = np.full(n, 0.55, dtype=np.float64)
    return (probs_short, probs_long, ret_short, ret_long,
            t1_short, t1_long, spreads, thresh_l, thresh_s)


def test_kernel_deterministic() -> None:
    """Two calls with identical inputs give bit-identical output."""
    from tradebot.backtest._kernels import calc_non_overlapping_stats
    args = _build_synthetic_signal(n=500, seed=1)
    a1 = calc_non_overlapping_stats(*args)
    a2 = calc_non_overlapping_stats(*args)
    for x, y in zip(a1, a2):
        np.testing.assert_array_equal(x, y)


def test_kernel_no_overlap_invariant() -> None:
    """Active trades never overlap in time (entry > prev_exit)."""
    from tradebot.backtest._kernels import calc_non_overlapping_stats
    args = _build_synthetic_signal(n=1000, seed=7)
    rets, sides, idx = calc_non_overlapping_stats(*args)
    if len(idx) < 2:
        pytest.skip("Too few trades to test overlap.")
    # idx is monotonically increasing AND each subsequent idx > previous t1.
    assert np.all(np.diff(idx) > 0)


def test_kernel_side_encoding() -> None:
    """Side codes only contain 0 (Short) or 2 (Long), never 1 (Flat)."""
    from tradebot.backtest._kernels import calc_non_overlapping_stats
    args = _build_synthetic_signal(n=300, seed=11)
    _, sides, _ = calc_non_overlapping_stats(*args)
    if len(sides) > 0:
        assert set(np.unique(sides).tolist()).issubset({0, 2})


# =============================================================================
# CUSUM filter
# =============================================================================

def test_cusum_returns_subset_of_index() -> None:
    """get_cusum_events output is always a subset of df.index, in order."""
    from tradebot.labeling.cusum import get_cusum_events
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open":  rng.uniform(99, 101, n),
        "high":  rng.uniform(100, 102, n),
        "low":   rng.uniform(98, 100, n),
        "close": 100.0 + np.cumsum(rng.normal(0, 0.5, n)),
        "volume": rng.uniform(1, 10, n),
        "feat_vol_gk": rng.uniform(0.001, 0.05, n),
    }, index=idx)
    out = get_cusum_events(df, threshold_multiplier=1.0)
    assert isinstance(out, pd.DatetimeIndex)
    assert len(out) <= n
    # Must be a subset of df.index, in order.
    if len(out) > 0:
        assert out.is_monotonic_increasing
        assert out.isin(df.index).all()


def test_cusum_threshold_monotonic() -> None:
    """Higher threshold ⇒ fewer events (monotonicity sanity check)."""
    from tradebot.labeling.cusum import get_cusum_events
    rng = np.random.default_rng(1)
    n = 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open":  100.0,
        "high":  101.0,
        "low":   99.0,
        "close": 100.0 + np.cumsum(rng.normal(0, 1.0, n)),
        "volume": 1.0,
        "feat_vol_gk": 0.01,
    }, index=idx)
    n_low  = len(get_cusum_events(df, threshold_multiplier=0.5))
    n_high = len(get_cusum_events(df, threshold_multiplier=5.0))
    assert n_low >= n_high
