"""tests/benchmark/test_kernel_speed.py — Kernel performance regression tests.

Marked @pytest.mark.slow — run nightly.
Alert threshold: > 30 % slowdown vs. previous week → flag.

These benchmarks use pytest-benchmark if available, otherwise fall back to
a simple time.perf_counter loop.
"""
from __future__ import annotations

import numpy as np
import pytest

_N_BARS = 5_000


@pytest.fixture(scope="module")
def large_ohlcv():
    import pandas as pd
    rng = np.random.default_rng(0)
    close = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, _N_BARS))
    high  = close * (1 + np.abs(rng.normal(0, 0.005, _N_BARS)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, _N_BARS)))
    open_ = close * (1 + rng.normal(0, 0.003, _N_BARS))
    idx   = pd.date_range("2024-01-01", periods=_N_BARS, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


@pytest.mark.slow
def test_gk_kernel_speed(large_ohlcv, benchmark=None) -> None:
    """GK volatility on 5k bars should complete < 500 ms."""
    import time
    from tradebot.volatility import get_garman_klass_volatility

    # Warm up JIT
    from tradebot.volatility.garman_klass import _garman_klass_kernel
    _garman_klass_kernel(
        large_ohlcv["open"].values[:100],
        large_ohlcv["high"].values[:100],
        large_ohlcv["low"].values[:100],
        large_ohlcv["close"].values[:100],
        14,
    )

    if benchmark is not None:
        benchmark(get_garman_klass_volatility, large_ohlcv, window=14)
    else:
        t0 = time.perf_counter()
        for _ in range(10):
            get_garman_klass_volatility(large_ohlcv, window=14)
        elapsed = (time.perf_counter() - t0) / 10
        assert elapsed < 0.5, f"GK kernel too slow: {elapsed:.3f}s per call"


@pytest.mark.slow
def test_sb_kernel_speed(benchmark=None) -> None:
    """Sequential bootstrap on 1k samples should complete < 2 s."""
    import time
    from tradebot.cv import get_sequential_bootstrap_indices

    t0_arr = np.arange(1_000, dtype=np.int64)
    t1_arr = np.minimum(t0_arr + 10, 999).astype(np.int64)

    # Warm up JIT
    get_sequential_bootstrap_indices(t0_arr[:50], t1_arr[:50], n_draws=50)

    if benchmark is not None:
        benchmark(get_sequential_bootstrap_indices, t0_arr, t1_arr, n_draws=1_000)
    else:
        t_start = time.perf_counter()
        get_sequential_bootstrap_indices(t0_arr, t1_arr, n_draws=1_000)
        elapsed = time.perf_counter() - t_start
        assert elapsed < 2.0, f"SB kernel too slow: {elapsed:.3f}s"
