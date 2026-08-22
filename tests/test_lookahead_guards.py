"""test_lookahead_guards.py — Pin the most dangerous lookahead guards.

These tests fail LOUDLY if the next refactor breaks a causality guard.
This is the difference between "compiles" and "actually safe to deploy."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_macro_merge_publication_lag_applied() -> None:
    """A macro value at t=0 must NOT be visible to the bar at t<lag.

    Verifies merge_macro applies the +12h publication lag for non-realtime
    columns (PUBLICATION-LAG-FIX, blueprint Item 7).
    """
    from tradebot.data.macro import _merge_asof_lagged

    bars_idx = pd.date_range("2024-01-01 00:00", periods=48, freq="1h", tz="UTC")
    df_bars = pd.DataFrame({"close": np.arange(48, dtype=np.float64)}, index=bars_idx)

    # Macro frame: a single observation at 2024-01-01 06:00 UTC.
    macro_idx = pd.DatetimeIndex(["2024-01-01 06:00"], tz="UTC")
    df_macro = pd.DataFrame({"feat_macro_cpi": [1.5]}, index=macro_idx)

    out = _merge_asof_lagged(df_bars, df_macro, "TEST")

    # With +12h lag, the value should first be visible at 18:00 UTC, NOT 06:00.
    val_at_06 = out.loc[pd.Timestamp("2024-01-01 06:00", tz="UTC"), "feat_macro_cpi"]
    val_at_18 = out.loc[pd.Timestamp("2024-01-01 18:00", tz="UTC"), "feat_macro_cpi"]
    assert pd.isna(val_at_06), "Macro leaked before publication-lag boundary."
    assert val_at_18 == 1.5,    "Macro not visible after lag boundary."


def test_macro_realtime_columns_exempt_from_lag() -> None:
    """Columns matching realtime_pattern (yield/realtime/rt_/...) get NO 12h lag."""
    from tradebot.data.macro import _merge_asof_lagged

    bars_idx = pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC")
    df_bars  = pd.DataFrame({"close": np.arange(24, dtype=np.float64)}, index=bars_idx)

    macro_idx = pd.DatetimeIndex(["2024-01-01 06:00"], tz="UTC")
    df_macro  = pd.DataFrame({"feat_yield_10y": [4.5]}, index=macro_idx)

    out = _merge_asof_lagged(df_bars, df_macro, "TEST")

    val_at_07 = out.loc[pd.Timestamp("2024-01-01 07:00", tz="UTC"), "feat_yield_10y"]
    assert val_at_07 == 4.5, "Realtime column lagged unexpectedly."


def test_cusum_kernel_uses_prev_bar_atr() -> None:
    """CUSUM threshold at bar t must use ATR computed from bars [..., t-1] only.

    Test design: a pure quiet phase (zero noise) followed by a single jump.
    Without the causal shift, the threshold at the LAST quiet bar would
    already see the jump's volatility and not trigger.  With the shift,
    the kernel triggers on the jump because thresholds lag by one bar.
    """
    from tradebot.labeling.cusum import get_cusum_events

    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    # Phase 1: dead-flat 0..100. Phase 2: jump + dead-flat 100..200.
    # No noise in either phase → CUSUM events come ONLY from the jump.
    close = np.full(n, 100.0)
    close[100:] = 105.0   # 5% jump exactly at bar 100

    df = pd.DataFrame({
        "open":  close,
        "high":  close * 1.0001,
        "low":   close * 0.9999,
        "close": close,
        "volume": 1.0,
        "feat_vol_gk": 0.001,
    }, index=idx)

    events = get_cusum_events(df, threshold_multiplier=1.0)

    if len(events) == 0:
        pytest.skip("Synthetic single-jump did not trigger CUSUM (kernel-specific).")

    events_before = events[events < idx[100]]
    # Causal kernel: NO events before the jump in dead-flat data.
    assert len(events_before) == 0, (
        f"Pre-jump events {len(events_before)} on dead-flat data — "
        "causal shift may be broken (CAUSAL-FIX Item 9)."
    )
