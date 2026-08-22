# tests/lookahead/test_eq_units_causality.py
"""G6 causality guard for the Wave-22 equity units (R-1).

The truncation property: for any cut date T, running a unit on data up to T
must produce bit-identical weights up to T compared with running it on the
full history. If future rows can change past weights, the unit is
lookahead-contaminated by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import eq_lowvol, eq_strev, eq_xsmom


def _panel(n_days: int = 480, n_sym: int = 15, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, (n_days, n_sym))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2021-01-01", periods=n_days, freq="B", tz="UTC")
    return pd.DataFrame(prices, index=idx, columns=[f"S{i}" for i in range(n_sym)])


@pytest.mark.parametrize("mod", [eq_xsmom, eq_strev, eq_lowvol], ids=lambda m: m.UNIT)
@pytest.mark.parametrize("cut", [320, 410])
def test_truncation_invariance(mod, cut: int) -> None:
    prices = _panel()
    full = mod.run(prices)
    trunc = mod.run(prices.iloc[:cut])

    # W28: the old version of this test excluded the truncated panel's final
    # partial month, calling the difference "by construction". It was not
    # construction, it was the phantom-rebalance defect (the last bar of any
    # truncated panel was the max of its period and so always a rebalance).
    # With the calendar decided by next-bar comparison, invariance holds over
    # the WHOLE common index and the carve-out is gone.
    common = trunc.weights.index

    pd.testing.assert_frame_equal(
        full.weights.loc[common], trunc.weights.loc[common]
    )
    pd.testing.assert_series_equal(
        full.net_returns.loc[common], trunc.net_returns.loc[common]
    )


@pytest.mark.parametrize("mod", [eq_xsmom, eq_strev, eq_lowvol], ids=lambda m: m.UNIT)
def test_signal_row_t_ignores_future(mod) -> None:
    """Perturb only the final row; no earlier signal row may change."""
    prices = _panel()
    sig = mod.signal_panel(prices)
    bumped = prices.copy()
    bumped.iloc[-1] *= 1.5
    sig_b = mod.signal_panel(bumped)
    pd.testing.assert_frame_equal(sig.iloc[:-1], sig_b.iloc[:-1])


@pytest.mark.parametrize("mod", [eq_xsmom, eq_strev, eq_lowvol], ids=lambda m: m.UNIT)
def test_determinism_bit_identical(mod) -> None:
    prices = _panel()
    a, b = mod.run(prices), mod.run(prices)
    pd.testing.assert_frame_equal(a.weights, b.weights)        # R-5
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)
