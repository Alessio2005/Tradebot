# tests/lookahead/test_eq_overnight_causality.py
"""G6 causality guards for the Wave-23c overnight unit (R-1, R-5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha import eq_overnight


def _panels(n_days: int = 300, n_sym: int = 15, seed: int = 5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B", tz="UTC")
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, (n_days, n_sym)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(n_sym)],
    )
    open_ = close.shift(1) * np.exp(rng.normal(0.0, 0.004, (n_days, n_sym)))
    open_.iloc[0] = close.iloc[0]
    return close, open_


def test_signal_row_t_ignores_future() -> None:
    close, open_ = _panels()
    sig = eq_overnight.signal_panel(close, open_)
    bumped_c, bumped_o = close.copy(), open_.copy()
    bumped_c.iloc[-1] *= 1.5
    bumped_o.iloc[-1] *= 1.5
    sig_b = eq_overnight.signal_panel(bumped_c, bumped_o)
    pd.testing.assert_frame_equal(sig.iloc[:-1], sig_b.iloc[:-1])


def test_truncation_invariance() -> None:
    close, open_ = _panels()
    full = eq_overnight.run(close, open_)
    cut = 250
    trunc = eq_overnight.run(close.iloc[:cut], open_.iloc[:cut])
    naive = trunc.weights.index.tz_convert(None)
    safe = trunc.weights.index[naive.to_period("M") < naive[-1].to_period("M")]
    pd.testing.assert_frame_equal(full.weights.loc[safe], trunc.weights.loc[safe])
    pd.testing.assert_series_equal(full.net_returns.loc[safe], trunc.net_returns.loc[safe])


def test_determinism_bit_identical() -> None:
    close, open_ = _panels()
    a, b = eq_overnight.run(close, open_), eq_overnight.run(close, open_)
    pd.testing.assert_frame_equal(a.weights, b.weights)
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)


def test_signal_is_pure_overnight_component() -> None:
    """Intraday-only moves (open->close) must not enter the score."""
    close, open_ = _panels()
    sig = eq_overnight.signal_panel(close, open_)
    # scale intraday move of the last 21 days for one symbol: close changes
    # BUT each overnight gap open(t)/close(t-1) kept identical
    c2, o2 = close.copy(), open_.copy()
    factor = 1.25
    c2.loc[c2.index[-21]:, "S0"] = close.loc[close.index[-21]:, "S0"] * factor
    o2.loc[o2.index[-20]:, "S0"] = open_.loc[open_.index[-20]:, "S0"] * factor
    # every overnight gap open(t)/close(t-1) is unchanged by this rescaling
    # (the intraday jump on day -21 absorbed it) -> the score must be
    # bit-identical: intraday moves are invisible to the overnight signal.
    sig2 = eq_overnight.signal_panel(c2, o2)
    pd.testing.assert_frame_equal(sig, sig2)
