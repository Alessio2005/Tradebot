# tests/unit/test_fx_tsmom.py
"""Wave 26a guards: FX TSMOM sign, causality, determinism, gross normalisation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha import fx_tsmom


def _trending_panel() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=600, freq="B", tz="UTC")
    t = np.arange(len(idx))
    # EUR steadily up, JPY steadily down, others flat-ish with mild noise
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "EUR": np.exp(0.0006 * t),
            "JPY": np.exp(-0.0006 * t),
            "GBP": np.exp(0.0001 * t + 0.001 * rng.standard_normal(len(idx)).cumsum() * 0),
            "CHF": np.exp(0.00005 * t),
            "AUD": np.exp(0.0004 * t),
            "CAD": np.exp(-0.0002 * t),
        },
        index=idx,
    )


def test_tsmom_goes_long_uptrend_short_downtrend() -> None:
    tr = _trending_panel()
    res = fx_tsmom.run(tr)
    last_w = res.weights.iloc[-1]
    assert last_w["EUR"] > 0, "steady uptrend must be long"
    assert last_w["JPY"] < 0, "steady downtrend must be short"


def test_tsmom_is_deterministic() -> None:
    tr = _trending_panel()
    a = fx_tsmom.run(tr)
    b = fx_tsmom.run(tr)
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)


def test_tsmom_gross_normalised_to_one() -> None:
    tr = _trending_panel()
    res = fx_tsmom.run(tr)
    active = res.weights.loc[(res.weights != 0).any(axis=1)]
    gross = active.abs().sum(axis=1)
    # every active rebalance row sums |w| to ~1
    assert np.allclose(gross[gross > 0], 1.0, atol=1e-9)


def test_tsmom_no_lookahead_last_day() -> None:
    tr = _trending_panel()
    base = fx_tsmom.run(tr).net_returns
    tr2 = tr.copy()
    tr2.iloc[-1] *= 1.5  # shock the final close
    after = fx_tsmom.run(tr2).net_returns
    # only the LAST return may change (held weights act on next-day return);
    # all earlier net returns must be identical (no future leakage)
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])
