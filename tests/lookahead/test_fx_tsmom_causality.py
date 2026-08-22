# tests/lookahead/test_fx_tsmom_causality.py
"""G6 causality guard for the G10 FX TSMOM unit (R-1).

Added in Wave 28 step 0.2 together with the phantom-rebalance repair. The
unit had no lookahead test at all, which is why the defect that
``test_truncation_invariance_no_future_leakage`` caught in ``cm_tsmom``
survived here unnoticed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import fx_tsmom


def _panel(n_days: int = 900, n_ccy: int = 9, seed: int = 26) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.006, (n_days, n_ccy))
    tr = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2019-01-01", periods=n_days, freq="B", tz="UTC")
    cols = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"][:n_ccy]
    return pd.DataFrame(tr, index=idx, columns=cols)


@pytest.mark.parametrize("cut", [640, 755, 800])
def test_truncation_invariance_no_future_leakage(cut: int) -> None:
    """Returns up to T must not change when data after T is removed.

    Before the W28 repair the final bar of the truncated panel was always the
    max of its month and therefore a phantom rebalance, so this failed on the
    cut bar. Invariance now holds over the whole common index.
    """
    panel = _panel()
    full = fx_tsmom.run(panel)
    trunc = fx_tsmom.run(panel.iloc[:cut])
    common = trunc.weights.index

    pd.testing.assert_frame_equal(
        full.weights.loc[common], trunc.weights.loc[common]
    )
    pd.testing.assert_series_equal(
        full.net_returns.loc[common], trunc.net_returns.loc[common]
    )


def test_weights_are_held_one_bar_before_earning() -> None:
    """The return on bar t must be earned by the weights formed on bar t-1."""
    panel = _panel()
    res = fx_tsmom.run(panel)
    rets = panel.pct_change(fill_method=None)

    expected = (res.weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    pd.testing.assert_series_equal(
        res.gross_returns, expected, check_names=False
    )
    same_bar = (res.weights * rets).sum(axis=1)
    assert not np.allclose(
        res.gross_returns.fillna(0.0), same_bar.fillna(0.0)
    ), "gross return matches same-bar weights — lookahead"


def test_future_prices_cannot_move_past_weights() -> None:
    """Perturbing a future price must leave every earlier weight untouched."""
    panel = _panel()
    base = fx_tsmom.run(panel).weights
    bumped = panel.copy()
    bumped.iloc[700:] *= 1.4
    after = fx_tsmom.run(bumped).weights

    pd.testing.assert_frame_equal(base.iloc[:700], after.iloc[:700])


def test_rebalance_calendar_never_uses_the_final_bar() -> None:
    """The ragged edge is not a rebalance: weights must be flat into it."""
    panel = _panel()
    w = fx_tsmom.run(panel).weights
    # last bar cannot differ from the one before it — it is a hold, never a
    # rebalance, because "is the next bar a new month" is unknowable there.
    pd.testing.assert_series_equal(
        w.iloc[-1], w.iloc[-2], check_names=False
    )
