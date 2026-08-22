"""R-1 / G6 causality guards for the Wave-27 cross-asset panel and TSMOM unit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.cm_tsmom import run
from tradebot.data.sources.base import ASOF_COL, EVENT_COL, validate_pit
from tradebot.data.xasset_proxy import to_tr_panel

pytestmark = pytest.mark.lookahead

_N = 900
_SYMS = ("AAA", "BBB", "CCC", "DDD")


def _synthetic_panel(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=_N, freq="B", tz="UTC")
    data = {}
    for i, s in enumerate(_SYMS):
        steps = rng.normal(0.0003 * (1 if i % 2 == 0 else -1), 0.01, _N)
        data[s] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=idx)


def _long_form(panel: pd.DataFrame) -> pd.DataFrame:
    long = (
        panel.rename_axis(EVENT_COL)
        .reset_index()
        .melt(id_vars=EVENT_COL, var_name="symbol", value_name="close")
    )
    long[ASOF_COL] = long[EVENT_COL] + pd.Timedelta(days=1)
    return long


def test_panel_satisfies_pit_contract():
    long = _long_form(_synthetic_panel())
    out = validate_pit(long, required=("symbol", "close"))
    assert (out[ASOF_COL] >= out[EVENT_COL]).all()


def test_asof_before_event_is_rejected():
    long = _long_form(_synthetic_panel())
    long.loc[5, ASOF_COL] = long.loc[5, EVENT_COL] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="lookahead"):
        validate_pit(long, required=("symbol", "close"))


def test_to_tr_panel_roundtrip_preserves_values():
    panel = _synthetic_panel()
    back = to_tr_panel(_long_form(panel))
    pd.testing.assert_frame_equal(
        back[list(_SYMS)].astype(float),
        panel[list(_SYMS)].astype(float),
        check_freq=False,
        check_names=False,
    )


def test_truncation_invariance_no_future_leakage():
    """Returns up to T must not change when data after T is removed.

    This is the operational definition of causality: if appending tomorrow's
    bar changed yesterday's P&L, the unit is reading the future.
    """
    panel = _synthetic_panel()
    cut = 700
    full = run(panel).net_returns.iloc[:cut]
    trunc = run(panel.iloc[:cut]).net_returns.iloc[:cut]
    pd.testing.assert_series_equal(full, trunc, check_freq=False)


def test_weights_are_held_one_bar_before_earning():
    """The return on bar t must be earned by the weights formed on bar t-1."""
    panel = _synthetic_panel()
    res = run(panel)
    rets = panel.pct_change(fill_method=None)
    expected_gross = (res.weights.shift(1).fillna(0.0) * rets).sum(axis=1)
    pd.testing.assert_series_equal(
        res.gross_returns, expected_gross, check_names=False
    )
    # and NOT by same-bar weights (would be lookahead)
    same_bar = (res.weights * rets).sum(axis=1)
    assert not np.allclose(
        res.gross_returns.fillna(0), same_bar.fillna(0)
    ), "gross return matches same-bar weights — lookahead"


def test_signal_uses_only_past_prices():
    """Perturbing a future price must not change an earlier weight."""
    panel = _synthetic_panel()
    base = run(panel).weights
    bumped = panel.copy()
    bumped.iloc[800:] *= 1.5
    after = run(bumped).weights
    pd.testing.assert_frame_equal(
        base.iloc[:790], after.iloc[:790], check_freq=False
    )
