"""Wave-27 unit tests for cm_tsmom_xasset (R-5 determinism + known outcomes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.cm_tsmom import COST, UNIT, effective_breadth, run
from tradebot.alpha.xs_unit import CostModel

_N = 900


def _trending(n: int = _N, drift: float = 0.0015, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B", tz="UTC")
    up = 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n)))
    down = 100 * np.exp(np.cumsum(rng.normal(-drift, 0.006, n)))
    return pd.DataFrame({"UP": up, "DOWN": down}, index=idx)


def test_requires_tz_aware_index():
    panel = _trending()
    with pytest.raises(ValueError, match="tz-aware"):
        run(panel.tz_localize(None))


def test_deterministic_bit_identical():
    panel = _trending()
    a, b = run(panel), run(panel)
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)
    pd.testing.assert_frame_equal(a.weights, b.weights)
    assert a.summary() == b.summary()


def test_known_outcome_long_uptrend_short_downtrend():
    """A persistent trend must produce the literature-signed positions."""
    res = run(_trending())
    tail = res.weights.iloc[-200:]
    assert (tail["UP"] > 0).mean() > 0.9, "should be long the uptrending name"
    assert (tail["DOWN"] < 0).mean() > 0.9, "should be short the downtrending name"


def test_gross_exposure_normalised_to_one():
    res = run(_trending())
    active = res.weights.abs().sum(axis=1)
    active = active[active > 0]
    assert np.allclose(active, 1.0, atol=1e-9)


def test_costs_strictly_reduce_returns():
    panel = _trending()
    free = run(panel, cost=CostModel(0.0, 0.0, 0.0))
    charged = run(panel, cost=COST)
    assert charged.net_returns.sum() < free.net_returns.sum()
    pd.testing.assert_series_equal(
        free.gross_returns, charged.gross_returns, check_names=False
    )


def test_borrow_charged_only_on_the_short_book():
    panel = _trending()
    res = run(panel, cost=CostModel(0.0, 0.0, 0.10))
    held = res.weights.shift(1).fillna(0.0)
    shorts = held.clip(upper=0.0).abs().sum(axis=1)
    assert (res.borrow_cost[shorts == 0] == 0).all()
    assert (res.borrow_cost[shorts > 0] > 0).all()


def test_effective_breadth_penalises_correlated_instruments():
    idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    rng = np.random.default_rng(11)
    base = rng.normal(0, 0.01, 500)
    clones = pd.DataFrame({f"C{i}": base + rng.normal(0, 1e-6, 500) for i in range(5)}, index=idx)
    independents = pd.DataFrame(
        {f"I{i}": rng.normal(0, 0.01, 500) for i in range(5)}, index=idx
    )
    assert effective_breadth(clones) < 1.5     # five clones ~= one bet
    assert effective_breadth(independents) > 4.0
    assert effective_breadth(clones) < effective_breadth(independents)


def test_summary_reports_the_gate_metrics():
    s = run(_trending()).summary()
    for key in (
        "net_sharpe", "gross_sharpe", "net_cagr", "ann_vol", "max_drawdown",
        "calmar", "dd_over_vol", "years_positive_frac", "borrow_drag_ann",
    ):
        assert key in s, f"summary missing gate metric {key!r}"
    assert s["unit"] == UNIT
