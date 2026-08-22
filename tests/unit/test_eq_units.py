# tests/unit/test_eq_units.py
"""Wave 22 synthetic guards: XS harness math + the three equity premia."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import eq_lowvol, eq_strev, eq_xsmom
from tradebot.alpha.xs_unit import CostModel, decile_weights, run_xs_unit

UTC = "UTC"


def _idx(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz=UTC)


def _panel(n_days: int = 600, n_sym: int = 20, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.002, -0.002, n_sym)  # sym 0 strongest uptrend
    rets = rng.normal(0.0, 0.01, (n_days, n_sym)) + drift
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=_idx(n_days), columns=[f"S{i}" for i in range(n_sym)])


# ── decile_weights ───────────────────────────────────────────────────────────

def test_decile_weights_dollar_neutral_gross_one() -> None:
    scores = pd.Series(np.arange(20.0), index=[f"S{i}" for i in range(20)])
    w = decile_weights(scores, top_frac=0.3)
    assert w.sum() == pytest.approx(0.0)
    assert w.abs().sum() == pytest.approx(1.0)
    assert w["S19"] > 0 and w["S0"] < 0          # top score long, bottom short
    assert (w != 0).sum() == 12                   # 6 long + 6 short


def test_decile_weights_thin_universe_is_flat() -> None:
    scores = pd.Series([1.0, 2.0, np.nan], index=["A", "B", "C"])
    w = decile_weights(scores, min_names=10)
    assert (w == 0).all()


# ── harness math (hand-checked) ──────────────────────────────────────────────

def test_harness_pnl_is_lagged_weights_times_returns() -> None:
    idx = _idx(5)
    prices = pd.DataFrame(
        {"A": [100, 110, 121, 133.1, 146.41],    # +10%/day
         "B": [100, 100, 100, 100, 100],
         "C": [100, 95, 90.25, 85.7375, 81.450625],  # -5%/day
         "D": [100, 100, 100, 100, 100]},
        index=idx, dtype=float,
    )
    sig = pd.DataFrame(
        {"A": 2.0, "B": 1.0, "C": -1.0, "D": -2.0}, index=idx
    )  # constant: long A,B / short C,D
    res = run_xs_unit(
        prices, sig, unit="toy", rebalance="D",
        cost=CostModel(commission_bps=0, half_spread_bps=0, borrow_fee_ann=0),
        top_frac=0.5, min_names=2,
    )
    # w = +0.25 A,B / -0.25 C,D from day0; gross_t = w(t-1)·r(t)
    # day1: 0.25*0.10 + 0.25*0 - 0.25*(-0.05) - 0.25*0 = 0.0375
    assert res.gross_returns.iloc[1] == pytest.approx(0.0375)
    assert res.gross_returns.iloc[0] == pytest.approx(0.0)  # no held position day0
    assert (res.gross_returns.iloc[1:] > 0).all()


def test_costs_reduce_net_and_turnover_positive() -> None:
    prices = _panel()
    res = eq_xsmom.run(prices, cost=CostModel(commission_bps=1, half_spread_bps=5))
    free = eq_xsmom.run(prices, cost=CostModel(0.0, 0.0, 0.0))
    assert res.daily_turnover.sum() > 0
    assert res.net_returns.sum() < free.net_returns.sum()
    assert free.net_returns.equals(free.gross_returns)


def test_beta_neutral_leg_scaling() -> None:
    prices = _panel(n_days=100)
    sig = pd.DataFrame(
        {c: float(i) for i, c in enumerate(prices.columns)}, index=prices.index
    )
    kw = dict(rebalance="D", cost=CostModel(0.0, 0.0, 0.0), top_frac=0.5, min_names=2)
    base = run_xs_unit(prices, sig, unit="b0", **kw)
    ones = run_xs_unit(prices, sig, unit="b1",
                       beta_panel=pd.DataFrame(1.0, index=prices.index,
                                               columns=prices.columns), **kw)
    twos = run_xs_unit(prices, sig, unit="b2",
                       beta_panel=pd.DataFrame(2.0, index=prices.index,
                                               columns=prices.columns), **kw)
    pd.testing.assert_frame_equal(base.weights, ones.weights)  # beta 1 = no-op
    pd.testing.assert_frame_equal(twos.weights, base.weights / 2.0)  # legs halved
    assert base.config["beta_neutral"] is False
    assert twos.config["beta_neutral"] is True


def test_membership_excludes_symbol() -> None:
    prices = _panel()
    member = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    member["S0"] = False
    res = eq_xsmom.run(prices, membership=member)
    assert (res.weights["S0"] == 0).all()


# ── signal definitions ───────────────────────────────────────────────────────

def test_xsmom_ranks_trend_winner_top_and_skips_last_month() -> None:
    prices = _panel()
    sig = eq_xsmom.signal_panel(prices)
    # exact identity: score(t) = P(t-21)/P(t-252) - 1 (deterministic; the
    # earlier idxmax-on-drift assertion was noise-fragile — drift spacing
    # 0.0002/day << noise 0.01/day x sqrt(231))
    expected = prices.iloc[-22] / prices.iloc[-253] - 1.0
    pd.testing.assert_series_equal(
        sig.iloc[-1], expected, check_names=False
    )
    # drift ordering holds in rank-correlation (robust, not per-name)
    last = sig.iloc[-1].dropna()
    drift_rank = pd.Series(range(len(last), 0, -1), index=last.index, dtype=float)
    assert last.rank().corr(drift_rank.rank(), method="spearman") > 0.5
    # skip month: a crash inside the FINAL 21 days must not change ANY score
    crashed = prices.copy()
    crashed.iloc[-21:, 0] *= 0.5
    pd.testing.assert_series_equal(
        eq_xsmom.signal_panel(crashed).iloc[-1], sig.iloc[-1]
    )
    # ...but the same crash placed 22-42 days back DOES enter the window
    crashed2 = prices.copy()
    crashed2.iloc[-42:-21, 0] *= 0.5
    assert eq_xsmom.signal_panel(crashed2).iloc[-1]["S0"] < sig.iloc[-1]["S0"]


def test_strev_shorts_recent_winner() -> None:
    prices = _panel()
    sig = eq_strev.signal_panel(prices)
    ret_1m = prices.iloc[-1] / prices.iloc[-22] - 1.0
    assert sig.iloc[-1].idxmin() == ret_1m.idxmax()


def test_lowvol_prefers_quiet_asset() -> None:
    idx = _idx(400)
    rng = np.random.default_rng(3)
    quiet = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 400)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 400)))
    prices = pd.DataFrame({"QUIET": quiet, "WILD": wild}, index=idx)
    sig = eq_lowvol.signal_panel(prices)
    assert sig.iloc[-1]["QUIET"] > sig.iloc[-1]["WILD"]


def test_configs_are_literature_fixed() -> None:
    # F12 guard: the constructions advertise their fixed params + prior
    prices = _panel()
    for mod in (eq_xsmom, eq_strev, eq_lowvol):
        res = mod.run(prices)
        assert res.config["rebalance"] == "ME"
        assert res.config["top_frac"] == 0.3
        assert "prior" in res.config and res.unit == mod.UNIT
