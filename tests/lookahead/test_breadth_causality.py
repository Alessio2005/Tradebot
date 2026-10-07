"""Causaliteit en survivorship-regels van robuust boek v2 (`systematic/breadth.py`).

Synthetisch universum van twaalf munten: late noteringen, één delisting halverwege,
uiteenlopende liquiditeit. De eigenschappen die hier worden bewezen, moeten op elke
markt gelden: geen besluit tot en met *t* hangt af van data na *t*, het universum is
point-in-time, en een verdwenen munt wordt gesloten in plaats van stil doorgerekend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.schemas.robust_book_v2 import robust_book_v2_config
from tradebot.systematic.book import CostSpec, run_book
from tradebot.systematic.breadth import (
    breadth_from_frames,
    chl_half_spread,
    combine_scaled,
    sleeve_targets,
    universe_mask,
)
from tradebot.utils.failfast import DataContractError

N_BARS = 600
SYMS = [f"C{i:02d}USDT" for i in range(11)] + ["BTCUSDT"]
CUTS = (300, 420)
SLEEVES = ("X1_XSMOM", "X2_XSCARRY", "X3_TREND_LS", "X4_TREND_LF")


def _cfg():
    cfg = robust_book_v2_config()
    # Een klein universum op een kleine markt; de regels zelf blijven die van v2.
    return cfg.model_copy(update={"universe": cfg.universe.model_copy(update={
        "top_n": 8, "min_history_bars": 40, "min_adv_usd": 1.0e6})})


def _frames(seed: int = 3, *, perturb_after: int | None = None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=N_BARS, freq="D", tz="UTC", name="asof_ts")
    common = rng.standard_t(4, size=N_BARS) * 0.02
    rets = np.column_stack([0.7 * common + rng.standard_t(4, size=N_BARS) * 0.03
                            + 0.001 * np.sin(np.arange(N_BARS) / (40 + 7 * i))
                            for i in range(len(SYMS))]).clip(-0.5, 0.5)
    if perturb_after is not None:
        noise = np.random.default_rng(99).normal(0.0, 0.08, size=rets[perturb_after + 1:].shape)
        rets[perturb_after + 1:] = noise
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=SYMS)
    for i, s in enumerate(SYMS[:6]):
        close.loc[close.index[: 20 * i], s] = np.nan     # late noteringen
    close.loc[close.index[450:], "C07USDT"] = np.nan     # delisting op bar 450
    spread = np.abs(rng.normal(0.0, 0.01, size=close.shape)) * close
    high, low = close + spread, close - spread
    qv = pd.DataFrame(np.exp(rng.normal(17.0, 0.6, size=close.shape)), index=idx,
                      columns=SYMS).mul(np.linspace(1.0, 3.0, len(SYMS)), axis=1)
    funding = pd.DataFrame(rng.normal(2e-4, 3e-4, size=close.shape), index=idx, columns=SYMS)
    if perturb_after is not None:
        qv.iloc[perturb_after + 1:] *= np.exp(np.random.default_rng(7).normal(
            0.0, 2.0, size=qv.iloc[perturb_after + 1:].shape))
        funding.iloc[perturb_after + 1:] = np.random.default_rng(8).normal(
            0.0, 5e-3, size=funding.iloc[perturb_after + 1:].shape)
        high.iloc[perturb_after + 1:] = close.iloc[perturb_after + 1:] * 1.3
    listed = close.notna()
    return (close, high.where(listed), low.where(listed), qv.where(listed),
            funding.where(listed))


def _market(perturb_after: int | None = None):
    c, h, lo, qv, f = _frames(perturb_after=perturb_after)
    return breadth_from_frames(c, h, lo, qv, f, _cfg(), source="synthetic")


def _equal_through(a: pd.DataFrame, b: pd.DataFrame, t: int) -> bool:
    x = a.iloc[: t + 1].to_numpy(dtype=float)
    y = b.iloc[: t + 1].to_numpy(dtype=float)
    return bool(np.array_equal(np.isnan(x), np.isnan(y))
                and np.array_equal(np.nan_to_num(x), np.nan_to_num(y)))


@pytest.mark.parametrize("t", CUTS)
def test_the_universe_is_point_in_time(t):
    c, _, _, qv, _ = _frames()
    c2, _, _, qv2, _ = _frames(perturb_after=t)
    a = universe_mask(c, qv, _cfg()).astype(float)
    b = universe_mask(c2, qv2, _cfg()).astype(float)
    assert _equal_through(a, b, t)
    assert not _equal_through(a, b, N_BARS - 1)


@pytest.mark.parametrize("t", CUTS)
def test_the_spread_estimate_is_causal(t):
    c, h, lo, _, _ = _frames()
    c2, h2, lo2, _, _ = _frames(perturb_after=t)
    a = chl_half_spread(c, h, lo, window=60, floor=1e-4, cap=25e-4)
    b = chl_half_spread(c2, h2, lo2, window=60, floor=1e-4, cap=25e-4)
    assert _equal_through(a, b, t)


@pytest.mark.parametrize("t", CUTS)
@pytest.mark.parametrize("name", SLEEVES)
def test_breadth_sleeves_do_not_read_the_future(name, t):
    cfg = _cfg()
    a = sleeve_targets(name, _market(), cfg).weights
    b = sleeve_targets(name, _market(perturb_after=t), cfg).weights
    assert _equal_through(a, b, t)


@pytest.mark.parametrize("t", CUTS)
def test_the_combination_does_not_read_the_future(t):
    cfg = _cfg()
    m, m2 = _market(), _market(perturb_after=t)
    parts = ("X1_XSMOM", "X2_XSCARRY", "X4_TREND_LF")
    a = combine_scaled("X5", [sleeve_targets(s, m, cfg) for s in parts], m, cfg).weights
    b = combine_scaled("X5", [sleeve_targets(s, m2, cfg) for s in parts], m2, cfg).weights
    assert _equal_through(a, b, t)


@pytest.mark.parametrize("name", ("X1_XSMOM", "X2_XSCARRY"))
def test_cross_sectional_sleeves_are_dollar_neutral(name):
    t = sleeve_targets(name, _market(), _cfg())
    w = t.weights[t.rebalance].dropna(how="all")
    w = w[w.abs().sum(axis=1) > 0]
    assert len(w) > 10
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)


def test_weights_live_only_inside_the_universe():
    m = _market()
    for name in SLEEVES:
        t = sleeve_targets(name, m, _cfg())
        outside = t.weights.where(~m.universe).abs().fillna(0.0)
        assert float(outside.to_numpy().max()) == 0.0, name


def test_a_delisted_coin_is_closed_not_carried():
    m = _market()
    b = m.book
    w = pd.DataFrame(0.0, index=b.index, columns=b.symbols)
    live = b.live(40)
    w["C07USDT"] = np.where(live["C07USDT"], 0.2, 0.0)
    w["BTCUSDT"] = np.where(live["BTCUSDT"], 0.2, 0.0)
    first_only = pd.Series(False, index=b.index)
    first_only.iloc[int(np.argmax(live["C07USDT"].to_numpy()))] = True
    costs = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=None, aum_usd=1e6)
    with pytest.raises(DataContractError):
        run_book(w, first_only, b, costs, lag=1)
    res = run_book(w, first_only, b, costs, lag=1, half_spread=m.half_spread,
                   exit_on_missing_price=True)
    assert res.audit["n_forced_exits"] == 1
    assert float(res.held["C07USDT"].iloc[460:].abs().max()) == 0.0
    assert float(res.held["BTCUSDT"].iloc[460]) > 0.0


def test_short_data_holes_are_bridged_and_long_ones_are_not():
    from tradebot.systematic.breadth import bridge_data_holes
    idx = pd.date_range("2022-01-01", periods=40, freq="D", tz="UTC")
    close = pd.DataFrame({"A": np.arange(40, dtype=float) + 100.0,
                          "B": np.arange(40, dtype=float) + 50.0}, index=idx)
    close.iloc[10:13, 0] = np.nan          # gat van 3 dagen: overbrugd
    close.iloc[15:30, 1] = np.nan          # gat van 15 dagen: delisting + herintrede
    out = bridge_data_holes(close, close)
    assert out["A"].iloc[10:13].tolist() == [109.0, 109.0, 109.0]
    assert out["B"].iloc[15:30].isna().all()
    # Vóór de eerste notering en na de laatste wordt niets verzonnen.
    late = close.copy()
    late.iloc[:5, 0] = np.nan
    late.iloc[35:, 0] = np.nan
    assert bridge_data_holes(late, late)["A"].iloc[:5].isna().all()
    assert bridge_data_holes(late, late)["A"].iloc[35:].isna().all()


def test_a_lagged_order_in_a_coin_that_vanished_is_not_executed():
    m = _market()
    b = m.book
    w = pd.DataFrame(0.0, index=b.index, columns=b.symbols)
    live = b.live(40)
    w["C07USDT"] = np.where(live["C07USDT"], 0.1, 0.0)   # besluit tot en met bar 449
    reb = pd.Series(True, index=b.index)
    costs = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=None, aum_usd=1e6)
    res = run_book(w, reb, b, costs, lag=2, half_spread=m.half_spread,
                   exit_on_missing_price=True)
    assert float(res.held["C07USDT"].iloc[452:].abs().max()) == 0.0
