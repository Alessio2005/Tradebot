"""De boekhouding van `systematic/book.py`: wat elke run moet kloppen, op elke markt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.systematic_fixtures import SYMBOLS, synthetic_market
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.systematic.book import CostSpec, run_book
from tradebot.utils.failfast import DataContractError

FREE = CostSpec(taker_fee=0.0, half_spread=0.0, impact=None, aum_usd=1e5)


def _daily(market):
    return pd.Series(True, index=market.index)


def _target(market, weights: dict[str, float]):
    w = pd.DataFrame(0.0, index=market.index, columns=list(SYMBOLS))
    for s, v in weights.items():
        w[s] = v
    return w.where(market.close.notna(), 0.0)


def _impact():
    return ImpactParams(eta=1.0, kappa_d=0.5, status=ImpactStatus.IMPACT_UNCALIBRATED,
                        method="test", data_hash="0" * 16, sample_size=1,
                        period_start="2021-01-01", period_end="2021-12-31",
                        instruments=("BTCUSDT",), eta_ci_low=0.5, eta_ci_high=2.0)


def test_without_costs_net_is_price_plus_funding():
    m = synthetic_market()
    res = run_book(_target(m, {"BTCUSDT": 0.6, "ETHUSDT": -0.4}), _daily(m), m, FREE, lag=1)
    f = res.frame
    np.testing.assert_allclose(f["net"], f["gross"] + f["funding"], atol=1e-15)
    assert (f[["fees", "spread", "slippage", "impact"]] == 0.0).all().all()


def test_single_asset_hold_compounds_to_the_price_ratio():
    m = synthetic_market()
    m = type(m)(close=m.close, ret=m.ret, funding=m.funding * 0.0, adv_usd=m.adv_usd,
                sigma_daily=m.sigma_daily, source_hashes={})
    target = _target(m, {"BTCUSDT": 1.0})
    first_only = pd.Series(False, index=m.index)
    first_only.iloc[0] = True
    res = run_book(target, first_only, m, FREE, lag=1)
    equity = float((1.0 + res.net).prod())
    ratio = float(m.close["BTCUSDT"].iloc[-1] / m.close["BTCUSDT"].iloc[0])
    assert equity == pytest.approx(ratio, rel=1e-10)
    # Eén aankoop, daarna geen enkele trade: een buy-and-hold handelt niet bij.
    assert res.frame["turnover"].iloc[1:].abs().max() == 0.0


def test_turnover_counts_the_drift_back_to_a_constant_weight():
    m = synthetic_market(late_listing=False)
    res = run_book(_target(m, {"BTCUSDT": 0.5, "ETHUSDT": 0.5}), _daily(m), m, FREE, lag=1)
    r = m.ret.iloc[5][["BTCUSDT", "ETHUSDT"]].to_numpy()
    port = 0.5 * r.sum() - float(0.5 * m.funding.iloc[5][["BTCUSDT", "ETHUSDT"]].sum())
    drifted = 0.5 * (1.0 + r) / (1.0 + port)
    expected = np.abs(0.5 - drifted).sum()
    assert res.frame["turnover"].iloc[5] == pytest.approx(expected, rel=1e-12)
    assert expected > 0.0


def test_a_long_pays_positive_funding():
    m = synthetic_market(late_listing=False)
    res = run_book(_target(m, {"BTCUSDT": 1.0}), _daily(m), m, FREE, lag=1)
    t = 10
    assert res.frame["funding"].iloc[t] == pytest.approx(
        -float(res.held["BTCUSDT"].iloc[t] * m.funding["BTCUSDT"].iloc[t]))


def test_lag_two_is_lag_one_on_the_previous_decision():
    m = synthetic_market()
    rng = np.random.default_rng(3)
    raw = pd.DataFrame(rng.normal(0, 0.3, size=m.close.shape), index=m.index,
                       columns=list(SYMBOLS)).where(m.close.notna(), 0.0)
    shifted = raw.shift(1).where(m.close.notna(), 0.0).fillna(0.0)
    a = run_book(raw, _daily(m), m, FREE, lag=2)
    b = run_book(shifted, _daily(m), m, FREE, lag=1)
    np.testing.assert_allclose(a.net.iloc[2:], b.net.iloc[2:], atol=1e-14)


def test_costs_are_charged_on_traded_notional():
    m = synthetic_market(late_listing=False)
    costs = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=None, aum_usd=1e5)
    res = run_book(_target(m, {"BTCUSDT": 1.0}), _daily(m), m, costs, lag=1)
    f = res.frame
    scale = 1.0 + f["gross"] + f["funding"]
    np.testing.assert_allclose(-f["fees"], 5.5e-4 * f["turnover"] * scale, rtol=1e-12)
    np.testing.assert_allclose(-f["spread"], 1e-4 * f["turnover"] * scale, rtol=1e-12)


def test_impact_grows_with_the_square_root_of_book_size():
    m = synthetic_market(late_listing=False)
    target = _target(m, {"BTCUSDT": 1.0, "ETHUSDT": -0.5})
    target.iloc[:35] = 0.0  # pas handelen als de 30-daagse ADV bestaat
    small = run_book(target, _daily(m), m, CostSpec(0.0, 0.0, _impact(), 1e6), lag=1)
    big = run_book(target, _daily(m), m, CostSpec(0.0, 0.0, _impact(), 4e6), lag=1)
    t = 40  # ruim na de ADV-opwarming
    assert big.frame["impact"].iloc[t] == pytest.approx(2.0 * small.frame["impact"].iloc[t], rel=1e-2)


def test_a_target_in_an_unlisted_symbol_crashes():
    m = synthetic_market()
    target = _target(m, {"BTCUSDT": 0.5})
    target["DOTUSDT"] = 0.1  # DOT noteert pas na 150 bars
    with pytest.raises(DataContractError):
        run_book(target, _daily(m), m, FREE, lag=1)


def test_lag_zero_is_refused():
    m = synthetic_market()
    with pytest.raises(DataContractError):
        run_book(_target(m, {"BTCUSDT": 1.0}), _daily(m), m, FREE, lag=0)


def test_partial_adjustment_closes_a_fixed_fraction_of_the_gap():
    m = synthetic_market(late_listing=False)
    target = _target(m, {"BTCUSDT": 1.0})
    res = run_book(target, _daily(m), m, FREE, lag=1, trade_rate=0.25)
    assert res.held["BTCUSDT"].iloc[1] == pytest.approx(0.25)
    full = run_book(target, _daily(m), m, FREE, lag=1)
    assert res.frame["turnover"].sum() < full.frame["turnover"].sum()


def test_trade_rate_outside_the_unit_interval_is_refused():
    m = synthetic_market()
    with pytest.raises(DataContractError):
        run_book(_target(m, {"BTCUSDT": 1.0}), _daily(m), m, FREE, lag=1, trade_rate=0.0)
