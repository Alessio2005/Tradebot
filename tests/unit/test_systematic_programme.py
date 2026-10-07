"""De orkestratie van het robuuste boek, op een synthetische markt.

Bewijst dat elke trial te bouwen en te draaien is, dat de batterij elke gevoeligheid
oplevert die de preregistratie noemt, en dat de score binnen 0..100 valt. Er wordt
hier niets over echte data gezegd.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.systematic_fixtures import config, synthetic_market
from tradebot.systematic.book import CostSpec
from tradebot.systematic.evaluate import pbo_record, robustness_score, summarize
from tradebot.systematic.programme import (
    CANDIDATES,
    REFERENCES,
    SLEEVES,
    _every,
    battery,
    build_trial,
    perturbation_family,
    run_targets,
)

COSTS = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=None, aum_usd=1e5)


@pytest.fixture(scope="module")
def setup():
    m = synthetic_market(n_bars=800)
    return m, config(), (m.index[250], m.index[780])


@pytest.mark.parametrize("name", REFERENCES + CANDIDATES)
def test_every_trial_builds_and_runs(setup, name):
    m, cfg, (a, b) = setup
    t = build_trial(name, m, cfg, included=SLEEVES)
    res = run_targets(t, m, COSTS, lag=1).window(a, b)
    s = summarize(res)
    assert np.isfinite(s["sharpe"]) and np.isfinite(s["max_drawdown"])
    assert s["max_gross_leverage"] <= cfg.sizing.gross_cap + 1e-9 or name in REFERENCES
    held = run_targets(t, m, COSTS, lag=1).held
    assert held.abs().max().max() <= cfg.sizing.per_asset_cap + 1e-6


def test_carry_is_dollar_neutral_on_its_decisions(setup):
    m, cfg, _ = setup
    t = build_trial("C4_CARRY_XS", m, cfg)
    w = t.weights[t.rebalance].dropna(how="all")
    w = w[w.abs().sum(axis=1) > 0]
    assert len(w) > 10
    np.testing.assert_allclose(w.sum(axis=1), 0.0, atol=1e-12)


def test_rebalancing_less_often_trades_less(setup):
    m, cfg, (a, b) = setup
    t = build_trial("C1_TREND_LS", m, cfg)
    daily = run_targets(t, m, COSTS, lag=1).window(a, b).frame["turnover"].sum()
    weekly = run_targets(_every(t, 7), m, COSTS, lag=1).window(a, b).frame["turnover"].sum()
    assert weekly < daily


def test_the_perturbation_family_matches_the_preregistration(setup):
    _, cfg, _ = setup
    trend = perturbation_family("C1_TREND_LS", cfg, ())
    assert {"lookbacks_x0.5", "lookbacks_x2.0", "vol_span_20", "max_scale_3.0"} <= set(trend)
    core = perturbation_family("C3_VOLMAN_CORE", cfg, ())
    assert not any(k.startswith("lookbacks") or k.startswith("carry") for k in core)
    combo = perturbation_family("C5_COMBO", cfg, ("C1_TREND_LS", "C4_CARRY_XS"))
    assert any(k.startswith("lookbacks") for k in combo) and any(k.startswith("carry") for k in combo)


def test_battery_delivers_every_sensitivity_and_a_bounded_score(setup):
    m, cfg, (a, b) = setup
    name = "C1_TREND_LS"
    base = run_targets(build_trial(name, m, cfg), m, COSTS, lag=1)
    bat, family = battery(name, m, cfg, COSTS, included=(), base=base, w_dev=(a, b))
    for key in ("perturbation", "costs", "capacity", "delay", "signal_noise", "missing_data",
                "rebalance", "universe", "start_dates", "end_dates", "monte_carlo_drawdown"):
        assert key in bat, key
    assert set(bat["universe"]) >= {"btc_eth_only", "without_BTCUSDT"}
    pbo = pbo_record(family)
    assert 0.0 <= pbo["pbo"] <= 1.0
    dev = summarize(base.window(a, b))
    score = robustness_score({
        "sharpe": dev["sharpe"], "sharpe_train": dev["sharpe"], "sharpe_validate": dev["sharpe"],
        "sharpe_2x_cost": bat["costs"]["x2"], "sharpe_lag2": bat["delay"]["lag_2"],
        "plateau_fraction": bat["plateau_fraction"], "p_sharpe_gt_0": dev["p_sharpe_gt_0"],
        "max_drawdown": dev["max_drawdown"], "pbo": pbo["pbo"],
        "leave_one_out_positive_fraction": bat["leave_one_out_positive_fraction"]})
    assert 0.0 <= score["total"] <= 100.0


def test_the_score_rewards_nothing_for_a_negative_sharpe():
    s = robustness_score({"sharpe": -0.3, "sharpe_train": 1.0, "sharpe_validate": 1.0,
                          "sharpe_2x_cost": 1.0, "sharpe_lag2": 1.0, "plateau_fraction": 0.0,
                          "p_sharpe_gt_0": 0.3, "max_drawdown": 0.5, "pbo": 1.0,
                          "leave_one_out_positive_fraction": 0.0})
    assert s["total"] == pytest.approx(0.0)


def test_window_is_inclusive_and_does_not_recompute(setup):
    m, cfg, (a, b) = setup
    res = run_targets(build_trial("C3_VOLMAN_CORE", m, cfg), m, COSTS, lag=1)
    w = res.window(a, b)
    assert w.frame.index[0] == a and w.frame.index[-1] == b
    pd.testing.assert_frame_equal(w.frame, res.frame.loc[a:b])
