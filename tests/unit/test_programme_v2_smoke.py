"""De v2-orkestratie op een synthetisch breed universum: alles bouwt, draait en scoort."""
from __future__ import annotations

import numpy as np
import pytest

from tests.lookahead.test_breadth_causality import _cfg, _market
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.systematic.book import CostSpec
from tradebot.systematic.evaluate import pbo_record, robustness_score, summarize
from tradebot.systematic.programme_v2 import CANDIDATES, REFERENCES, battery, build, regimes_v2, run

IMPACT = ImpactParams(eta=2.991922, kappa_d=0.5, status=ImpactStatus.IMPACT_UNCALIBRATED,
                      method="test", data_hash="0" * 16, sample_size=1,
                      period_start="2021-01-01", period_end="2021-12-31",
                      instruments=("BTCUSDT",), eta_ci_low=1.9, eta_ci_high=4.3)
# Zoals productie: de repo-impactparameters met de literatuur-Y als override.
COSTS = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=IMPACT, aum_usd=1e6, impact_eta=1.0)


@pytest.fixture(scope="module")
def setup():
    m = _market()
    return m, _cfg(), (m.book.index[200], m.book.index[590])


@pytest.mark.parametrize("name", REFERENCES + CANDIDATES)
def test_every_v2_trial_builds_and_runs(setup, name):
    m, cfg, (a, b) = setup
    t = build(name, m, cfg, included=("X1_XSMOM", "X2_XSCARRY", "X4_TREND_LF"))
    res = run(t, m, COSTS, lag=1).window(a, b)
    s = summarize(res)
    assert np.isfinite(s["sharpe"])
    if name not in REFERENCES:
        assert s["max_gross_leverage"] <= cfg.sizing.gross_cap + 1e-9
    assert res.held.abs().max().max() <= max(cfg.sizing.per_asset_cap, 1.0) + 1e-6
    assert set(regimes_v2(res, m)) == {"bull", "bear", "high_vol", "low_vol", "high_corr", "low_corr"}


def test_v2_battery_and_score(setup):
    m, cfg, (a, b) = setup
    name = "X1_XSMOM"
    base = run(build(name, m, cfg), m, COSTS, lag=1)
    bat, family = battery(name, m, cfg, COSTS, included=(), base=base, w_dev=(a, b),
                          alt_universes={"same": m})
    for key in ("perturbation", "costs", "capacity", "delay", "signal_noise", "missing_data",
                "rebalance", "universe", "start_dates", "end_dates", "monte_carlo_drawdown"):
        assert key in bat, key
    assert 0.0 <= pbo_record(family)["pbo"] <= 1.0
    dev = summarize(base.window(a, b))
    score = robustness_score({
        "sharpe": dev["sharpe"], "sharpe_train": dev["sharpe"], "sharpe_validate": dev["sharpe"],
        "sharpe_2x_cost": bat["costs"]["x2"], "sharpe_lag2": bat["delay"]["lag_2"],
        "plateau_fraction": bat["plateau_fraction"], "p_sharpe_gt_0": dev["p_sharpe_gt_0"],
        "max_drawdown": dev["max_drawdown"], "pbo": 0.5,
        "leave_one_out_positive_fraction": bat["universe_positive_fraction"]})
    assert 0.0 <= score["total"] <= 100.0
