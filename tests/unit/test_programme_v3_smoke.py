"""v3 op een synthetisch breed universum: dagelijks doorgetrokken doelen, kappa, batterij."""
from __future__ import annotations

import numpy as np
import pytest

from tests.lookahead.test_breadth_causality import _cfg, _market
from tests.unit.test_programme_v2_smoke import COSTS
from tradebot.schemas.robust_book_v2 import RobustBookV3Config, robust_book_v3_config
from tradebot.systematic import programme_v2 as v2
from tradebot.systematic.evaluate import summarize
from tradebot.systematic.programme_v3 import CANDIDATES, SLEEVE_OF, build, make_runner


def _cfg3() -> RobustBookV3Config:
    base = _cfg()
    v3 = robust_book_v3_config()
    return RobustBookV3Config(**{**base.model_dump(), "v3": v3.v3.model_dump(),
                                 "planned_trials": 3})


@pytest.fixture(scope="module")
def setup():
    m = _market()
    return m, _cfg3(), (m.book.index[200], m.book.index[590])


@pytest.mark.parametrize("name", CANDIDATES)
def test_v3_targets_are_daily_and_partial_adjustment_trades_less(setup, name):
    m, cfg, (a, b) = setup
    t = build(name, m, cfg)
    started = t.rebalance[t.rebalance]
    assert bool(t.rebalance.loc[started.index[0]:].all())
    runner = make_runner(cfg)
    slow = runner(t, m, COSTS, lag=1).window(a, b)
    fast = runner(t, m, COSTS, lag=1, trade_rate=1.0).window(a, b)
    assert slow.frame["turnover"].sum() < fast.frame["turnover"].sum()
    assert np.isfinite(summarize(slow)["sharpe"])


def test_v3_battery_carries_kappa_and_spread_stress(setup):
    m, cfg, (a, b) = setup
    name = "Y5_COMBO"
    runner = make_runner(cfg)
    base = runner(build(name, m, cfg), m, COSTS, lag=1)
    sleeve, inc = SLEEVE_OF[name]
    bat, family = v2.battery(name, m, cfg, COSTS, included=inc, base=base, w_dev=(a, b),
                             alt_universes={"same": m}, builder=build, runner=runner,
                             family_name=sleeve, run_family={"trade_rate_1.0": {"trade_rate": 1.0}},
                             extra_costs={"half_spread_stress": COSTS})
    assert "trade_rate_1.0" in bat["perturbation"] and "trade_rate_1.0" in family
    assert "half_spread_stress" in bat["costs"]
    assert any(k.startswith("carry_window") for k in bat["perturbation"])
