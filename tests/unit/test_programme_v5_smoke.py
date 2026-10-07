"""De v5-orkestratie op de synthetische markt: elke kandidaat bouwt, draait en scoort,
en de preregistratie past bij de config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.lookahead.test_basis_causality import _basis
from tests.lookahead.test_breadth_causality import _cfg
from tests.unit.test_programme_v2_smoke import IMPACT
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.schemas.robust_book_v2 import robust_book_v5_config
from tradebot.systematic.book import CostSpec
from tradebot.systematic.evaluate import pbo_record, summarize
from tradebot.systematic.harvest import BasisCosts
from tradebot.systematic.programme_v5 import (
    CANDIDATES,
    CONFIG,
    ROOT,
    SPEC_PATH,
    _family,
    battery,
    simulate,
)

PERP = CostSpec(taker_fee=5.5e-4, half_spread=3e-4, impact=IMPACT, aum_usd=1e6, impact_eta=1.0)
SPOT = CostSpec(taker_fee=10e-4, half_spread=5e-4, impact=IMPACT, aum_usd=1e6, impact_eta=1.0)
COSTS = BasisCosts(perp=PERP, spot=SPOT)


def _v5():
    cfg = robust_book_v5_config()
    small = _cfg().universe
    # Synthetische funding is ~7 %/jaar: lagere drempels, zodat er iets gehouden wordt.
    return cfg.model_copy(update={
        "universe": small,
        "basis": cfg.basis.model_copy(update={"enter_apr": 0.05, "exit_apr": 0.0,
                                              "min_spot_adv_usd": 1e5}),
    })


@pytest.fixture(scope="module")
def setup():
    m = _basis()
    return m, _v5(), (m.index[200], m.index[590])


@pytest.mark.parametrize("name", CANDIDATES)
def test_every_v5_candidate_builds_runs_and_holds_something(setup, name):
    m, cfg, (a, b) = setup
    res = simulate(name, m, cfg, COSTS).window(a, b)
    s = summarize(res)
    assert np.isfinite(s["sharpe"])
    assert s["avg_gross_leverage"] > 0.0
    if name != "H2_BASIS_TREND":
        # Long spot, short perp: netto exposure nul (op de basis na).
        assert abs(s["avg_net_leverage"]) < 0.01
        assert s["max_gross_leverage"] <= 2 * cfg.basis.notional * (1 + cfg.basis.band) + 1e-9


def test_overrides_change_the_book(setup):
    m, cfg, (a, b) = setup
    base = simulate("H1_BASIS", m, cfg, COSTS).window(a, b).net
    for ov in ({"slots": 2}, {"every": 7}, {"missing_seed": 1}, {"noise_seed": 1}):
        other = simulate("H1_BASIS", m, cfg, COSTS, ov=ov).window(a, b).net
        assert not np.allclose(base, other), ov


def test_unknown_candidate_is_refused(setup):
    m, cfg, _ = setup
    from tradebot.utils.failfast import DataContractError
    with pytest.raises(DataContractError):
        simulate("H9_NOPE", m, cfg, COSTS)


@pytest.mark.parametrize("name", CANDIDATES)
def test_v5_battery_covers_every_check(setup, name):
    m, cfg, (a, b) = setup
    base = simulate(name, m, cfg, COSTS)
    stress = BasisCosts(perp=PERP, spot=SPOT.scaled(multiplier=2.0))
    bat, family = battery(name, m, cfg, COSTS, stress, base=base, w_dev=(a, b),
                          alt={"same": m})
    for key in ("perturbation", "costs", "capacity", "delay", "signal_noise", "missing_data",
                "rebalance", "universe", "start_dates", "end_dates", "monte_carlo_drawdown"):
        assert key in bat, key
    assert set(bat["perturbation"]) == set(_family(name, cfg))
    assert 0.0 <= pbo_record(family)["pbo"] <= 1.0


def test_the_preregistration_matches_the_config():
    cfg = robust_book_v5_config(ROOT / CONFIG)
    spec = load_preregistration_spec(ROOT / SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"planned": cfg.planned_trials})
    assert spec.planned_trials == cfg.planned_trials
    names = {c.name for c in spec.stop_criteria}
    assert {"backcast_sharpe_collapse", "holdout_sharpe_collapse",
            "promotion_requires_all_clear"} <= names
    assert Path(ROOT / CONFIG).is_file()
