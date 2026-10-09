"""De v7-orkestratie op de synthetische markt: beide kandidaten bouwen en draaien via de
gedeelde v6-orkestratie; de spreiding en de tranche doen wat ze beloven; de preregistratie
past bij de config en versoepelt geen enkele v6-poort."""
from __future__ import annotations

import numpy as np
import pytest

from tests.lookahead.test_breadth_causality import _cfg
from tests.lookahead.test_levered_carry_causality import _mm
from tests.unit.test_programme_v6_smoke import COSTS, PERP, SPOT
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.schemas.robust_book_v2 import robust_book_v7_config
from tradebot.systematic import programme_v6 as v6
from tradebot.systematic.harvest import BasisCosts
from tradebot.systematic.programme_v7 import CANDIDATES, CONFIG, SPEC_PATH, V7, _family, simulate
from tradebot.utils.failfast import DataContractError

ROOT = v6.ROOT


def _v7():
    cfg = robust_book_v7_config()
    return cfg.model_copy(update={
        "universe": _cfg().universe,
        "basis": cfg.basis.model_copy(update={"enter_apr": 0.05, "exit_apr": 0.0,
                                              "min_spot_adv_usd": 1e5}),
        "leverage": cfg.leverage.model_copy(update={"majors": ("BTCUSDT",),
                                                    "financing_floor_apr": 0.02}),
        "v7": cfg.v7.model_copy(update={"lever_spread_apr": 0.02}),
    })


@pytest.fixture(scope="module")
def setup():
    m = _mm()
    return m, _v7(), (m.index[200], m.index[590])


@pytest.mark.parametrize("name", CANDIDATES)
def test_every_v7_candidate_builds_runs_and_stays_inside_the_mandate(setup, name):
    m, cfg, (a, b) = setup
    res = simulate(name, m, cfg, COSTS).window(a, b)
    s = v6.summary(res, bootstrap=False)
    assert np.isfinite(s["sharpe"]) and s["avg_gross_leverage"] > 0.0
    assert abs(s["avg_net_leverage"]) < 0.01
    traded = res.frame["turnover"] > 0
    assert (res.frame.loc[traded, "gross_leverage"] <= cfg.leverage.gross_cap + 1e-9).all()
    step = (1.0 / cfg.basis.slots) / cfg.v7.slice_days
    # Elke munt hooguit één stap per dag, behalve een gedwongen sluiting (koers weg).
    listed = (m.basis.perp.book.close.notna() & m.basis.spot_close.notna()).loc[a:b]
    traded_legs = (res.trades / 2.0).where(listed, 0.0)
    assert traded_legs.to_numpy().max() <= step + 1e-9, "elke munt hooguit een stap per dag"


def test_the_tranche_adds_leverage_only_on_top_of_the_base(setup):
    m, cfg, (a, b) = setup
    s1 = simulate("S1_CARRY_PM_SLICED", m, cfg, COSTS).window(a, b)
    s2 = simulate("S2_CARRY_PM_TRANCHE", m, cfg, COSTS).window(a, b)
    assert s2.frame["spot_notional"].mean() > s1.frame["spot_notional"].mean()
    assert s2.frame["spot_notional"].max() <= 2.0 * (1 + cfg.basis.band) + 1e-9


def test_a_dearer_loan_never_buys_more_tranche(setup):
    m, cfg, (a, b) = setup
    cheap = simulate("S2_CARRY_PM_TRANCHE", m, cfg, COSTS).window(a, b)
    dear = simulate("S2_CARRY_PM_TRANCHE", m, cfg, COSTS,
                    ov={"financing": "stress"}).window(a, b)
    assert dear.frame["spot_notional"].mean() <= cheap.frame["spot_notional"].mean() + 1e-9


def test_unsliced_override_turns_slicing_off(setup):
    m, cfg, (a, b) = setup
    whole = simulate("S1_CARRY_PM_SLICED", m, cfg, COSTS, ov={"slice_days": 1}).window(a, b)
    step = (1.0 / cfg.basis.slots) / cfg.v7.slice_days
    assert (whole.trades.to_numpy() / 2.0).max() > step


def test_the_family_adds_the_new_parameters(setup):
    _, cfg, _ = setup
    assert set(_family("S1_CARRY_PM_SLICED", cfg)) == set(v6._family()) | {"slice_days_3",
                                                                          "slice_days_10"}
    assert {"lever_spread_0.05", "lever_spread_0.15"} <= set(_family("S2_CARRY_PM_TRANCHE", cfg))


def test_v7_battery_through_the_shared_orchestration(setup):
    m, cfg, (a, b) = setup
    name = "S2_CARRY_PM_TRANCHE"
    base = simulate(name, m, cfg, COSTS)
    stress = BasisCosts(perp=PERP, spot=SPOT.scaled(multiplier=2.0))
    bat, family = v6.battery(name, m, cfg, COSTS, stress, base=base, w_dev=(a, b),
                             alt={"same": m}, prog=V7)
    assert "execution_unsliced" in bat and "financing_stress" in bat
    assert set(bat["perturbation"]) == set(_family(name, cfg))
    assert len(family) == 1 + len(_family(name, cfg))


def test_unknown_candidate_is_refused(setup):
    m, cfg, _ = setup
    with pytest.raises(DataContractError):
        simulate("K1_CARRY_PM_1X", m, cfg, COSTS)


def test_the_v7_preregistration_keeps_every_v6_gate():
    cfg = robust_book_v7_config(ROOT / CONFIG)
    assert tuple(cfg.leverage.candidates) == CANDIDATES == V7.candidates
    spec = load_preregistration_spec(ROOT / SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"p": 1})
    assert spec.planned_trials == cfg.planned_trials == len(CANDIDATES)
    old = load_preregistration_spec(ROOT / v6.SPEC_PATH, data_hashes=(("x", "0"),),
                                    parameters={"p": 1})
    new = {c.name: (c.metric, c.operator, c.threshold, c.action) for c in spec.stop_criteria}
    assert new == {c.name: (c.metric, c.operator, c.threshold, c.action)
                   for c in old.stop_criteria}
