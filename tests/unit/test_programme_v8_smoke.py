"""De v8-orkestratie op de synthetische markt: de kandidaat draait via de gedeelde
orkestratie, de sprongrisicogrens houdt munten boven de grens buiten het boek, en de
preregistratie versoepelt geen enkele v6/v7-poort."""
from __future__ import annotations

import numpy as np
import pytest

from tests.lookahead.test_levered_carry_causality import _mm
from tests.unit.test_programme_v6_smoke import COSTS, PERP, SPOT
from tests.unit.test_programme_v7_smoke import _v7
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.schemas.robust_book_v2 import robust_book_v8_config
from tradebot.systematic import programme_v6 as v6
from tradebot.systematic import programme_v7 as v7
from tradebot.systematic.harvest import BasisCosts
from tradebot.systematic.programme_v8 import CANDIDATES, CONFIG, SPEC_PATH, V8, _family, simulate

ROOT = v6.ROOT
NAME = CANDIDATES[0]


def _v8(cap: float):
    base = _v7()
    cfg = robust_book_v8_config()
    return cfg.model_copy(update={
        "universe": base.universe, "basis": base.basis,
        "leverage": base.leverage.model_copy(update={"candidates": dict(cfg.leverage.candidates)}),
        "v7": base.v7.model_copy(update={"tranche": cfg.v7.tranche}),
        "v8": cfg.v8.model_copy(update={"max_daily_sigma": cap}),
    })


@pytest.fixture(scope="module")
def setup():
    m = _mm()
    sig = np.fmax(m.basis.spot_sigma, m.basis.perp.book.sigma_daily)
    return m, sig, (m.index[200], m.index[590])


def test_the_jump_cap_keeps_the_book_out_of_coins_above_the_cap(setup):
    """De besluitregel zelf is getoetst in `test_levered_carry_causality` (geen doel boven de
    grens). Op boekniveau: de blootstelling aan munt-dagen boven de grens krimpt sterk (wat
    overblijft is de uitstap in vijf stappen nadat een gehouden munt de grens passeert)."""
    m, sig, (a, b) = setup
    cap = float(sig.stack().quantile(0.6))
    above = (sig > cap).to_numpy()
    on = simulate(NAME, m, _v8(cap), COSTS)
    off = simulate(NAME, m, _v8(cap), COSTS, ov={"max_sigma": None})
    exposure = lambda r: float(r.held.shift(-1).fillna(0.0).to_numpy()[above].sum())  # noqa: E731
    assert exposure(off) > 0.0
    assert exposure(on) < 0.5 * exposure(off)
    assert off.window(a, b).frame["spot_notional"].mean() > on.window(a, b).frame["spot_notional"].mean()


def test_without_the_cap_v8_is_v7_s2(setup):
    m, _, (a, b) = setup
    cfg = _v8(0.5)
    base = simulate(NAME, m, cfg, COSTS, ov={"max_sigma": None}).window(a, b)
    s2 = v7.simulate_book(NAME, m, cfg, COSTS).window(a, b)
    assert np.allclose(base.frame["net"], s2.frame["net"])


def test_v8_battery_through_the_shared_orchestration(setup):
    m, sig, (a, b) = setup
    cfg = _v8(float(sig.stack().quantile(0.8)))
    stress = BasisCosts(perp=PERP, spot=SPOT.scaled(multiplier=2.0))
    bat, family = v6.battery(NAME, m, cfg, COSTS, stress, base=simulate(NAME, m, cfg, COSTS),
                             w_dev=(a, b), alt={"same": m}, prog=V8)
    assert {"jump_cap_off", "execution_unsliced", "financing_stress"} <= set(bat)
    assert {"max_sigma_0.15", "max_sigma_0.30", "lever_spread_0.05"} <= set(bat["perturbation"])
    assert len(family) == 1 + len(_family(NAME, cfg))


def test_the_v8_preregistration_keeps_every_gate():
    cfg = robust_book_v8_config(ROOT / CONFIG)
    assert tuple(cfg.leverage.candidates) == CANDIDATES == V8.candidates
    spec = load_preregistration_spec(ROOT / SPEC_PATH, data_hashes=(("x", "0"),),
                                     parameters={"p": 1})
    assert spec.planned_trials == cfg.planned_trials == 1
    old = load_preregistration_spec(ROOT / v7.SPEC_PATH, data_hashes=(("x", "0"),),
                                    parameters={"p": 1})
    assert ({c.name: (c.metric, c.operator, c.threshold, c.action) for c in spec.stop_criteria}
            == {c.name: (c.metric, c.operator, c.threshold, c.action) for c in old.stop_criteria})
