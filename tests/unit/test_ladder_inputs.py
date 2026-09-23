# tests/unit/test_ladder_inputs.py
"""De invoer van de vier-lagen-ladder, op de gecertificeerde store. Fase 11, stap 3.

`backtest/ladder_inputs.py` is verhuisde code: `load_market` en de assemblage
uit `main()` van `apps/run_phase5_baseline.py`. Dat de verhuizing
gedragsneutraal is, is bij de verhuizing gemeten (alle argumenten van
`run_all_layers` vóór en na gevangen en vergeleken: identiek). Deze tests leggen
de grootheden vast waar de ladder op rust, zodat een latere wijziging aan deze
module niet stil een ander paneel oplevert: het venster, het aantal bars, de
vier tracks, en dat het beleid dat een aanroeper meegeeft het beleid is dat
beslist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.backtest.ladder_inputs import (
    LadderInputs,
    load_ladder_inputs,
    run_ladder_track,
)
from tradebot.risk.binding_audit import TracingRiskEngine, policy_from_registry
from tradebot.risk.engine import risk_config_hash

LAPSED = "1b60cb664fbf9a2a"


@pytest.fixture(scope="module")
def inputs() -> LadderInputs:
    return load_ladder_inputs(ROOT)


def test_the_window_is_the_certified_one(inputs: LadderInputs) -> None:
    """1.743 bars, 2021-11-15 t/m 2026-08-23: het venster van elke ladderrun
    sinds fase 10 (`phase5_revaluation.json`: `n_bars`, `period`)."""
    assert len(inputs.usable) == 1743
    assert str(inputs.usable[0].date()) == "2021-11-15"
    assert str(inputs.usable[-1].date()) == "2026-08-23"


def test_every_usable_bar_has_a_volatility_and_an_adv(inputs: LadderInputs) -> None:
    """De soevereine laag weigert een bar zonder `sigma_hat` of ADV. `usable`
    is precies de doorsnede waarop beide bestaan."""
    market = inputs.market
    assert not market["sigma"].loc[inputs.usable].isna().any().any()
    assert not market["adv"].loc[inputs.usable].isna().any().any()


def test_the_four_tracks(inputs: LadderInputs) -> None:
    assert sorted(inputs.tracks) == [
        "long_only_equal_weight", "long_only_risk_parity",
        "xs_momentum_equal_weight", "xs_momentum_risk_parity"]


def test_the_policy_passed_in_is_the_policy_that_decides(
    inputs: LadderInputs,
) -> None:
    """Onder het VERVALLEN beleid, zodat een module die stil terugvalt op
    `conf/risk/default.yaml` hier rood wordt."""
    cfg = policy_from_registry(LAPSED)
    seen: dict[str, str] = {}

    def factory(layer: str, risk_cfg: object) -> TracingRiskEngine:
        engine = TracingRiskEngine(risk_cfg)  # type: ignore[arg-type]
        seen[layer] = engine.config_hash
        return engine

    rows = run_ladder_track(inputs, "long_only_equal_weight", cfg,
                            engine_factory=factory)
    assert seen == {"L1_sovereign": LAPSED, "L3_execution": LAPSED}
    l3 = next(r for r in rows if r.layer == "L3_execution")
    assert l3.audit["risk_policy_hash"] == risk_config_hash(cfg) == LAPSED
