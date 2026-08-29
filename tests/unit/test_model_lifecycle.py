"""De promotie-state-machine en de deur die ernaast openstond — Stage B-4.

Twee dingen worden hier bewaakt.

1. `registry/lifecycle.py` — de bewijsketen `REGISTERED -> TESTED -> CANDIDATE
   -> PAPER -> CHAMPION`. Eenrichtingsverkeer, elke overgang met bewijslast.
2. `registry/promotion.py::promote` — de DEPLOYMENT-as. Die kon een model tot
   `prod` promoveren op `sharpe >= 0.5 AND max_dd <= 0.25 AND n_obs >= 200`:
   drie in-sample-getallen, geen DSR, geen SPA, geen `M`. Dat is fase-no-go 5
   via een tweede deur, en `TestProdCannotBeReachedAroundTheGate` sluit hem.

De belangrijkste tests in dit bestand zijn de weigeringen. Een state machine die
alleen op de gelukkige weg is getoetst, bewijst dat de gelukkige weg bestaat —
niet dat de andere wegen dicht zijn.
"""
from __future__ import annotations

import pytest

from tradebot.registry.catalog import ModelCatalog, ModelRecord
from tradebot.registry.lifecycle import (
    DM_ALPHA,
    FALSIFIED,
    MIN_PAPER_DAYS,
    TRANSITIONS,
    LifecycleState,
    ModelLifecycle,
    TransitionEvidence,
)
from tradebot.registry.promotion import promote
from tradebot.registry.trial_counter import TrialCount
from tradebot.utils.failfast import DataContractError
from tradebot.validation.gates import PROMOTION_GATE_NAMES, GateResult, GateVerdict


def _evidence(**over) -> TransitionEvidence:
    base = dict(
        git_sha="deadbeef", config_hash="1b60cb664fbf9a2a",
        data_hash="d4ta", preregistration_id="prereg-abc",
    )
    base.update(over)
    return TransitionEvidence(**base)          # type: ignore[arg-type]


def _at(state: str) -> ModelLifecycle:
    """Een model dat via de voorgeschreven weg in `state` is beland."""
    lc = ModelLifecycle(model_id="unit-under-test")
    if state == LifecycleState.REGISTERED:
        return lc
    lc.transition(LifecycleState.TESTED,
                  _evidence(lookahead_suite_passed=True))
    if state == LifecycleState.TESTED:
        return lc
    lc.transition(LifecycleState.CANDIDATE,
                  _evidence(gate_result_passed=True,
                            gate_result_verdict=GateVerdict.PROMOTED))
    if state == LifecycleState.CANDIDATE:
        return lc
    lc.transition(LifecycleState.PAPER, _evidence(paper_config_hash="cfg-frozen"))
    if state == LifecycleState.PAPER:
        return lc
    lc.transition(LifecycleState.CHAMPION,
                  _evidence(clean_paper_days=MIN_PAPER_DAYS, dm_p_value=0.01,
                            dm_vs_champion="phase3_baseline"))
    return lc


# --------------------------------------------------------------------------- #
# De keten loopt één kant op
# --------------------------------------------------------------------------- #
class TestTheChainRunsOneWay:
    def test_the_full_forward_path_is_walkable(self) -> None:
        lc = _at(LifecycleState.CHAMPION)
        assert lc.state == LifecycleState.CHAMPION
        assert [h["to"] for h in lc.history] == [
            LifecycleState.TESTED, LifecycleState.CANDIDATE,
            LifecycleState.PAPER, LifecycleState.CHAMPION,
        ]

    @pytest.mark.parametrize(
        ("state", "back"),
        [
            (LifecycleState.TESTED, LifecycleState.REGISTERED),
            (LifecycleState.CANDIDATE, LifecycleState.TESTED),
            (LifecycleState.PAPER, LifecycleState.CANDIDATE),
            (LifecycleState.CHAMPION, LifecycleState.PAPER),
        ],
    )
    def test_going_back_does_not_exist(self, state: str, back: str) -> None:
        """Terug kunnen betekent blijven proberen tot het lukt."""
        lc = _at(state)
        with pytest.raises(DataContractError, match="bestaat niet"):
            lc.transition(back, _evidence())

    def test_skipping_a_state_does_not_exist(self) -> None:
        """De bewijslast is cumulatief; een sprong slaat een bewijsstap over."""
        lc = ModelLifecycle(model_id="skipper")
        with pytest.raises(DataContractError, match="bestaat niet"):
            lc.transition(LifecycleState.CHAMPION,
                          _evidence(clean_paper_days=99, dm_p_value=0.001,
                                    dm_vs_champion="x"))

    def test_an_unknown_state_crashes(self) -> None:
        lc = ModelLifecycle(model_id="x")
        with pytest.raises(DataContractError, match="Onbekende doeltoestand"):
            lc.transition("PRODUCTION", _evidence())

    def test_next_state_names_the_only_way_forward(self) -> None:
        assert _at(LifecycleState.PAPER).next_state == LifecycleState.CHAMPION
        assert _at(LifecycleState.CHAMPION).next_state is None


# --------------------------------------------------------------------------- #
# FALSIFIED
# --------------------------------------------------------------------------- #
class TestFalsification:
    @pytest.mark.parametrize("state", [
        LifecycleState.REGISTERED, LifecycleState.TESTED,
        LifecycleState.CANDIDATE, LifecycleState.PAPER,
        LifecycleState.CHAMPION,
    ])
    def test_a_model_may_always_be_refuted(self, state: str) -> None:
        lc = _at(state)
        lc.falsify("DSR < drempel na correctie voor M", _evidence())
        assert lc.state == FALSIFIED
        assert lc.is_terminal

    def test_falsified_is_final(self) -> None:
        """Er is geen `unfalsify`. Opnieuw aanbieden vereist een NIEUWE
        pre-registratie, en telt dus opnieuw mee in M."""
        lc = _at(LifecycleState.CANDIDATE)
        lc.falsify("weerlegd", _evidence())
        assert TRANSITIONS[FALSIFIED] == frozenset()
        for target in (LifecycleState.REGISTERED, LifecycleState.TESTED,
                       LifecycleState.CANDIDATE, LifecycleState.PAPER,
                       LifecycleState.CHAMPION):
            with pytest.raises(DataContractError, match="bestaat niet"):
                lc.transition(target, _evidence())

    def test_falsification_without_a_reason_crashes(self) -> None:
        lc = _at(LifecycleState.TESTED)
        with pytest.raises(DataContractError, match="FALSIFIED zonder reden"):
            lc.transition(FALSIFIED, _evidence())

    def test_the_history_keeps_the_refutation(self) -> None:
        """Een model dat ooit FALSIFIED was, blijft dat zichtbaar hebben."""
        lc = _at(LifecycleState.PAPER)
        lc.falsify("onverklaarde execution drift", _evidence())
        assert any(h["to"] == FALSIFIED for h in lc.history)
        assert lc.as_dict()["history"][-1]["falsification_reason"]


# --------------------------------------------------------------------------- #
# De bewijslast per overgang
# --------------------------------------------------------------------------- #
class TestEveryForwardStepCostsEvidence:
    def test_tested_requires_a_green_lookahead_suite(self) -> None:
        lc = ModelLifecycle(model_id="x")
        with pytest.raises(DataContractError, match="GROENE"):
            lc.transition(LifecycleState.TESTED,
                          _evidence(lookahead_suite_passed=False))
        with pytest.raises(DataContractError, match="GROENE"):
            lc.transition(LifecycleState.TESTED, _evidence())

    def test_candidate_requires_a_passed_gate_result(self) -> None:
        lc = _at(LifecycleState.TESTED)
        with pytest.raises(DataContractError, match="GESLAAGD"):
            lc.transition(LifecycleState.CANDIDATE,
                          _evidence(gate_result_passed=False,
                                    gate_result_verdict=GateVerdict.FALSIFIED))

    def test_unproven_is_not_a_pass_to_candidate(self) -> None:
        """No-go 13 op de state machine: onbewezen is niet bewezen goed."""
        lc = _at(LifecycleState.TESTED)
        with pytest.raises(DataContractError, match="GESLAAGD"):
            lc.transition(LifecycleState.CANDIDATE,
                          _evidence(gate_result_passed=False,
                                    gate_result_verdict=GateVerdict.UNPROVEN))

    def test_paper_requires_a_config_hash_frozen_before_the_clock(self) -> None:
        """No-go 10: een drempel die achteraf wordt bepaald, is geen drempel."""
        lc = _at(LifecycleState.CANDIDATE)
        with pytest.raises(DataContractError, match="paper_config_hash"):
            lc.transition(LifecycleState.PAPER, _evidence())

    @pytest.mark.parametrize("field", [
        "git_sha", "config_hash", "data_hash", "preregistration_id"])
    def test_missing_provenance_crashes_on_every_transition(
        self, field: str
    ) -> None:
        lc = ModelLifecycle(model_id="x")
        with pytest.raises(DataContractError, match=field):
            lc.transition(LifecycleState.TESTED,
                          _evidence(lookahead_suite_passed=True, **{field: ""}))


# --------------------------------------------------------------------------- #
# PAPER -> CHAMPION: de poort die Stage D consumeert
# --------------------------------------------------------------------------- #
class TestTheChampionChallengerGate:
    """Audit §18.1 regel 3, technisch afgedwongen.

    De twee negatieve controles die exit-criterium D10 met zoveel woorden eist —
    59 dagen en p = 0,06 — staan hier, omdat de poort hier wordt gebouwd en niet
    in Stage D nog eens.
    """

    def test_fifty_nine_days_is_not_sixty(self) -> None:
        lc = _at(LifecycleState.PAPER)
        with pytest.raises(DataContractError, match="OPEENVOLGENDE schone"):
            lc.transition(LifecycleState.CHAMPION,
                          _evidence(clean_paper_days=MIN_PAPER_DAYS - 1,
                                    dm_p_value=0.0001,
                                    dm_vs_champion="phase3_baseline"))
        assert lc.state == LifecycleState.PAPER

    def test_p_equals_six_hundredths_is_not_a_promotion(self) -> None:
        """Ook niet bij een indrukwekkende equity curve."""
        lc = _at(LifecycleState.PAPER)
        with pytest.raises(DataContractError, match="Diebold-Mariano"):
            lc.transition(LifecycleState.CHAMPION,
                          _evidence(clean_paper_days=365, dm_p_value=0.06,
                                    dm_vs_champion="phase3_baseline"))
        assert lc.state == LifecycleState.PAPER

    def test_the_threshold_is_strict_not_inclusive(self) -> None:
        """`p < 0.05`, niet `p <= 0.05`. Op de grens is er geen bewijs."""
        lc = _at(LifecycleState.PAPER)
        with pytest.raises(DataContractError, match="Diebold-Mariano"):
            lc.transition(LifecycleState.CHAMPION,
                          _evidence(clean_paper_days=60, dm_p_value=DM_ALPHA,
                                    dm_vs_champion="phase3_baseline"))

    def test_a_dm_result_without_an_opponent_is_not_a_comparison(self) -> None:
        lc = _at(LifecycleState.PAPER)
        with pytest.raises(DataContractError, match="identiteit van"):
            lc.transition(LifecycleState.CHAMPION,
                          _evidence(clean_paper_days=60, dm_p_value=0.001))

    def test_missing_evidence_is_not_treated_as_zero(self) -> None:
        """`None` mag nooit als 'nog geen bezwaar' worden gelezen."""
        lc = _at(LifecycleState.PAPER)
        with pytest.raises(DataContractError):
            lc.transition(LifecycleState.CHAMPION,
                          _evidence(dm_vs_champion="phase3_baseline"))

    def test_the_gate_does_promote_when_the_evidence_is_there(self) -> None:
        """De weigering is gericht, niet totaal — anders is groen betekenisloos."""
        lc = _at(LifecycleState.PAPER)
        lc.transition(LifecycleState.CHAMPION,
                      _evidence(clean_paper_days=60, dm_p_value=0.049,
                                dm_vs_champion="phase3_baseline"))
        assert lc.state == LifecycleState.CHAMPION


# --------------------------------------------------------------------------- #
# GateResult kan niet met de hand worden vervalst
# --------------------------------------------------------------------------- #
def _gate_result(verdict: str, **gates_over) -> GateResult:
    gates = dict.fromkeys(PROMOTION_GATE_NAMES, True)
    gates.update(gates_over)
    return GateResult(
        model_id="forged", verdict=verdict, gates=gates,
        reasons=dict.fromkeys(PROMOTION_GATE_NAMES, "n.v.t."),
        trial_count=TrialCount(2776, "frozen", "prereg-abc", 2363, 413),
        dsr_result=None, spa_result=None,
        git_sha="deadbeef", config_hash="1b60cb664fbf9a2a",
        data_hash="d4ta", preregistration_id="prereg-abc",
    )


class TestAGateResultCannotBeForged:
    def test_promoted_with_a_failed_gate_crashes(self) -> None:
        """Vier van de vijf met verdict=PROMOTED is niet te onderscheiden van
        een echt oordeel zodra het object eenmaal bestaat."""
        with pytest.raises(DataContractError, match="verdict=PROMOTED terwijl"):
            _gate_result(GateVerdict.PROMOTED, dsr=False)

    def test_an_unknown_verdict_crashes(self) -> None:
        with pytest.raises(DataContractError, match="Onbekend oordeel"):
            _gate_result("LOOKS_FINE")

    def test_a_consistent_refusal_is_constructible(self) -> None:
        r = _gate_result(GateVerdict.FALSIFIED, spa=False)
        assert r.passed is False
        assert r.failed_gates == ("spa",)


# --------------------------------------------------------------------------- #
# De tweede deur naar productie
# --------------------------------------------------------------------------- #
def _staged(catalog: ModelCatalog, symbol: str) -> None:
    """Een model in `staging` met metrics die STAGING_TO_PROD ruim haalt.

    Ruim, opzettelijk: anders zou een test groen kunnen worden doordat de
    legacy-drempels toevallig niet gehaald zijn in plaats van doordat de
    L11-poort weigert.
    """
    catalog.register(ModelRecord(
        symbol=symbol, side="LONG", version=1, stage="staging",
        artifact_path="m.joblib", git_sha="abc", dvc_hash="d", feature_hash="f",
        metrics={"sharpe": 3.0, "max_dd": 0.05, "n_obs": 5000},
        extra={"engine": "event_driven"},
    ))


class TestProdCannotBeReachedAroundTheGate:
    """Stage B-4. Tot deze commit was `prod` bereikbaar op drie in-sample-
    getallen; de vijf poorten stonden ernaast in plaats van ervoor."""

    def test_prod_without_a_gate_result_crashes(self, tmp_path) -> None:
        catalog = ModelCatalog(tmp_path / "catalog.json")
        _staged(catalog, "BTCUSDT")
        with pytest.raises(DataContractError, match="zonder GateResult"):
            promote(catalog, "BTCUSDT", "LONG", "prod")

    def test_prod_with_a_failed_gate_result_crashes(self, tmp_path) -> None:
        catalog = ModelCatalog(tmp_path / "catalog.json")
        _staged(catalog, "ETHUSDT")
        with pytest.raises(DataContractError, match="niet is geslaagd"):
            promote(catalog, "ETHUSDT", "LONG", "prod",
                    gate_result=_gate_result(GateVerdict.FALSIFIED, dsr=False))

    def test_unproven_does_not_reach_prod_either(self, tmp_path) -> None:
        """Onbewezen is niet bewezen goed — ook niet op de deployment-as."""
        catalog = ModelCatalog(tmp_path / "catalog.json")
        _staged(catalog, "SOLUSDT")
        with pytest.raises(DataContractError, match="niet is geslaagd"):
            promote(catalog, "SOLUSDT", "LONG", "prod",
                    gate_result=_gate_result(GateVerdict.UNPROVEN,
                                             dsr=False, spa=False))

    def test_a_passed_gate_result_does_promote(self, tmp_path) -> None:
        """De weigering is gericht. Zonder deze test zou een `promote` die alles
        weigert ook groen zijn."""
        catalog = ModelCatalog(tmp_path / "catalog.json")
        _staged(catalog, "XRPUSDT")
        promoted = promote(catalog, "XRPUSDT", "LONG", "prod",
                           gate_result=_gate_result(GateVerdict.PROMOTED))
        assert promoted is not None
        assert promoted.stage == "prod"

    def test_staging_still_works_without_a_gate_result(self, tmp_path) -> None:
        """`staging` is geen productie. De screening-drempels blijven daar
        gelden, en de L11-poort wordt daar niet geëist."""
        catalog = ModelCatalog(tmp_path / "catalog.json")
        catalog.register(ModelRecord(
            symbol="ADAUSDT", side="LONG", version=1, stage="research",
            artifact_path="m.joblib", git_sha="abc", dvc_hash="d",
            feature_hash="f",
            metrics={"oos_logloss": 0.40, "sharpe": 2.0},
            extra={"engine": "event_driven"},
        ))
        assert promote(catalog, "ADAUSDT", "LONG", "staging") is not None
