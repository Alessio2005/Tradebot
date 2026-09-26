"""Het pre-registratiecontract — Phase 3, stap 1.

Een governance-gate die nooit is aangetoond, is `model_risk_policy.md` opnieuw:
papier zonder tanden (audit bevinding D-1). Deze tests bewijzen dat de gate
daadwerkelijk sluit. Zij draaien op een synthetische fixture: sinds de herstart
van 2026-09-26 staat er geen bevroren pre-registratie in de repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tradebot.registry.preregistration import (
    PreRegistration,
    StopCriterion,
    freeze_preregistration,
    load_preregistration_spec,
    require_preregistration,
)
from tradebot.utils.failfast import ConfigContractError, DataContractError

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "tests" / "fixtures" / "preregistration_example.yaml"
GOVERNANCE = ROOT / "artefacts" / "governance"

GIT_SHA = "abc1234"
LEDGER_TOTAL = 2711
DATA_HASHES = (("crypto/ohlcv/BTCUSDT/1d", "0" * 32),)
PARAMETERS = {"alpha": {"lookback_bars": 60}}


def _spec(tmp_path: Path, **overrides: object) -> Path:
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    raw["preregistration"].update(overrides)
    out = tmp_path / "spec.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def _load(path: Path) -> PreRegistration:
    return load_preregistration_spec(
        path, data_hashes=DATA_HASHES, parameters=PARAMETERS
    )


class TestTheGateCloses:
    def test_missing_id_crashes(self) -> None:
        with pytest.raises(DataContractError, match="zonder pre-registratie-ID"):
            require_preregistration("", directory=GOVERNANCE)

    def test_unknown_id_crashes(self, tmp_path: Path) -> None:
        with pytest.raises(DataContractError, match="Onbekende pre-registratie-ID"):
            require_preregistration("f" * 32, directory=tmp_path)

    def test_tampered_artefact_crashes(self, tmp_path: Path) -> None:
        """Na het bevriezen gewijzigde inhoud hasht niet meer naar zijn eigen ID."""
        prereg = _load(SPEC)
        path = freeze_preregistration(
            prereg, git_sha=GIT_SHA, ledger_total_at_freeze=LEDGER_TOTAL,
            directory=tmp_path,
        )
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["content"]["planned_trials"] = 999
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(DataContractError, match="hasht niet naar zijn eigen ID"):
            require_preregistration(prereg.preregistration_id, directory=tmp_path)

    def test_freeze_is_idempotent_but_not_overwritable(self, tmp_path: Path) -> None:
        prereg = _load(SPEC)
        first = freeze_preregistration(
            prereg, git_sha=GIT_SHA, ledger_total_at_freeze=LEDGER_TOTAL,
            directory=tmp_path)
        second = freeze_preregistration(
            prereg, git_sha="other99", ledger_total_at_freeze=LEDGER_TOTAL + 1,
            directory=tmp_path)
        assert first == second
        stored = json.loads(first.read_text(encoding="utf-8"))
        assert stored["git_sha"] == GIT_SHA, "een tweede freeze mag niets herschrijven"

    def test_freeze_without_git_sha_or_ledger_total_crashes(self, tmp_path: Path) -> None:
        prereg = _load(SPEC)
        with pytest.raises(DataContractError, match="git_sha"):
            freeze_preregistration(prereg, git_sha="",
                                   ledger_total_at_freeze=LEDGER_TOTAL,
                                   directory=tmp_path)
        with pytest.raises(DataContractError, match="hypothese-ledger"):
            freeze_preregistration(prereg, git_sha=GIT_SHA,
                                   ledger_total_at_freeze=0, directory=tmp_path)


class TestTheSpecIsStrict:
    def test_unknown_key_crashes(self, tmp_path: Path) -> None:
        path = _spec(tmp_path, definitely_not_a_real_key=123)
        with pytest.raises(ConfigContractError, match="Onbekende sleutel"):
            _load(path)

    def test_missing_stop_criteria_crashes(self, tmp_path: Path) -> None:
        path = _spec(tmp_path, stop_criteria=[])
        with pytest.raises(ConfigContractError, match="stop-criteria"):
            _load(path)

    def test_zero_planned_trials_crashes(self, tmp_path: Path) -> None:
        path = _spec(tmp_path, planned_trials=0)
        with pytest.raises(ConfigContractError, match="nul geplande trials"):
            _load(path)

    def test_no_data_hashes_crashes(self) -> None:
        with pytest.raises(DataContractError, match="data_hash"):
            load_preregistration_spec(SPEC, data_hashes=(), parameters=PARAMETERS)

    def test_there_is_no_action_that_relaxes_a_threshold(self) -> None:
        """De actieruimte bevat bewust geen 'drempel bijstellen'."""
        with pytest.raises(ConfigContractError, match="Onbekende actie"):
            StopCriterion(name="x", metric="net_oos_sharpe", operator="<",
                          threshold=0.0, action="relax_threshold",
                          rationale="zou fraude zijn")

    def test_criterion_without_rationale_crashes(self) -> None:
        with pytest.raises(ConfigContractError, match="onderbouwing"):
            StopCriterion(name="x", metric="net_oos_sharpe", operator="<",
                          threshold=0.0, action="falsify", rationale="")


class TestIdentity:
    def test_id_is_content_addressed_and_ignores_git_sha(self, tmp_path: Path) -> None:
        """Dezelfde inhoud op een andere commit is dezelfde pre-registratie."""
        a = _load(SPEC)
        b = _load(SPEC)
        assert a.preregistration_id == b.preregistration_id

    def test_changed_content_changes_the_id(self, tmp_path: Path) -> None:
        base = _load(SPEC)
        other = _load(_spec(tmp_path, planned_trials=base.planned_trials + 1))
        assert other.preregistration_id != base.preregistration_id

    def test_changed_data_hash_changes_the_id(self) -> None:
        base = _load(SPEC)
        other = load_preregistration_spec(
            SPEC, data_hashes=(("crypto/ohlcv/BTCUSDT/1d", "f" * 32),),
            parameters=PARAMETERS)
        assert other.preregistration_id != base.preregistration_id

    def test_changed_parameters_change_the_id(self) -> None:
        base = _load(SPEC)
        other = load_preregistration_spec(
            SPEC, data_hashes=DATA_HASHES,
            parameters={"alpha": {"lookback_bars": 120}})
        assert other.preregistration_id != base.preregistration_id


class TestStopCriteriaBind:
    @pytest.mark.parametrize(
        "operator,threshold,measured,expected",
        [("<", 0.0, -0.1, True), ("<", 0.0, 0.1, False),
         ("<=", 0.0, 0.0, True), (">", 0.5, 0.6, True),
         (">=", 0.05, 0.05, True), (">=", 0.05, 0.04, False)],
    )
    def test_binds(self, operator: str, threshold: float, measured: float,
                   expected: bool) -> None:
        c = StopCriterion(name="c", metric="m", operator=operator,
                          threshold=threshold, action="archive", rationale="test")
        assert c.binds(measured) is expected

    def test_unknown_criterion_lookup_crashes(self) -> None:
        prereg = _load(SPEC)
        with pytest.raises(ConfigContractError, match="Onbekend stop-criterium"):
            prereg.stop_criterion("a_criterion_invented_after_the_fact")

