# tests/unit/test_registry_edges.py
"""De registry-paden die het hoofdpad niet raakt.

Catalogus, experimentlog, ledger, lineage, pre-registratie en promotie hebben
elk hun eigen hoofdtest. Hier staan de randen: een catalogus met een kapotte
regel, een run die nooit is gestart, MLflow als spiegel, een staging-bestand
dat geen lijst is, een gewijzigd artefact, en elke weigering van `promote`.
"""
from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradebot.exceptions import LineageHashMismatch
from tradebot.registry import lineage
from tradebot.registry import weekly_programme as programme
from tradebot.registry.catalog import ModelCatalog, ModelRecord
from tradebot.registry.experiment import ExperimentTracker
from tradebot.registry.hypothesis_ledger import HypothesisLedger
from tradebot.registry.ledger_reset import ActiveCount
from tradebot.registry.preregistration import (
    freeze_metadata,
    freeze_preregistration,
    load_preregistration_spec,
    require_preregistration,
)
from tradebot.registry.promotion import promote
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "tests" / "fixtures" / "preregistration_example.yaml"


def _record(stage: str = "research", version: int = 1, **metrics: float) -> ModelRecord:
    return ModelRecord(symbol="BTC", side="LONG", version=version, stage=stage,
                       metrics=dict(metrics))


class TestCatalog:
    def test_an_empty_catalog_starts_at_version_one(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        assert catalog.query() == []
        assert catalog.next_version("BTC", "LONG") == 1

    def test_queries_filter_and_skip_broken_lines(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_record(version=1))
        catalog.register(_record(stage="staging", version=2))
        catalog.register(ModelRecord(symbol="ETH", side="SHORT", version=1))
        with open(tmp_path / "model_catalog.jsonl", "a", encoding="utf-8") as fh:
            fh.write("\n{broken\n[1, 2]\n")
        assert catalog.next_version("BTC", "LONG") == 3
        everything = catalog.query(latest=False)
        assert len(everything) == 3
        assert [r.version for r in catalog.query(symbol="BTC", latest=False)] == [1, 2]
        assert [r.symbol for r in catalog.query(side="SHORT", latest=False)] == ["ETH"]
        assert [r.version for r in catalog.query(stage="staging", latest=False)] == [2]


class _FakeMlflow(types.ModuleType):
    """Neemt elke aanroep op; MLflow is een SPIEGEL naast de JSONL-ledger."""

    def __init__(self) -> None:
        super().__init__("mlflow")
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return _call


class TestExperimentTracker:
    def test_metrics_and_artifacts_need_a_started_run(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker("exp", tmp_path)
        with pytest.raises(RuntimeError, match="start_run"):
            tracker.log_metrics({"sharpe": 1.0})
        with pytest.raises(RuntimeError, match="start_run"):
            tracker.log_artifact(tmp_path / "model.cbm")
        tracker.end_run()  # zonder actieve run: niets te doen
        assert tracker.load_runs().empty
        assert tracker.best_run() is None

    def test_only_finished_runs_compete_for_best(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker("exp", tmp_path)
        tracker.start_run(0, {"depth": 3})
        tracker.log_metrics({"sharpe": 2.0})
        tracker.end_run(status="FAILED")
        assert tracker.best_run() is None
        tracker.start_run(1, {"depth": 4})
        tracker.log_metrics({"sharpe": 0.5})
        tracker.log_artifact(tmp_path / "m1.cbm")
        tracker.end_run()
        best = tracker.best_run()
        assert best is not None and best.trial_number == 1
        assert best.artifact_paths == [str(tmp_path / "m1.cbm")]

    def test_mlflow_mirrors_every_step_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _FakeMlflow()
        monkeypatch.setitem(sys.modules, "mlflow", fake)
        monkeypatch.setattr("tradebot.registry.experiment.has_module",
                            lambda name: name == "mlflow")
        tracker = ExperimentTracker("exp", tmp_path, use_mlflow=True)
        assert tracker._try_import_mlflow() is True
        tracker.start_run(7, {"depth": 3}, tags={"wave": "1"})
        tracker.log_metrics({"sharpe": 1.0}, step=1)
        tracker.log_artifact("model.cbm")
        tracker.end_run()
        names = [c[0] for c in fake.calls]
        assert names == ["set_tracking_uri", "set_experiment", "start_run",
                         "log_params", "set_tags", "log_metrics",
                         "log_artifact", "end_run"]
        assert tracker.load_runs().iloc[0]["trial_number"] == 7


def _ledger(tmp_path: Path) -> HypothesisLedger:
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(json.dumps({"version": 1, "seed_total": 100, "entries": []}),
                    encoding="utf-8")
    return HypothesisLedger(path)


_STAGED = {"wave": 1, "unit": "u", "market": "crypto", "config_hash": "c",
           "n_trials": 5, "result": "interim", "git_sha": "deadbeef",
           "data_hash": "d4ta", "preregistration_id": "prereg-abc"}


class TestLedgerStaging:
    def test_a_staging_file_must_be_a_list(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging.json"
        staging.write_text(json.dumps(_STAGED), encoding="utf-8")
        with pytest.raises(ValueError, match="JSON list"):
            _ledger(tmp_path).merge_staging(staging)

    def test_an_undeletable_staging_file_is_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sommige mounts verbieden `unlink`. Dan wordt het bestand geleegd, zodat
        een tweede merge nooit dubbel kan tellen."""
        ledger = _ledger(tmp_path)
        staging = tmp_path / "staging.json"
        staging.write_text(json.dumps([_STAGED]), encoding="utf-8")

        def _refuse(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("read-only mount")

        monkeypatch.setattr(Path, "unlink", _refuse)
        assert ledger.merge_staging(staging) == 105
        assert json.loads(staging.read_text(encoding="utf-8")) == []
        assert ledger.merge_staging(staging) == 105


def test_the_active_count_is_a_frozen_trial_count(tmp_path: Path) -> None:
    count = ActiveCount(total=4, archived_total=2776,
                        reset_path=tmp_path / "reset.json").to_trial_count()
    assert count.value == 4
    assert count.seed_total == 0
    assert count.registered_total == 4
    assert count.source == "frozen"


class TestLineage:
    def test_the_file_hash_is_a_stable_128_bit_digest(self, tmp_path: Path) -> None:
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"x" * 200_000)
        digest = lineage.get_file_hash(artifact, chunk=4096)
        assert len(digest) == 32
        assert digest == lineage.get_file_hash(artifact)

    def test_a_changed_artifact_or_feature_set_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        artifact = tmp_path / "model.bin"
        artifact.write_bytes(b"model-v1")
        good = lineage.get_file_hash(artifact)
        monkeypatch.setattr(lineage, "get_git_sha", lambda short=True: "beef123")
        lineage.verify_lineage(artifact, "beef123", good, "f1", "f1")
        lineage.verify_lineage(artifact, "", "", "", "")
        with caplog.at_level("WARNING"):
            lineage.verify_lineage(artifact, "cafe456", good, "f1", "f1")
        assert "Git SHA mismatch" in caplog.text
        with pytest.raises(LineageHashMismatch, match="content hash"):
            lineage.verify_lineage(artifact, "", "0" * 32, "", "")
        with pytest.raises(LineageHashMismatch, match="Feature schema"):
            lineage.verify_lineage(artifact, "", good, "f1", "f2")


class TestPreregistrationRoundTrip:
    def test_a_frozen_preregistration_reads_back_with_its_metadata(
        self, tmp_path: Path,
    ) -> None:
        prereg = load_preregistration_spec(
            SPEC, data_hashes=(("crypto/ohlcv/BTCUSDT/1d", "0" * 32),),
            parameters={"alpha": {"lookback_bars": 60}})
        freeze_preregistration(prereg, git_sha="abc1234",
                               ledger_total_at_freeze=2711, directory=tmp_path)
        back = require_preregistration(prereg.preregistration_id, directory=tmp_path)
        assert back == prereg
        first = prereg.stop_criteria[0]
        assert back.stop_criterion(first.name) == first
        meta = freeze_metadata(prereg.preregistration_id, directory=tmp_path)
        assert meta["git_sha"] == "abc1234"
        assert meta["ledger_total_at_freeze"] == 2711
        assert meta["m_trials"] == 2711 + prereg.planned_trials


_PASSED = SimpleNamespace(passed=True, verdict="PROMOTED", failed_gates=())


class TestPromotionRefusals:
    def test_nothing_to_promote(self, tmp_path: Path) -> None:
        assert promote(ModelCatalog(tmp_path), "BTC", "LONG", "staging") is None

    def test_staging_needs_the_screening_thresholds(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_record(oos_logloss=0.75, sharpe=1.0))
        assert promote(catalog, "BTC", "LONG", "staging") is None

    def test_prod_needs_staging_first(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_record(sharpe=1.0, max_dd=0.1, n_obs=500))
        assert promote(catalog, "BTC", "LONG", "prod",
                       gate_result=_PASSED) is None  # type: ignore[arg-type]

    def test_prod_needs_the_prod_thresholds(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_record(stage="staging", sharpe=0.1, max_dd=0.1, n_obs=500))
        assert promote(catalog, "BTC", "LONG", "prod",
                       gate_result=_PASSED) is None  # type: ignore[arg-type]

    def test_an_unknown_stage_is_an_error(self, tmp_path: Path) -> None:
        catalog = ModelCatalog(tmp_path)
        catalog.register(_record(oos_logloss=0.5, sharpe=1.0))
        with pytest.raises(ValueError, match="Unknown target_stage"):
            promote(catalog, "BTC", "LONG", "live")


class TestWeeklyProgrammeEntrypoint:
    def test_main_wires_the_holdout_and_the_freeze(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        calls: list[str] = []
        cfg = SimpleNamespace(holdout_split_utc="2026-06-24", planned_trials=4)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(programme, "weekly_meta_config", lambda: cfg)
        monkeypatch.setattr(programme, "current_git_sha", lambda: "beef123")
        monkeypatch.setattr(programme, "refreeze_unread_holdout",
                            lambda path, **kw: calls.append(f"holdout:{path.name}"))
        monkeypatch.setattr(programme, "DataRegister", lambda path: path)

        def _freeze(**kw):
            calls.append(f"freeze:{kw['git_sha']}")
            return tmp_path / "preregistration_x.json"

        monkeypatch.setattr(programme, "book_and_freeze", _freeze)
        programme.main(["freeze"])
        assert calls == ["holdout:holdout_lock.json", "freeze:beef123"]
        assert "preregistration_x.json; M = 4" in capsys.readouterr().out

    def test_any_other_command_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="Gebruik"):
            programme.main(["thaw"])

    @pytest.mark.filterwarnings(
        "ignore:.*found in sys.modules after import of package:RuntimeWarning")
    def test_the_module_runs_as_a_script(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["weekly_programme", "--help"])
        with pytest.raises(DataContractError, match="Gebruik"):
            runpy.run_module("tradebot.registry.weekly_programme",
                             run_name="__main__")
