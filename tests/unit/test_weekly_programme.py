from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.features.base import DataRegister
from tradebot.registry.hypothesis_ledger import HypothesisLedger
from tradebot.registry.preregistration import load_preregistration_spec
from tradebot.registry.trial_counter import frozen_trial_count
from tradebot.registry.weekly_programme import (
    book_and_freeze,
    certified_data_hashes,
    programme_parameters,
    refreeze_unread_holdout,
)
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.utils.failfast import DataContractError
from tradebot.validation.weekly_verdict import DEVELOPMENT_METRICS, HOLDOUT_METRICS

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "conf/research/preregistration_weekly_meta.yaml"
CFG = weekly_meta_config()


def _empty_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(json.dumps({"version": 1, "seed_total": 0, "seed_note": "t", "entries": []}),
                    encoding="utf-8")
    return path


def test_the_spec_loads_with_every_criterion_measured() -> None:
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    prereg = load_preregistration_spec(SPEC, data_hashes=certified_data_hashes(register, CFG.symbols),
                                       parameters=programme_parameters(CFG))
    decisive = {c.metric for c in prereg.stop_criteria if c.action != "promote"}
    assert decisive == set(DEVELOPMENT_METRICS) | set(HOLDOUT_METRICS)
    assert prereg.planned_trials == CFG.planned_trials


def test_the_thresholds_match_the_config() -> None:
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    prereg = load_preregistration_spec(SPEC, data_hashes=certified_data_hashes(register, CFG.symbols),
                                       parameters=programme_parameters(CFG))
    by = {c.name: c.threshold for c in prereg.stop_criteria}
    assert by["overfit_probability"] == CFG.pbo_max
    assert by["drawdown_risk"] == CFG.mc_max_probability_1y
    assert by["insufficient_trades"] == CFG.min_trades
    assert by["negative_control_shuffle"] == pytest.approx(0.5 - CFG.shuffle_auc_band[0])
    assert by["holdout_brier_worse"] == CFG.holdout_brier_margin


def test_booking_then_freezing_fixes_m(tmp_path) -> None:
    ledger = _empty_ledger(tmp_path)
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    path = book_and_freeze(spec_path=SPEC, cfg=CFG, register=register, ledger_path=ledger,
                           prereg_dir=tmp_path, git_sha="abc1234")
    assert HypothesisLedger(ledger).total_n_hypotheses() == 4
    assert frozen_trial_count(path).value == 4
    with pytest.raises(DataContractError, match="eerste"):
        book_and_freeze(spec_path=SPEC, cfg=CFG, register=register, ledger_path=ledger,
                        prereg_dir=tmp_path, git_sha="abc1234")


def test_an_unread_holdout_can_be_refrozen_and_a_read_one_cannot(tmp_path) -> None:
    lock = tmp_path / "holdout_lock.json"
    lock.write_text(json.dumps({"split_utc": "2025-09-05T00:00:00+00:00", "git_sha": "x",
                                "frozen_utc": "t", "reads": []}), encoding="utf-8")
    refreeze_unread_holdout(lock, split_utc=CFG.holdout_split_utc, git_sha="abc1234")
    assert json.loads(lock.read_text(encoding="utf-8"))["split_utc"] == CFG.holdout_split_utc
    data = json.loads(lock.read_text(encoding="utf-8"))
    data["reads"] = [{"hypothesis_id": "h"}]
    lock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DataContractError, match="gelezen"):
        refreeze_unread_holdout(lock, split_utc="2026-07-01T00:00:00+00:00", git_sha="abc1234")
