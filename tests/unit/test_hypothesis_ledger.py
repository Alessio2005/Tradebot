# tests/unit/test_hypothesis_ledger.py
"""Guards on the append-only hypothesis ledger (Wave 20, step 0.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.registry import HypothesisLedger, LedgerEntry

SEED = 100


@pytest.fixture()
def ledger_path(tmp_path: Path) -> Path:
    doc = {"version": 1, "seed_total": SEED, "entries": []}
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _entry(unit: str = "u1", cfg: dict | None = None, n: int = 1) -> LedgerEntry:
    return LedgerEntry.from_config(
        wave=20, unit=unit, market="crypto", config=cfg or {"a": 1}, n_trials=n
    )


def test_missing_ledger_is_never_autocreated(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        HypothesisLedger(tmp_path / "nope.json")


def test_total_is_seed_plus_appends(ledger_path: Path) -> None:
    ledger = HypothesisLedger(ledger_path)
    assert ledger.total_n_hypotheses() == SEED
    assert ledger.append(_entry(n=3)) == SEED + 3
    assert ledger.append(_entry(cfg={"a": 2}, n=2)) == SEED + 5
    assert ledger.total_n_hypotheses() == SEED + 5


def test_entry_validation() -> None:
    with pytest.raises(ValueError):
        _entry(n=0)
    with pytest.raises(ValueError):
        LedgerEntry.from_config(20, "u", "bad_market", {"a": 1})
    with pytest.raises(ValueError):
        LedgerEntry.from_config(20, "u", "crypto", {"a": 1}, result="maybe")


def test_staging_roundtrip_and_duplicate_rejection(
    ledger_path: Path, tmp_path: Path
) -> None:
    ledger = HypothesisLedger(ledger_path)
    staging = tmp_path / "staging_w22a.json"
    HypothesisLedger.append_to_staging(staging, _entry(unit="s1", n=2))
    HypothesisLedger.append_to_staging(staging, _entry(unit="s2", n=1))
    assert ledger.merge_staging(staging) == SEED + 3
    assert not staging.exists()

    # same (wave, unit, config_hash) again -> refused
    staging2 = tmp_path / "staging_dup.json"
    HypothesisLedger.append_to_staging(staging2, _entry(unit="s1", n=2))
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.merge_staging(staging2)


def test_append_only_guards(ledger_path: Path) -> None:
    ledger = HypothesisLedger(ledger_path)
    ledger.append(_entry())
    doc = ledger.load()
    doc["entries"] = []
    with pytest.raises(ValueError, match="append-only"):
        ledger._write_atomic(doc)
    doc2 = ledger.load()
    doc2["seed_total"] = 1
    with pytest.raises(ValueError, match="immutable"):
        ledger._write_atomic(doc2)
