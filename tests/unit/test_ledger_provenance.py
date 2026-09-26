"""Ledger-integriteit — exit criterium B8, Phase 7/8 Stage B-8.

    Een entry zonder `git_sha`, `data_hash`, `config_hash` of
    `preregistration_id` CRASHT.

WAT ER HIERVOOR STOND
=====================
`LedgerEntry` had die vier velden niet. Drie ervan werden als vrije tekst in
`notes` gepropt:

    notes="Baseline-resultaat; preregistration_id=56395fa2...; git_sha=42555d2"

Dat is geen contract maar een gewoonte, en een gewoonte kan geen entry weigeren.
Er was dus geen manier om te weten of een entry uit een reproduceerbare run
kwam — terwijl `M` in élke DSR in dit platform zit. Een niet-herleidbare
entry maakt die telling een bewering.

DE LEDGER IS OP 2026-09-26 OPNIEUW BEGONNEN
===========================================
Op besluit van de eigenaar zijn alle eerdere entries verwijderd en staat de
teller op 0. Er bestaan dus geen entries zonder herkomst meer, en
`TestTheGapIsAClosedSet` houdt dat zo: het aantal mag alleen omlaag, en het
staat op nul. Elke NIEUWE entry draagt de vier velden.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.registry.hypothesis_ledger import (
    DEFAULT_LEDGER_PATH,
    PROVENANCE_FIELDS,
    HypothesisLedger,
    LedgerEntry,
)

ROOT = Path(__file__).resolve().parents[2]

#: Nulstand na de herstart van de ledger op 2026-09-26. RATCHET: alleen omlaag,
#: en er is geen omlaag meer.
LEGACY_ENTRIES_WITHOUT_PROVENANCE = 0


def _entry(**over) -> LedgerEntry:
    base = dict(
        wave=99, unit="unit-under-test", market="crypto",
        config_hash="1b60cb664fbf9a2a", n_trials=1, result="interim",
        git_sha="deadbeef", data_hash="d4ta",
        preregistration_id="prereg-abc",
    )
    base.update(over)
    return LedgerEntry(**base)          # type: ignore[arg-type]


def _fresh_ledger(tmp_path: Path) -> HypothesisLedger:
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(
        json.dumps({"version": 1, "seed_total": 100, "entries": []}),
        encoding="utf-8")
    return HypothesisLedger(path)


# --------------------------------------------------------------------------- #
# B8: de vier velden zijn verplicht
# --------------------------------------------------------------------------- #
class TestProvenanceIsMandatory:
    @pytest.mark.parametrize("field", PROVENANCE_FIELDS)
    def test_an_empty_provenance_field_crashes(self, field: str) -> None:
        with pytest.raises(ValueError, match="herkomstveld"):
            _entry(**{field: ""})

    @pytest.mark.parametrize("field", PROVENANCE_FIELDS)
    def test_whitespace_is_not_provenance(self, field: str) -> None:
        """`" "` is geen hash. Een veld dat je met een spatie kunt vullen, is
        geen verplicht veld."""
        with pytest.raises(ValueError, match="herkomstveld"):
            _entry(**{field: "   "})

    def test_the_message_names_every_missing_field_at_once(self) -> None:
        """Vier keer raden is drie keer te vaak."""
        with pytest.raises(ValueError) as exc:
            LedgerEntry(wave=1, unit="u", market="crypto", config_hash="",
                        n_trials=1, result="interim")
        message = str(exc.value)
        for field in PROVENANCE_FIELDS:
            assert field in message

    def test_a_complete_entry_is_accepted(self) -> None:
        """De weigering is gericht. Zonder deze test zou een LedgerEntry die
        alles weigert ook groen zijn."""
        entry = _entry()
        assert entry.git_sha == "deadbeef"
        assert entry.ts_utc                      # automatisch gezet

    def test_from_config_cannot_forget_the_provenance(self) -> None:
        """De drie velden staan vóór elk argument met een default, dus Python
        zelf dwingt af dat ze worden meegegeven."""
        with pytest.raises(TypeError):
            LedgerEntry.from_config(          # type: ignore[call-arg]
                wave=1, unit="u", market="crypto", config={"a": 1})

    def test_from_config_derives_the_config_hash(self) -> None:
        entry = LedgerEntry.from_config(
            wave=1, unit="u", market="crypto", config={"a": 1},
            git_sha="deadbeef", data_hash="d4ta",
            preregistration_id="prereg-abc")
        assert entry.config_hash and entry.config_hash != "d4ta"


# --------------------------------------------------------------------------- #
# Het contract geldt ook op de schrijfpaden
# --------------------------------------------------------------------------- #
class TestTheContractHoldsOnEveryWritePath:
    def test_append_stores_the_provenance(self, tmp_path: Path) -> None:
        ledger = _fresh_ledger(tmp_path)
        ledger.append(_entry(n_trials=3))
        stored = ledger.entries()[-1]
        for field in PROVENANCE_FIELDS:
            assert stored[field], f"{field} is niet weggeschreven"

    def test_merge_staging_refuses_an_entry_without_provenance(
        self, tmp_path: Path
    ) -> None:
        """De staging-route is de PARALLELLE schrijfweg, en dus de weg waarlangs
        een entry zonder herkomst het gemakkelijkst binnenkomt."""
        ledger = _fresh_ledger(tmp_path)
        staging = tmp_path / "staging.json"
        staging.write_text(json.dumps([{
            "wave": 1, "unit": "smuggled", "market": "crypto",
            "config_hash": "c", "n_trials": 5, "result": "interim",
        }]), encoding="utf-8")
        with pytest.raises(ValueError, match="herkomstveld"):
            ledger.merge_staging(staging)
        assert ledger.entries() == [], "de entry is toch binnengekomen"

    def test_merge_staging_accepts_a_complete_entry(
        self, tmp_path: Path
    ) -> None:
        ledger = _fresh_ledger(tmp_path)
        staging = tmp_path / "staging.json"
        staging.write_text(json.dumps([{
            "wave": 1, "unit": "legit", "market": "crypto",
            "config_hash": "c", "n_trials": 5, "result": "interim",
            "git_sha": "deadbeef", "data_hash": "d4ta",
            "preregistration_id": "prereg-abc",
        }]), encoding="utf-8")
        assert ledger.merge_staging(staging) == 105

    def test_the_ledger_stays_append_only(self, tmp_path: Path) -> None:
        """De eigenschap waarop `M` rust: hij kan niet krimpen."""
        ledger = _fresh_ledger(tmp_path)
        ledger.append(_entry(n_trials=2))
        doc = ledger.load()
        doc["entries"] = []
        with pytest.raises(ValueError, match="append-only"):
            ledger._write_atomic(doc)

    def test_the_seed_is_immutable(self, tmp_path: Path) -> None:
        ledger = _fresh_ledger(tmp_path)
        doc = ledger.load()
        doc["seed_total"] = 1
        with pytest.raises(ValueError, match="seed_total is immutable"):
            ledger._write_atomic(doc)


# --------------------------------------------------------------------------- #
# De historische entries
# --------------------------------------------------------------------------- #
class TestTheGapIsAClosedSet:
    """RATCHET. Het aantal entries zonder herkomst mag alleen omlaag.

    Een `git_sha` verzinnen voor een entry is herkomst FABRICEREN, en een
    verzonnen hash is niet van een echte te onderscheiden; dus wordt een entry
    zonder herkomst niet aangevuld maar geweigerd.
    """

    def test_no_new_entry_lacks_provenance(self) -> None:
        doc = json.loads((ROOT / DEFAULT_LEDGER_PATH).read_text(encoding="utf-8"))
        without = [
            e["unit"] for e in doc["entries"]
            if any(not str(e.get(f, "")).strip() for f in PROVENANCE_FIELDS)
        ]
        assert len(without) <= LEGACY_ENTRIES_WITHOUT_PROVENANCE, (
            f"{len(without)} entries zonder herkomst tegen een nulstand van "
            f"{LEGACY_ENTRIES_WITHOUT_PROVENANCE}. Nieuw sinds de meting: "
            f"{without[LEGACY_ENTRIES_WITHOUT_PROVENANCE:]}")

    def test_the_ledger_loads_and_counts(self) -> None:
        """De gecommitte ledger is leesbaar en `M` is de som van seed en entries."""
        ledger = HypothesisLedger(ROOT / DEFAULT_LEDGER_PATH)
        doc = ledger.load()
        assert ledger.total_n_hypotheses() == int(doc["seed_total"]) + sum(
            int(e["n_trials"]) for e in doc["entries"])

    def test_the_restart_is_documented_in_the_artefact(self) -> None:
        doc = json.loads((ROOT / DEFAULT_LEDGER_PATH).read_text(encoding="utf-8"))
        assert doc["seed_total"] == 0
        assert "2026-09-26" in doc.get("seed_note", "")
