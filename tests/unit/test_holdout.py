"""Het poortsample. Eén split, één meting per hypothese, en een weigering die
niet te omzeilen is zonder een commit die zichtbaar is in een diff."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.holdout import (
    HoldoutAlreadyFrozen,
    HoldoutAlreadyUsed,
    backcast_gate_slice,
    development_slice,
    freeze_holdout,
    gate_slice,
    resume_registered_read,
)

IDX = pd.date_range("2021-10-15", periods=1615, freq="D", tz="UTC")


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"A": rng.normal(0, 0.01, 1615)}, index=IDX)


def _lock(tmp_path: Path) -> Path:
    out = tmp_path / "holdout_lock.json"
    freeze_holdout(split_utc="2025-09-05T00:00:00+00:00", out=out,
                   git_sha="deadbee")
    return out


def test_development_and_gate_slices_are_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    lock, frame = _lock(tmp_path), _frame()
    dev = development_slice(frame, lock_path=lock)
    gate = gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    assert dev.index.max() < gate.index.min()
    assert len(dev) + len(gate) == len(frame)


def test_a_second_read_of_the_gate_slice_for_the_same_hypothesis_is_refused(
    tmp_path: Path,
) -> None:
    """De enige eigenschap die telt. Twee metingen op het poortsample maken M
    op dat sample groter dan 1, en dan is het geen poortsample meer."""
    lock, frame = _lock(tmp_path), _frame()
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    with pytest.raises(HoldoutAlreadyUsed):
        gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")


def test_a_different_hypothesis_may_read_it_once(tmp_path: Path) -> None:
    lock, frame = _lock(tmp_path), _frame()
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.3")


def test_every_read_is_logged_with_a_timestamp(tmp_path: Path) -> None:
    lock, frame = _lock(tmp_path), _frame()
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    import json
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["reads"][0]["hypothesis_id"] == "H-10.1"
    assert payload["reads"][0]["read_utc"]


def test_the_backcast_slice_is_the_part_before_the_split_and_is_read_once(
    tmp_path: Path,
) -> None:
    """Het spiegelbeeld: het beschermde stuk ligt VÓÓR de split, met dezelfde R7."""
    lock, frame = _lock(tmp_path), _frame()
    back = backcast_gate_slice(frame, lock_path=lock, hypothesis_id="B-1")
    assert back.index.equals(development_slice(frame, lock_path=lock).index)
    with pytest.raises(HoldoutAlreadyUsed):
        backcast_gate_slice(frame, lock_path=lock, hypothesis_id="B-1")
    # De weigering geldt per lock, niet per richting: dezelfde id op hetzelfde slot
    # mag ook niet via de andere kant een tweede lezing kopen.
    with pytest.raises(HoldoutAlreadyUsed):
        gate_slice(frame, lock_path=lock, hypothesis_id="B-1")


# --- Fix ronde 1, bevinding 1: de vier require()-guards in freeze_holdout -
#
# Elk pint zowel het geraiste type ALS dat de boodschap de geschonden
# aanname noemt (R-5) -- niet alleen dat er uberhaupt iets is geraist.
def test_freeze_holdout_refuses_an_empty_split_utc(tmp_path: Path) -> None:
    with pytest.raises(DataContractError) as exc:
        freeze_holdout(split_utc="   ", out=tmp_path / "holdout_lock.json",
                       git_sha="deadbee")
    assert "split_utc" in str(exc.value)


def test_freeze_holdout_refuses_a_naive_timestamp(tmp_path: Path) -> None:
    """Zonder tijdzone is een timestamp niet ondubbelzinnig te vergelijken
    met een UTC-geindexeerd paneel."""
    with pytest.raises(DataContractError) as exc:
        freeze_holdout(split_utc="2025-09-05T00:00:00",
                       out=tmp_path / "holdout_lock.json", git_sha="deadbee")
    assert "tijdzone" in str(exc.value)


def test_freeze_holdout_refuses_an_empty_git_sha(tmp_path: Path) -> None:
    with pytest.raises(DataContractError) as exc:
        freeze_holdout(split_utc="2025-09-05T00:00:00+00:00",
                       out=tmp_path / "holdout_lock.json", git_sha="  ")
    assert "git_sha" in str(exc.value)


def test_freeze_holdout_refuses_a_second_freeze_of_the_same_path(
    tmp_path: Path,
) -> None:
    """Ruling P29. Dit is een geweigerde HERHAALDE actie op een bestaande
    STAAT, geen geschonden datacontract -- vandaar een NAMED exceptie,
    `HoldoutAlreadyFrozen`, net als `registry.ledger_reset.ResetAlreadyExists`
    voor de ledger-reset, en niet `DataContractError`."""
    out = tmp_path / "holdout_lock.json"
    freeze_holdout(split_utc="2025-09-05T00:00:00+00:00", out=out,
                   git_sha="deadbee")
    with pytest.raises(HoldoutAlreadyFrozen) as exc:
        freeze_holdout(split_utc="2025-09-06T00:00:00+00:00", out=out,
                       git_sha="deadbee")
    assert str(out) in str(exc.value)


# --- Ruling P24: de header is bevroren, het bestand is dat niet -----------
#
# `holdout_lock.json` muteert naar ontwerp: `reads` groeit bij elke
# `gate_slice`-aanroep, dus een hele-bestand-hash zoals op `ledger_reset.json`
# kan hier niet. Wat WEL vastligt zijn de drie header-velden die bij het
# bevriezen zijn geschreven en die geen enkele latere `gate_slice`-aanroep
# hoort te raken. Deze test pint precies dat, op het gecommitte artefact --
# analoog aan `test_the_frozen_reset_artefact_is_pinned_and_linked_to_the_ledger`
# in `tests/unit/test_ledger_reset.py`, maar op de velden die hier daadwerkelijk
# onveranderlijk zijn in plaats van op een hash van het geheel.
LOCK = Path("artefacts/governance/holdout_lock.json")


def test_the_frozen_holdout_header_is_pinned() -> None:
    assert LOCK.is_file(), (
        f"{LOCK} ontbreekt. Stap 4B.4 bevriest de split precies een keer, "
        f"vóór de eerste poortmeting (R7)."
    )
    import json
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    assert payload["split_utc"] == "2025-09-05T00:00:00+00:00", (
        "De splitdatum is pre-geregistreerd (ruling P23) en hoort na het "
        "bevriezen nooit meer te veranderen."
    )
    assert payload["git_sha"], "git_sha ontbreekt op het bevroren artefact."
    assert payload["frozen_utc"] == "2026-09-09T18:55:43.793046+00:00", (
        "frozen_utc is de timestamp van het EENMALIGE bevriezen (stap 4B.4) "
        "en hoort na dat moment nooit meer te veranderen."
    )


def test_a_registered_read_can_be_resumed_once_and_visibly(tmp_path: Path) -> None:
    """Een crash na de registratie: dezelfde lezing afmaken, eenmaal, met reden in het slot."""
    import json
    lock, frame = _lock(tmp_path), _frame()
    with pytest.raises(DataContractError):
        resume_registered_read(lock, hypothesis_id="H-1", reason="crash")  # nooit gelezen
    gate_slice(frame, lock_path=lock, hypothesis_id="H-1")
    with pytest.raises(DataContractError):
        resume_registered_read(lock, hypothesis_id="H-1", reason=" ")
    resume_registered_read(lock, hypothesis_id="H-1", reason="crash in de samenvatting")
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["resumes"][0]["reason"] == "crash in de samenvatting"
    assert len(payload["reads"]) == 1
    with pytest.raises(HoldoutAlreadyUsed):
        resume_registered_read(lock, hypothesis_id="H-1", reason="nog eens")
