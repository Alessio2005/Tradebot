# tests/unit/test_clean.py
"""Een opruimscript is het gevaarlijkste gereedschap in de repository.

Phase 9, stap 14. `make clean` bestond, maar `make` niet — gemeten op
2026-09-04: `make: command not found`. Het opruimdoel was dus net zo
onuitvoerbaar als de LOC-poort ernaast, terwijl `.mypy_cache` ondertussen naar
**2.773 bestanden / 106 MB** was gegroeid.

De vervanger is `scripts/clean.py`, en die verdient meer toetsing dan de rest
van deze fase bij elkaar: hij verwijdert bestanden. Een `rm -rf` met een fout
pad is de enige handeling in dit project die niet met `git show` terug te halen
is, want de doelen zijn juist de dingen die NIET in versiebeheer staan.

WAT DEZE TEST AFDWINGT
======================
1. Hij ruimt op wat hij hoort op te ruimen.
2. Hij raakt **`data/pit_store/`** niet aan. Dat is fence 1 van deze fase: 228
   ruwe parquet/json-bestanden staan daar bewust in versiebeheer (AD-12), en
   verwijderen is een architectuurbesluit, geen opruiming.
3. Hij raakt **`artefacts/governance/`** niet aan. Fence 2: append-only.
4. Hij raakt niets aan dat door git wordt getrackt.
5. `--dry-run` verwijdert niets.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.clean import PROTECTED, TARGETS, sweep


def _touch(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


class TestItRemovesWhatItShould:
    def test_it_reports_the_cache_directories_it_would_remove(self, tmp_path: Path) -> None:
        _touch(tmp_path / ".mypy_cache" / "a.json")
        _touch(tmp_path / ".ruff_cache" / "b.json")
        removed = sweep(tmp_path, dry_run=True)
        assert {p.name for p in removed} == {".mypy_cache", ".ruff_cache"}

    def test_it_removes_pycache_directories_anywhere_in_the_tree(self, tmp_path: Path) -> None:
        _touch(tmp_path / "src" / "pkg" / "__pycache__" / "m.cpython-313.pyc")
        removed = sweep(tmp_path, dry_run=True)
        assert any(p.name == "__pycache__" for p in removed)

    def test_a_dry_run_deletes_nothing(self, tmp_path: Path) -> None:
        victim = _touch(tmp_path / ".mypy_cache" / "a.json")
        sweep(tmp_path, dry_run=True)
        assert victim.exists()

    def test_a_real_run_deletes_it(self, tmp_path: Path) -> None:
        victim = _touch(tmp_path / ".mypy_cache" / "a.json")
        sweep(tmp_path, dry_run=False)
        assert not victim.exists()
        assert not victim.parent.exists()

    def test_it_is_idempotent_on_an_already_clean_tree(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        assert sweep(tmp_path, dry_run=False) == []


class TestTheFencesHold:
    """Fence 1 en 2 van Phase 9. Overtreding is vernietiging van bewijsmateriaal."""

    def test_it_never_touches_the_pit_store(self, tmp_path: Path) -> None:
        keep = _touch(tmp_path / "data" / "pit_store" / "BTCUSDT.parquet")
        _touch(tmp_path / "data" / "pit_store" / "__pycache__" / "x.pyc")
        sweep(tmp_path, dry_run=False)
        assert keep.exists(), "fence 1 geschonden: data/pit_store is aangeraakt"

    def test_it_never_touches_the_governance_ledger(self, tmp_path: Path) -> None:
        keep = _touch(tmp_path / "artefacts" / "governance" / "hypothesis_ledger.json")
        _touch(tmp_path / "artefacts" / "governance" / "__pycache__" / "x.pyc")
        sweep(tmp_path, dry_run=False)
        assert keep.exists(), "fence 2 geschonden: artefacts/governance is aangeraakt"

    def test_a_pycache_inside_a_protected_path_is_not_even_reported(
        self, tmp_path: Path
    ) -> None:
        """Niet alleen overslaan bij het verwijderen — ook niet voorstellen."""
        _touch(tmp_path / "data" / "pit_store" / "__pycache__" / "x.pyc")
        assert sweep(tmp_path, dry_run=True) == []

    def test_the_protected_list_names_both_fences(self) -> None:
        assert "data/pit_store" in PROTECTED
        assert "artefacts/governance" in PROTECTED

    def test_no_target_is_a_tracked_source_directory(self) -> None:
        """Een doel als `src` of `tests` zou de repository wissen."""
        assert not ({"src", "tests", "apps", "scripts", "research", "docs",
                     "conf", "reports", "artefacts", "data"} & set(TARGETS))


class TestItLeavesGitAlone:
    def test_it_never_removes_a_tracked_file(self, tmp_path: Path) -> None:
        tracked = _touch(tmp_path / "src" / "tradebot" / "engine.py", "print(1)\n")
        _touch(tmp_path / "logs" / "run.log")
        sweep(tmp_path, dry_run=False)
        assert tracked.exists()

    def test_the_git_directory_is_protected(self) -> None:
        assert ".git" in PROTECTED
