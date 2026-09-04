# tests/unit/test_file_size_ratchet.py
"""De LOC-poort moet een ratchet zijn, geen vrijbrief.

Phase 9, stap 13. De poort die er stond, bewaakte niets, en dat is op drie
manieren gemeten:

1. **Hij draaide niet.** De logica stond in de `Makefile` en `make` bestaat niet
   in deze omgeving. Een poort die niet uitvoerbaar is, is documentatie.
2. **Zijn whitelist kende zijn eigen scope niet.** De lijst noemde
   `risk/portfolio.py`, dat sinds Phase 4 niet meer bestaat — `DEFERRED_ISSUES.md`
   DI-10 zegt dat zelf al.
3. **De ratchet die DI-3 claimt, bestond niet.** DI-3 noteert *"gemeten
   2026-09-01: nog 2 bestanden >800 LOC"*; gemeten 2026-09-04 zijn het er
   **negen**, en de twee die DI-3 noemt staan er met andere getallen in
   (`labeling/meta.py` 838 → 1.181, `backtest/evaluation.py` 833 → 1.054). Een
   ratchet die verdere groei zou blokkeren, heeft die groei niet geblokkeerd.

WAT DEZE POORT WEL DOET
=======================
Een blanket-whitelist van negen bestanden is een poort die alles doorlaat.
Daarom is de uitzondering hier een **CAP per bestand** op de gemeten waarde:
het bestand mag blijven bestaan op zijn huidige omvang en mag niet groeien. Wie
er een regel bij schrijft, wordt rood en moet ofwel splitsen ofwel de cap
bewust verhogen — een gelezen besluit in een diff.

Elke uitzondering draagt bovendien een `# LOC-EXCEPTION:`-regel in de header van
het bestand zelf, zodat de reden staat waar iemand hem tegenkomt.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.check_file_size import CAPS, HEADER_EXEMPT, LIMIT, main, oversized


def _write(path: Path, lines: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * lines, encoding="utf-8")


class TestTheGateCatchesGrowth:
    def test_a_file_over_the_limit_without_a_cap_is_reported(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "pkg" / "big.py", LIMIT + 1)
        assert [p for p, _, _ in oversized(tmp_path, caps={})] == ["src/pkg/big.py"]

    def test_a_file_at_exactly_the_limit_passes(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "pkg" / "edge.py", LIMIT)
        assert oversized(tmp_path, caps={}) == []

    def test_a_capped_file_at_its_cap_passes(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "pkg" / "legacy.py", 1000)
        assert oversized(tmp_path, caps={"src/pkg/legacy.py": 1000}) == []

    def test_a_capped_file_that_grows_by_one_line_is_reported(self, tmp_path: Path) -> None:
        """Dit is de ratchet. Zonder deze test is de cap een vrijbrief."""
        _write(tmp_path / "src" / "pkg" / "legacy.py", 1001)
        found = oversized(tmp_path, caps={"src/pkg/legacy.py": 1000})
        assert [p for p, _, _ in found] == ["src/pkg/legacy.py"]
        assert found[0][1] == 1001 and found[0][2] == 1000

    def test_a_capped_file_that_shrinks_passes(self, tmp_path: Path) -> None:
        """Krimpen mag altijd; de cap is een plafond, geen doel."""
        _write(tmp_path / "src" / "pkg" / "legacy.py", 900)
        assert oversized(tmp_path, caps={"src/pkg/legacy.py": 1000}) == []


class TestTheExitCode:
    def test_it_returns_zero_when_nothing_is_oversized(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "pkg" / "small.py", 10)
        assert main(["--root", str(tmp_path), "--caps", ""]) == 0

    def test_it_returns_one_when_something_is_oversized(self, tmp_path: Path) -> None:
        _write(tmp_path / "src" / "pkg" / "big.py", LIMIT + 1)
        assert main(["--root", str(tmp_path), "--caps", ""]) == 1


class TestTheCapsMatchTheRepository:
    """De cap-tabel mag niet verrotten, in geen van beide richtingen."""

    def test_every_capped_file_exists(self) -> None:
        """De oude whitelist noemde `risk/portfolio.py`, dat niet bestaat."""
        missing = [rel for rel in CAPS if not (ROOT / rel).is_file()]
        assert not missing, f"cap voor een niet-bestaand bestand: {missing}"

    def test_no_capped_file_is_below_the_limit(self) -> None:
        """Een cap op een bestand dat de limiet niet meer haalt, hoort weg."""
        shrunk = [
            rel for rel in CAPS
            if len((ROOT / rel).read_text(encoding="utf-8").splitlines()) <= LIMIT
        ]
        assert not shrunk, (
            f"deze bestanden zitten weer onder {LIMIT} LOC; haal hun cap weg: {shrunk}"
        )

    def test_every_capped_file_carries_its_reason_in_its_own_header(self) -> None:
        """De reden hoort waar iemand hem tegenkomt, niet alleen in een lijst.

        Uitgezonderd is `portfolio/legacy_sizing.py`: DI-10 houdt dat bestand
        ONGEWIJZIGD zodat de Phase 3-baseline herrekenbaar blijft, en er een
        commentaarregel in schrijven zou die afspraak schenden. Zijn reden staat
        bij `HEADER_EXEMPT` in de scanner.
        """
        without = [
            rel for rel in CAPS
            if rel not in HEADER_EXEMPT
            and "# LOC-EXCEPTION:" not in (ROOT / rel).read_text(encoding="utf-8")
        ]
        assert not without, f"geen `# LOC-EXCEPTION:`-regel in: {without}"

    def test_the_header_exemption_is_not_a_blanket(self) -> None:
        """Precies een bestand mag zonder header, en het draagt zijn reden elders."""
        assert HEADER_EXEMPT == {"src/tradebot/portfolio/legacy_sizing.py"}
        assert HEADER_EXEMPT <= set(CAPS)

    def test_the_repository_passes_its_own_gate(self) -> None:
        """Exit-criterium 7 op de echte boom."""
        assert oversized(ROOT, CAPS) == []


class TestTheGateCanGoRed:
    """Zonder deze test bewaakt de test hierboven niets."""

    def test_adding_a_line_to_a_capped_file_would_fail_the_gate(self) -> None:
        rel, cap = next(iter(CAPS.items()))
        pretend = dict(CAPS)
        pretend[rel] = cap - 1          # alsof het bestand een regel is gegroeid
        found = oversized(ROOT, pretend)
        assert [p for p, _, _ in found] == [rel]

    def test_an_uncapped_oversized_file_would_fail_the_gate(self) -> None:
        found = oversized(ROOT, caps={})
        assert set(CAPS) == {p for p, _, _ in found}, (
            "zonder caps horen exact de gecapte bestanden op te lichten"
        )
