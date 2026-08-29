"""De negatieve controle op de gate-runner zelf — Stage B-5.

`research_gates.yml` draait `apps/run_gates.py`, die `run_research_gates`
draait. Is die runner stuk, dan is de workflow groen zonder één poort te
draaien, en dan is D-1 opnieuw een bewering.

Dat is geen theoretisch risico. Stage A vond precies deze klasse fout in de
lint-gate van `hygiene.yml`: hij draaide, hij was groen, en zijn oordeel hing af
van de releasedatum van `ruff`. Een gate die niets meet en groen staat, is
duurder dan geen gate, omdat iedereen erop vertrouwt.

DE BELANGRIJKSTE TEST HIER
==========================
`test_a_fully_skipped_suite_is_a_failure`. `pytest` geeft **exit 0** wanneer
alle tests worden overgeslagen — een ontbrekende PIT-store, een verkeerd pad,
een `skipif` die per ongeluk altijd waar is. Zonder de `min_passed`-ondergrens
staat de research-gate dan groen met nul bewijs. Dat is fase-no-go 18.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tradebot.validation.gate_runner import (
    GATE_SPECS,
    GateSpec,
    run_research_gates,
)

ROOT = Path(__file__).resolve().parents[2]

_ALL_SKIPPED = '''
import pytest

@pytest.mark.skipif(True, reason="PIT-store leeg")
def test_one():
    assert True

@pytest.mark.skipif(True, reason="PIT-store leeg")
def test_two():
    assert True
'''

_ALL_PASS = '''
def test_one():
    assert True

def test_two():
    assert True

def test_three():
    assert True
'''

_ONE_FAILS = '''
def test_one():
    assert True

def test_two():
    assert False, "bewust rood"
'''


def _spec(name: str, target: Path, *, min_passed: int) -> GateSpec:
    return GateSpec(
        name=name,
        args=("-m", "pytest", str(target), "-q", "-p", "no:randomly"),
        why="synthetische poort voor de negatieve controle",
        min_passed=min_passed,
    )


@pytest.fixture
def suite(tmp_path: Path):
    def _write(body: str, name: str) -> Path:
        path = tmp_path / f"test_{name}.py"
        path.write_text(body, encoding="utf-8")
        return path
    return _write


class TestTheRunnerCanGoRed:
    def test_a_failing_suite_fails_the_gate(self, tmp_path, suite) -> None:
        target = suite(_ONE_FAILS, "one_fails")
        runs = run_research_gates(
            tmp_path, specs=(_spec("synthetic", target, min_passed=1),))
        assert len(runs) == 1
        assert runs[0].passed is False
        assert runs[0].returncode != 0
        assert runs[0].counts["failures"] == 1

    def test_a_fully_skipped_suite_is_a_failure(self, tmp_path, suite) -> None:
        """DE KERNTEST. Exit 0 met nul geslaagde tests is geen groen licht.

        Zou `min_passed` ontbreken, dan zou deze poort slagen terwijl er nul
        bewijs is geproduceerd — en precies dat gebeurt in CI zodra `dvc pull`
        faalt en de hele lookahead-suite overslaat.
        """
        target = suite(_ALL_SKIPPED, "all_skipped")
        runs = run_research_gates(
            tmp_path, specs=(_spec("synthetic", target, min_passed=1),))
        assert runs[0].returncode == 0, "pytest gaf geen exit 0 op een volledig overgeslagen suite; deze test meet dan iets anders dan hij beweert"
        assert runs[0].passed is False, (
            "een volledig overgeslagen suite werd als geslaagde poort geteld")
        assert "GESLAAGD" in runs[0].reason
        assert runs[0].counts["skipped"] == 2
        assert runs[0].counts["passed"] == 0

    def test_the_same_suite_without_a_floor_would_have_passed(
        self, tmp_path, suite
    ) -> None:
        """Bewijs dat de ondergrens het verschil maakt en niet iets anders."""
        target = suite(_ALL_SKIPPED, "all_skipped_nofloor")
        runs = run_research_gates(
            tmp_path, specs=(_spec("synthetic", target, min_passed=0),))
        assert runs[0].passed is True

    def test_a_passing_suite_passes(self, tmp_path, suite) -> None:
        """De weigering is gericht. Een runner die alles rood maakt, is net zo
        nutteloos als een die alles groen maakt."""
        target = suite(_ALL_PASS, "all_pass")
        runs = run_research_gates(
            tmp_path, specs=(_spec("synthetic", target, min_passed=3),))
        assert runs[0].passed is True
        assert runs[0].counts["passed"] == 3

    def test_one_test_short_of_the_floor_is_red(self, tmp_path, suite) -> None:
        """De ondergrens is een drempel en geen richtlijn — dezelfde regel die
        voor DM p < 0,05 geldt."""
        target = suite(_ALL_PASS, "all_pass_short")
        runs = run_research_gates(
            tmp_path, specs=(_spec("synthetic", target, min_passed=4),))
        assert runs[0].passed is False

    def test_a_missing_target_is_a_failure_not_a_skip(self, tmp_path) -> None:
        """pytest geeft exit 4 op een niet-bestaand pad. Een typefout in een
        spec mag geen groene poort opleveren."""
        runs = run_research_gates(
            tmp_path,
            specs=(_spec("synthetic", tmp_path / "test_nope.py", min_passed=1),))
        assert runs[0].passed is False


class TestNonPytestGates:
    def test_a_non_zero_exit_from_a_script_fails(self, tmp_path) -> None:
        script = tmp_path / "boom.py"
        script.write_text("import sys; sys.exit(3)", encoding="utf-8")
        runs = run_research_gates(tmp_path, specs=(
            GateSpec(name="script", args=(str(script),), why="synthetisch",
                     is_pytest=False),))
        assert runs[0].passed is False
        assert runs[0].returncode == 3

    def test_a_zero_exit_from_a_script_passes(self, tmp_path) -> None:
        script = tmp_path / "ok.py"
        script.write_text("print('ok')", encoding="utf-8")
        runs = run_research_gates(tmp_path, specs=(
            GateSpec(name="script", args=(str(script),), why="synthetisch",
                     is_pytest=False),))
        assert runs[0].passed is True


class TestFailFast:
    def test_stopping_early_reports_which_gates_did_not_run(
        self, tmp_path, suite
    ) -> None:
        """Niet-gedraaide poorten mogen nooit als geslaagd worden gelezen."""
        red = _spec("red", suite(_ONE_FAILS, "ff_red"), min_passed=1)
        green = _spec("green", suite(_ALL_PASS, "ff_green"), min_passed=1)
        runs = run_research_gates(
            tmp_path, specs=(red, green), stop_on_first_failure=True)
        assert [r.name for r in runs] == ["red"]

    def test_without_fail_fast_every_gate_reports(self, tmp_path, suite) -> None:
        """CI hoort ALLE rode poorten te melden, niet alleen de eerste: anders
        kost elke reparatie een volledige CI-ronde om de volgende te vinden."""
        red = _spec("red", suite(_ONE_FAILS, "af_red"), min_passed=1)
        green = _spec("green", suite(_ALL_PASS, "af_green"), min_passed=1)
        runs = run_research_gates(tmp_path, specs=(red, green))
        assert [r.name for r in runs] == ["red", "green"]
        assert [r.passed for r in runs] == [False, True]


class TestTheRealSpecsArePointedAtSomethingThatExists:
    """Een spec die naar een verdwenen pad wijst, faalt met een onduidelijke
    melding in plaats van met de reden dat het pad weg is."""

    @pytest.mark.parametrize("spec", GATE_SPECS, ids=[s.name for s in GATE_SPECS])
    def test_every_target_path_exists(self, spec: GateSpec) -> None:
        targets = [a for a in spec.args
                   if a.endswith(".py") or a.startswith("tests/")]
        assert targets, f"{spec.name} heeft geen doelpad in zijn argumenten"
        for target in targets:
            assert (ROOT / target).exists(), (
                f"{spec.name} wijst naar {target}, dat niet bestaat")

    def test_every_pytest_gate_declares_a_floor(self) -> None:
        """Een pytest-poort zonder ondergrens kan groen staan op nul bewijs."""
        for spec in GATE_SPECS:
            if spec.is_pytest:
                assert spec.min_passed > 0, (
                    f"{spec.name} draait pytest zonder min_passed en kan dus "
                    f"groen staan terwijl alles is overgeslagen")

    def test_the_gate_names_are_unique(self) -> None:
        names = [s.name for s in GATE_SPECS]
        assert len(names) == len(set(names))
