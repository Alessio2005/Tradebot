"""De research-gates achter één aanroep — Stage B-5.

`.github/workflows/research_gates.yml` draait DEZE functie, en een lokale
gate-run hoort dat ook te doen: een lokale run die iets anders meet dan de
CI-run, geeft een groen licht dat niets voorspelt.

DE DRIE POORTEN
===============
1. `lookahead_suite`  — de zes benoemde D-1-poorten plus de bestaande
                        causaliteitsdekking.
2. `promotion_gates`  — de poort zelf, de state machine en Hansen's SPA-kern.
3. `gate_killgate`    — de negatieve controle: het `shift(-1)`-model MOET worden
                        geweigerd.

EEN GEDRAAIDE POORT DIE NIETS MAT, IS EEN GEFAALDE POORT
========================================================
`min_passed` is het onderdeel van deze module dat het meeste werk doet.

`pytest` geeft exit 0 wanneer ALLE tests worden overgeslagen. Een ontbrekende
PIT-store, een verkeerd pad, een `-k`-filter die niets selecteert, een
`skipif`-conditie die per ongeluk altijd waar is — in al die gevallen staat de
gate groen en is er nul bewijs geproduceerd. Dat is fase-no-go 18 (*"een
permanent overgeslagen test houdt de suite ten onrechte groen"*), en Stage A
vond dezelfde klasse fout al in de lint-gate.

Elke poort declareert daarom hoeveel tests er MINIMAAL moeten SLAGEN. Onder dat
aantal faalt de poort, ook bij exit 0.

De tellingen komen uit JUnit-XML en niet uit een regex op de uitvoer: het
uitvoerformaat van pytest is geen contract.

Ref: `fase_7_8_consolidatie_productie.md` §B-5, §B-6; audit §17.1.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["GATE_SPECS", "GateRun", "GateSpec", "run_research_gates"]


@dataclass(frozen=True)
class GateSpec:
    """Eén poort: wat hij draait en hoeveel bewijs hij minimaal moet leveren."""

    name: str
    #: Argumenten NA de interpreter. `{python}` wordt niet gesubstitueerd; de
    #: aanroeper geeft het interpreterpad mee.
    args: tuple[str, ...]
    why: str
    #: Nul betekent: dit is geen pytest-run en er valt niets te tellen.
    min_passed: int = 0
    is_pytest: bool = True


#: De drie poorten, in de volgorde waarin ze draaien.
GATE_SPECS: tuple[GateSpec, ...] = (
    GateSpec(
        name="lookahead_suite",
        args=("-m", "pytest", "tests/lookahead", "-q", "-p", "no:randomly"),
        why="De zes D-1-poorten. Dit is de enige poort die een lekkend model "
            "tegenhoudt; de statistiek doet dat aantoonbaar niet.",
        min_passed=400,
    ),
    GateSpec(
        name="promotion_gates",
        args=("-m", "pytest", "tests/unit/test_promotion_gates.py",
              "tests/unit/test_model_lifecycle.py",
              "tests/unit/test_spa_hansen.py",
              "-q", "-p", "no:randomly"),
        why="De poort, de state machine en Hansen's SPA-kern.",
        min_passed=80,
    ),
    GateSpec(
        name="gate_killgate",
        args=("-m", "pytest", "tests/killgates/test_gate_cannot_be_bypassed.py",
              "-q", "-p", "no:randomly"),
        why="De negatieve controle op de poort zelf: het shift(-1)-model MOET "
            "worden geweigerd (exit criterium B7).",
        min_passed=15,
    ),
)


@dataclass(frozen=True)
class GateRun:
    """De uitkomst van één poort, met alles wat nodig is om hem na te rekenen."""

    name: str
    command: tuple[str, ...]
    returncode: int
    passed: bool
    reason: str
    counts: dict[str, int] = field(default_factory=dict)
    stdout_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "returncode": self.returncode,
            "passed": self.passed,
            "reason": self.reason,
            "counts": dict(self.counts),
        }

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        counts = (f"  ({self.counts.get('passed', 0)} passed, "
                  f"{self.counts.get('skipped', 0)} skipped)"
                  if self.counts else "")
        return f"[{mark}] {self.name}{counts}\n       {self.reason}"


def _junit_counts(path: Path) -> dict[str, int]:
    """Tellingen uit JUnit-XML. Het uitvoerformaat van pytest is geen contract."""
    if not path.is_file():
        return {}
    # nosec B314 -- BEOORDEELD EN AANVAARD, niet genegeerd.
    # `path` is de JUnit-XML die pytest zelf een paar regels eerder in
    # DEZELFDE run heeft geschreven; er komt geen invoer van buiten dit
    # proces langs. `defusedxml` toevoegen zou een HARDE dependency zijn
    # (pyproject.toml par. "DOCTRINE": elke dependency hier is hard) voor een
    # dreiging die op dit pad niet bestaat. Verandert de herkomst van dit
    # bestand ooit, dan vervalt deze motivering en hoort defusedxml er wel.
    root = ET.parse(path).getroot()  # nosec B314
    suites = [root] if root.tag == "testsuite" else list(root)
    total = failures = errors = skipped = 0
    for s in suites:
        total += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
    return {
        "total": total, "failures": failures, "errors": errors,
        "skipped": skipped, "passed": total - failures - errors - skipped,
    }


def _run_one(spec: GateSpec, *, root: Path, python: str) -> GateRun:
    args = list(spec.args)
    xml_path: Path | None = None
    tmpdir: tempfile.TemporaryDirectory[str] | None = None

    if spec.is_pytest:
        tmpdir = tempfile.TemporaryDirectory(prefix="tradebot-gates-")
        xml_path = Path(tmpdir.name) / f"{spec.name}.xml"
        args += [f"--junit-xml={xml_path}"]

    command = (python, *args)
    try:
        proc = subprocess.run(
            command, cwd=str(root), capture_output=True, text=True,
            check=False,
        )
        # De tellingen worden gelezen VOORDAT de tijdelijke map verdwijnt.
        counts = _junit_counts(xml_path) if xml_path else {}
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    stdout_tail = (proc.stdout or "")[-2000:]

    if proc.returncode != 0:
        return GateRun(
            spec.name, command, proc.returncode, False,
            f"exit {proc.returncode} — {spec.why}", counts, stdout_tail)

    # EXIT 0 IS NIET GENOEG. Zie de moduledocstring: een suite waarvan alles is
    # overgeslagen, geeft exit 0 en nul bewijs.
    if spec.is_pytest and spec.min_passed:
        n_passed = counts.get("passed", 0)
        if n_passed < spec.min_passed:
            return GateRun(
                spec.name, command, proc.returncode, False,
                f"exit 0, maar slechts {n_passed} test(s) GESLAAGD tegen een "
                f"ondergrens van {spec.min_passed} "
                f"({counts.get('skipped', 0)} overgeslagen van "
                f"{counts.get('total', 0)}). Een poort die niets meet, is geen "
                f"poort — {spec.why}",
                counts, stdout_tail)

    return GateRun(spec.name, command, proc.returncode, True, spec.why,
                   counts, stdout_tail)


def run_research_gates(
    root: Path,
    *,
    python: str | None = None,
    specs: tuple[GateSpec, ...] = GATE_SPECS,
    stop_on_first_failure: bool = False,
) -> list[GateRun]:
    """Draai elke poort en geef per poort een oordeel terug.

    Parameters
    ----------
    stop_on_first_failure
        Standaard False: CI hoort ALLE poorten te melden, niet alleen de eerste.
        Wie lokaal snel wil itereren, zet hem op True.
    """
    interpreter = python or sys.executable
    runs: list[GateRun] = []
    for spec in specs:
        run = _run_one(spec, root=root, python=interpreter)
        runs.append(run)
        if stop_on_first_failure and not run.passed:
            break
    return runs
