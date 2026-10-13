# tests/unit/test_ci_runs_every_test.py
"""De CI-pijplijn draait elke test, en geen workflow selecteert een lege verzameling.

Twee fouten maakten dit nodig, allebei gemeten op `main`:

* `ci.yml` draaide `-m "not slow and not regression"` en niets anders. De 17
  getagde tests (bit-equivalentie, garch, benchmarks, de e2e-smoke) liepen alleen
  in nachtelijke workflows en konden dus pas na een merge rood worden.
* Die nachtelijke workflows selecteerden op markers die niet bestonden of niet
  gebruikt werden: `-m benchmark` koos 0 van 2.822 tests (pytest-exit 5) en
  `-m e2e` is onbekend onder `--strict-markers` (pytest-exit 4). Beide jobs
  stonden dagen rood zonder dat een test iets fout deed.

Deze test leest de workflows zoals ze er staan en controleert de selecties tegen
de markers in `pyproject.toml` en het gebruik ervan onder `tests/`.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest
import tomllib
import yaml
from _pytest.mark.expression import Expression

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
THIS_FILE = Path(__file__).resolve()
_KEYWORDS = {"and", "or", "not"}

_PYTEST_CALL = re.compile(r"(?:^|\s)(?:python\s+-m\s+)?pytest\b(?P<args>[^\n]*)")
_MARKER_EXPR = re.compile(r"""-m\s+(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>\S+))""")


def _registered_markers() -> set[str]:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lines = cfg["tool"]["pytest"]["ini_options"]["markers"]
    return {line.split(":", 1)[0].strip() for line in lines}


def _joined(run: str) -> str:
    return re.sub(r"\\\s*\n\s*", " ", run)


def _selections(workflow: Path) -> list[tuple[str, str | None]]:
    """(job, `-m` expression or None) for every `pytest` call in a workflow's run steps."""
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    out: list[tuple[str, str | None]] = []
    for job_name, job in doc["jobs"].items():
        for step in job.get("steps", []):
            run = step.get("run")
            if not run:
                continue
            for call in _PYTEST_CALL.finditer(_joined(run)):
                m = _MARKER_EXPR.search(call.group("args"))
                expr = (m.group("dq") or m.group("sq") or m.group("bare")) if m else None
                out.append((job_name, expr))
    return out


def _selects(expr: str | None, markers: frozenset[str]) -> bool:
    if expr is None:
        return True
    return Expression.compile(expr).evaluate(lambda name, **_: name in markers)


def _used_marker_names() -> set[str]:
    used: set[str] = set()
    for path in (ROOT / "tests").rglob("*.py"):
        if path.resolve() == THIS_FILE:
            continue
        used.update(re.findall(r"pytest\.mark\.([A-Za-z_]\w*)", path.read_text(encoding="utf-8")))
    return used


def _expression_names(expr: str) -> set[str]:
    return {tok for tok in re.findall(r"[A-Za-z_]\w*", expr) if tok not in _KEYWORDS}


class TestEveryTestRunsInCi:
    def test_ci_workflow_selects_every_kind_of_test(self) -> None:
        selections = [expr for _job, expr in _selections(WORKFLOWS / "ci.yml")]
        assert selections, "ci.yml runs no pytest at all"
        registered = sorted(_registered_markers())
        kinds = [frozenset()] + [frozenset({m}) for m in registered]
        kinds += [frozenset(pair) for pair in itertools.combinations(registered, 2)]
        unselected = [sorted(k) for k in kinds if not any(_selects(e, k) for e in selections)]
        assert not unselected, (
            f"ci.yml selects none of the tests carrying these markers: {unselected}. "
            "Add them to a job's -m expression or to a new job.")

    def test_the_fast_and_the_slow_job_are_both_there(self) -> None:
        jobs = {job for job, _ in _selections(WORKFLOWS / "ci.yml")}
        assert {"test-fast", "test-slow"} <= jobs

    def test_ci_does_not_swallow_failures(self) -> None:
        text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        assert "continue-on-error" not in text
        assert not re.search(r"pytest[^\n]*\|\|\s*true", text)

    def test_the_check_can_go_red(self) -> None:
        """A selection that skips `slow` leaves slow tests unselected, so the check above would fail."""
        only_fast = ["not slow and not regression"]
        assert not any(_selects(e, frozenset({"slow"})) for e in only_fast)
        assert _selects(None, frozenset({"anything"}))


class TestNoWorkflowSelectsNothing:
    @pytest.mark.parametrize("workflow", sorted(WORKFLOWS.glob("*.yml")), ids=lambda p: p.name)
    def test_every_marker_in_a_selection_is_registered_and_used(self, workflow: Path) -> None:
        registered, used = _registered_markers(), _used_marker_names()
        for job, expr in _selections(workflow):
            if expr is None:
                continue
            for name in _expression_names(expr):
                assert name in registered, (
                    f"{workflow.name}/{job}: -m {expr!r} names the unregistered marker {name!r} "
                    "(--strict-markers turns that into a pytest usage error, exit 4)")
                if name in _positively_selected(expr):
                    assert name in used, (
                        f"{workflow.name}/{job}: -m {expr!r} selects on {name!r}, "
                        "but no test carries it (pytest exits 5 on an empty selection)")


def _positively_selected(expr: str) -> set[str]:
    """Marker names whose presence is REQUIRED somewhere in the expression, i.e. not only under `not`."""
    return {tok for tok in _expression_names(expr)
            if not re.search(rf"\bnot\s+{re.escape(tok)}\b", expr) or re.search(rf"(?<!not )\b{re.escape(tok)}\b", expr)}
