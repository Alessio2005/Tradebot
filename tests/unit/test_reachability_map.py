# tests/unit/test_reachability_map.py
"""De bereikbaarheidskaart moet bewijsbaar zijn vóór zij iets mag laten verwijderen.

Phase 9 verwijdert code op grond van één getal: de klasse die
`scripts/reachability_map.py` aan een module toekent. Een scanner die zijn eigen
resolver niet bewijst, levert geen meting maar een mening — en op een mening een
`git rm` doen is precies de stille degradatie waar de fail-fast-doctrine uit
Phase 0 tegen is opgetuigd.

WAT DEZE TEST AFDWINGT
======================
Vier resolvergevallen op een echte fixture-boom in `tmp_path`, want dat zijn de
vier manieren waarop een naïeve scanner een bereikte module ten onrechte
onbereikbaar noemt:

1. **Relatieve import.** `from .b import thing` noemt `b` nergens bij zijn
   volledige naam. Een regex ziet dat niet; een AST-resolver die de punt niet
   tegen het pakketpad oplost evenmin.
2. **`__init__`-re-export.** Een consument importeert `pkg.sub`; het bestand dat
   het werk doet is `pkg/sub/impl.py` en wordt uitsluitend via de re-export in
   `pkg/sub/__init__.py` bereikt.
3. **Hydra `_target_`-string.** Een dotted path in een YAML is een import zonder
   `import`-statement. In de echte boom staat er op dit moment nul van, maar een
   poort die dat geval niet kent, gaat stuk op de dag dat iemand er één toevoegt.
4. **Echt onbereikbaar.** Een module die door niets wordt genoemd, moet als
   klasse E uitkomen — anders meet de scanner niets en is klasse E per
   constructie leeg.

Daarnaast de vier klassegrenzen zelf (A/B/C/D) en de `--strict`-exitcode, want
`.github/workflows/inventory.yml` hangt eraan: een poort die niet rood kan
worden, is decoratie.

Ref: fase-opdracht Phase 9, stap 4 en deliverable 2.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reachability_map import Klass, scan


# ─────────────────────────────────────────────────────────────────────────────
# Fixture-boom
# ─────────────────────────────────────────────────────────────────────────────
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Een miniatuur van de echte repo-indeling, met één module per resolvergeval.

    Bereikbaarheid die deze boom bedoelt:

        dag_reached      A  — app_in_dag.py staat in dvc.yaml
        relative_target  A  — via de relatieve import in dag_reached.py
        hydra_only       A  — via de `_target_`-string in conf/model.yaml
        pkg.sub          B  — via app_off_dag.py, die niet in dvc.yaml staat
        pkg.sub.impl     B  — via de re-export in pkg/sub/__init__.py
        script_only      C  — alleen vanuit scripts/
        test_only        D  — alleen vanuit tests/
        orphan           E  — door niets
    """
    root = tmp_path

    pkg = root / "src" / "pkg"
    _write(pkg / "__init__.py", "")

    # 1. relatieve import: dag_reached noemt `relative_target` alleen met een punt.
    _write(
        pkg / "dag_reached.py",
        "from .relative_target import helper\n\n\ndef run() -> int:\n    return helper()\n",
    )
    _write(pkg / "relative_target.py", "def helper() -> int:\n    return 1\n")

    # 2. __init__-re-export: niemand noemt pkg.sub.impl rechtstreeks.
    _write(pkg / "sub" / "__init__.py", "from .impl import Worker\n\n__all__ = ['Worker']\n")
    _write(pkg / "sub" / "impl.py", "class Worker:\n    pass\n")

    # 3. Hydra: uitsluitend genoemd als dotted string in conf/.
    _write(pkg / "hydra_only.py", "class Estimator:\n    pass\n")

    # 4. onbereikbaar.
    _write(pkg / "orphan.py", "def never_called() -> None:\n    pass\n")

    # klasse C en D.
    _write(pkg / "script_only.py", "def research() -> None:\n    pass\n")
    _write(pkg / "test_only.py", "def contract() -> None:\n    pass\n")

    _write(root / "apps" / "app_in_dag.py", "from pkg.dag_reached import run\n")
    _write(root / "apps" / "app_off_dag.py", "import pkg.sub\n")
    _write(root / "scripts" / "research_script.py", "from pkg.script_only import research\n")
    _write(root / "tests" / "test_contract.py", "from pkg.test_only import contract\n")

    _write(root / "conf" / "model.yaml", "estimator:\n  _target_: pkg.hydra_only.Estimator\n")
    _write(root / "dvc.yaml", "stages:\n  build:\n    cmd: python apps/app_in_dag.py\n")

    return root


def _by_module(root: Path) -> dict[str, object]:
    return {row.module: row for row in scan(root)}


# ─────────────────────────────────────────────────────────────────────────────
# De vier resolvergevallen
# ─────────────────────────────────────────────────────────────────────────────
def test_relative_import_is_followed(tree: Path) -> None:
    """`from .relative_target import helper` bereikt de module.

    Zonder relatieve-importresolutie komt deze module als E uit en zou Phase 9
    hem verwijderen, terwijl de DAG erop draait.
    """
    rows = _by_module(tree)
    assert rows["pkg.relative_target"].klass is Klass.A


def test_init_reexport_is_followed(tree: Path) -> None:
    """`import pkg.sub` bereikt `pkg/sub/impl.py` via de re-export."""
    rows = _by_module(tree)
    assert rows["pkg.sub.impl"].klass is Klass.B


def test_hydra_target_string_counts_as_reach(tree: Path) -> None:
    """Een dotted `_target_`-string in conf/ is een import zonder importstatement."""
    rows = _by_module(tree)
    assert rows["pkg.hydra_only"].klass is not Klass.E


def test_unreachable_module_is_classified_e(tree: Path) -> None:
    """Een module die door niets wordt genoemd, is E — anders meet de scanner niets."""
    rows = _by_module(tree)
    assert rows["pkg.orphan"].klass is Klass.E


# ─────────────────────────────────────────────────────────────────────────────
# De klassegrenzen
# ─────────────────────────────────────────────────────────────────────────────
def test_dvc_reachable_module_is_class_a(tree: Path) -> None:
    rows = _by_module(tree)
    assert rows["pkg.dag_reached"].klass is Klass.A


def test_app_off_the_dag_gives_class_b(tree: Path) -> None:
    """Een app die niet in dvc.yaml staat levert B, niet A."""
    rows = _by_module(tree)
    assert rows["pkg.sub"].klass is Klass.B


def test_script_only_module_is_class_c(tree: Path) -> None:
    rows = _by_module(tree)
    assert rows["pkg.script_only"].klass is Klass.C


def test_test_only_module_is_class_d(tree: Path) -> None:
    rows = _by_module(tree)
    assert rows["pkg.test_only"].klass is Klass.D


# ─────────────────────────────────────────────────────────────────────────────
# De rij draagt zijn eigen bewijs
# ─────────────────────────────────────────────────────────────────────────────
def test_row_names_the_seed_that_reaches_it(tree: Path) -> None:
    """Een klasse zonder seed is niet navolgbaar; het inventarisrapport eist de seed."""
    rows = _by_module(tree)
    assert rows["pkg.relative_target"].seed == "apps/app_in_dag.py"
    assert rows["pkg.orphan"].seed == ""


def test_loc_is_the_raw_line_count(tree: Path) -> None:
    """LOC is `wc -l`; de nulmeting van 71.487 is op die definitie gemaakt."""
    rows = _by_module(tree)
    expected = len((tree / "src" / "pkg" / "orphan.py").read_text(encoding="utf-8").splitlines())
    assert rows["pkg.orphan"].loc == expected


# ─────────────────────────────────────────────────────────────────────────────
# De poort
# ─────────────────────────────────────────────────────────────────────────────
def test_strict_exits_nonzero_when_class_e_is_not_empty(tree: Path) -> None:
    """`--strict` is de CI-poort. Kan hij niet rood worden, dan bewaakt hij niets."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reachability_map.py"), "--root", str(tree), "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "pkg.orphan" in proc.stdout + proc.stderr


def test_strict_exits_zero_once_the_orphan_is_gone(tree: Path) -> None:
    """De negatieve controle: dezelfde poort staat groen zodra E leeg is."""
    (tree / "src" / "pkg" / "orphan.py").unlink()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reachability_map.py"), "--root", str(tree), "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_json_output_carries_every_module(tree: Path, tmp_path: Path) -> None:
    """Het JSON-artefact is de invoer voor het inventarisrapport."""
    out = tmp_path / "map.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reachability_map.py"),
            "--root", str(tree),
            "--json", str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {m["module"] for m in payload["modules"]} == set(_by_module(tree))


# ─────────────────────────────────────────────────────────────────────────────
# Dekking erbij — het inventarisrapport heeft klasse ÉN dekking per rij nodig
# ─────────────────────────────────────────────────────────────────────────────
def test_row_carries_coverage_when_a_coverage_json_is_given(tree: Path, tmp_path: Path) -> None:
    """Exit-criterium 4 eist per rij een klasse en een dekking; de rij draagt beide."""
    cov = tmp_path / "coverage.json"
    cov.write_text(
        json.dumps(
            {
                "files": {
                    "src/pkg/orphan.py": {"summary": {"percent_covered": 12.5}},
                    "src/pkg/dag_reached.py": {"summary": {"percent_covered": 100.0}},
                }
            }
        ),
        encoding="utf-8",
    )
    rows = {r.module: r for r in scan(tree, coverage=cov)}
    assert rows["pkg.orphan"].coverage == pytest.approx(12.5)
    assert rows["pkg.dag_reached"].coverage == pytest.approx(100.0)


def test_coverage_is_none_for_a_module_the_report_does_not_mention(tree: Path, tmp_path: Path) -> None:
    """Nul is een meting, ontbreken is er geen. Het verschil mag niet wegvallen."""
    cov = tmp_path / "coverage.json"
    cov.write_text(json.dumps({"files": {}}), encoding="utf-8")
    rows = {r.module: r for r in scan(tree, coverage=cov)}
    assert rows["pkg.orphan"].coverage is None


def test_coverage_is_none_when_no_report_is_given(tree: Path) -> None:
    rows = {r.module: r for r in scan(tree)}
    assert rows["pkg.orphan"].coverage is None


def test_json_output_carries_the_coverage_column(tree: Path, tmp_path: Path) -> None:
    """`reports/phase9_inventory.md` wordt uit dit JSON gebouwd; zonder dekking
    kan exit-criterium 4 (klasse EN dekking per rij) er niet uit."""
    cov = tmp_path / "coverage.json"
    cov.write_text(
        json.dumps({"files": {"src/pkg/orphan.py": {"summary": {"percent_covered": 42.0}}}}),
        encoding="utf-8",
    )
    out = tmp_path / "map.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reachability_map.py"),
            "--root", str(tree),
            "--coverage", str(cov),
            "--json", str(out),
        ],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    row = next(m for m in payload["modules"] if m["module"] == "pkg.orphan")
    assert row["coverage"] == pytest.approx(42.0)


# ─────────────────────────────────────────────────────────────────────────────
# De research/-track (stap 9)
# ─────────────────────────────────────────────────────────────────────────────
def test_a_module_reached_only_from_research_is_class_c(tree: Path) -> None:
    """`research/` is een seedmap naast `scripts/`, en levert dezelfde klasse.

    Stap 9 verhuist de wave-onderzoeksscripts van `scripts/` naar `research/`.
    Kent de scanner die map niet, dan verliezen de modules die zij bereiken hun
    seed en zakken zij naar D of E -- waarna `--strict` verwijdering zou eisen
    van code die research wel degelijk gebruikt.
    """
    _write(tree / "src" / "pkg" / "research_only.py", "def probe() -> None:\n    pass\n")
    _write(tree / "research" / "w99_eval.py", "from pkg.research_only import probe\n")
    rows = {r.module: r for r in scan(tree)}
    assert rows["pkg.research_only"].klass is Klass.C
    assert rows["pkg.research_only"].seed == "research/w99_eval.py"
