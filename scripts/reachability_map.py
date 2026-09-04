"""AST-bereikbaarheidskaart over `src/` — Phase 9, deliverable 1.

Deze scanner beantwoordt één vraag per module: **door welk entrypoint wordt hij
bereikt?** Phase 9 verwijdert code uitsluitend op grond van dat antwoord, dus de
scanner is geen rapportagehulpje maar het meetinstrument waar de opruiming op
rust. Zijn resolver wordt bewezen in `tests/unit/test_reachability_map.py`.

WAAROM AST EN GEEN REGEX
========================
`from .b import thing` noemt `b` nergens bij zijn volledige naam. Een tekstzoek
naar `import tradebot.x.y` ziet die regel niet, classificeert `b` als
onbereikbaar, en een opruimfase die daarop vertrouwt verwijdert een module die
in de DAG draait. Hetzelfde geldt voor `from pkg import sub` — de import van een
submodule zonder dat de submodulenaam ooit als dotted pad in de tekst staat.

DE VIJF KLASSEN
===============
Toegekend op precedentie A > B > C > D > E: een module die zowel vanuit de DAG
als vanuit een test bereikbaar is, is A. Wat overblijft is E.

    A  authoritative   bereikbaar vanuit een `dvc.yaml`-stage
    B  operationeel    bereikbaar vanuit een app die niet in de DAG staat
    C  research        uitsluitend bereikbaar vanuit `scripts/`
    D  test-only       uitsluitend bereikbaar vanuit `tests/`
    E  onbereikbaar    door niets bereikt

HYDRA-`_target_`-STRINGS
========================
Een dotted path in `conf/**/*.yaml` is een import zonder importstatement. Zulke
targets tellen als seed in klasse **A**: `conf/` is de configuratie van de
authoritative keten, en zonder te traceren wélk entrypoint de config leest, is A
de conservatieve toekenning — "bij twijfel behouden" is een regel van deze fase.
In de boom van 2026-09-04 staat er nul van; de afhandeling bestaat zodat de
poort niet stukgaat op de dag dat iemand er één toevoegt.

WAT DEZE SCANNER NIET ZIET
==========================
Import-bereikbaarheid, meer niet. Drie klassen treffers zijn daarom onbetrouwbaar
en mogen nooit op grond van deze uitvoer alleen worden verwijderd:
`importlib`/`__subclasses__`-registratie, aanroep als `python -m tradebot.x.y`
vanaf de commandoregel, en `__init__.py`-bestanden waarvan consumenten de
submodules rechtstreeks importeren. Zie `reports/phase9_inventory.md` §false
positives voor de handmatige audit die daarbij hoort.

GEBRUIK
=======
    python scripts/reachability_map.py --json reports/phase9_reachability.json \\
                                       --markdown reports/phase9_reachability.md
    python scripts/reachability_map.py --strict     # exit 1 zodra klasse E niet leeg is
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Mappen buiten `src/` waarvan de bestanden als seed dienen.
ENTRYPOINT_DIRS = ("apps", "scripts", "tests")

#: `_target_: pkg.mod.Klasse` in een Hydra-config.
HYDRA_TARGET = re.compile(r"^\s*(?:-\s*)?_target_\s*:\s*['\"]?([A-Za-z_][\w.]*)['\"]?\s*$", re.MULTILINE)

#: `python apps/foo.py` of `python -m apps.foo` in een dvc.yaml-`cmd`.
DVC_CMD_PATH = re.compile(r"(?:python\s+(?:-m\s+)?)([\w./]+)")


class Klass(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


@dataclass(frozen=True)
class Row:
    """Eén regel van de kaart: de module, zijn omvang, zijn klasse, zijn seed."""

    module: str
    path: str
    loc: int
    klass: Klass
    seed: str
    #: Regeldekking in procent uit een `coverage.py`-JSON, of None wanneer er
    #: geen rapport is meegegeven of het rapport dit bestand niet noemt. Nul is
    #: een meting, ontbreken is er geen; dat verschil mag niet wegvallen.
    coverage: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Naamgeving
# ─────────────────────────────────────────────────────────────────────────────
def _module_name(py: Path, base: Path) -> str:
    """`src/pkg/sub/impl.py` -> `pkg.sub.impl`; `src/pkg/__init__.py` -> `pkg`."""
    parts = list(py.relative_to(base).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(module: str, is_package: bool) -> str:
    """De package waartegen een relatieve import in deze module oplost."""
    if is_package:
        return module
    return module.rpartition(".")[0]


def _ancestors(dotted: str) -> list[str]:
    """`a.b.c` -> [`a`, `a.b`] — bovenliggende packages draaien mee bij import."""
    parts = dotted.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


# ─────────────────────────────────────────────────────────────────────────────
# Importextractie
# ─────────────────────────────────────────────────────────────────────────────
def _imported_names(source: str, package: str) -> set[str]:
    """Elke dotted naam die dit bestand importeert, relatief opgelost."""
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                trim = node.level - 1
                base = ".".join(parts[: len(parts) - trim]) if trim <= len(parts) else ""
            else:
                base = ""
            full = ".".join(p for p in (base, node.module or "") if p)
            if not full:
                continue
            names.add(full)
            # `from pkg.sub import impl` -- impl kan een submodule zijn.
            for alias in node.names:
                if alias.name != "*":
                    names.add(f"{full}.{alias.name}")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# De graaf
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph(root: Path) -> tuple[dict[str, set[str]], dict[str, str], dict[str, Path]]:
    """Bouw de importgraaf over `src/` plus de entrypointbestanden.

    Knopen zijn dotted namen: `tradebot.risk.var` voor `src/`, en `apps.doctor`
    voor `apps/doctor.py`. Retourneert (edges, node->pad, pad->bestand).
    """
    src = root / "src"
    files: dict[str, Path] = {}

    if src.is_dir():
        for py in sorted(src.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            files[_module_name(py, src)] = py

    for d in ENTRYPOINT_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for py in sorted(base.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            files[_module_name(py, root)] = py

    edges: dict[str, set[str]] = {}
    for node, py in files.items():
        is_package = py.name == "__init__.py"
        package = _package_of(node, is_package)
        source = py.read_text(encoding="utf-8", errors="replace")
        targets: set[str] = set()
        for name in _imported_names(source, package):
            for candidate in (name, *_ancestors(name)):
                if candidate in files and candidate != node:
                    targets.add(candidate)
        edges[node] = targets

    paths = {node: py.relative_to(root).as_posix() for node, py in files.items()}
    return edges, paths, files


# ─────────────────────────────────────────────────────────────────────────────
# Seeds
# ─────────────────────────────────────────────────────────────────────────────
def _dvc_seeds(root: Path, known: set[str]) -> list[tuple[str, str]]:
    """De entrypoints die een `dvc.yaml`-stage aanroept, als (knoop, label)."""
    dvc = root / "dvc.yaml"
    if not dvc.is_file():
        return []
    seeds: set[str] = set()
    for line in dvc.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("cmd:"):
            continue
        for token in DVC_CMD_PATH.findall(stripped):
            dotted = token[:-3] if token.endswith(".py") else token
            dotted = dotted.replace("/", ".").strip(".")
            if dotted in known:
                seeds.add(dotted)
    return [(n, "") for n in sorted(seeds)]


def _hydra_seeds(root: Path, known: set[str]) -> list[tuple[str, str]]:
    """Dotted `_target_`-strings uit `conf/**/*.yaml`, gelabeld met hun YAML."""
    conf = root / "conf"
    if not conf.is_dir():
        return []
    seeds: dict[str, str] = {}
    for yml in sorted(conf.rglob("*.y*ml")):
        label = yml.relative_to(root).as_posix()
        for dotted in HYDRA_TARGET.findall(yml.read_text(encoding="utf-8", errors="replace")):
            for candidate in (dotted, *_ancestors(dotted)):
                if candidate in known:
                    seeds.setdefault(candidate, f"{label} (_target_)")
    return sorted(seeds.items())


def _dir_seeds(known: set[str], prefix: str) -> list[tuple[str, str]]:
    return [(n, "") for n in sorted(known) if n.split(".")[0] == prefix]


# ─────────────────────────────────────────────────────────────────────────────
# Afsluiting
# ─────────────────────────────────────────────────────────────────────────────
def _closure(
    edges: dict[str, set[str]], seeds: list[tuple[str, str]], paths: dict[str, str]
) -> dict[str, str]:
    """Transitieve afsluiting, met per bereikte module de seed die hem eerst raakt.

    De seed telt zelf mee: een Hydra-`_target_` wijst een `src/`-module aan die
    daarmee bereikt IS, niet alleen zijn imports.
    """
    reached: dict[str, str] = {}
    for seed, given_label in seeds:
        if seed not in edges:
            continue
        label = given_label or paths.get(seed, seed)
        reached.setdefault(seed, label)
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for target in sorted(edges.get(node, ())):
                if target in reached:
                    continue
                reached[target] = label
                queue.append(target)
    return reached


def _coverage_by_path(report: Path | None) -> dict[str, float]:
    """`coverage.py`-JSON -> {pad met forward slashes: percentage}."""
    if report is None:
        return {}
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for raw, entry in payload.get("files", {}).items():
        summary = entry.get("summary", {})
        if "percent_covered" in summary:
            out[raw.replace("\\", "/")] = float(summary["percent_covered"])
    return out


def scan(root: Path | str = ROOT, coverage: Path | str | None = None) -> list[Row]:
    """De volledige kaart: één `Row` per module in `src/`."""
    root = Path(root)
    cov = _coverage_by_path(Path(coverage) if coverage else None)
    edges, paths, files = _build_graph(root)
    known = set(edges)

    src_modules = sorted(n for n, py in files.items() if (root / "src") in py.parents or py.is_relative_to(root / "src"))

    seed_sets: list[tuple[Klass, list[str]]] = [
        (Klass.A, _dvc_seeds(root, known) + _hydra_seeds(root, known)),
        (Klass.B, _dir_seeds(known, "apps")),
        (Klass.C, _dir_seeds(known, "scripts")),
        (Klass.D, _dir_seeds(known, "tests")),
    ]

    assigned: dict[str, tuple[Klass, str]] = {}
    for klass, seeds in seed_sets:
        for module, label in _closure(edges, seeds, paths).items():
            if module not in assigned:
                assigned[module] = (klass, label)

    rows: list[Row] = []
    for module in src_modules:
        klass, seed = assigned.get(module, (Klass.E, ""))
        text = files[module].read_text(encoding="utf-8", errors="replace")
        rows.append(
            Row(
                module=module,
                path=paths[module],
                loc=len(text.splitlines()),
                klass=klass,
                seed=seed,
                coverage=cov.get(paths[module]),
            )
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Rapportage
# ─────────────────────────────────────────────────────────────────────────────
def _summary(rows: list[Row]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for klass in Klass:
        members = [r for r in rows if r.klass is klass]
        out[klass.value] = {"modules": len(members), "loc": sum(r.loc for r in members)}
    return out


def _as_json(rows: list[Row]) -> str:
    payload = {
        "summary": _summary(rows),
        "totals": {"modules": len(rows), "loc": sum(r.loc for r in rows)},
        "modules": [
            {
                "module": r.module,
                "path": r.path,
                "loc": r.loc,
                "klass": r.klass.value,
                "seed": r.seed,
                "coverage": r.coverage,
            }
            for r in rows
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _as_markdown(rows: list[Row]) -> str:
    summary = _summary(rows)
    lines = [
        "# Bereikbaarheidskaart `src/`",
        "",
        "Gegenereerd door `scripts/reachability_map.py`. Klassen op precedentie A > B > C > D > E.",
        "",
        "| Klasse | Modules | LOC |",
        "|---|---:|---:|",
    ]
    for klass in Klass:
        s = summary[klass.value]
        lines.append(f"| {klass.value} | {s['modules']} | {s['loc']} |")
    lines += [
        f"| **totaal** | **{len(rows)}** | **{sum(r.loc for r in rows)}** |",
        "",
        "| Module | Pad | LOC | Klasse | Dekking | Bereikt via |",
        "|---|---|---:|:---:|---:|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.klass.value, r.module)):
        cov = "—" if r.coverage is None else f"{r.coverage:.1f}%"
        lines.append(
            f"| `{r.module}` | `{r.path}` | {r.loc} | {r.klass.value} | {cov} | {r.seed or '—'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository-wortel (default: deze repo)")
    parser.add_argument("--json", type=Path, help="schrijf het JSON-artefact hierheen")
    parser.add_argument("--markdown", type=Path, help="schrijf de markdown-tabel hierheen")
    parser.add_argument("--coverage", type=Path, help="coverage.py-JSON om de dekkingskolom te vullen")
    parser.add_argument("--strict", action="store_true", help="exit 1 zodra klasse E niet leeg is")
    args = parser.parse_args(argv)

    rows = scan(args.root, coverage=args.coverage)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_as_json(rows), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_as_markdown(rows), encoding="utf-8")

    summary = _summary(rows)
    for klass in Klass:
        s = summary[klass.value]
        print(f"{klass.value}: {s['modules']:>4} modules  {s['loc']:>7} LOC")
    print(f"totaal: {len(rows)} modules, {sum(r.loc for r in rows)} LOC")

    unreachable = [r for r in rows if r.klass is Klass.E]
    if args.strict and unreachable:
        print(f"\nFAIL: {len(unreachable)} onbereikbare module(s) in klasse E:")
        for r in unreachable:
            print(f"  {r.module}  ({r.path}, {r.loc} LOC)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
