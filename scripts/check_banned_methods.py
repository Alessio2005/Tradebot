"""Phase 2 / Stage B-5 — AST-scanner op verboden validatiemethoden.

Audit §17.1: *"Purged Walk-Forward met embargo is ESSENTIAL en de ENIGE
toegestane CV in de promotiepijplijn."* Deze scanner maakt die zin afdwingbaar.

WAAROM AST EN GEEN REGEX
========================
De masterprompt schrijft AST voor, en dat is geen stijlvoorkeur. Een regex op
`KFold` vindt ook:

  * `docs`-tekst en docstrings die de methode BESPREKEN — dit bestand zelf staat
    er vol mee, en `compliance/mrm_report.py` rapporteert letterlijk
    `"train_test_split": "Purged CPCV with 10 folds"` als bewijs dat het NIET
    gebeurt;
  * een variabelenaam of een sleutel in een dict;
  * uitgecommentarieerde code.

Een scanner die daarop afgaat, wordt binnen een week met `# noqa` uitgezet, en
dan bewaakt hij niets meer. De AST ziet alleen wat er wordt UITGEVOERD: imports,
aanroepen en keyword-argumenten.

WAT VERBODEN IS, EN WAAROM
==========================
`RANDOM_CV` — splitsen zonder de tijd te respecteren. Een willekeurige fold
bevat bars van vóór én ná elke andere fold; het model traint dan letterlijk op
de toekomst van zijn eigen testset. Op financiële reeksen is dat geen subtiel
lek maar een geheel andere meting.

`SHUFFLE` — `shuffle=True` doet hetzelfde in één keyword. Hij is gevaarlijker
dan een `KFold`-import omdat hij op elke splitter kan staan, ook op een die
verder correct is gekozen.

`UNPURGED_TS` — `TimeSeriesSplit` respecteert de tijd WEL en lekt toch: hij kent
geen purge en geen embargo. Bij een label met horizon `H` loopt het label van de
laatste `H` trainbars het testvenster in. Dit is de subtielste van de drie en de
enige die er correct uitziet in code review.

`SEARCH_CV` — `GridSearchCV` en verwanten defaulten intern naar `KFold`. De
verboden splitsing staat dan nergens in de broncode.

RATCHET
=======
`ALLOWLIST` is LEEG en hoort dat te blijven. Elk item zou een pad zijn waarlangs
een niet-temporele splitsing een promotiebesluit kan bereiken. Groeit hij, dan
is dat een besluit dat in `docs/ARCHITECTURAL_DECISIONS.md` hoort te staan, niet
in deze tabel.

Gebruik:
    python scripts/check_banned_methods.py            # rapport, exit 0
    python scripts/check_banned_methods.py --strict   # exit 1 bij een treffer
    python scripts/check_banned_methods.py --strict <pad> [<pad> ...]
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ("src/tradebot", "apps", "scripts")
SKIP_PARTS = {"__pycache__", ".git", ".venv", "node_modules"}

#: Naam -> (categorie, reden). De reden komt woordelijk in de melding: wie de
#: scanner rood ziet, moet niet hoeven zoeken waarom dit verboden is.
BANNED_NAMES: dict[str, tuple[str, str]] = {
    # -- willekeurige splitsingen ------------------------------------------- #
    "KFold": ("RANDOM_CV", "splitst zonder de tijd te respecteren"),
    "StratifiedKFold": ("RANDOM_CV", "stratificeert over de tijdsas heen"),
    "RepeatedKFold": ("RANDOM_CV", "herhaalt een niet-temporele splitsing"),
    "RepeatedStratifiedKFold": ("RANDOM_CV", "idem, met stratificatie"),
    "GroupKFold": ("RANDOM_CV", "groepeert, maar ordent niet in de tijd"),
    "ShuffleSplit": ("RANDOM_CV", "trekt willekeurige train/test-partities"),
    "StratifiedShuffleSplit": ("RANDOM_CV", "idem, met stratificatie"),
    "GroupShuffleSplit": ("RANDOM_CV", "idem, per groep"),
    "LeaveOneOut": ("RANDOM_CV", "laat één observatie weg, ongeacht wanneer"),
    "LeavePOut": ("RANDOM_CV", "idem, voor p observaties"),
    "train_test_split": (
        "RANDOM_CV",
        "splitst standaard willekeurig; ook met shuffle=False ontbreekt elke "
        "purge en embargo"),
    # -- CV-drivers die intern naar KFold defaulten ------------------------- #
    "cross_val_score": ("SEARCH_CV", "defaultet naar KFold"),
    "cross_validate": ("SEARCH_CV", "defaultet naar KFold"),
    "cross_val_predict": ("SEARCH_CV", "defaultet naar KFold"),
    "GridSearchCV": ("SEARCH_CV", "defaultet naar KFold"),
    "RandomizedSearchCV": ("SEARCH_CV", "defaultet naar KFold"),
    "HalvingGridSearchCV": ("SEARCH_CV", "defaultet naar KFold"),
    "HalvingRandomSearchCV": ("SEARCH_CV", "defaultet naar KFold"),
    # -- temporeel, maar zonder purge of embargo ---------------------------- #
    "TimeSeriesSplit": (
        "UNPURGED_TS",
        "respecteert de tijd maar kent geen purge en geen embargo; het label "
        "van de laatste H trainbars loopt het testvenster in"),
}

#: Namen die ALLEEN verboden zijn wanneer ze uit een bepaalde module komen.
#: naam -> (moduleprefix, categorie, reden).
#:
#: `shuffle` staat hier en niet in `BANNED_NAMES`, omdat de eerste versie van
#: deze scanner elke `.shuffle(`-aanroep vlagde en daarmee
#: `selection/mda.py::rng.shuffle(block_indices)` als overtreding aanmerkte. Dat
#: is een BLOK-permutatie met embargo voor MDA-feature-importance — de correcte
#: techniek, geen splitsing. Een scanner die zulke treffers produceert, wordt
#: binnen een week uitgezet, en dan bewaakt hij niets meer. Vals alarm is voor
#: een ratchet duurder dan een gemist geval.
BANNED_FROM_IMPORTS: dict[str, tuple[str, str, str]] = {
    "shuffle": ("sklearn", "RANDOM_CV",
                "sklearn.utils.shuffle vernietigt de tijdsvolgorde"),
}

#: RATCHET. Pad (POSIX, relatief aan de repo-root) -> (budget, motivering).
#: Leeg, en dat is de bedoeling. Zie de moduledocstring.
ALLOWLIST: dict[str, tuple[int, str]] = {}


@dataclass(frozen=True)
class Finding:
    path: str
    lineno: int
    name: str
    category: str
    reason: str
    how: str

    def render(self) -> str:
        return (f"{self.path}:{self.lineno}: [{self.category}] {self.how} "
                f"`{self.name}` — {self.reason}")


class _Visitor(ast.NodeVisitor):
    """Alleen wat wordt UITGEVOERD telt: imports, aanroepen, keywords.

    Een docstring die `KFold` noemt is een `ast.Constant` en komt hier nooit
    langs. Dat is precies het verschil met een regex, en het is de reden dat deze
    scanner niet hoeft te worden uitgezet om over hem heen te kunnen schrijven.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.findings: list[Finding] = []

    def _record(self, name: str, lineno: int, how: str) -> None:
        entry = BANNED_NAMES.get(name)
        if entry is None:
            return
        category, reason = entry
        self.findings.append(
            Finding(self.path, lineno, name, category, reason, how))

    # -- imports ------------------------------------------------------------ #
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self._record(alias.name, node.lineno, "importeert")
            scoped = BANNED_FROM_IMPORTS.get(alias.name)
            if scoped is not None and module.startswith(scoped[0]):
                self.findings.append(Finding(
                    self.path, node.lineno, f"{module}.{alias.name}",
                    scoped[1], scoped[2], "importeert"))
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name.rsplit(".", 1)[-1], node.lineno, "importeert")
        self.generic_visit(node)

    # -- aanroepen ---------------------------------------------------------- #
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            self._record(func.id, node.lineno, "roept aan")
        elif isinstance(func, ast.Attribute):
            self._record(func.attr, node.lineno, "roept aan")

        # `shuffle=True` op welke splitter dan ook. Dit is de gevaarlijkste van
        # de drie klassen: hij staat op één regel, hij ziet er onschuldig uit en
        # hij kan op een splitter staan die verder correct is gekozen.
        for kw in node.keywords:
            if kw.arg == "shuffle" and isinstance(kw.value, ast.Constant) \
                    and kw.value.value is True:
                self.findings.append(Finding(
                    self.path, node.lineno, "shuffle=True", "SHUFFLE",
                    "vernietigt de tijdsvolgorde van de splitsing",
                    "geeft mee"))
        self.generic_visit(node)


def scan_file(path: Path, *, root: Path) -> list[Finding]:
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:                       # pragma: no cover
        return [Finding(rel, exc.lineno or 0, "<syntaxfout>", "PARSE",
                        f"bestand is niet te parsen: {exc.msg}", "kan niet lezen")]
    visitor = _Visitor(rel)
    visitor.visit(tree)
    return visitor.findings


def iter_python_files(targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        if target.is_file() and target.suffix == ".py":
            files.append(target)
            continue
        for path in sorted(target.rglob("*.py")):
            if SKIP_PARTS.isdisjoint(path.parts):
                files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=None,
                    help="bestanden of mappen; standaard src/tradebot, apps, scripts")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 zodra een bestand zijn budget overschrijdt")
    ap.add_argument("--root", default=None,
                    help="repo-root voor relatieve paden (standaard: deze repo)")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else ROOT
    targets = ([Path(p).resolve() for p in args.paths] if args.paths
               else [root / t for t in DEFAULT_TARGETS])

    findings: list[Finding] = []
    n_files = 0
    for path in iter_python_files(targets):
        n_files += 1
        findings.extend(scan_file(path, root=root))

    per_file: dict[str, list[Finding]] = {}
    for f in findings:
        per_file.setdefault(f.path, []).append(f)

    overruns: list[str] = []
    for path, items in sorted(per_file.items()):
        budget = ALLOWLIST.get(path, (0, ""))[0]
        if len(items) > budget:
            overruns.append(path)

    if findings:
        print("VERBODEN VALIDATIEMETHODEN:")
        for f in sorted(findings, key=lambda x: (x.path, x.lineno)):
            print(f"  {f.render()}")
        print()

    print(f"{len(findings)} treffer(s) in {len(per_file)} bestand(en); "
          f"{n_files} bestand(en) gescand.")
    print(f"Toegestaan door de ratchet: {sum(b for b, _ in ALLOWLIST.values())}.")

    if overruns:
        print()
        print("OVERSCHRIJDINGEN:")
        for path in overruns:
            budget = ALLOWLIST.get(path, (0, ""))[0]
            print(f"  {path}: {len(per_file[path])} treffer(s), budget {budget}")
        print()
        print("Audit §17.1: Purged Walk-Forward met embargo is de ENIGE "
              "toegestane CV in de promotiepijplijn.")
        print("Gebruik `tradebot.validation.walk_forward.purged_walk_forward`.")
        if args.strict:
            return 1
    elif not findings:
        print("Geen verboden methode gevonden.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
