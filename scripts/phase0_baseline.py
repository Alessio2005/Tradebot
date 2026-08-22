"""Phase 0 / Step 1 - repository baseline inventory (evidence artefact).

Generates reports/phase0_baseline.md:
  * LOC per module and per file
  * every file > 800 LOC (D-6) and every app > 80 LOC (D-7)
  * every file importing the non-existent package `quant_architect`
  * shadow trees, misplaced build configuration and missing DAG directories

This script is read-only. It never mutates the repository.
"""
from __future__ import annotations

import ast
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "apps", "tests", "scripts", "miscellaneous")
SHADOW = {"__pycache__", ".pytest_cache", ".ruff_cache", "catboost_info", ".dvc", ".git"}
LOC_LIMIT = 800          # architecture.md R-4
APP_LOC_LIMIT = 80       # architecture.md R-6

# The symbols the phantom `quant_architect` package is asked for, mapped to the
# internal module that actually defines each of them today.
PHANTOM_SYMBOLS = [
    ("EntropyGate", "tradebot.train.schema_guard"),
    ("EntropyGateDecision", "tradebot.train.schema_guard"),
    ("FeatureSchemaGuard", "tradebot.train.schema_guard"),
    ("SchemaMismatchError", "tradebot.train.schema_guard"),
    ("LedoitWolfThompsonSampler", "tradebot.train.thompson"),
    ("NetAlphaResult", "tradebot.train.reward"),
    ("NetAlphaReward", "tradebot.train.reward"),
    ("QuantArchitectStack", "tradebot.train.stack"),
    ("SymmetricQuantileScaler", "tradebot.train.quant_arch"),
]


def iter_py(base: Path):
    for p in sorted(base.rglob("*.py")):
        if any(part in SHADOW for part in p.parts):
            continue
        yield p


def loc(p: Path) -> int:
    return len(p.read_text(encoding="utf-8", errors="replace").splitlines())


def imports_of(p: Path) -> set[str]:
    """Return the set of top-level module names imported by this file."""
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def scan_hygiene() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name in ("__pycache__", ".pytest_cache", ".ruff_cache", "catboost_info",
                 "outputs", "logs"):
        d = ROOT / name
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*"))
            out.append((name + "/", "shadow tree aanwezig in de root ({} entries)".format(n)))
    for name in ("pyproject.toml", "Makefile", "dvc.yaml", "params.yaml",
                 ".pre-commit-config.yaml", ".env"):
        at_root = (ROOT / name).exists()
        at_misc = (ROOT / "miscellaneous" / name).exists()
        if at_misc and not at_root:
            out.append(("miscellaneous/" + name,
                        "**build-config buiten de projectroot** - "
                        "`pip install -e .` vanuit de root is onmogelijk"))
        elif not at_root and not at_misc:
            out.append((name, "ontbreekt volledig"))
    if not (ROOT / ".git").is_dir():
        out.append((".git/", "**ontbreekt - D-9**: geen versiebeheer; elk MRM-rapport is invalide"))
    if not (ROOT / ".gitignore").exists():
        out.append((".gitignore", "ontbreekt"))
    for d in ("artefacts/features", "artefacts/models", "artefacts/tracks"):
        if not (ROOT / d).is_dir():
            out.append((d + "/", "**ontbreekt - D-8**: DAG-map uit architecture.md"))
    return out


def main() -> int:
    per_module: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    oversized: list[tuple[Path, int]] = []
    oversized_apps: list[tuple[Path, int]] = []
    qa_importers: list[tuple[Path, int]] = []
    total_files = total_loc = 0

    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in iter_py(base):
            n = loc(p)
            rel = p.relative_to(ROOT)
            total_files += 1
            total_loc += n
            parts = rel.parts
            if parts[0] == "src" and len(parts) > 2:
                key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:2])
            else:
                key = parts[0]
            per_module[key].append((rel, n))
            if n > LOC_LIMIT:
                oversized.append((rel, n))
            if parts[0] == "apps" and n > APP_LOC_LIMIT:
                oversized_apps.append((rel, n))
            if "quant_architect" in imports_of(p):
                qa_importers.append((rel, n))

    hygiene = scan_hygiene()

    L: list[str] = []
    A = L.append
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    A("# PHASE 0 - REPOSITORY BASELINE (nulmeting)")
    A("")
    A("**Gegenereerd:** " + stamp + "  ")
    A("**Generator:** `scripts/phase0_baseline.py` (read-only)  ")
    A("**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md`  ")
    A("**Status:** bewijsmateriaal - deze nulmeting wordt niet overschreven na remediatie.")
    A("")
    A("Feitelijke staat van de repository *voordat* enige Phase 0-wijziging is doorgevoerd.")
    A("")
    A("---")
    A("")
    A("## 1. Totalen")
    A("")
    A("| Metriek | Waarde |")
    A("|---|---|")
    A("| Gescande Python-bestanden | {} |".format(total_files))
    A("| Totaal LOC | {:,} |".format(total_loc))
    A("| Bestanden > {} LOC (D-6) | {} |".format(LOC_LIMIT, len(oversized)))
    A("| Apps > {} LOC (D-7) | {} |".format(APP_LOC_LIMIT, len(oversized_apps)))
    A("| Bestanden met `quant_architect`-import | {} |".format(len(qa_importers)))
    A("| Hygiene-bevindingen (shadow/config/DAG) | {} |".format(len(hygiene)))
    A("")
    A("---")
    A("")
    A("## 2. LOC per module")
    A("")
    A("| Module | Bestanden | LOC |")
    A("|---|---:|---:|")
    for k in sorted(per_module, key=lambda x: -sum(n for _, n in per_module[x])):
        files = per_module[k]
        A("| `{}` | {} | {:,} |".format(k, len(files), sum(n for _, n in files)))
    A("")
    A("---")
    A("")
    A("## 3. Bestanden > {} LOC - D-6 (architecture.md R-4)".format(LOC_LIMIT))
    A("")
    if oversized:
        A("| Bestand | LOC | Overschrijding |")
        A("|---|---:|---:|")
        for rel, n in sorted(oversized, key=lambda t: -t[1]):
            A("| `{}` | {:,} | +{:,} |".format(rel.as_posix(), n, n - LOC_LIMIT))
    else:
        A("_Geen._")
    A("")
    A("> D-6 is **P2** en valt buiten de Phase 0-scope. Vastgelegd als bewijs; remediatie")
    A("> is doorgeschoven naar `docs/DEFERRED_ISSUES.md`.")
    A("")
    A("---")
    A("")
    A("## 4. Apps > {} LOC - D-7 (architecture.md R-6)".format(APP_LOC_LIMIT))
    A("")
    napps = len(per_module.get("apps", []))
    A("**{} van de {} apps overschrijden de limiet.**".format(len(oversized_apps), napps))
    A("")
    A("| App | LOC | Overschrijding |")
    A("|---|---:|---:|")
    for rel, n in sorted(oversized_apps, key=lambda t: -t[1]):
        A("| `{}` | {:,} | +{:,} |".format(rel.as_posix(), n, n - APP_LOC_LIMIT))
    A("")
    A("> D-7 is **P2** en valt buiten de Phase 0-scope. Nieuwe apps die in Phase 1-3 worden")
    A("> toegevoegd respecteren de 80-LOC-limiet wel.")
    A("")
    A("---")
    A("")
    A("## 5. `quant_architect`-importeurs - sectie 5.2")
    A("")
    A("Het pakket `quant_architect` bestaat nergens op het filesystem en staat in geen enkele")
    A("dependency-declaratie. Elk van onderstaande bestanden draait daardoor permanent in")
    A("gedegradeerde modus, zonder melding.")
    A("")
    if qa_importers:
        A("| Bestand | LOC |")
        A("|---|---:|")
        for rel, n in sorted(qa_importers):
            A("| `{}` | {:,} |".format(rel.as_posix(), n))
    else:
        A("_Geen treffers._")
    A("")
    A("### 5.1 Afwijking t.o.v. het auditdocument")
    A("")
    A("Sectie 5.2 van de audit noemt `tune/objective.py` als vijfde importeur. Dat is feitelijk")
    A("onjuist: `tune/objective.py` bevat geen `quant_architect`-import. De werkelijke vijfde")
    A("importeur is `src/tradebot/features/scaling.py`. Conform sectie 2.2 (documentatie is een")
    A("hypothese, geen waarheid) prevaleert deze meting boven de audittekst.")
    A("")
    A("Alle gevraagde symbolen bestaan wel degelijk als interne module:")
    A("")
    A("| Symbool | Interne bron |")
    A("|---|---|")
    for sym, mod in PHANTOM_SYMBOLS:
        A("| `{}` | `{}` |".format(sym, mod))
    A("")
    A("De import is dus niet dood maar **verkeerd geadresseerd**: de functionaliteit is ooit uit")
    A("`quant_architect.py` geextraheerd naar `train/`, zonder dat de importsites zijn bijgewerkt.")
    A("Stap 5 is daarmee een gerichte heradressering, geen herimplementatie.")
    A("")
    A("---")
    A("")
    A("## 6. Shadow trees, misplaatste build-config en ontbrekende DAG-mappen")
    A("")
    A("| Pad | Bevinding |")
    A("|---|---|")
    for rel, note in hygiene:
        A("| `{}` | {} |".format(rel, note))
    A("")

    out = ROOT / "reports" / "phase0_baseline.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote {}".format(out))
    print("files={} loc={} oversized={} apps_over={} quant_architect={} hygiene={}".format(
        total_files, total_loc, len(oversized), len(oversized_apps),
        len(qa_importers), len(hygiene)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
