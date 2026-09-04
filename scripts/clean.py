"""Werkboomhygiëne — Phase 9, stap 14.

Verwijdert de caches die geen enkele meting nodig heeft en die wél elke `find`,
`grep` en IDE-index vervuilen. Gemeten op 2026-09-04, vóór de eerste run:

    .mypy_cache      2.773 bestanden   106 MB
    logs/               68 bestanden   4,1 MB
    .hypothesis        555 bestanden   906 KB
    .pytest_cache                      359 KB

Alle zeven doelen staan al in `.gitignore` — dat is nagemeten met
`git check-ignore` en het was geen git-probleem. Het was een zoekprobleem: een
`grep -rn` over de boom haalde 2.773 valse treffers binnen uit één cache.

WAAROM DIT EEN SCRIPT IS EN GEEN MAKEFILE-REGEL
===============================================
`make clean` bestond, maar `make` bestaat niet in deze omgeving
(`make: command not found`). Een opruimdoel dat niet uitvoerbaar is, ruimt niets
op — precies de faalvorm die deze fase bij de LOC-poort ook aantrof.

WAAROM HIJ ZO VOORZICHTIG IS
============================
Dit is het enige gereedschap in de repository dat bestanden verwijdert die NIET
in versiebeheer staan, en dus niet met `git show` terug te halen zijn. Twee
paden zijn daarom hard beschermd, en `tests/unit/test_clean.py` toetst dat:

* **`data/pit_store/`** — fence 1. De 228 ruwe parquet/json-bestanden staan daar
  bewust in git (AD-12, `.gitignore` regel 53-58); onder DVC stond de store op
  een remote die niet bestond. Verwijderen is een architectuurbesluit.
* **`artefacts/governance/`** — fence 2. Append-only: de hypothesis-ledger, de
  preregistraties, `data_hashes.json`.

GEBRUIK
=======
    python scripts/clean.py --dry-run     # laat zien wat er weg zou gaan
    python scripts/clean.py               # doe het
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Mappen in de wortel die in hun geheel weg mogen. Geen van deze bevat iets
#: dat niet opnieuw te genereren is.
TARGETS: tuple[str, ...] = (
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis",
    "catboost_info",
    "logs",
    "outputs",
)

#: Paden die dit script NOOIT betreedt, ook niet voor een `__pycache__` erin.
#: De eerste twee zijn fence 1 en fence 2 van Phase 9.
PROTECTED: tuple[str, ...] = (
    ".git",
    "data/pit_store",
    "artefacts/governance",
)


def _is_protected(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(rel == p or rel.startswith(p + "/") for p in PROTECTED)


def sweep(root: Path | str = ROOT, dry_run: bool = False) -> list[Path]:
    """Verwijder de caches; retourneer wat er weg is (of weg zou gaan).

    `dry_run=True` raakt niets aan en levert dezelfde lijst op, zodat wat je
    ziet exact is wat er gebeurt.
    """
    root = Path(root)
    hits: list[Path] = []

    for name in TARGETS:
        target = root / name
        if target.is_dir() and not _is_protected(target, root):
            hits.append(target)

    for cache in sorted(root.rglob("__pycache__")):
        if cache.is_dir() and not _is_protected(cache, root):
            hits.append(cache)

    if not dry_run:
        for path in hits:
            shutil.rmtree(path, ignore_errors=True)

    return hits


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true",
                        help="laat zien wat er weg zou gaan zonder iets te verwijderen")
    args = parser.parse_args(argv)

    hits = sweep(args.root, dry_run=True)
    if not hits:
        print("werkboom is al schoon.")
        return 0

    total = 0
    for path in hits:
        n = _size(path)
        total += n
        print(f"  {path.relative_to(args.root).as_posix():<50} {n / 1e6:>8.1f} MB")

    verb = "zou verwijderen" if args.dry_run else "verwijderd"
    if not args.dry_run:
        sweep(args.root, dry_run=False)
    print(f"\n{len(hits)} pad(en), {total / 1e6:.1f} MB {verb}.")
    print("data/pit_store/ en artefacts/governance/ zijn niet aangeraakt (fence 1 en 2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
