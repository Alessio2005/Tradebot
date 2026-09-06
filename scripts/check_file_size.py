"""LOC-poort met een ratchet per bestand — Phase 9, stap 13.

`docs/architecture.md` R-4 stelt een grens van 800 regels per module. De poort
die dat moest bewaken stond in de `Makefile`, en die bewaakte niets:

* `make` bestaat niet in deze omgeving, dus de regel is nooit uitgevoerd;
* de whitelist noemde `risk/portfolio.py`, dat sinds Phase 4 niet bestaat;
* `DEFERRED_ISSUES.md` DI-3 claimt *"gemeten 2026-09-01: nog 2 bestanden >800
  LOC"* terwijl het er op 2026-09-04 **negen** zijn — en de twee die DI-3 noemt
  staan er met andere getallen in. De ratchet die *"verdere groei blokkeert"*
  heeft die groei niet geblokkeerd.

WAAROM EEN CAP EN GEEN WHITELIST
================================
Negen bestanden vrijstellen is een poort die alles doorlaat. `CAPS` legt daarom
per bestand de GEMETEN omvang vast. Het bestand mag blijven zoals het is en mag
niet groeien; wie er een regel bij schrijft, wordt rood en moet ofwel splitsen
ofwel de cap bewust verhogen. Dat laatste is dan een gelezen besluit in een
diff, in plaats van een grens die stil opschuift.

Krimpen mag altijd. Zakt een bestand onder de 800, dan hoort zijn cap weg — en
`tests/unit/test_file_size_ratchet.py` maakt daar een rode test van, zodat de
tabel niet in de andere richting verrot.

GEBRUIK
=======
    python scripts/check_file_size.py            # exit 1 zodra iets groeit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: `docs/architecture.md` R-4.
LIMIT = 800

#: Bestanden boven de limiet, met hun GEMETEN omvang op 2026-09-04 als plafond.
#: Elk bestand draagt daarnaast een `# LOC-EXCEPTION:`-regel in zijn eigen
#: header met de reden; die binding wordt getoetst.
CAPS: dict[str, int] = {
    "src/tradebot/labeling/meta.py": 1185,
    "src/tradebot/train/ensemble.py": 1117,
    # Fase 10, stap 4A: +78 voor `InferenceConfig` en de accessors
    # `inference_config()` / `backtest_config()`. Bewust verhoogd en niet
    # gesplitst: de cap-tabel bestaat om zo'n verhoging een gelezen besluit in
    # een diff te maken, en dit bestand IS het ene configuratiecontract.
    "src/tradebot/schemas/config.py": 1176,
    # Fase 10, stap 4A: nieuw. De ene implementatie van de Sharpe-SE, de
    # Sharpe-verschiltoets en de circulaire blokbootstrap (R-3). De reden dat
    # hij niet gesplitst is, staat in zijn eigen `# LOC-EXCEPTION:`-regel.
    "src/tradebot/validation/inference.py": 1044,
    "src/tradebot/features/regime.py": 1060,
    "src/tradebot/backtest/evaluation.py": 1057,
    "src/tradebot/tune/objective.py": 958,
    "src/tradebot/live/engine.py": 917,
    "src/tradebot/portfolio/legacy_sizing.py": 824,
    "src/tradebot/validation/data_adequacy.py": 807,
}

#: `portfolio/legacy_sizing.py` draagt GEEN `# LOC-EXCEPTION:`-regel, en dat is
#: opzet. DI-10 houdt dat bestand bewust ONGEWIJZIGD zodat de Phase 3-baseline
#: herrekenbaar blijft; er een commentaarregel in schrijven zou die afspraak
#: schenden voor een reden die net zo goed hier kan staan. De reden:
#: legacy-sizinglogica die als geheel is verhuisd uit `risk/portfolio.py` en die
#: niet wordt aangeraakt tot de baseline niet langer nodig is.
HEADER_EXEMPT: frozenset[str] = frozenset({"src/tradebot/portfolio/legacy_sizing.py"})


def oversized(root: Path | str, caps: dict[str, int]) -> list[tuple[str, int, int]]:
    """(pad, gemeten LOC, plafond) voor elk bestand dat zijn plafond overschrijdt.

    Het plafond is `caps[pad]` wanneer het bestand daarin staat, en anders
    `LIMIT`.
    """
    root = Path(root)
    found: list[tuple[str, int, int]] = []
    for py in sorted((root / "src").rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(root).as_posix()
        loc = len(py.read_text(encoding="utf-8", errors="replace").splitlines())
        ceiling = caps.get(rel, LIMIT)
        if loc > ceiling:
            found.append((rel, loc, ceiling))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--caps", default=None,
        help="laat leeg ('') om zonder caps te draaien; standaard de tabel in dit bestand",
    )
    args = parser.parse_args(argv)

    caps = CAPS if args.caps is None else {}
    found = oversized(args.root, caps)

    if not found:
        print(f"LOC-poort groen: niets boven {LIMIT} regels zonder cap, "
              f"niets boven zijn cap ({len(caps)} gecapte bestanden).")
        return 0

    print(f"FAIL: {len(found)} bestand(en) boven hun plafond:")
    for rel, loc, ceiling in found:
        how = "cap" if ceiling != LIMIT else "limiet R-4"
        print(f"  {rel}: {loc} regels > {ceiling} ({how})")
    print(
        "\nSplits het bestand, of verhoog zijn cap in scripts/check_file_size.py "
        "met een reden in de `# LOC-EXCEPTION:`-regel van het bestand zelf."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
