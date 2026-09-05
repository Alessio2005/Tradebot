"""AD-23-poort: het meetdomein, afdwingbaar gemaakt.

TWEE CONTROLES
==============
1. RESOLUTIE. Geen enkel bestand in `conf/` declareert een barresolutie die
   niet exact `1d` is. Eén meetklok (B-1).
2. HEROPENING. Elke heropeningsconditie in de registers noemt ten minste één
   source-id uit `conf/governance/measurement_domain.yaml`, of staat expliciet
   gemarkeerd als buiten het domein, of is een expliciete weigering die niets
   belooft (zie WEIGERINGEN hieronder).

WAAROM EEN WHITELIST
====================
Een blacklist van verboden informatiebronnen is nooit volledig -- zij noemt de
vormen die iemand al had bedacht -- en zij dwingt dit project om de uitgesloten
ruimte te blijven benoemen. Een whitelist is per constructie volledig: wat geen
bron uit het domein noemt, is niet toelaatbaar, ongeacht hoe het heet.

WAAROM DE POORT OP KOLOM 4 TRIGGERT EN NIET OP HET WOORD "HEROPEN"
====================================================================
Een eerdere ontwerpversie triggerde alleen op rijen die het woord "heropen"
bevatten. Dat woord staat in `FALSIFICATION_REGISTER.md` uitsluitend in de
KOLOMKOP ("Heropening alleen als...") en in GEEN van de twintig F-rijen zelf --
de heropeningsconditie staat daar ongelabeld in kolom 4. Een poort die op dat
woord trigt, slaat dus elke echte rij over en meldt een vervuild register als
schoon. Deze poort leest daarom altijd kolom 4 van elke `F<n>` / `H<n>` /
`DI-<n>`-rij, ongeacht de woordkeuze in die kolom.

WEIGERINGEN (categorie 3)
==========================
Een conditie die expliciet niets toezegt ("nooit", "n.v.t.") kan ook niets
buiten het domein toezeggen, en is daarom geldig zonder een bron te noemen.
Dit is een SMALLE categorie: de patronen hieronder zijn met de hand tegen elke
huidige F-rij geverifieerd en matchen uitsluitend rijen met exact die vorm. Een
rij die met "n.v.t." of "nooit" begint maar vervolgens wél een concreet
alternatief buiten het domein noemt (zoals F8: "n.v.t. binnen crypto; in
equities/FX/commodities is XSMOM een ander, gedocumenteerd premium") is GEEN
weigering en moet worden geraakt -- vandaar de smalle patronen in plaats van
een generieke "begint met nooit/n.v.t." regel. Nieuwe weigeringsvormen krijgen
een eigen, met de hand toegevoegd patroon; dat is een bewuste code-wijziging en
geen sluiproute.

BEKENDE OPEN PLEK -- SLUIT IN STAP 16
======================================
`RESOLUTION` matcht `bar_resolution` / `bar_interval` / `bar_timeframe`, niet
`bar_seconds`. `conf/conf_config.yaml:136` (`bar_seconds: 3600.0`) en
`conf/env/prod.yaml:12` (`bar_seconds: 5`) declareren dus allebei een
niet-dagelijkse klok die deze poort NIET ziet. Dat gat is bekend en blijft
bewust open: mandaatbesluit B-1 verwijdert `live/feed.py::FeedConfig.bar_seconds`
pas in stap 16, en de regex hier uitbreiden zou deze poort vóór die stap rood
zetten op sleutels die de fase nog niet heeft opgeruimd. Stap 16 sluit dit gat.

Rood worden is de bedoeling. `tests/unit/test_domain_consistency.py` bewijst dat
deze poort het kan, in beide richtingen.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "conf" / "governance" / "measurement_domain.yaml"

ROW = re.compile(r"^\|\s*(F\d+|H\d+|DI-\d+)\s*\|")
RESOLUTION = re.compile(r"bar_(?:resolution|interval|timeframe)\s*:\s*([^\s#]+)")

# Smalle, met de hand geverifieerde weigeringspatronen -- zie de docstring
# hierboven ("WEIGERINGEN"). Elke regel noemt de rij in het echte register die
# hem heeft opgeleverd, zodat een toekomstige lezer kan navertellen waarom het
# patroon precies zo smal is.
REFUSAL_PATTERNS = (
    re.compile(r"^nooit\s*[-–—]"),  # F4: "nooit -- AUC ..."
    re.compile(r"^nooit in deze vorm\b"),  # F9: "nooit in deze vorm; ..."
    re.compile(r"^n\.v\.t\.\s*;"),  # F12: "n.v.t.; harvest simpel ..."
)


@dataclass(frozen=True)
class Domain:
    resolution: str
    source_ids: tuple[str, ...]
    marker: str


def load_domain(path: Path = DOMAIN) -> Domain:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Domain(
        resolution=str(payload["observation"]["bar_resolution"]),
        source_ids=tuple(str(s["id"]) for s in payload["sources"]),
        marker=str(payload["out_of_domain_marker"]).lower(),
    )


def scan_resolutions(conf_root: Path, domain: Domain) -> list[str]:
    offenders: list[str] = []
    for path in sorted(conf_root.rglob("*.y*ml")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = RESOLUTION.search(line)
            if match and match.group(1).strip().strip("\"'") != domain.resolution:
                offenders.append(f"{path}:{number}: {match.group(1)}")
    return offenders


def _reopening_condition(line: str) -> str | None:
    """De laatste kolom van een markdown-tabelrij
    `| id | hypothese | bewijs | conditie |`.

    `line.split("|")` op zo'n rij geeft `['', id, hypothese, bewijs, conditie,
    '']` -- zes elementen, en de conditie staat op index 4 (= index -2, vóór
    de lege staart na de slotpipe). Sommige bewijs-kolommen dragen zelf een
    letterlijke `|` (bv. F19: "DOW max |t|=1.67"), wat het aantal elementen
    optrekt; `cells[-2]` blijft in dat geval nog steeds de laatste kolom vóór
    de slotpipe, dus de conditie. Minder dan zes elementen betekent een rij
    die geen vier kolommen heeft; die is per definitie geen geldige
    heropeningsconditie en telt daarom als offender in plaats van
    stilzwijgend te worden overgeslagen.
    """
    cells = line.split("|")
    if len(cells) < 6:
        return None
    return cells[-2].strip()


def _is_refusal(condition: str) -> bool:
    return any(pattern.search(condition) for pattern in REFUSAL_PATTERNS)


def scan_register(register: Path, domain: Domain) -> list[str]:
    offenders: list[str] = []
    for line in register.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if not match:
            continue
        identifier = match.group(1)
        condition = _reopening_condition(line)
        if condition is None:
            offenders.append(identifier)
            continue
        if _is_refusal(condition):
            continue
        lowered = condition.lower()
        if domain.marker in lowered:
            continue
        if any(source in lowered for source in domain.source_ids):
            continue
        offenders.append(identifier)
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--domain", type=Path, default=DOMAIN)
    parser.add_argument("--conf-root", type=Path, default=REPO / "conf")
    parser.add_argument(
        "--register", type=Path,
        default=REPO / "docs" / "FALSIFICATION_REGISTER.md",
    )
    args = parser.parse_args(argv)

    domain = load_domain(args.domain)
    problems: list[str] = []

    for offender in scan_resolutions(args.conf_root, domain):
        problems.append(
            f"{offender}: declareert een barresolutie die niet "
            f"'{domain.resolution}' is (AD-22)"
        )
    for identifier in scan_register(args.register, domain):
        problems.append(
            f"{identifier}: heropeningsconditie noemt geen bron uit het "
            f"meetdomein {domain.source_ids} en is niet gemarkeerd als "
            f"'{domain.marker}' (AD-23)"
        )

    for problem in problems:
        print(problem)
    if problems and args.strict:
        return 1
    print(f"domeinconsistentie: {len(problems)} openstaande regel(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
