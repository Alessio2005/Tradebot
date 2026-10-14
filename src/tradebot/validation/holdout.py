"""Het bevroren poortsample -- AD-24 R7/R8, de externe verificatie op M_new.

WAAROM DIT BESTAAT
==================
De ledger-reset (`registry/ledger_reset.py`) is een belofte over de
toekomstige zoekruimte: `M_new = 25` telt alleen wat het beweert te tellen
zolang niemand stiekem meer trials draait dan geregistreerd. Die belofte is
door de reset zelf niet te verifiëren -- dat is de zwakke plek die R8 benoemt.
Deze module repareert dat zonder op discipline te vertrouwen: het laatste
stuk van `W_FULL` wordt afgesloten VOOR de eerste meting, en op dat stuk is
`M = 1` per constructie, want `gate_slice` staat per hypothese ten hoogste
één lezing toe (R7). Wie het poortsample twee keer leest voor dezelfde
hypothese, doet dat niet stiekem: `HoldoutAlreadyUsed` weigert de tweede
lezing, en de eerste staat al met een timestamp in het bevroren bestand.

DE SPLIT, EN WAT HIJ KOST (ruling P23)
=======================================
`--split-utc 2025-09-05T00:00:00+00:00` is de pre-geregistreerde grens. Op
`W_FULL` (1743 bars, ruling T1-B) geeft dat:

    development:  1390 bars, t_years = 3,8082, t=2-drempel = 1,0249
    gate:          353 bars, t_years = 0,9671  ("de laatste ~12 maanden")

De t=2-drempel op de ontwikkelsample stijgt van 0,9152 (op heel `W_FULL`) naar
1,0249 -- een 12,0% zwaardere bewijslast, niet de 14% die een eerdere,
inmiddels ingetrokken lezing op `n_obs=1615` noemde. Het poortvenster is
0,9671 jaar, NOOIT 1,00 jaar: 353 dagbars is bijna maar niet precies een jaar,
en dit bestand rondt dat niet stilzwijgend af.

WAAROM DE HELE-BESTAND-HASH VAN `ledger_reset.json` HIER NIET WERKT (ruling P24)
=================================================================================
`ledger_reset.json` is bevroren en blijft dat: zijn hash is een geldig hek,
en `tests/unit/test_ledger_reset.py` pint hem tegen de ledger. Dit bestand
is anders: `reads` GROEIT met opzet, bij elke `gate_slice`-aanroep, dus een
hele-bestand-hash zou groen worden op het moment dat hij zou moeten falen.
De drie velden die WEL bevroren horen te blijven -- `split_utc`, `git_sha`,
`frozen_utc` -- staan vast vanaf `freeze_holdout()` en worden door geen
functie in deze module ooit herschreven; `tests/unit/test_holdout.py` pint
die drie op het gecommitte artefact.

Eerlijk gezegd: de `reads`-log is append-only BIJ CONVENTIE, niet bij
constructie. Er zit geen handtekening, geen hash-keten en geen ander
mechanisme onder dat een verwijderde entry detecteert of voorkomt --
niemand heeft daarom gevraagd en deze fase voegt geen machinerie toe die
niet nodig is (YAGNI). Wie een entry uit `reads` verwijdert, herstelt
daarmee een verbruikte lezing, en niets in deze module merkt dat. De
bescherming zit in dat zo'n verwijdering een zichtbare regel in een
gecommitte diff is (zie de moduledocstring-openingszin), niet in cryptografie.
`docs/MEASUREMENT_CONTRACT.md` §2 herhaalt deze scoping in proza.

R7, LETTERLIJK
===============
"De poortsample wordt per hypothese ten hoogste eenmaal gelezen. Elke lezing
wordt vóór de run geregistreerd in `holdout_lock.json`." `gate_slice`
schrijft de lees-entry daarom VOORDAT hij de data teruggeeft: een uitvoerder
die het proces afbreekt na de meting maar vóór het loggen, heeft dan alsnog
een geregistreerde lezing. Dat is de goede kant om deze race op te lossen --
de fout die telt is een ONGEREGISTREERDE lezing, niet een geregistreerde
lezing waarvan de data nooit is gebruikt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = [
    "HoldoutAlreadyFrozen",
    "HoldoutAlreadyUsed",
    "HoldoutLock",
    "backcast_gate_slice",
    "development_slice",
    "freeze_holdout",
    "gate_slice",
    "resume_registered_read",
]


class HoldoutAlreadyUsed(RuntimeError):
    """R7 -- de poortsample wordt per hypothese ten hoogste eenmaal gelezen.

    Geen `TradebotContractError`-subklasse: dit is geen geschonden
    datacontract maar een geweigerde HERHAALDE actie, hetzelfde onderscheid
    dat `registry.ledger_reset.ResetAlreadyExists` al maakt voor de reset."""


class HoldoutAlreadyFrozen(RuntimeError):
    """R5 -- er is precies EEN bevroren split (ruling P29).

    Zelfde onderscheid als hierboven bij `HoldoutAlreadyUsed`, en exact het
    patroon van `registry.ledger_reset.ResetAlreadyExists`: "het pad bestaat
    al" is een geweigerde HERHAALDE actie op een bestaande STAAT, geen
    geschonden datacontract op de aangeleverde argumenten. Een aanroeper die
    `DataContractError` vangt om malvormde invoer af te handelen, mag "de
    split is al bevroren" daarom niet stilzwijgend meeslikken."""


@dataclass(frozen=True)
class HoldoutLock:
    """De onveranderlijke header van een bevroren poortsample.

    Draagt bewust GEEN `reads`: dat veld groeit na het bevriezen (ruling
    P24), en deze dataclass is de momentopname van het moment van bevriezen,
    niet een live-aanzicht op het bestand."""

    split_utc: str
    git_sha: str
    frozen_utc: str


def _read_lock(lock_path: Path) -> dict:
    require(
        lock_path.exists(),
        f"{lock_path} ontbreekt. Er is geen bevroren split om op te lezen -- "
        f"draai eerst `apps/freeze_holdout.py` (stap 4B.4). Zonder een "
        f"bevroren split bestaat er geen grens tussen ontwikkeling en poort, "
        f"en dus geen manier om R7 af te dwingen.",
        DataContractError, path=str(lock_path),
    )
    return json.loads(lock_path.read_text(encoding="utf-8"))


def _split_ts(payload: dict) -> pd.Timestamp:
    return pd.Timestamp(payload["split_utc"])


def freeze_holdout(*, split_utc: str, out: Path, git_sha: str) -> HoldoutLock:
    """Bevries de split. Onherhaalbaar op hetzelfde pad; het bestand is het slot.

    Schrijft `{split_utc, git_sha, frozen_utc, reads: []}`. De eerste drie
    velden zijn de onveranderlijke header (ruling P24); `reads` is de enige
    lijst die daarna nog groeit, en uitsluitend via `gate_slice`."""
    require(
        bool(split_utc.strip()),
        "Een lege split_utc heeft geen grens tussen ontwikkeling en poort "
        "-- er is dan niets om te bevriezen (R5).",
        DataContractError,
    )
    # Laat `pd.Timestamp` de ISO-8601-vorm valideren; een niet-parseerbare
    # of tijdzone-loze string hoort hier hard te crashen, niet als string
    # in het bevroren bestand te belanden om pas bij de eerste `gate_slice`
    # te falen.
    ts = pd.Timestamp(split_utc)
    require(
        ts.tzinfo is not None,
        f"split_utc={split_utc!r} heeft geen tijdzone. Een naieve timestamp "
        f"kan niet ondubbelzinnig tegen een UTC-geindexeerd paneel worden "
        f"vergeleken (R5) -- geef expliciet een offset, zoals +00:00.",
        DataContractError,
    )
    if out.exists():
        raise HoldoutAlreadyFrozen(
            f"{out} bestaat al. Er is precies EEN bevroren split (dezelfde "
            f"R5 als `registry.ledger_reset.freeze_reset`, die deze exacte "
            f"'bestaat al'-staat ook als een NAMED `ResetAlreadyExists` "
            f"raist en niet via een generieke contractfout): een tweede "
            f"keer bevriezen zou de header kunnen verschuiven nadat er al "
            f"lezingen tegen de oude grens zijn geregistreerd."
        )
    require(
        bool(git_sha.strip()),
        "Een split zonder git_sha is niet naar de commit te herleiden die "
        "hem heeft bevroren (R12: commit per stap).",
        DataContractError,
    )
    lock = HoldoutLock(
        split_utc=split_utc,
        git_sha=git_sha,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split_utc": lock.split_utc,
                "git_sha": lock.git_sha,
                "frozen_utc": lock.frozen_utc,
                "reads": [],
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    return lock


def development_slice(frame: pd.DataFrame, *, lock_path: Path) -> pd.DataFrame:
    """Alles vóór de bevroren split. Vrij herhaaldelijk te lezen -- R7 geldt
    uitsluitend voor het poortsample, niet voor de ontwikkelsample; het
    trial-budget (`registry.trial_budget`) is de rem op DIT venster."""
    payload = _read_lock(lock_path)
    split = _split_ts(payload)
    return frame.loc[frame.index < split]


def gate_slice(
    frame: pd.DataFrame, *, lock_path: Path, hypothesis_id: str
) -> pd.DataFrame:
    """Alles vanaf de bevroren split -- ten hoogste eenmaal per hypothese (R7).

    Schrijft de lees-entry naar `lock_path` VOORDAT hij de data teruggeeft
    (zie de moduledocstring, "R7, LETTERLIJK"): de goede kant van deze race
    is een geregistreerde lezing waarvan de uitvoerder de uitkomst nooit
    heeft gezien, niet een gebruikte uitkomst die nergens staat.

    De eerlijke grens van deze garantie (twin van ruling P24's append-only
    caveat): de weigering sleutelt op een DOOR DE AANROEPER OPGEGEVEN
    `hypothesis_id`, niet op enige onafhankelijk geverifieerde identiteit.
    "Ten hoogste een lezing" betekent dus precies: ten hoogste een lezing per
    GEDECLAREERDE `hypothesis_id`. Een aanroeper die `"H-10.1"` herdoopt naar
    `"H-10.1b"` koopt daarmee een tweede lezing, en deze functie bouwt daar
    geen machinerie tegen -- geen register van toegestane ids, geen
    ondertekening (YAGNI, zoals P24 al koos voor de `reads`-log zelf). De
    mitigatie is zichtbaarheid, niet voorkoming: elke lezing staat met haar
    eigen `hypothesis_id` in het gecommitte `holdout_lock.json`, dus een
    hernoeming is zichtbaar in die diff, net zoals een verwijderde entry dat
    al was onder P24."""
    require(
        bool(hypothesis_id.strip()),
        "Een lezing zonder hypothesis_id is niet aan R7 te toetsen: zonder "
        "identiteit kan een tweede lezing niet worden herkend als een "
        "tweede lezing.",
        DataContractError,
    )
    payload = _register_read(lock_path, hypothesis_id)
    split = _split_ts(payload)
    return frame.loc[frame.index >= split]


def _register_read(lock_path: Path, hypothesis_id: str) -> dict:
    """R7: weiger een tweede lezing, en schrijf de eerste weg VOORDAT er data terugkomt."""
    payload = _read_lock(lock_path)
    already_read = {entry["hypothesis_id"] for entry in payload["reads"]}
    if hypothesis_id in already_read:
        raise HoldoutAlreadyUsed(
            f"{hypothesis_id!r} heeft het poortsample al gelezen "
            f"({lock_path}). R7 staat ten hoogste een lezing per hypothese "
            f"toe; een tweede lezing maakt M op dit sample groter dan 1, en "
            f"dan is het geen poortsample meer. Een nieuwe hypothese-id "
            f"registreert een nieuwe, eigen hypothese -- geen omzeiling."
        )
    payload["reads"].append({
        "hypothesis_id": hypothesis_id,
        "read_utc": datetime.now(timezone.utc).isoformat(),
    })
    lock_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def backcast_gate_slice(
    frame: pd.DataFrame, *, lock_path: Path, hypothesis_id: str
) -> pd.DataFrame:
    """Het spiegelbeeld van `gate_slice`: alles VÓÓR de bevroren split, ten
    hoogste eenmaal per hypothese (R7).

    Voor een sample dat aan het BEGIN van de reeks ligt en nooit is gemeten
    (een backcast): hetzelfde lockformaat, dezelfde registratie vóór de data.
    Een backcast-lock hoort alleen via deze functie gelezen te worden;
    `development_slice` op zo'n lock zou het beschermde stuk vrij teruggeven.
    Het onderscheid zit in het lockpad, en het lockpad staat in de code van
    het programma dat hem bevroor."""
    require(
        bool(hypothesis_id.strip()),
        "Een lezing zonder hypothesis_id is niet aan R7 te toetsen.",
        DataContractError,
    )
    payload = _register_read(lock_path, hypothesis_id)
    split = _split_ts(payload)
    return frame.loc[frame.index < split]


def resume_registered_read(lock_path: Path, *, hypothesis_id: str, reason: str) -> dict:
    """Maak een AL geregistreerde lezing af, eenmaal, zichtbaar -- geen tweede lezing.

    Voor precies dit geval: de lezing staat in `reads` (R7 is dus al geteld), maar de
    berekening crashte voordat er een uitkomst was. Een nieuwe `gate_slice` weigert terecht;
    een nieuwe hypothese-id zou een tweede lezing kopen. Deze functie eist dat de id al
    gelezen heeft, schrijft een `resumes`-entry met de reden VOORDAT de aanroeper verder
    rekent, en weigert een tweede hervatting van dezelfde id. Kandidaat, parameters en
    poorten veranderen hier niet; dat is aan de aanroeper, en zichtbaar in zijn diff."""
    require(bool(reason.strip()), "Een hervatting zonder reden is niet te beoordelen.",
            DataContractError)
    payload = _read_lock(lock_path)
    read = {entry["hypothesis_id"] for entry in payload["reads"]}
    require(hypothesis_id in read, "Alleen een al geregistreerde lezing kan worden "
            "afgemaakt; een eerste lezing gaat via gate_slice.", DataContractError,
            hypothesis_id=hypothesis_id)
    resumes = payload.setdefault("resumes", [])
    if hypothesis_id in {entry["hypothesis_id"] for entry in resumes}:
        raise HoldoutAlreadyUsed(
            f"{hypothesis_id!r} is al eenmaal hervat ({lock_path}); een tweede "
            f"hervatting is een tweede lezing."
        )
    resumes.append({
        "hypothesis_id": hypothesis_id,
        "resumed_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    })
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
