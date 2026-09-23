# src/tradebot/registry/policy_carry.py
"""Elke meting draagt de policy-hash van de configuratie die haar produceerde.

Fase 11, stap 2. AD-27, R-13.

WAAROM DIT BESTAND BESTAAT
==========================
`risk_registry.py` hiernaast legt vast WELKE risicoconfiguraties hebben
bestaan. Dat is niet hetzelfde als vastleggen ONDER WELKE configuratie een
meting is gedaan, en het verschil is op 2026-09-12 duur geworden.

`b18aeda` verving het propfirm-budget door een eigen-kapitaalmandaat:
`sigma_target` 0,08 -> 0,20, `max_leverage` 1,5 -> 4,0, `max_position_age_h`
48 -> 720. De `config_hash` sprong van `1b60cb664fbf9a2a` naar
`9961e1613bc907a5`. Het register noteerde de nieuwe rij netjes. Daarna liep de
mandaatwijziging door de hele meetbasis zonder een enkele melding: de
vier-lagen-ladder, twee fase-6-campagnes, twee fase-9-baselines en de
fase-10-campagne meten nog altijd onder het vervallen beleid.

Eén ding merkte het op -- `tests/regression/test_dust_breaks_relative_limits.py`
-- en alleen omdat iemand daar een hash had vastgepind. Dat is toeval van
dekking. Deze module maakt er een mechanisme van.

DE DRIE EISEN
=============
1. Een document dat een RISICOBESLUIT bevat, draagt een `config_hash`. Geen
   hash is een `DataContractError` -- zo staat `phase10_h10_1.json` er vandaag
   bij: 1.566 halts geteld, geen beleid genoemd.
2. Die hash staat in `risk_config_registry.json`. Een hash die daar niet staat
   betekent dat de producerende configuratie nooit is geregistreerd. Dat is het
   enige signaal dat een risicobesluit uit een NIET-GETRACKTE configbron
   (`conf/env/`, DI-32) achterlaat: git ziet zo'n bron niet, het register wel.
3. Meet een artefact onder een VERVALLEN beleid, dan zegt het dat expliciet via
   `superseded_by_policy`. Blijven staan als herkomst mag; stilzwijgen niet.

WAAROM DE MARKERING BEDERFELIJK IS
==================================
`superseded_by_policy.superseded_by` moet gelijk zijn aan de HUIDIGE hash, en
`measured_under` moet exact de hashes noemen die het artefact draagt. Een
volgende mandaatwijziging maakt daarmee elke markering ongeldig en dwingt een
nieuw besluit per artefact. Dat is opzet: een stempel dat een mandaatwijziging
overleeft, is precies de constructie die deze fase repareert.

WAT DEZE MODULE NIET DOET
=========================
Zij leidt niet af welke configuratie een historisch artefact heeft gedraaid --
dat kan niet uit het artefact zelf. Zij eist dat het artefact het zegt. De
`git_sha` is daarvoor géén vervanger: `risk_config_registry.json` registreert
`9961e1613bc907a5` onder `git_sha 7f6181d`, en op dát commit levert
`conf/risk/default.yaml` nog `1b60cb664fbf9a2a` (`b18aeda`, het commit dat de
waarden wél draagt, is geen voorouder van `7f6181d`). De hash die met de meting
meereist is het enige betrouwbare anker.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..utils.failfast import DataContractError, require

__all__ = [
    "DECISION_MARKERS",
    "DISPOSITIONS",
    "HASH_SOURCES",
    "POLICY_HASH_KEYS",
    "SUPERSEDED_KEY",
    "carries_risk_decision",
    "current_policy_hash",
    "declared_policy_hashes",
    "latest_registered_policy",
    "registered_policy_hashes",
    "require_policy_carry",
    "require_registered_policy",
    "superseded_marking",
]

#: Repository-root. Zelfde conventie als `schemas/config.py::_REPO_ROOT`.
_REPO_ROOT = Path(__file__).resolve().parents[3]

RISK_REGISTRY_PATH = _REPO_ROOT / "artefacts/governance/risk_config_registry.json"
RISK_CONFIG_PATH = _REPO_ROOT / "conf/risk/default.yaml"

#: Sleutels die bewijzen dat de SOEVEREINE LAAG heeft gedraaid, niet dat er
#: ergens een drempel wordt genoemd. Een preregistratie die `sigma_target`
#: citeert beslist niets; een artefact met een haltteller wel. Vandaar
#: tellers en uitkomsten, geen configuratiewaarden -- anders wordt elke
#: datacatalogus onterecht rood en verliest de poort haar betekenis.
DECISION_MARKERS = frozenset({
    "n_sovereign_halted",   # backtest/engine.py: de haltteller per track
    "n_sovereign_clipped",  # idem, de clipteller
    "permitted_exposure",   # de uitkomst van RiskEngine.decide
    "halt_reason",          # welke kill switch vuurde
    "risk_audit_header",    # RiskEngine.audit_header(), meegeschreven bij stress
    "binding_constraint",   # welke limiet bond (fase 11, stap 3)
    "binding_constraints",  # dezelfde grootheid, meervoud, in de fase-9-baselines
})

#: In volgorde van specificiteit. Artefacten uit verschillende fasen gebruiken
#: verschillende namen voor dezelfde grootheid; dat is DI-28 en wordt hier
#: gelezen, niet gerepareerd.
POLICY_HASH_KEYS = ("risk_policy_hash", "risk_config_hash", "config_hash", "policy_hash")

SUPERSEDED_KEY = "superseded_by_policy"

#: `PROVENANCE_ONLY` -- blijft staan zoals hij is, als herkomst.
#: `REDERIVE`        -- wordt opnieuw afgeleid; tot dat is gebeurd is het
#:                      artefact geen meetlat meer.
DISPOSITIONS = ("PROVENANCE_ONLY", "REDERIVE")

#: Waar de toegeschreven hash vandaan komt. `declared` -- de run schreef hem
#: zelf mee. `derived_from_git_sha` -- hij is achteraf gereconstrueerd uit het
#: `git_sha` van het artefact, en dan moet `derivation` erbij. Dat onderscheid
#: staat in de data en niet in een rapport, omdat het twee verschillende
#: sterktes van bewijs zijn.
HASH_SOURCES = ("declared", "derived_from_git_sha")

#: `risk_config_hash` is de eerste 16 hex-tekens van een sha256. Een lengtetoets
#: alleen telt ook een label als `"h10_3_lapsed_no_"` (hypothesis_ledger.json)
#: als hash, en dan noemt de poort het verkeerde defect.
_HASH_PATTERN = re.compile(r"[0-9a-f]{16}")


def _walk(node: Any) -> Iterable[tuple[str, Any]]:
    """Elke (sleutel, waarde) in een genest JSON-document, diepte-eerst."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key), value
            yield from _walk(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk(value)


def carries_risk_decision(doc: Any) -> bool:
    """Bevat dit document een risicobesluit? Zie `DECISION_MARKERS`."""
    return any(key in DECISION_MARKERS for key, _ in _walk(doc))


def declared_policy_hashes(doc: Any) -> frozenset[str]:
    """Elke policy-hash die het document zelf noemt, op welke diepte dan ook.

    Een artefact met twee verschillende hashes is niet ongeldig maar wel
    verdacht, en de poort behandelt ze alle twee: elke hash moet geregistreerd
    zijn en elke vervallen hash moet gemarkeerd zijn.
    """
    return frozenset(
        value for key, value in _walk(doc)
        if key in POLICY_HASH_KEYS
        and isinstance(value, str) and _HASH_PATTERN.fullmatch(value) is not None
    )


def superseded_marking(doc: Any) -> Mapping[str, Any] | None:
    """De `superseded_by_policy`-markering, als het document er een draagt."""
    if isinstance(doc, Mapping):
        mark = doc.get(SUPERSEDED_KEY)
        if isinstance(mark, Mapping):
            return mark
    return None


def registered_policy_hashes(path: Path | str | None = None) -> frozenset[str]:
    """Elke `config_hash` in het risicoconfiguratie-register."""
    p = Path(path) if path is not None else RISK_REGISTRY_PATH
    require(p.is_file(),
            "Het risicoconfiguratie-register ontbreekt. Zonder dat register "
            "kan geen enkele meting worden geverifieerd: er is dan geen lijst "
            "van beleidsregels die ooit hebben bestaan.",
            DataContractError, path=str(p))
    doc = json.loads(p.read_text(encoding="utf-8"))
    return frozenset(str(e["config_hash"]) for e in doc["entries"])


def latest_registered_policy(path: Path | str | None = None) -> str:
    """De `config_hash` van de LAATST geregistreerde risicoconfiguratie.

    Dit is niet hetzelfde als `current_policy_hash()`, en het verschil is het
    hele punt. `current_policy_hash()` leest `conf/risk/default.yaml`; deze
    functie leest het register. Lopen ze uiteen, dan is er een risicobeleid in
    gebruik dat niemand heeft vastgelegd -- een ONAANGEKONDIGDE
    beleidswijziging. Een test die de twee tegen elkaar zet, kan daarop rood
    worden; een test die twee keer hetzelfde bestand leest, kan dat niet.

    Het register is append-only, dus de laatste rij is de jongste. Dat wordt
    hier niet aangenomen maar gecontroleerd: is hij het niet, dan is de
    append-only-eigenschap geschonden en betekent "de laatste" niets meer.
    """
    p = Path(path) if path is not None else RISK_REGISTRY_PATH
    doc = json.loads(
        p.read_text(encoding="utf-8")) if p.is_file() else {"entries": []}
    entries = doc.get("entries", [])
    require(
        bool(entries),
        "Het risicoconfiguratie-register is leeg. Er is dan geen vastgelegd "
        "beleid om een besluit tegen te toetsen.",
        DataContractError, path=str(p),
    )
    stamps = [str(e.get("registered_at", "")) for e in entries]
    require(
        stamps[-1] == max(stamps),
        "De laatste rij van het register is niet de jongste. Het register is "
        "append-only; is dat niet meer waar, dan is de historie herschreven "
        "en zegt 'het laatst geregistreerde beleid' niets.",
        DataContractError,
        path=str(p), last=stamps[-1], newest=max(stamps),
    )
    return str(entries[-1]["config_hash"])


def current_policy_hash(path: Path | str | None = None) -> str:
    """De `config_hash` van het beleid dat `conf/risk/default.yaml` vandaag levert."""
    from ..risk.engine import risk_config_hash
    from ..schemas.config import RiskConfig, load_config

    return risk_config_hash(
        load_config(path if path is not None else RISK_CONFIG_PATH, RiskConfig))


def require_registered_policy(
    config_hash: str, *, source: str, registered: Iterable[str],
) -> None:
    """Weiger een risicobesluit uit een configuratie die nooit is geregistreerd.

    Dit is de poort op het CONFIGKANAAL, en zij is er om de reden die DI-32
    beschrijft: `conf/env/` staat niet in git (`.gitignore` regel 35 matcht
    `ENV/` op elke diepte), draagt `# @package _global_` en kan daarmee een
    `risk:`-blok leveren dat geen enkele diff laat zien. Wat zo'n bron
    samenstelt is een beleid dat niemand heeft geregistreerd -- en dat is
    precies wat deze functie meet.
    """
    known = frozenset(registered)
    require(
        config_hash in known,
        "Dit risicobesluit komt uit een configuratie die niet in "
        "`artefacts/governance/risk_config_registry.json` staat. Een "
        "niet-geregistreerd beleid is niet auditbaar en kan uit een bron komen "
        "die git niet ziet (conf/env/, DI-32). Registreer de configuratie "
        "voordat er een meting op wordt gedaan (AD-27, R-13).",
        DataContractError,
        source=source, config_hash=config_hash, registered=sorted(known),
    )


def require_policy_carry(
    doc: Any, *, source: str, registered: Iterable[str], current: str,
) -> frozenset[str]:
    """Verifieer dat `doc` het beleid draagt waaronder het is gemeten.

    Geeft de gedragen hashes terug, zodat een aanroeper ze kan rapporteren.
    Een document zonder risicobesluit gaat vrijuit: de poort eist herkomst van
    wie beslist, niet van wie beschrijft.

    Raises
    ------
    DataContractError
        Als het document een risicobesluit draagt en (a) geen hash noemt,
        (b) een hash noemt die nooit is geregistreerd, of (c) onder een
        vervallen beleid meet zonder een geldige `superseded_by_policy`.
    """
    known = frozenset(registered)
    declared = declared_policy_hashes(doc)
    if not carries_risk_decision(doc):
        return declared

    mark = superseded_marking(doc)
    marked: Mapping[str, Any] = mark if mark is not None else {}
    attributed = declared | frozenset(
        str(h) for h in marked.get("measured_under", ()))

    require(
        bool(attributed),
        "Dit artefact bevat een risicobesluit maar noemt geen config_hash. "
        "Een meting zonder haar configuratie is geen meting (R-13): niemand "
        "kan nagaan welke limieten golden toen deze getallen ontstonden. "
        f"Zoek de hash op via het producerende commit en schrijf hem als "
        f"`risk_policy_hash` mee, of markeer het artefact met `{SUPERSEDED_KEY}`.",
        DataContractError,
        source=source, markers=sorted(DECISION_MARKERS & {k for k, _ in _walk(doc)}),
    )

    for unknown in sorted(attributed - known):
        require(
            False,
            "Dit artefact draagt een policy-hash die nooit is geregistreerd. "
            "Ofwel is het register onvolledig, ofwel heeft de producerende run "
            "gedraaid op een configuratie die buiten `conf/risk/default.yaml` "
            "om is samengesteld (DI-32).",
            DataContractError,
            source=source, config_hash=unknown, registered=sorted(known),
        )

    lapsed = attributed - {current}
    if not lapsed:
        return attributed

    require(
        mark is not None,
        "Dit artefact meet onder een VERVALLEN risicobeleid en zegt dat niet. "
        "Het boek dat het meet, bestaat niet meer. Zet er een "
        f"`{SUPERSEDED_KEY}`-blok in met `measured_under`, `superseded_by`, "
        f"`disposition` ({' of '.join(DISPOSITIONS)}) en `reason`, of leid het "
        "artefact opnieuw af (AD-27, fase 11 stap 2.3).",
        DataContractError,
        source=source, lapsed=sorted(lapsed), current=current,
    )

    require(
        frozenset(str(h) for h in marked.get("measured_under", ())) == lapsed,
        "De markering dekt niet de hashes die dit artefact werkelijk draagt. "
        "Een markering die naast het artefact staat in plaats van erop, "
        "verbergt precies wat zij hoort te melden.",
        DataContractError,
        source=source, marked=sorted(marked.get("measured_under", ())),
        lapsed=sorted(lapsed),
    )
    require(
        str(marked.get("superseded_by", "")) == current,
        "De markering noemt niet het beleid dat NU geldt als opvolger. Dat is "
        "geen formaliteit: zo wordt elke markering ongeldig zodra het mandaat "
        "opnieuw wijzigt, en moet iemand per artefact opnieuw beslissen of hij "
        "herkomst blijft of opnieuw wordt afgeleid.",
        DataContractError,
        source=source, superseded_by=marked.get("superseded_by"), current=current,
    )
    require(
        str(marked.get("disposition", "")) in DISPOSITIONS,
        "De markering draagt geen geldige `disposition`. Exit-criterium 4 van "
        "deze fase eist dat PER ARTEFACT is opgeschreven welke van de twee is "
        "gekozen.",
        DataContractError,
        source=source, disposition=marked.get("disposition"),
        allowed=list(DISPOSITIONS),
    )
    require(
        bool(str(marked.get("reason", "")).strip()),
        "De markering draagt geen reden. Een markering zonder reden is een "
        "stempel (R-5).",
        DataContractError, source=source,
    )
    require(
        str(marked.get("hash_source", "")) in HASH_SOURCES,
        "De markering zegt niet WAAR de toegeschreven hash vandaan komt. Een "
        "hash die uit de run zelf komt en een hash die achteraf uit het "
        "producerende commit is afgeleid, zijn niet even sterk bewijs, en het "
        "verschil hoort in de data te staan in plaats van in een rapport.",
        DataContractError,
        source=source, hash_source=marked.get("hash_source"),
        allowed=list(HASH_SOURCES),
    )
    require(
        marked.get("hash_source") != "derived_from_git_sha"
        or bool(str(marked.get("derivation", "")).strip()),
        "Een AFGELEIDE toeschrijving moet zijn afleiding opschrijven. Zonder "
        "die regel is het een reconstructie die niet na te rekenen is, en "
        "daarmee precies de bewering die deze poort weigert.",
        DataContractError, source=source,
    )
    return attributed
