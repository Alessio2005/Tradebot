# src/tradebot/registry/preregistration.py
"""Pre-registratiecontract — de hypothese wordt vastgelegd VOOR de eerste run.

Phase 3, stap 1. (Oorspronkelijk een deliverable van de Research & Falsification
Foundation; hij wordt hier gebouwd omdat Phase 3 zonder hem niet kan starten —
zie `reports/phase2_feature_validation.md` §0 voor de fase-nummering.)

WAAROM DIT BESTAAT
------------------
Zonder pre-registratie is er geen verschil tussen *"ik voorspelde X en mat X"*
en *"ik mat X en noemde het achteraf mijn voorspelling"*. Het tweede is geen
onderzoek maar een verhaal, en het is de meest voorkomende manier waarop een
backtest zichzelf bevestigt.

Een pre-registratie legt daarom vast, en wel voordat er ook maar een run heeft
plaatsgevonden:

  * de **hypothese** en expliciet de **nulhypothese**;
  * het **universum**, de **periode** en de **granulariteit**;
  * de volledige **parameterruimte** — geen sweep die achteraf wordt uitgebreid;
  * het **geplande aantal trials**, dat meetelt in de `M` van de DSR;
  * de **stop-criteria**: welke uitkomst leidt tot welk besluit. Een drempel
    verlagen omdat het resultaat tegenviel is fraude; de toegestane zet is
    de-scopen, nooit versoepelen.

De `preregistration_id` is een hash over die inhoud PLUS de `data_hash` van elke
gecertificeerde reeks waarop hij betrekking heeft. Pre-registreren op de ene
dataset en meten op de andere is daardoor detecteerbaar en niet iets dat je moet
onthouden.

De `git_sha` zit bewust NIET in de ID. Een pre-registratie is een document, geen
codeartefact: zijn identiteit hoort niet te verschuiven omdat er elders in de
repo een commit is gemaakt. De `git_sha` op het moment van bevriezen wordt wel
naast de ID vastgelegd. (Dit is het omgekeerde van de keuze bij
`features.registry.feature_hash`, waar de codeversie juist WEL meetelt omdat het
artefact daar door de code wordt geproduceerd.)

CONTRACT
--------
* Bevriezen is eenmalig. Een tweede `freeze` met afwijkende inhoud crasht;
  met identieke inhoud is het een no-op, zodat een herstarte run veilig is.
* `require_preregistration(...)` crasht op een onbekende ID. Een gate-run zonder
  geldige pre-registratie is per definitie ongeldig.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 6, 17, 18.1 (regel 1), 26.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.failfast import ConfigContractError, DataContractError, require
from ..utils.hashing import DATA_HASH_LENGTH, hash_config
from ..utils.time import now_utc

__all__ = [
    "DEFAULT_PREREGISTRATION_DIR",
    "PREREGISTRATION_ID_LENGTH",
    "PreRegistration",
    "StopCriterion",
    "freeze_preregistration",
    "load_preregistration_spec",
    "require_preregistration",
]

#: Waar bevroren pre-registraties landen.
DEFAULT_PREREGISTRATION_DIR = Path("artefacts/governance")

#: Lengte van de `preregistration_id`, gelijk aan die van de `data_hash`.
PREREGISTRATION_ID_LENGTH = DATA_HASH_LENGTH

#: De enige toegestane vergelijkingsoperatoren in een stop-criterium.
_OPERATORS = ("<", "<=", ">", ">=")

#: Wat er gebeurt zodra een stop-criterium bindt. Er is bewust geen actie
#: "drempel bijstellen": dat is precies de zet die pre-registratie uitsluit.
_ACTIONS = ("falsify", "archive", "descope", "promote")

#: Toegestane sleutels in een stop-criterium in de YAML.
_CRITERION_KEYS = frozenset(
    {"name", "metric", "operator", "threshold", "action", "rationale"}
)

#: Toegestane sleutels op het topniveau van een pre-registratie-YAML.
#: `parameters` staat er bewust NIET bij. De parameterruimte wordt niet in de
#: YAML overgeschreven maar bij het bevriezen INGESPOTEN vanuit `conf/`, zodat
#: er precies een bron van waarheid is en de registratie niet uit de pas kan
#: lopen met de configuratie waarmee daadwerkelijk wordt gedraaid.
_SPEC_KEYS = frozenset(
    {
        "wave", "title", "hypothesis", "null_hypothesis", "universe",
        "granularity", "period_start", "period_end", "evaluation_start",
        "primary_metric", "planned_trials", "stop_criteria", "data_series",
    }
)


@dataclass(frozen=True)
class StopCriterion:
    """Een vooraf vastgelegde beslisregel: welke uitkomst leidt tot welk besluit."""

    name: str
    metric: str
    operator: str
    threshold: float
    action: str
    rationale: str

    def __post_init__(self) -> None:
        require(bool(self.name), "Stop-criterium zonder naam.", ConfigContractError)
        require(
            bool(self.metric),
            "Stop-criterium zonder metriek; een regel die niet naar een gemeten "
            "grootheid verwijst, kan niet binden.",
            ConfigContractError,
            name=self.name,
        )
        require(
            self.operator in _OPERATORS,
            "Onbekende operator in stop-criterium.",
            ConfigContractError,
            name=self.name,
            operator=self.operator,
            allowed=list(_OPERATORS),
        )
        require(
            self.action in _ACTIONS,
            "Onbekende actie in stop-criterium. Er bestaat bewust GEEN actie die "
            "de drempel bijstelt: dat is de zet die pre-registratie uitsluit.",
            ConfigContractError,
            name=self.name,
            action=self.action,
            allowed=list(_ACTIONS),
        )
        require(
            bool(self.rationale),
            "Stop-criterium zonder onderbouwing. Waarom deze drempel en niet een "
            "andere, is precies wat achteraf niet meer eerlijk te beantwoorden is.",
            ConfigContractError,
            name=self.name,
        )

    def binds(self, measured: float) -> bool:
        """True wanneer de gemeten waarde dit criterium doet binden."""
        if self.operator == "<":
            return measured < self.threshold
        if self.operator == "<=":
            return measured <= self.threshold
        if self.operator == ">":
            return measured > self.threshold
        return measured >= self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": float(self.threshold),
            "action": self.action,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PreRegistration:
    """Een onveranderlijke pre-registratie. De ID is een hash over de inhoud."""

    wave: str
    title: str
    hypothesis: str
    null_hypothesis: str
    universe: tuple[str, ...]
    granularity: str
    period_start: str
    period_end: str
    evaluation_start: str
    primary_metric: str
    parameters: Mapping[str, Any]
    planned_trials: int
    stop_criteria: tuple[StopCriterion, ...]
    data_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "universe", tuple(self.universe))
        object.__setattr__(self, "stop_criteria", tuple(self.stop_criteria))
        object.__setattr__(self, "data_hashes", tuple(sorted(self.data_hashes)))
        for field_name in ("wave", "title", "hypothesis", "null_hypothesis",
                           "granularity", "period_start", "period_end",
                           "evaluation_start", "primary_metric"):
            require(
                bool(getattr(self, field_name)),
                f"Pre-registratie mist het verplichte veld {field_name!r}.",
                ConfigContractError,
                wave=self.wave,
            )
        require(
            len(self.universe) > 0,
            "Pre-registratie zonder universum.",
            ConfigContractError,
            wave=self.wave,
        )
        require(
            self.planned_trials > 0,
            "Een pre-registratie met nul geplande trials beschrijft geen "
            "onderzoek. Het aantal telt bovendien mee in de M van de DSR.",
            ConfigContractError,
            wave=self.wave,
            planned_trials=self.planned_trials,
        )
        require(
            len(self.stop_criteria) > 0,
            "Pre-registratie zonder stop-criteria. Zonder vooraf vastgelegde "
            "beslisregels is elke uitkomst achteraf te herinterpreteren, en dan "
            "is de pre-registratie decoratie.",
            ConfigContractError,
            wave=self.wave,
        )
        require(
            len(self.data_hashes) > 0,
            "Pre-registratie zonder gecertificeerde data_hash. De registratie zou "
            "dan niet vastleggen WAAROP hij betrekking heeft.",
            DataContractError,
            wave=self.wave,
        )
        names = [c.name for c in self.stop_criteria]
        require(
            len(set(names)) == len(names),
            "Dubbele naam in de stop-criteria.",
            ConfigContractError,
            wave=self.wave,
            names=names,
        )

    # --------------------------------------------------------------- identiteit
    def content(self) -> dict[str, Any]:
        """De inhoud waarover de ID wordt berekend. Geen metadata, geen git_sha."""
        return {
            "wave": self.wave,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "null_hypothesis": self.null_hypothesis,
            "universe": list(self.universe),
            "granularity": self.granularity,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "evaluation_start": self.evaluation_start,
            "primary_metric": self.primary_metric,
            "parameters": {k: self.parameters[k] for k in sorted(self.parameters)},
            "planned_trials": int(self.planned_trials),
            "stop_criteria": [c.as_dict() for c in self.stop_criteria],
            "data_hashes": [list(pair) for pair in self.data_hashes],
        }

    @property
    def preregistration_id(self) -> str:
        return hash_config(self.content(), length=PREREGISTRATION_ID_LENGTH)

    def artefact_path(self, directory: Path | str = DEFAULT_PREREGISTRATION_DIR) -> Path:
        return Path(directory) / f"preregistration_{self.preregistration_id}.json"

    def stop_criterion(self, name: str) -> StopCriterion:
        for c in self.stop_criteria:
            if c.name == name:
                return c
        require(
            False,
            "Onbekend stop-criterium. Een criterium dat niet is pre-geregistreerd, "
            "bestaat niet voor deze wave.",
            ConfigContractError,
            name=name,
            available=[c.name for c in self.stop_criteria],
        )
        raise AssertionError("unreachable")  # pragma: no cover


def load_preregistration_spec(
    path: Path | str,
    *,
    data_hashes: Sequence[tuple[str, str]],
    parameters: Mapping[str, Any],
) -> PreRegistration:
    """Laad een pre-registratie-YAML en koppel er data en parameters aan.

    `parameters` wordt door de aanroeper uit `conf/` geresolveerd en NIET in de
    YAML herhaald: een tweede kopie van dezelfde drempels zou uit de pas kunnen
    lopen met de configuratie waarmee daadwerkelijk wordt gedraaid, en dan pint
    de registratie iets anders vast dan er is gemeten.

    Onbekende sleutels worden geweigerd. Een typo in een stop-criterium mag nooit
    stilzwijgend betekenen dat het criterium niet bestaat.
    """
    import yaml

    p = Path(path)
    require(
        p.is_file(),
        "Pre-registratiebestand bestaat niet.",
        ConfigContractError,
        path=str(p),
    )
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    require(
        isinstance(raw, dict) and bool(raw),
        "Pre-registratiebestand is leeg of geen mapping.",
        ConfigContractError,
        path=str(p),
    )
    payload = raw.get("preregistration", raw)
    require(
        isinstance(payload, dict),
        "De sleutel `preregistration` bevat geen mapping.",
        ConfigContractError,
        path=str(p),
    )
    unknown = sorted(set(payload) - _SPEC_KEYS)
    require(
        not unknown,
        "Onbekende sleutel(s) in de pre-registratie. Een typo mag nooit "
        "stilzwijgend betekenen dat een afspraak niet bestaat.",
        ConfigContractError,
        path=str(p),
        unknown=unknown,
        allowed=sorted(_SPEC_KEYS),
    )
    missing = sorted(_SPEC_KEYS - set(payload) - {"data_series"})
    require(
        not missing,
        "Ontbrekende sleutel(s) in de pre-registratie.",
        ConfigContractError,
        path=str(p),
        missing=missing,
    )

    criteria: list[StopCriterion] = []
    for raw_c in payload["stop_criteria"]:
        require(
            isinstance(raw_c, dict),
            "Elk stop-criterium moet een mapping zijn.",
            ConfigContractError,
            path=str(p),
        )
        unknown_c = sorted(set(raw_c) - _CRITERION_KEYS)
        require(
            not unknown_c,
            "Onbekende sleutel(s) in een stop-criterium.",
            ConfigContractError,
            path=str(p),
            unknown=unknown_c,
        )
        criteria.append(
            StopCriterion(
                name=str(raw_c["name"]),
                metric=str(raw_c["metric"]),
                operator=str(raw_c["operator"]),
                threshold=float(raw_c["threshold"]),
                action=str(raw_c["action"]),
                rationale=str(raw_c["rationale"]),
            )
        )

    return PreRegistration(
        wave=str(payload["wave"]),
        title=str(payload["title"]),
        hypothesis=str(payload["hypothesis"]),
        null_hypothesis=str(payload["null_hypothesis"]),
        universe=tuple(str(s) for s in payload["universe"]),
        granularity=str(payload["granularity"]),
        period_start=str(payload["period_start"]),
        period_end=str(payload["period_end"]),
        evaluation_start=str(payload["evaluation_start"]),
        primary_metric=str(payload["primary_metric"]),
        parameters=dict(parameters),
        planned_trials=int(payload["planned_trials"]),
        stop_criteria=tuple(criteria),
        data_hashes=tuple(data_hashes),
    )


def freeze_preregistration(
    prereg: PreRegistration,
    *,
    git_sha: str,
    ledger_total_at_freeze: int,
    directory: Path | str = DEFAULT_PREREGISTRATION_DIR,
) -> Path:
    """Bevries de pre-registratie op schijf. Eenmalig, onveranderlijk.

    Bestaat het artefact al met IDENTIEKE inhoud, dan is dit een no-op — een
    herstarte run mag niet crashen op iets dat al klopt. Bestaat het met
    afwijkende inhoud onder dezelfde ID, dan is er iets fundamenteel mis met de
    hash en crasht deze functie.
    """
    require(
        bool(git_sha),
        "Bevriezen zonder git_sha. De codetoestand op het moment van vastleggen "
        "is onderdeel van het bewijs dat de registratie VOOR de run kwam.",
        DataContractError,
        wave=prereg.wave,
    )
    require(
        ledger_total_at_freeze > 0,
        "Bevriezen zonder de stand van de hypothese-ledger. Zonder die stand is "
        "achteraf niet vast te stellen hoeveel trials er al waren gedaan, en is "
        "de M van de DSR niet te controleren.",
        DataContractError,
        wave=prereg.wave,
        ledger_total=ledger_total_at_freeze,
    )
    document = {
        "preregistration_id": prereg.preregistration_id,
        "content": prereg.content(),
        "frozen_utc": now_utc().isoformat(),
        "git_sha": git_sha,
        "ledger_total_at_freeze": int(ledger_total_at_freeze),
    }
    path = prereg.artefact_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        require(
            existing["content"] == document["content"],
            "Er bestaat al een bevroren pre-registratie met deze ID maar met "
            "AFWIJKENDE inhoud. Dat kan niet kloppen: de ID is een hash over "
            "precies die inhoud.",
            DataContractError,
            path=str(path),
            preregistration_id=prereg.preregistration_id,
        )
        return path
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def require_preregistration(
    preregistration_id: str,
    *,
    directory: Path | str = DEFAULT_PREREGISTRATION_DIR,
) -> PreRegistration:
    """Laad een bevroren pre-registratie of CRASH.

    Dit is de poort uit sectie 18.1 regel 1: een run zonder geldige
    pre-registratie-ID komt er niet doorheen. Er is geen `None`-retour en geen
    `default`; beide zouden een aanroeper de kans geven de controle te negeren.
    """
    require(
        bool(preregistration_id),
        "Gate-run zonder pre-registratie-ID. Een resultaat dat niet aan een "
        "vooraf vastgelegde hypothese te koppelen is, is geen bewijs maar een "
        "waarneming achteraf.",
        DataContractError,
    )
    path = Path(directory) / f"preregistration_{preregistration_id}.json"
    require(
        path.is_file(),
        "Onbekende pre-registratie-ID; er is geen bevroren registratie met deze "
        "ID. Registreer de hypothese VOORDAT de run plaatsvindt.",
        DataContractError,
        preregistration_id=preregistration_id,
        directory=str(directory),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    content = document["content"]
    prereg = PreRegistration(
        wave=content["wave"],
        title=content["title"],
        hypothesis=content["hypothesis"],
        null_hypothesis=content["null_hypothesis"],
        universe=tuple(content["universe"]),
        granularity=content["granularity"],
        period_start=content["period_start"],
        period_end=content["period_end"],
        evaluation_start=content["evaluation_start"],
        primary_metric=content["primary_metric"],
        parameters=content["parameters"],
        planned_trials=int(content["planned_trials"]),
        stop_criteria=tuple(
            StopCriterion(
                name=c["name"], metric=c["metric"], operator=c["operator"],
                threshold=float(c["threshold"]), action=c["action"],
                rationale=c["rationale"],
            )
            for c in content["stop_criteria"]
        ),
        data_hashes=tuple(tuple(pair) for pair in content["data_hashes"]),
    )
    require(
        prereg.preregistration_id == preregistration_id,
        "De inhoud van het bevroren artefact hasht niet naar zijn eigen ID. Het "
        "bestand is na het bevriezen gewijzigd.",
        DataContractError,
        path=str(path),
        recomputed=prereg.preregistration_id,
        stored=preregistration_id,
    )
    return prereg
