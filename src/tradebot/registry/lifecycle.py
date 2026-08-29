"""De promotie-state-machine — Phase 2 deliverable, gebouwd in Stage B-4.

    REGISTERED -> TESTED -> CANDIDATE -> PAPER -> CHAMPION
         |          |          |          |         |
         +----------+----------+----------+---------+--> FALSIFIED

WAAROM EEN STATE MACHINE EN NIET EEN VLAG
=========================================
`registry/promotion.py` kende tot Stage B alleen `research -> staging -> prod`:
drie STAGES die zeggen waar een model DRAAIT. Dit bestand modelleert iets
anders — hoeveel BEWIJS er voor een model bestaat. De twee lopen niet gelijk: een
model kan in `staging` draaien met het bewijsniveau van `REGISTERED`, en precies
dat gat is waar een onbewezen model productie in glipt.

De regels:

* **Eenrichtingsverkeer.** Van `PAPER` terug naar `CANDIDATE` gaan bestaat niet.
  Een model dat faalt gaat naar `FALSIFIED`, en dat is een eindtoestand. Terug
  kunnen betekent dat je kunt blijven proberen tot het lukt, en dat is
  data-snooping met extra stappen.
* **Elke overgang heeft een bewijslast**, hieronder als code en niet als
  documentatie. Ontbreekt het bewijs, dan crasht de overgang.
* **`FALSIFIED` is bereikbaar vanuit elke toestand** — een model mag altijd
  worden weerlegd.
* **`FALSIFIED` is definitief.** Er is geen `unfalsify`. Een weerlegd model
  opnieuw aanbieden vereist een NIEUWE pre-registratie met een nieuwe hypothese,
  en telt dus opnieuw mee in `M`.

DE OVERGANG DIE STAGE D NODIG HEEFT
===================================
`PAPER -> CHAMPION` is de poort uit audit §18.1 regel 3:

    "Een challenger-model vervangt het champion-model pas na minimaal 60 dagen
     OOS paper-trading waarin het de champion statistisch significant verslaat
     (Diebold-Mariano p < 0.05)."

Hij staat hier, één keer gebouwd, zodat Stage D hem consumeert in plaats van een
tweede versie te schrijven.

WAT `UNPROVEN` HIER DOET
========================
Een model dat de Data Adequacy Gate niet haalt, gaat NIET naar `FALSIFIED`
(no-go 13). Het blijft in zijn huidige toestand met een `UNPROVEN`-notitie. Dat
is geen administratief detail: `FALSIFIED` betekent *"weerlegd"*, en een model
weerleggen op grond van te weinig data is een conclusie trekken uit de afwezigheid
van bewijs.

Ref: audit §18.1, §26; `fase_2_research_falsification.md`; Stage B-4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..utils.failfast import DataContractError, require

__all__ = [
    "FALSIFIED",
    "LifecycleState",
    "ModelLifecycle",
    "TRANSITIONS",
    "TransitionEvidence",
]


class LifecycleState:
    """De zes toestanden. Geen `Enum`, zodat een onbekende string crasht in
    plaats van stilzwijgend als lid te worden aangenomen."""

    REGISTERED = "REGISTERED"
    TESTED = "TESTED"
    CANDIDATE = "CANDIDATE"
    PAPER = "PAPER"
    CHAMPION = "CHAMPION"
    FALSIFIED = "FALSIFIED"


FALSIFIED = LifecycleState.FALSIFIED

#: De enige toegestane vooruitgang. Elke andere overgang is een fout.
_FORWARD: dict[str, str] = {
    LifecycleState.REGISTERED: LifecycleState.TESTED,
    LifecycleState.TESTED: LifecycleState.CANDIDATE,
    LifecycleState.CANDIDATE: LifecycleState.PAPER,
    LifecycleState.PAPER: LifecycleState.CHAMPION,
}

#: Alle toegestane overgangen, inclusief de weg naar FALSIFIED.
TRANSITIONS: dict[str, frozenset[str]] = {
    LifecycleState.REGISTERED: frozenset({LifecycleState.TESTED, FALSIFIED}),
    LifecycleState.TESTED: frozenset({LifecycleState.CANDIDATE, FALSIFIED}),
    LifecycleState.CANDIDATE: frozenset({LifecycleState.PAPER, FALSIFIED}),
    LifecycleState.PAPER: frozenset({LifecycleState.CHAMPION, FALSIFIED}),
    LifecycleState.CHAMPION: frozenset({FALSIFIED}),
    FALSIFIED: frozenset(),          # eindtoestand
}

#: Minimum aantal schone paper-trading-dagen vóór PAPER -> CHAMPION.
#: Audit §18.1 regel 3: een MINIMUM, geen richtlijn.
MIN_PAPER_DAYS = 60

#: Diebold-Mariano-drempel voor dezelfde overgang. Een DREMPEL, geen streefwaarde.
DM_ALPHA = 0.05


@dataclass(frozen=True)
class TransitionEvidence:
    """Het bewijs dat één overgang rechtvaardigt.

    Welke velden verplicht zijn, hangt af van de overgang; `ModelLifecycle`
    controleert dat. Alles is optioneel in het type en verplicht in de regel,
    zodat een ontbrekend veld een duidelijke fout geeft in plaats van een
    constructiefout ver van de oorzaak.
    """

    #: Verplicht voor ELKE overgang. Zonder herkomst is een overgang niet auditbaar.
    git_sha: str
    config_hash: str
    data_hash: str
    preregistration_id: str

    #: REGISTERED -> TESTED
    lookahead_suite_passed: bool | None = None
    #: TESTED -> CANDIDATE
    gate_result_passed: bool | None = None
    gate_result_verdict: str | None = None
    #: CANDIDATE -> PAPER
    paper_config_hash: str | None = None
    #: PAPER -> CHAMPION
    clean_paper_days: int | None = None
    dm_p_value: float | None = None
    dm_vs_champion: str | None = None
    #: -> FALSIFIED
    falsification_reason: str | None = None

    reason: str = ""
    ts_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelLifecycle:
    """De toestand van één model, met zijn volledige overgangshistorie.

    De historie is append-only. Er is geen methode die een overgang verwijdert:
    een model dat ooit `FALSIFIED` was, blijft dat zichtbaar hebben, ook wanneer
    een latere variant onder een nieuwe pre-registratie wél slaagt.
    """

    model_id: str
    state: str = LifecycleState.REGISTERED
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        require(
            self.state in TRANSITIONS,
            f"Onbekende toestand {self.state!r}. Toegestaan: "
            f"{sorted(TRANSITIONS)}.",
            DataContractError,
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------ #
    # De enige manier om van toestand te veranderen
    # ------------------------------------------------------------------ #
    def transition(self, target: str, evidence: TransitionEvidence) -> None:
        """Ga naar `target` of crash. Er is geen derde uitkomst.

        Raises
        ------
        DataContractError
            Bij een niet-toegestane overgang, bij ontbrekende herkomst, of bij
            ontbrekend bewijs voor deze specifieke overgang.
        """
        require(
            target in TRANSITIONS,
            f"Onbekende doeltoestand {target!r}.",
            DataContractError, model_id=self.model_id,
        )
        allowed = TRANSITIONS[self.state]
        # Buiten de f-string opgebouwd: een regelafbreking BINNEN een
        # replacement field is pas geldig vanaf Python 3.12, en dit project
        # belooft 3.10 (`requires-python = ">=3.10"`). De referentie-interpreter
        # is 3.13, dus zo'n constructie draait hier stil door en breekt pas op de
        # laagste ondersteunde versie.
        allowed_text = (
            ", ".join(sorted(allowed)) or "niets - dit is een eindtoestand")
        require(
            target in allowed,
            f"{self.model_id}: overgang {self.state} -> {target} bestaat niet. "
            f"Toegestaan vanuit {self.state}: {allowed_text}. De keten is "
            f"eenrichtingsverkeer: teruggaan zou betekenen dat je kunt blijven "
            f"proberen tot het lukt.",
            DataContractError,
            model_id=self.model_id, current=self.state, target=target,
        )

        for name in ("git_sha", "config_hash", "data_hash", "preregistration_id"):
            require(
                bool(str(getattr(evidence, name, "")).strip()),
                f"{self.model_id}: overgang {self.state} -> {target} mist "
                f"{name!r}. Een overgang zonder resolvable herkomst is niet "
                f"reproduceerbaar en telt daarom niet.",
                DataContractError,
                model_id=self.model_id,
            )

        if target == FALSIFIED:
            require(
                bool(str(evidence.falsification_reason or "").strip()),
                f"{self.model_id}: FALSIFIED zonder reden. Een weerlegging "
                f"zonder vastgelegde grond is niet naderhand te toetsen.",
                DataContractError, model_id=self.model_id,
            )
        else:
            self._require_forward_evidence(target, evidence)

        self.history.append({
            "from": self.state, "to": target, **evidence.as_dict()})
        self.state = target

    def _require_forward_evidence(
        self, target: str, ev: TransitionEvidence
    ) -> None:
        """De bewijslast per vooruitgang. Dit is de kern van deze module."""
        if target == LifecycleState.TESTED:
            require(
                ev.lookahead_suite_passed is True,
                f"{self.model_id}: REGISTERED -> TESTED vereist een GROENE "
                f"lookahead-suite. Zonder causaliteitsbewijs meet elke latere "
                f"toets een model dat de toekomst mag zien.",
                DataContractError, model_id=self.model_id,
            )

        elif target == LifecycleState.CANDIDATE:
            require(
                ev.gate_result_passed is True,
                f"{self.model_id}: TESTED -> CANDIDATE vereist een GESLAAGD "
                f"GateResult (alle vijf poorten). Kreeg verdict="
                f"{ev.gate_result_verdict!r}. Er is geen deelscore.",
                DataContractError, model_id=self.model_id,
            )

        elif target == LifecycleState.PAPER:
            require(
                bool(str(ev.paper_config_hash or "").strip()),
                f"{self.model_id}: CANDIDATE -> PAPER vereist een "
                f"`paper_config_hash` die VÓÓR de start van de klok is "
                f"vastgelegd. Een drempel die achteraf wordt bepaald, is geen "
                f"drempel (no-go 10).",
                DataContractError, model_id=self.model_id,
            )

        elif target == LifecycleState.CHAMPION:
            # De poort uit audit §18.1 regel 3. Stage D consumeert deze.
            require(
                ev.clean_paper_days is not None
                and ev.clean_paper_days >= MIN_PAPER_DAYS,
                f"{self.model_id}: PAPER -> CHAMPION vereist minimaal "
                f"{MIN_PAPER_DAYS} OPEENVOLGENDE schone paper-dagen, kreeg "
                f"{ev.clean_paper_days}. 60 dagen is een MINIMUM, geen "
                f"richtlijn (audit §18.1 regel 3). De teller herstart bij elke "
                f"crash en bij elke onverklaarde execution drift.",
                DataContractError, model_id=self.model_id,
            )
            require(
                ev.dm_p_value is not None and ev.dm_p_value < DM_ALPHA,
                f"{self.model_id}: PAPER -> CHAMPION vereist Diebold-Mariano "
                f"p < {DM_ALPHA} tegen de zittende champion, kreeg "
                f"p={ev.dm_p_value}. Dat is een DREMPEL, geen streefwaarde: "
                f"p = 0,06 is geen promotie, ook niet bij een indrukwekkende "
                f"equity curve.",
                DataContractError, model_id=self.model_id,
            )
            require(
                bool(str(ev.dm_vs_champion or "").strip()),
                f"{self.model_id}: PAPER -> CHAMPION vereist de identiteit van "
                f"de champion waartegen is getoetst. Een DM-uitslag zonder "
                f"tegenpartij is geen vergelijking.",
                DataContractError, model_id=self.model_id,
            )

    # ------------------------------------------------------------------ #
    def falsify(self, reason: str, evidence: TransitionEvidence) -> None:
        """Weerleg dit model. Bereikbaar vanuit elke toestand behalve FALSIFIED."""
        from dataclasses import replace

        self.transition(FALSIFIED, replace(evidence, falsification_reason=reason))

    @property
    def is_terminal(self) -> bool:
        return self.state == FALSIFIED

    @property
    def next_state(self) -> str | None:
        """De enige vooruitgang die vanuit hier bestaat, of None."""
        return _FORWARD.get(self.state)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "state": self.state,
            "is_terminal": self.is_terminal,
            "next_state": self.next_state,
            "history": list(self.history),
        }
