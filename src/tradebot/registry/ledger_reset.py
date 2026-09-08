"""De eenmalige ledger-reset -- AD-24.

WAAROM DIT EEN EIGEN MODULE IS
==============================
`trial_counter.py` telt trials. Deze module doet iets anders en gevaarlijkers:
hij verlaagt een drempel. Dat hoort niet verstopt te zitten in de teller, want
dan is de handeling niet meer zichtbaar in een diff.

DE VEILIGHEID ZIT NIET IN DE REKENKUNDE
=======================================
Bij `M_new = 25` staat de DSR-eis op **1,67** geannualiseerd op het venster van
dit programma (W_FULL, n_obs = 1743, ruling T1-B). Op de legacy purged-WF
OOS-mask (n_obs = 1615) -- het venster waaraan revisie 1 van deze fase nog
rekende -- is dezelfde eis **1,73**. Beide liggen hoger dan het VERWACHTE
maximum van 2.776 pure ruistrekkingen op datzelfde venster: **1,62** op
W_FULL, **1,68** op de legacy mask -- maar de kans dat dat maximum boven de
eis uitkomt is **ongeveer een op de drie** (0,3095 op W_FULL, 0,3092 op de
legacy mask; zelfs de oorspronkelijke 1,74-bij-1615-koppeling van de fasetekst
geeft nog 0,2953 -- geen van deze een op de vijf, en die eerdere lezing was
ONGEMETEN en toevallig geflatteerd -- ruling T3-G). Een ongeregistreerde
zoektocht van die omvang haalt dus regelmatig een resultaat dat deze poort
passeert.

(Gemeten met `tradebot.backtest.metrics.deflated_sharpe`,
`sr_variance = 1/n_obs`, `skew = 0`, `kurtosis = 3`, `bars_per_year = 365`; de
drempels zoals hierboven, en de overschrijdingskans als
`1 - Phi(eis * sqrt(t_jaren)) ** M` met `t_jaren = n_obs / 365`. Zie
`tests/unit/test_ledger_reset.py` voor de reproductie op beide vensters.
`docs/MEASUREMENT_CONTRACT.md` §2.5 scoopt n_obs = 1615 tot het lezen van
legacy-artefacten -- geen drempel die een nieuwe meting mag kiezen -- en
ruling T1-B legt W_FULL (n_obs = 1743) vast als het ENE venster van dit
programma. Beide getallen staan hier zodat een lezer die een oud artefact
naast dit bestand legt, geen tegenspraak ziet die er niet is.)

De reset is daarom uitsluitend geldig zolang `M_new` het WERKELIJKE aantal
geprobeerde varianten telt, en die belofte wordt niet door dit bestand gedragen
maar door vier dingen buiten dit bestand: de pre-registratie, het bevriezen
vóór de eerste fit, het falsificatieregister (R2), en het afgesloten
poortsample (R8, stap 4B).

WAAROM DIT BESTAND ZICHZELF CONTROLEERT (ruling T3-B/T3-D)
===========================================================
Een bevroren bestand dat niemand leest, is geen slot maar een notitie:
`tests/unit/test_monitoring_thresholds_are_frozen.py` formuleert het zo --
zonder de test die weigert groen te worden zodra de gemeten hash afwijkt, is
het bevriezen van een hash een bestand dat niemand leest. `active_trial_count`
controleert daarom `archived_total` tegen `HypothesisLedger.total_n_hypotheses()`
in plaats van het zelfrapportage-veld van het artefact te geloven, en
`tests/unit/test_ledger_reset.py` pint bovendien dat het gecommitte artefact
bestaat, `m_new == 25` draagt, en zijn eigen hash gelijk is aan de `data_hash`
op het AD-24-amendement in de ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..utils.failfast import DataContractError, require
from .hypothesis_ledger import DEFAULT_LEDGER_PATH, HypothesisLedger
from .trial_counter import TrialCount

__all__ = [
    "ActiveCount",
    "ResetAlreadyExists",
    "ResetRecord",
    "active_trial_count",
    "freeze_reset",
]


class ResetAlreadyExists(RuntimeError):
    """R5 -- er is er precies een."""


@dataclass(frozen=True)
class ResetRecord:
    m_new: int
    rationale: str
    git_sha: str
    frozen_utc: str
    archived_total: int


@dataclass(frozen=True)
class ActiveCount:
    total: int
    archived_total: int
    reset_path: Path

    def to_trial_count(self) -> TrialCount:
        """De post-reset `M` als `TrialCount` -- R-3, geen tweede `M`-implementatie.

        `seed_total=0` omdat de gearchiveerde 2.776 hier niet in doorwerken:
        R3 archiveert die telling, zij telt niet mee in de post-reset `M`. De
        volledige `m_new` staat daarom in `registered_total`, wat ook de
        enige indeling is die door `TrialCount.__post_init__` wordt
        geaccepteerd (`seed_total + registered_total == value`).
        """
        return TrialCount(
            value=self.total,
            source="frozen",
            origin=str(self.reset_path),
            seed_total=0,
            registered_total=self.total,
        )


def freeze_reset(
    *, m_new: int, rationale: str, git_sha: str, out: Path,
    ledger_path: Path | None = None,
) -> ResetRecord:
    """Bevries de reset. Onherhaalbaar; het bestand is het slot."""
    require(
        m_new >= 2,
        f"M_new = {m_new} is onbruikbaar voor een DSR. Bailey-Lopez de Prado "
        f"is gedefinieerd vanaf twee trials; onder die grens bestaat er geen "
        f"expected maximum om tegen te corrigeren (R1). En R5 maakt een "
        f"bevroren M_new < 2 onherstelbaar zonder een tweede reset -- dus deze "
        f"eis staat hier, niet pas bij het eerste gebruik van de M.",
        DataContractError, m_new=m_new,
    )
    require(
        bool(rationale.strip()),
        "Een reset zonder grondslag is niet auditbaar (R3).",
        DataContractError,
    )
    if out.exists():
        raise ResetAlreadyExists(
            f"{out} bestaat al. R5: er is precies EEN reset. Een tweede maakt "
            f"M een parameter in plaats van een meting."
        )
    ledger = ledger_path or DEFAULT_LEDGER_PATH
    archived = HypothesisLedger(ledger).total_n_hypotheses()
    record = ResetRecord(
        m_new=int(m_new),
        rationale=rationale,
        git_sha=git_sha,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        archived_total=archived,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "m_new": record.m_new,
                "rationale": record.rationale,
                "git_sha": record.git_sha,
                "frozen_utc": record.frozen_utc,
                "archived_total": record.archived_total,
                "protocol": "AD-24 R1-R8",
                "note": (
                    "De oude telling is GEARCHIVEERD, niet gewist. "
                    "FALSIFICATION_REGISTER.md F1-F20 blijft onverkort bindend; "
                    "dat is de prijs van deze reset (R2). Het bevroren "
                    "poortsample (R8) is de enige externe verificatie."
                ),
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    return record


def active_trial_count(*, reset_path: Path, ledger_path: Path) -> ActiveCount:
    """De `M` waarmee wordt gemeten na de reset, plus wat er is gearchiveerd.

    `archived_total` in het bevroren bestand is een ZELFRAPPORTAGE. Deze
    functie gelooft haar niet: zij herrekent `archived_total` uit
    `ledger_path` via dezelfde `HypothesisLedger.total_n_hypotheses()` die
    ook `trial_counter.live_trial_count()` gebruikt (R-3, geen tweede
    implementatie), en weigert wanneer de twee uiteenlopen (ruling T3-D). Een
    afwijking betekent dat het artefact of de ledger is veranderd sinds het
    bevriezen, en dan is er geen gecontroleerde `M` meer om mee te meten.
    """
    require(
        reset_path.exists(),
        "Geen bevroren reset gevonden; meten met een impliciete M is precies "
        "de vrijheidsgraad die de DSR hoort weg te nemen (R6).",
        DataContractError, path=str(reset_path),
    )
    payload = json.loads(reset_path.read_text(encoding="utf-8"))
    claimed = int(payload["archived_total"])
    measured = HypothesisLedger(ledger_path).total_n_hypotheses()
    require(
        measured == claimed,
        f"{reset_path} claimt archived_total={claimed}, maar {ledger_path} "
        f"telt op tot {measured}. Het bevroren bestand is een zelfrapportage "
        f"van het moment van bevriezen; zonder deze controle kan zij "
        f"stilzwijgend uit de pas raken met de ledger die zij beweert samen "
        f"te vatten (R3, R5).",
        DataContractError, claimed=claimed, measured=measured,
    )
    return ActiveCount(
        total=int(payload["m_new"]),
        archived_total=claimed,
        reset_path=reset_path,
    )
