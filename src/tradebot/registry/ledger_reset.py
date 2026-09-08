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
eis uitkomt is in de orde van een op de vijf. Een ongeregistreerde zoektocht
van die omvang haalt dus met enige regelmaat een resultaat dat deze poort
passeert.

(Gemeten met `tradebot.backtest.metrics.deflated_sharpe`,
`sr_variance = 1/n_obs`, `skew = 0`, `kurtosis = 3`, `bars_per_year = 365`;
zie `tests/unit/test_ledger_reset.py` voor de reproductie op beide vensters.
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
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..utils.failfast import DataContractError, require
from .trial_counter import TrialCount

__all__ = ["ResetAlreadyExists", "ResetRecord", "active_trial_count", "freeze_reset"]


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


def _archived_total(ledger_path: Path) -> int:
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    return int(payload["seed_total"]) + sum(
        int(entry["n_trials"]) for entry in payload["entries"]
    )


def freeze_reset(
    *, m_new: int, rationale: str, git_sha: str, out: Path,
    ledger_path: Path | None = None,
) -> ResetRecord:
    """Bevries de reset. Onherhaalbaar; het bestand is het slot."""
    require(
        m_new > 0,
        "M_new moet positief zijn. Een programma zonder geplande trials heeft "
        "geen kandidaten, en een DSR met M = 0 is geen correctie (R1).",
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
    ledger = ledger_path or Path("artefacts/governance/hypothesis_ledger.json")
    archived = _archived_total(ledger)
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
    """De `M` waarmee wordt gemeten na de reset, plus wat er is gearchiveerd."""
    require(
        reset_path.exists(),
        "Geen bevroren reset gevonden; meten met een impliciete M is precies "
        "de vrijheidsgraad die de DSR hoort weg te nemen (R6).",
        DataContractError, path=str(reset_path),
    )
    payload = json.loads(reset_path.read_text(encoding="utf-8"))
    return ActiveCount(
        total=int(payload["m_new"]),
        archived_total=int(payload["archived_total"]),
        reset_path=reset_path,
    )
