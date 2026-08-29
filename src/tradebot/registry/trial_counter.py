"""De eerlijke trial-teller `M` — Phase 2 deliverable, gebouwd in Phase 7/8 Stage B-3.

WAAROM DIT BESTAAT
==================
De Deflated Sharpe Ratio corrigeert een waargenomen Sharpe voor het aantal
strategievarianten dat is geprobeerd. Die correctie is monotoon in `M`: hoe
groter `M`, hoe hoger de Sharpe moet zijn om dezelfde DSR te halen.

**Een te lage `M` maakt de DSR structureel te optimistisch.** Op 1.743 bars is
dat dodelijk — de standaardfout van een Sharpe schaalt met `1/sqrt(n_obs)`, dus
bij korte reeksen is de expected-maximum-correctie precies het deel dat
onderscheidt tussen een vondst en ruis.

Er zijn twee manieren om `M` fout te krijgen, en zij zijn NIET symmetrisch:

* **`M` te laag** — de gate laat modellen door die er niet doorheen horen. Dit
  is een fout die geld kost en die pas live zichtbaar wordt.
* **`M` te hoog** — de gate weigert modellen die het verdienen. Dit kost een
  gemiste kans, en is zichtbaar op het moment dat hij optreedt.

Dit platform kiest daarom expliciet voor de tweede fout wanneer er twijfel is.
Zie `M_UNCERTAINTY_NOTE`.

DE TWEE BRONNEN VAN `M`, EN WAAROM ZE VERSCHILLEN
=================================================
`live_trial_count()`  — de LOPENDE teller: seed + elke geregistreerde trial.
                        Groeit bij elke append. Gebruik hem om te REGISTREREN.

`frozen_trial_count()` — de `M` die in een BEVROREN pre-registratie staat.
                        Verandert nooit. Gebruik hem om te MÉTEN.

> **De Phase 3-les, letterlijk overgenomen:** `M` voor een meting komt uit een
> bevroren pre-registratie, niet live uit de ledger. Anders is de meting niet
> reproduceerbaar — dezelfde code op dezelfde data geeft morgen een andere DSR
> omdat er intussen een trial is bijgeschreven. Een resultaat dat afhangt van
> wanneer je het uitrekent, is geen resultaat.

DE RECONSTRUCTIEMETHODE VAN DE SEED
===================================
`seed_total = 2363` is geen meting maar een **reconstructie**, uitgevoerd in
Wave 28 nadat het canonieke ledgerbestand verloren was gegaan. De bron is
`docs/WAVE_LOG.md`, itemsgewijs geciteerd:

    2000 (audit §10/§11) + W14:204 + W15:21 + W16:... = 2363

De reconstructie sluit rekenkundig aan op het onafhankelijk vermelde W26-totaal
van 2702. Dat is een **consistentiecontrole, geen bewijs**: beide getallen komen
uit hetzelfde logboek.

Ref: audit §17.1 (DSR als promotiegate), `fase_2_research_falsification.md`
(deliverable `registry/trial_counter.py`), `reports/phase3_exit_report.md` §5.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.failfast import ConfigContractError, DataContractError, require
from .hypothesis_ledger import DEFAULT_LEDGER_PATH, HypothesisLedger

__all__ = [
    "M_UNCERTAINTY_NOTE",
    "TrialCount",
    "frozen_trial_count",
    "live_trial_count",
    "reconstruction_provenance",
]


#: Wordt woordelijk meegeschreven in elk artefact dat een DSR rapporteert.
M_UNCERTAINTY_NOTE = (
    "M is een ONDERGRENS, geen exacte telling. De seed van 2363 is in Wave 28 "
    "gereconstrueerd uit docs/WAVE_LOG.md nadat het canonieke ledgerbestand "
    "verloren ging; trials die nooit in dat logboek zijn beland, ontbreken. "
    "Een ontbrekende trial maakt de DSR te OPTIMISTISCH, nooit te streng. "
    "Elke DSR in dit platform is daarom een bovengrens op de werkelijke "
    "significantie, en een randgeval (p net onder 0,05) moet als NIET "
    "significant worden gelezen."
)


@dataclass(frozen=True)
class TrialCount:
    """Een `M` met zijn herkomst erbij. Een `M` zonder herkomst is een getal."""

    #: De waarde die in de DSR gaat.
    value: int
    #: `"live"` of `"frozen"`. Zie de moduledocstring.
    source: str
    #: Waar hij vandaan komt: pad naar de ledger of de pre-registratie-ID.
    origin: str
    #: Het gereconstrueerde deel, dat niet uit directe registratie komt.
    seed_total: int
    #: Het deel dat wél per trial is geregistreerd.
    registered_total: int

    def __post_init__(self) -> None:
        require(
            self.value >= 2,
            f"M = {self.value} is onbruikbaar voor een DSR. Bailey-Lopez de Prado "
            f"is gedefinieerd vanaf twee trials; onder die grens bestaat er geen "
            f"expected maximum om tegen te corrigeren.",
            DataContractError,
            m=self.value,
        )
        require(
            self.source in ("live", "frozen"),
            f"source moet 'live' of 'frozen' zijn, kreeg {self.source!r}.",
            ConfigContractError,
        )
        require(
            self.seed_total + self.registered_total == self.value,
            f"M ({self.value}) is niet de som van seed ({self.seed_total}) en "
            f"geregistreerd ({self.registered_total}). Een M waarvan de delen niet "
            f"optellen, is niet reconstrueerbaar en dus niet auditbaar.",
            DataContractError,
        )

    @property
    def is_reproducible(self) -> bool:
        """Alleen een bevroren `M` levert morgen dezelfde meting op."""
        return self.source == "frozen"

    def as_dict(self) -> dict[str, Any]:
        """De vorm die in een artefact of ledger-entry wordt meegeschreven."""
        return {
            "M": self.value,
            "M_source": self.source,
            "M_origin": self.origin,
            "M_seed_total": self.seed_total,
            "M_registered_total": self.registered_total,
            "M_uncertainty": M_UNCERTAINTY_NOTE,
        }


def live_trial_count(path: Path | str = DEFAULT_LEDGER_PATH) -> TrialCount:
    """De LOPENDE `M`: seed plus elke geregistreerde trial.

    Gebruik deze om een nieuwe trial te REGISTREREN, nooit om een resultaat te
    meten — hij verandert zodra iemand anders iets bijschrijft.
    """
    ledger = HypothesisLedger(path)
    doc = ledger.load()
    seed = int(doc["seed_total"])
    registered = sum(int(e["n_trials"]) for e in doc["entries"])
    return TrialCount(
        value=seed + registered,
        source="live",
        origin=str(Path(path)),
        seed_total=seed,
        registered_total=registered,
    )


def frozen_trial_count(preregistration_path: Path | str) -> TrialCount:
    """De `M` zoals hij in een BEVROREN pre-registratie staat.

    Dit is de enige `M` waarmee een gerapporteerd resultaat mag worden berekend.

    FAIL-FAST op een pre-registratie zonder `M`. Er is geen terugval naar de
    live-teller: dat zou precies de niet-reproduceerbaarheid herintroduceren die
    deze functie moet uitsluiten, en het zou stilzwijgend gebeuren.
    """
    p = Path(preregistration_path)
    require(
        p.is_file(),
        f"Bevroren pre-registratie bestaat niet: {p}. Een DSR zonder bevroren M "
        f"is niet reproduceerbaar en telt daarom niet als bewijs.",
        ConfigContractError,
    )
    doc = json.loads(p.read_text(encoding="utf-8"))
    content = doc.get("content", doc)
    params = content.get("parameters", {})

    m_value = params.get("M", params.get("m_trials", content.get("M")))
    require(
        m_value is not None,
        f"Pre-registratie {p.name} bevat geen M in `parameters`. Registreer hem "
        f"met `live_trial_count()` op het moment van bevriezen; achteraf een M "
        f"kiezen is precies de vrijheidsgraad die de DSR hoort weg te nemen.",
        DataContractError,
        preregistration=p.name,
    )
    seed = int(params.get("M_seed_total", 0))
    registered = int(params.get("M_registered_total", int(m_value) - seed))
    return TrialCount(
        value=int(m_value),
        source="frozen",
        origin=doc.get("preregistration_id", p.stem),
        seed_total=seed,
        registered_total=registered,
    )


def reconstruction_provenance(
    path: Path | str = DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """De herkomst van de seed, zodat een rapport hem woordelijk kan citeren."""
    doc = HypothesisLedger(path).load()
    return {
        "seed_total": int(doc["seed_total"]),
        "seed_note": doc.get("seed_note", ""),
        "reconstructed_utc": doc.get("reconstructed_utc", ""),
        "reconstruction_note": doc.get("reconstruction_note", ""),
        "uncertainty": M_UNCERTAINTY_NOTE,
    }
