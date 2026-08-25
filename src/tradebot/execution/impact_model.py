# src/tradebot/execution/impact_model.py
"""Het bindende marktimpactmodel — L9.

Phase 5, deliverable 4. De vorm is voorgeschreven en niet onderhandelbaar
(audit §15.1, fase-opdracht §11):

    Impact = eta * sigma_daily * sqrt(order_notional / adv_notional)

WAAROM DIT BESTAAT NAAST `market_impact.py`
-------------------------------------------
`execution/market_impact.py::square_root_impact` rekent dezelfde formule, maar
met `eta: float = 0.142` als DEFAULT-argument. Dat getal is een
literatuurwaarde (Bouchaud-Bonart op BTC-perps) die nooit op dit universum is
gekalibreerd. Een default maakt hem onzichtbaar: een caller die `eta` vergeet,
krijgt geen fout maar een plausibel ogend getal, en het resultaat draagt geen
enkel spoor van het feit dat er niets is gemeten.

Fase-opdracht §11 is daar expliciet over: *"Een ontbrekende eta is
`ConfigContractError` en nooit: nul, fallback, default, silently estimated."*
Deze module heeft daarom geen enkel default-argument, en `ImpactParams` kan niet
worden geconstrueerd zonder volledige provenance.

`market_impact.py` blijft bestaan voor zijn andere inhoud (Ledoit-Wolf
shrinkage, de Ehlers/Kalman smoothers, `RankQuantileScaler`). Zijn
impactfuncties zijn vanaf Phase 5 geen besluitpad meer; zie
`reports/phase5_engine_diff.md`.

WAT eta EN kappa_d BETEKENEN
----------------------------
De audit noemt het model `(eta, kappa_d)` maar schrijft alleen `eta` in de
formule uit. `kappa_d` is hier vastgelegd als de **permanente fractie**: het
deel van de impact dat NIET reverteert binnen de kalibratiehorizon van één dag.
Dat is de standaard Bouchaud-decompositie en het sluit aan op het bestaande
veld `permanent_share` in `market_impact.py`. De definitie staat hier omdat de
audit haar niet geeft, en een ongedefinieerde parameter is geen parameter.

    permanente impact = kappa_d * Impact
    tijdelijke impact = (1 - kappa_d) * Impact

Voor de kostenrekening van één order telt de VOLLEDIGE impact: je betaalt de
tijdelijke component ook, ook al reverteert de prijs erna. `kappa_d` bepaalt
hoeveel van de beweging blijft staan voor de VOLGENDE order, en is daarmee
relevant zodra een positie over meerdere bars wordt opgebouwd.

DE STAAT VAN DE KALIBRATIE OP DIT MOMENT
----------------------------------------
`IMPACT_UNCALIBRATED`. `docs/DATA_REGISTER.md` §6 stelt vast dat er geen
orderboek-L1/L2 en geen trade-prints in de gecertificeerde store staan; alleen
daily OHLCV, funding en open interest. Zonder eigen orders en hun prijsrespons
is `eta` niet identificeerbaar.

Wat WEL meetbaar is, is een bovengrens. Zie `apps/calibrate_impact.py` en
`reports/TCA_CALIBRATION_REPORT.md`. Die bovengrens wordt gebruikt, expliciet
als bovengrens gelabeld, en elk resultaat dat erop draait draagt de status mee.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 15, 15.1, 19 (L9), 26, 27.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..utils.failfast import ConfigContractError, DataContractError, require

__all__ = [
    "ImpactEstimate",
    "ImpactParams",
    "ImpactStatus",
    "square_root_impact",
]


class ImpactStatus(str, Enum):
    """Herkomst van `eta`. Draagt mee naar elk rapport dat impact gebruikt."""

    #: Geschat op eigen orders en hun gemeten prijsrespons.
    CALIBRATED = "CALIBRATED"
    #: Niet te schatten op de beschikbare data; `eta` is een CONSERVATIEVE
    #: BOVENGRENS en geen schatting. Elk resultaat dat hierop draait, moet dit
    #: label zichtbaar dragen (fase-opdracht §11).
    IMPACT_UNCALIBRATED = "IMPACT_UNCALIBRATED"


@dataclass(frozen=True, slots=True)
class ImpactParams:
    """`eta`, `kappa_d` en de volledige herkomst ervan.

    Elk veld is verplicht. Er is geen constructor-default, ook niet voor de
    provenance-velden: een impactparameter zonder `data_hash` is een getal
    waarvan niemand kan navertellen waar het vandaan komt, en dat is precies
    het probleem dat `market_impact.py::eta=0.142` veroorzaakte.
    """

    eta: float
    kappa_d: float
    status: ImpactStatus
    method: str
    data_hash: str
    sample_size: int
    period_start: str
    period_end: str
    instruments: tuple[str, ...]
    eta_ci_low: float
    eta_ci_high: float

    def __post_init__(self) -> None:
        require(
            np.isfinite(self.eta) and self.eta > 0.0,
            "eta moet eindig en strikt positief zijn. Nul is geen geldige "
            "impactcoefficient maar de aanname dat handelen gratis is.",
            ConfigContractError,
            key="impact.eta", eta=self.eta,
        )
        require(
            np.isfinite(self.kappa_d) and 0.0 <= self.kappa_d <= 1.0,
            "kappa_d is de permanente FRACTIE en ligt in [0, 1].",
            ConfigContractError,
            key="impact.kappa_d", kappa_d=self.kappa_d,
        )
        require(
            isinstance(self.status, ImpactStatus),
            "status moet een ImpactStatus zijn, geen vrije tekst.",
            ConfigContractError,
            got=type(self.status).__name__,
        )
        require(
            bool(self.method) and bool(self.data_hash),
            "Een impactparameter zonder methode of data_hash is niet auditbaar.",
            ConfigContractError,
            key="impact", method=self.method, data_hash=self.data_hash,
        )
        require(
            int(self.sample_size) > 0,
            "Een impactparameter geschat op nul waarnemingen.",
            ConfigContractError,
            key="impact.sample_size", sample_size=self.sample_size,
        )
        require(
            len(self.instruments) > 0,
            "Een impactparameter zonder instrumenten.",
            ConfigContractError, key="impact.instruments",
        )
        require(
            np.isfinite(self.eta_ci_low) and np.isfinite(self.eta_ci_high)
            and self.eta_ci_low <= self.eta <= self.eta_ci_high,
            "Het betrouwbaarheidsinterval bevat de puntschatting niet.",
            ConfigContractError,
            key="impact.eta_ci", low=self.eta_ci_low, eta=self.eta,
            high=self.eta_ci_high,
        )
        require(
            bool(self.period_start) and bool(self.period_end),
            "Een impactparameter zonder kalibratieperiode.",
            ConfigContractError, key="impact.period",
        )

    @property
    def is_calibrated(self) -> bool:
        return self.status is ImpactStatus.CALIBRATED

    def as_record(self) -> dict[str, Any]:
        return {
            "eta": float(self.eta),
            "kappa_d": float(self.kappa_d),
            "status": self.status.value,
            "method": self.method,
            "data_hash": self.data_hash,
            "sample_size": int(self.sample_size),
            "period_start": self.period_start,
            "period_end": self.period_end,
            "instruments": list(self.instruments),
            "eta_ci_low": float(self.eta_ci_low),
            "eta_ci_high": float(self.eta_ci_high),
        }


@dataclass(frozen=True, slots=True)
class ImpactEstimate:
    """De impact van één order, met de herkomst van de parameter erbij.

    `status` reist mee zodat een consument nooit hoeft te raden of dit getal op
    metingen of op een bovengrens berust.
    """

    impact_fraction: float
    impact_bps: float
    permanent_fraction: float
    temporary_fraction: float
    participation: float
    status: ImpactStatus

    def cost(self, order_notional: float) -> float:
        """Absolute impactkosten in quote-valuta, altijd niet-negatief."""
        return float(abs(order_notional) * self.impact_fraction)

    def as_record(self) -> dict[str, Any]:
        return {
            "impact_fraction": float(self.impact_fraction),
            "impact_bps": float(self.impact_bps),
            "permanent_fraction": float(self.permanent_fraction),
            "temporary_fraction": float(self.temporary_fraction),
            "participation": float(self.participation),
            "status": self.status.value,
        }


def square_root_impact(
    *,
    order_notional: float,
    adv_notional: float,
    sigma_daily: float,
    params: ImpactParams,
) -> ImpactEstimate:
    """`Impact = eta * sigma_daily * sqrt(order_notional / adv_notional)`.

    Alle argumenten zijn keyword-only en geen enkel heeft een default. Dat is
    opzettelijk: de drie meest voorkomende manieren om een impactmodel
    stilzwijgend uit te schakelen zijn `eta=0`, `sigma=0` en een `adv` die naar
    oneindig gaat, en alle drie zijn hier een crash.

    Parameters
    ----------
    order_notional : omvang van de order in quote-valuta. Het TEKEN doet niet
        ter zake - impact kost geld in beide richtingen - en de absolute waarde
        wordt genomen.
    adv_notional : Average Daily Volume in DEZELFDE valuta. Dimensionele
        consistentie is hier geen detail: `market_impact.py` nam `Q` in
        basis-asset en `V` in bar-volume, en die verhouding is per symbool
        anders geschaald. In quote-notional is `Q/V` dimensieloos en over
        symbolen vergelijkbaar.
    sigma_daily : dagelijkse volatiliteit als decimaal (niet geannualiseerd).
    params : de gekalibreerde parameters, inclusief herkomst.

    Raises
    ------
    ConfigContractError
        Bij een ontbrekende of ongeldige `params`.
    DataContractError
        Bij een niet-eindige of niet-positieve `adv_notional` of een negatieve
        `sigma_daily`. Er is GEEN `min_volume`-guard die deelt door een
        epsilon: een ADV van nul betekent dat het instrument die dag niet
        handelde, en de impact van handelen in een niet-handelend instrument is
        niet klein maar ongedefinieerd.
    """
    require(
        isinstance(params, ImpactParams),
        "square_root_impact vereist expliciete ImpactParams. Er is geen "
        "default-eta: een ontbrekende impactparameter is een contractbreuk, "
        "geen aanleiding tot een aanname (fase-opdracht §11).",
        ConfigContractError,
        got=type(params).__name__,
    )
    q = abs(float(order_notional))
    require(
        np.isfinite(q),
        "Niet-eindige ordergrootte.",
        DataContractError, order_notional=order_notional,
    )
    v = float(adv_notional)
    require(
        np.isfinite(v) and v > 0.0,
        "Ontbrekend of niet-positief ADV. De impact van een order in een "
        "instrument dat niet handelde, is niet klein maar ongedefinieerd; er "
        "wordt hier niet door een epsilon gedeeld.",
        DataContractError,
        adv_notional=adv_notional,
    )
    s = float(sigma_daily)
    require(
        np.isfinite(s) and s >= 0.0,
        "Niet-eindige of negatieve dagvolatiliteit.",
        DataContractError, sigma_daily=sigma_daily,
    )

    participation = q / v
    impact = float(params.eta) * s * math.sqrt(participation)
    return ImpactEstimate(
        impact_fraction=impact,
        impact_bps=impact * 1e4,
        permanent_fraction=impact * float(params.kappa_d),
        temporary_fraction=impact * (1.0 - float(params.kappa_d)),
        participation=participation,
        status=params.status,
    )
