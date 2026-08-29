"""H1 — de QLIKE-competitie: de GARCH-familie tegen EWMA(0.94). Stap 7.

WAT HIER WEL EN NIET STAAT
===========================
De wetenschap staat elders en is daar getoetst:

    volatility/garch.py        fits per fold, forecasts, convergentie
    volatility/realized.py     de variantieproxies
    validation/vol_metrics.py  QLIKE, MSE-SD, MAE-SD, Mincer-Zarnowitz, DM-HLN
    validation/data_adequacy.py  de power-analyse

Dit module doet de vier dingen die DAARTUSSEN zitten, en dat is precies waar een
competitie stukgaat zonder dat er iets rood wordt:

  1. de EWMA-forecast op de bar zetten die hij voorspelt (:func:`ewma_variance_forecast`);
  2. beide modellen op DEZELFDE bars scoren -- uitsluitend de testvensters van de
     walk-forward (:func:`oos_mask`);
  3. de gemeten power uitrekenen, want de pre-registratie maakt het verschil
     tussen `FALSIFIED` en `UNPROVEN` afhankelijk van een getal
     (:func:`power_for_differential`);
  4. de vier stop-criteria in hun vastgelegde volgorde toepassen
     (:func:`judge_challenger`).

DE TITELVERDEDIGER KRIJGT GEEN GROTERE STEEKPROEF
==================================================
EWMA heeft op elke bar na de burn-in een waarde; een GARCH-variant heeft er een
op de testbars van geconvergeerde folds. Zou EWMA op ALLE bars worden gescoord,
dan wint of verliest hij op een andere steekproef dan zijn uitdager -- inclusief
de trainbars waarop de uitdager per constructie niet mag worden beoordeeld.
:func:`oos_mask` maakt van die steekproef één keer een expliciete grootheid.

DE RICHTING VAN EEN TWEEZIJDIGE TOETS
======================================
`p < 0,05` betekent "de twee verschillen", niet "de uitdager wint". Zonder het
teken van het gemiddelde verliesverschil erbij zou een variant die AANTOONBAAR
SLECHTER is dan EWMA als promotiekandidaat uit de toets komen. Dat teken staat
daarom in :func:`judge_challenger` naast de p-waarde, en niet in de bespreking
van de uitkomst achteraf.

Ref: `conf/research/preregistration_h1_garch_vs_ewma.yaml` (bevroren als
`cef1a3b9a6811d7bde1afc92a2a9503f`); `Prompts-fases/fase_6_advanced_research.md`
stap 7; Hansen & Lunde (2005); Harvey, Leybourne & Newbold (1997); Patton (2011).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..cv.walk_forward import WalkForwardCV
from ..registry.phase6_power import H1_ASSUMPTIONS
from ..schemas.config import PowerConfig, adequacy_config, econometrics_config
from ..utils.failfast import DataContractError, require
from ..volatility.ewma import ewma_variance_causal
from .data_adequacy import PowerAnalysis, mean_difference_power
from .vol_metrics import (
    DieboldMarianoResult,
    LossSeries,
    diebold_mariano_hln,
    loss_series,
)

__all__ = [
    "COMPETITION_LOSSES",
    "EXPECTED_EFFECT_SD",
    "H1Verdict",
    "NegativeControls",
    "VerdictStatus",
    "build_losses",
    "ewma_variance_forecast",
    "judge_challenger",
    "mde_sd_units",
    "oos_mask",
    "power_for_differential",
    "proxy_scale_ratio",
    "run_negative_controls",
    "shuffled_forecast",
    "swap_control_rejection_rate",
]

_ECONO = econometrics_config()
_ADEQUACY = adequacy_config()

#: De drie verliesfuncties uit stap 7. QLIKE is de PRIMAIRE maat en de enige
#: die het oordeel draagt: hij is proxy-robuust (Patton 2011), en dat is precies
#: wat telt wanneer de realisatie een range-estimator is in plaats van echte
#: realized variance. MSE-SD en MAE-SD staan ernaast omdat de pre-registratie ze
#: vraagt en omdat een uitkomst die op alle drie hetzelfde zegt, robuuster is
#: dan een die alleen op de primaire maat bestaat.
COMPETITION_LOSSES: tuple[str, ...] = ("qlike", "mse_sd", "mae_sd")

#: Het verwachte effect uit de bevroren pre-registratie, in standaarddeviaties
#: van de per-bar QLIKE-verschilreeks. Het staat in `registry/phase6_power.py`
#: omdat het in de `preregistration_id` is gehasht; het hier overtypen zou een
#: tweede waarheid maken die stil uiteen kan lopen met de eerste.
EXPECTED_EFFECT_SD: float = float(H1_ASSUMPTIONS["expected_effect_sd_units"])

VerdictStatus = Literal["PROMOTED", "FALSIFIED", "UNPROVEN", "DESCOPED"]

#: De namen komen LETTERLIJK uit `stop_criteria` in de pre-registratie. Een
#: oordeel dat naar een criterium verwijst dat daar niet staat, is een criterium
#: dat na de run is bedacht.
_ARCH_GATE = "arch_gate_no_conditional_heteroskedasticity"
_NO_IMPROVEMENT = "no_significant_qlike_improvement"
_UNDERPOWERED = "underpowered_test_cannot_falsify"
_CONVERGENCE = "convergence_too_low_to_compare"

#: Dit criterium staat NIET in de bevroren pre-registratie, en dat wordt hier
#: expliciet vastgelegd in plaats van weggemoffeld. Het toetst de PREMISSE die
#: de pre-registratie zelf uitspreekt om de range-proxy te rechtvaardigen:
#: *"QLIKE is robuust tegen proxy-ruis zolang de proxy conditioneel zuiver is
#: (Patton 2011), en de range-estimators zijn dat onder een driftloze GBM binnen
#: de dag."* Die premisse is meetbaar, en de eerste echte run mat haar als
#: geschonden.
#:
#: Een criterium dat ná het zien van de uitkomst wordt toegevoegd, is normaal
#: gesproken precies de manoeuvre die pre-registratie uitsluit. Dit criterium
#: kan echter UITSLUITEND een promotie tegenhouden en er nooit een maken: het
#: zet `PROMOTED` en `FALSIFIED` allebei om in `UNPROVEN`. Het kan de conclusie
#: dus alleen voorzichtiger maken, nooit gunstiger.
_PROXY_PREMISE = "proxy_premise_violated"

#: Ook dit criterium staat niet in de bevroren pre-registratie, en om dezelfde
#: reden als hierboven: het is gemeten en niet voorzien. Een forecast van 10^25
#: maal de gemiddelde variantie is geen forecast maar een numeriek artefact, en
#: QLIKE verbergt dat bijna -- hij groeit slechts logaritmisch in een
#: overschatting, dus een handvol geexplodeerde bars verschuift het gemiddelde
#: nauwelijks. Precies dat maakt het gevaarlijk: het getal ziet er bruikbaar uit.
#:
#: Het gevolg is een DE-SCOPE en geen falsificatie. Het model heeft op die
#: horizon geen bruikbaar getal geleverd; dat afrekenen als "slechter dan EWMA"
#: zou een oordeel vellen over een meting die niet bestaat.
_DEGENERATE_FORECAST = "degenerate_forecast_level"


# --------------------------------------------------------------------------- #
# 1. De EWMA-forecast
# --------------------------------------------------------------------------- #
def ewma_variance_forecast(
    returns: pd.Series, *, lam: float, burn_in_bars: int, horizon: int,
) -> pd.Series:
    """De `horizon`-staps EWMA-variantieforecast, op de bar die hij voorspelt.

    Rij `t` draagt de forecast die op bar ``t - horizon`` is gemaakt -- exact de
    conventie van :func:`~tradebot.volatility.garch.walk_forward_variance_forecasts`,
    zodat beide modellen dezelfde vraag beantwoorden.

    WAAROM DE FORECAST VLAK IS
    ---------------------------
    RiskMetrics is een IGARCH met ``alpha + beta = 1``: de onvoorwaardelijke
    variantie bestaat niet en de `h`-staps forecast is voor elke `h` gelijk aan
    de huidige conditionele variantie. Er wordt dus NIET met `horizon`
    vermenigvuldigd -- dat zou een variantie OVER `horizon` bars zijn, een
    andere grootheid, en de competitie beslechten op een eenheidsfout in plaats
    van op een model.

    De burn-in blijft NaN en wordt niet gevuld; QLIKE laat die bars vallen en
    het rapport telt hoeveel het er waren.
    """
    require(
        horizon >= 1,
        "Een variantieforecast met een horizon onder 1 bar bestaat niet.",
        DataContractError, horizon=horizon,
    )
    variance = ewma_variance_causal(
        returns, lam=lam, burn_in_bars=burn_in_bars)
    forecast: pd.Series = variance.shift(horizon)
    forecast.name = f"ewma_{lam}_h{horizon}"
    return forecast


# --------------------------------------------------------------------------- #
# 2. De steekproef
# --------------------------------------------------------------------------- #
def oos_mask(n: int, cv: WalkForwardCV) -> np.ndarray:
    """De bars die in ENIG testvenster van de walk-forward liggen.

    Niet de bars vanaf het eerste testvenster, en niet alles behalve de
    trainbars: de embargozone tussen het geëmbargeerde trainvenster en het
    testvenster hoort bij GEEN van beide, en telt hier dus niet mee. Op twaalf
    folds met een embargo van vijf bars gaat het om zestig bars die anders
    stilzwijgend in de competitie zouden meetellen.
    """
    mask = np.zeros(int(n), dtype=bool)
    for fold in cv.split(int(n)):
        mask[fold.test_idx] = True
    require(
        bool(mask.any()),
        "De walk-forward-structuur wees geen enkele out-of-sample bar aan. Een "
        "competitie zonder testvensters is geen leeg resultaat maar een "
        "configuratiefout.",
        DataContractError, n=int(n),
        train_size=cv.train_size, test_size=cv.test_size,
    )
    return mask


# --------------------------------------------------------------------------- #
# 3. De gemeten power
# --------------------------------------------------------------------------- #
def _serial_design_effect(ar1: float) -> float:
    """``(1 + rho) / (1 - rho)``, met een ondergrens van 1.

    Een POSITIEVE autocorrelatie in de verliesverschilreeks maakt opeenvolgende
    bars minder informatief en verkleint het effectieve aantal observaties. Een
    negatieve doet formeel het omgekeerde, maar die winst wordt hier NIET
    geclaimd: hij zou een onderpowerde toets op grond van een toevallig teken
    tot `FALSIFIED` kunnen promoveren, en dat is precies de uitkomst die
    no-go 8 van de fase uitsluit.
    """
    require(
        abs(ar1) < 1.0,
        "Een AR(1) van 1 of meer in absolute waarde: de verliesverschilreeks "
        "is dan niet stationair en de designfactor is niet gedefinieerd.",
        DataContractError, ar1=ar1,
    )
    return max((1.0 + ar1) / (1.0 - ar1), 1.0)


def power_for_differential(
    *,
    n_obs: int,
    ar1: float,
    effective_series: float = 1.0,
    expected_effect_sd: float = EXPECTED_EFFECT_SD,
    cfg: PowerConfig | None = None,
) -> PowerAnalysis:
    """De power van DEZE DM-toets, met de GEMETEN AR(1) in plaats van een scenario.

    De pre-registratie legt drie scenario's vast (AR(1) = 0,0 / 0,2 / 0,4) omdat
    de feitelijke autocorrelatie vooraf onbekend is. Na de run is zij gemeten en
    hoort het oordeel op dat getal te steunen -- niet op het gunstigste
    scenario. `effective_series` is 1 voor een toets op één symbool en het
    gemeten aantal effectief onafhankelijke reeksen voor de gepoolde toets.
    """
    require(
        n_obs > 0,
        "Power-analyse op een lege verliesverschilreeks.",
        DataContractError, n_obs=n_obs,
    )
    require(
        effective_series > 0.0,
        "Power-analyse op een niet-positief aantal effectieve reeksen.",
        DataContractError, effective_series=effective_series,
    )
    design_effect = _serial_design_effect(ar1)
    n_effective = n_obs * effective_series / design_effect
    return mean_difference_power(
        n_effective, expected_effect_sd,
        cfg if cfg is not None else _ADEQUACY.power,
        assumptions={
            "loss_differential_ar1": float(ar1),
            "serial_design_effect": design_effect,
            "effective_independent_series": float(effective_series),
            "n_obs": int(n_obs),
        },
    )


def mde_sd_units(
    *,
    n_obs: int,
    ar1: float,
    effective_series: float = 1.0,
    cfg: PowerConfig | None = None,
) -> float:
    """Het minimaal detecteerbare effect, in SD's van de verliesverschilreeks."""
    return power_for_differential(
        n_obs=n_obs, ar1=ar1, effective_series=effective_series, cfg=cfg,
    ).minimum_detectable_effect


# --------------------------------------------------------------------------- #
# 4. De negatieve controle
# --------------------------------------------------------------------------- #
def proxy_scale_ratio(
    proxy: np.ndarray, squared_return: np.ndarray, *, mask: np.ndarray,
) -> float:
    """``mean(proxy) / mean(r^2)`` op de gescoorde bars. 1,0 is zuiver.

    De gekwadrateerde return is de referentie en niet omgekeerd, omdat hij per
    constructie zuiver is voor de grootheid die de modellen voorspellen:
    ``E[r_t^2 | F_{t-1}] = sigma_t^2``. Hij is de RUISIGSTE proxy die er is --
    zijn relatieve efficiëntie is per definitie 1 -- maar ruis maakt een
    schatter niet scheef, en het is de scheefheid die QLIKE's rangorde breekt.

    Alleen de bars uit `mask` tellen: de competitie wordt daar beslecht, en een
    proxy die op de trainbars wel klopt en op de testbars niet, zou anders door
    de controle glippen.
    """
    proxy = np.asarray(proxy, dtype=np.float64)
    squared_return = np.asarray(squared_return, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    usable = mask & np.isfinite(proxy) & np.isfinite(squared_return)
    require(
        int(np.count_nonzero(usable)) > 0,
        "Geen enkele bar waarop proxy en gekwadrateerde return allebei "
        "bestaan; de zuiverheid van de proxy is dan niet te meten.",
        DataContractError, n_mask=int(mask.sum()),
    )
    reference = float(np.mean(squared_return[usable]))
    require(
        reference > 0.0,
        "De gemiddelde gekwadrateerde return is nul; er valt niets tegen af "
        "te zetten.",
        DataContractError,
    )
    return float(np.mean(proxy[usable])) / reference


def shuffled_forecast(values: np.ndarray, *, seed: int) -> np.ndarray:
    """Dezelfde forecastwaarden, andere volgorde. NaN's blijven waar ze staan.

    Een geschudde forecast heeft exact dezelfde marginale verdeling als het
    origineel en heeft alleen zijn TIMING verloren. Wat een competitie tussen
    beide meet, is dus uitsluitend of het model weet WANNEER de variantie hoog
    is -- en dat is precies de eigenschap waarop de competitie zou moeten
    beslissen. De NaN-posities blijven ongemoeid, anders zou de geschudde
    variant ook nog een andere steekproef krijgen dan het origineel.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = values.copy()
    rng = np.random.default_rng(seed)
    out[finite] = rng.permutation(values[finite])
    return out


# --------------------------------------------------------------------------- #
# 5. Het oordeel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class H1Verdict:
    """Het oordeel over één uitdager, met de criteria die eraan bonden."""

    status: VerdictStatus
    #: De namen van de stop-criteria die BINDEN, in de volgorde waarin ze zijn
    #: getoetst. Leeg betekent: geen enkel criterium stond promotie in de weg.
    binding: tuple[str, ...]
    rationale: str
    arch_p_value: float
    convergence_ratio: float
    boundary_ratio: float
    dm: DieboldMarianoResult | None
    power: PowerAnalysis | None
    #: `None` wanneer de premisse niet is gemeten (bijvoorbeeld bij een
    #: gesloten ARCH-poort, waar niets is gefit).
    proxy_scale_ratio: float | None = None
    #: De HOOGSTE forecast op de gescoorde bars, gedeeld door de gemiddelde
    #: gekwadrateerde return. Een maat voor of het model daar nog een getal
    #: produceerde in plaats van een numeriek artefact.
    forecast_level_ratio: float | None = None
    #: Of de negatieve controle op deze reeks en horizon slaagde. `False`
    #: betekent dat de toets een forecast met vernietigde timing niet van het
    #: origineel kan onderscheiden, en dan draagt een niet-significante uitslag
    #: geen falsificatie.
    power_control_passed: bool | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "binding": list(self.binding),
            "rationale": self.rationale,
            "arch_p_value": self.arch_p_value,
            "convergence_ratio": self.convergence_ratio,
            "boundary_ratio": self.boundary_ratio,
            "dm_p_value": self.dm.p_value if self.dm else float("nan"),
            "mean_loss_differential": (
                self.dm.mean_loss_differential if self.dm else float("nan")),
            "loss_differential_ar1": (
                self.dm.loss_differential_ar1 if self.dm else float("nan")),
            "mde_sd_units": (
                self.power.minimum_detectable_effect if self.power
                else float("nan")),
            "expected_effect_sd_units": EXPECTED_EFFECT_SD,
            "power_deficit": (
                self.power.power_deficit if self.power else float("nan")),
            "proxy_scale_ratio": (
                self.proxy_scale_ratio if self.proxy_scale_ratio is not None
                else float("nan")),
            "forecast_level_ratio": (
                self.forecast_level_ratio
                if self.forecast_level_ratio is not None else float("nan")),
            "power_control_passed": self.power_control_passed,
            "dm": self.dm.as_record() if self.dm else None,
        }


def judge_challenger(
    *,
    arch_p_value: float,
    convergence_ratio: float,
    boundary_ratio: float,
    dm: DieboldMarianoResult | None,
    proxy_scale_ratio: float | None = None,
    forecast_level_ratio: float | None = None,
    power_control_passed: bool | None = None,
    effective_series: float = 1.0,
    expected_effect_sd: float = EXPECTED_EFFECT_SD,
) -> H1Verdict:
    """Pas de vier stop-criteria toe, in de volgorde van de pre-registratie.

    De volgorde is geen implementatiedetail. De ARCH-poort gaat eerst omdat een
    gesloten poort betekent dat een GARCH-structuur op die reeks niet
    gerechtvaardigd IS -- er valt dan niets te falsifiëren, en een `FALSIFIED`
    zou het model iets aanrekenen dat de data niet vroeg. De convergentiepoort
    gaat daarna, want een QLIKE-reeks uit de helft van de vensters is een
    selectie van de gemakkelijke vensters en niet vergelijkbaar. Pas als beide
    open staan, mag de DM-uitslag iets betekenen.

    En dan nog beslist die uitslag niet alleen: een niet-significante uitkomst
    is `FALSIFIED` wanneer de toets het verwachte effect KON zien, en
    `UNPROVEN` wanneer niet. Dat onderscheid is het verschil tussen "getoetst en
    niet gevonden" en "niet kunnen toetsen".
    """
    if arch_p_value >= _ECONO.alpha:
        return H1Verdict(
            status="DESCOPED", binding=(_ARCH_GATE,),
            rationale=(
                f"De Engle ARCH-toets vindt geen conditionele "
                f"heteroskedasticiteit (p = {arch_p_value:.4f} >= "
                f"{_ECONO.alpha}). Een GARCH-structuur is op deze reeks niet "
                f"gerechtvaardigd; dat is de-scopen en geen falsificatie."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm, power=None,
        )

    garch_cfg = _ADEQUACY.garch
    if (convergence_ratio < garch_cfg.min_convergence_ratio
            or boundary_ratio > garch_cfg.max_boundary_solution_ratio):
        return H1Verdict(
            status="DESCOPED", binding=(_CONVERGENCE,),
            rationale=(
                f"Convergentie {convergence_ratio:.1%} (eis "
                f"{garch_cfg.min_convergence_ratio:.0%}) en randoplossingen "
                f"{boundary_ratio:.1%} (max "
                f"{garch_cfg.max_boundary_solution_ratio:.0%}). De QLIKE-reeks "
                f"is een selectie van de vensters die deze variant aankon."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm, power=None,
        )

    if forecast_level_ratio is not None and (
            forecast_level_ratio > _ADEQUACY.garch.max_forecast_level_ratio):
        return H1Verdict(
            status="DESCOPED", binding=(_DEGENERATE_FORECAST,),
            rationale=(
                f"De hoogste variantieforecast is {forecast_level_ratio:.3g}× "
                f"de gemiddelde gekwadrateerde return, tegen een grens van "
                f"{_ADEQUACY.garch.max_forecast_level_ratio:.0f}×. Dat is geen "
                f"forecast meer: de gesimuleerde meerstaps-verwachting van een "
                f"model waarvan de recursie in ln(sigma^2) loopt, schat een "
                f"moment dat niet hoeft te bestaan. Er valt niets te "
                f"vergelijken, dus ook niets te falsifiëren."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm, power=None,
            proxy_scale_ratio=proxy_scale_ratio,
            forecast_level_ratio=forecast_level_ratio,
            power_control_passed=power_control_passed,
        )

    if proxy_scale_ratio is not None and abs(proxy_scale_ratio - 1.0) > (
            _ADEQUACY.proxy.max_scale_deviation):
        return H1Verdict(
            status="UNPROVEN", binding=(_PROXY_PREMISE,),
            rationale=(
                f"De primaire proxy meet gemiddeld {proxy_scale_ratio:.2f}× de "
                f"gekwadrateerde return, tegen een toegestane afwijking van "
                f"{_ADEQUACY.proxy.max_scale_deviation:.0%}. QLIKE heeft zijn "
                f"minimum op E[proxy]; bij een verschoven niveau rangschikt hij "
                f"op kalibratie tegen dat verschoven doel en niet op "
                f"voorspelkwaliteit. Er valt op deze meetlat niets te "
                f"promoveren en evenmin iets te falsifiëren."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm,
            power=(
                power_for_differential(
                    n_obs=dm.n_obs, ar1=dm.loss_differential_ar1,
                    effective_series=effective_series,
                    expected_effect_sd=expected_effect_sd)
                if dm is not None else None),
            proxy_scale_ratio=proxy_scale_ratio,
            forecast_level_ratio=forecast_level_ratio,
            power_control_passed=power_control_passed,
        )

    if dm is None:
        return H1Verdict(
            status="UNPROVEN", binding=(_NO_IMPROVEMENT, _UNDERPOWERED),
            rationale=(
                "Er is geen vergelijking tot stand gekomen: de twee "
                "verliesreeksen deelden te weinig bars. Geen toets is geen "
                "bewijs van afwezigheid."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=None, power=None,
            proxy_scale_ratio=proxy_scale_ratio,
            forecast_level_ratio=forecast_level_ratio,
            power_control_passed=power_control_passed,
        )

    power = power_for_differential(
        n_obs=dm.n_obs, ar1=dm.loss_differential_ar1,
        effective_series=effective_series, expected_effect_sd=expected_effect_sd,
    )
    challenger_better = dm.mean_loss_differential < 0.0

    if dm.significant and challenger_better:
        return H1Verdict(
            status="PROMOTED", binding=(),
            rationale=(
                f"Lagere QLIKE dan EWMA(0.94) met een gemiddeld "
                f"verliesverschil van {dm.mean_loss_differential:.6f} en "
                f"DM-HLN p = {dm.p_value:.4g}. Geen enkel stop-criterium "
                f"bindt."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm, power=power,
            proxy_scale_ratio=proxy_scale_ratio,
            forecast_level_ratio=forecast_level_ratio,
            power_control_passed=power_control_passed,
        )

    if dm.significant:
        return H1Verdict(
            status="FALSIFIED", binding=(_NO_IMPROVEMENT,),
            rationale=(
                f"De toets verwerpt (p = {dm.p_value:.4g}), maar in het "
                f"NADEEL van de uitdager: zijn gemiddelde QLIKE ligt "
                f"{dm.mean_loss_differential:.6f} HOGER dan die van "
                f"EWMA(0.94). Een tweezijdige verwerping is geen bewijs van "
                f"superioriteit."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm, power=power,
            proxy_scale_ratio=proxy_scale_ratio,
            forecast_level_ratio=forecast_level_ratio,
            power_control_passed=power_control_passed,
        )

    # De GEMETEN power slaat de BEREKENDE. `power.informative` leidt af uit het
    # aantal observaties en de AR(1); de negatieve controle meet of deze toets
    # een forecast waarvan de timing volledig is vernietigd, uberhaupt kan
    # onderscheiden van het origineel. Spreken die twee elkaar tegen, dan wint
    # de meting: een toets die dat verschil niet ziet, ziet het veel kleinere
    # GARCH-EWMA-verschil zeker niet.
    if not power.informative or power_control_passed is False:
        return H1Verdict(
            status="UNPROVEN", binding=(_NO_IMPROVEMENT, _UNDERPOWERED),
            rationale=(
                f"Niet significant (p = {dm.p_value:.4g}), en deze opzet kon "
                f"het verwachte effect ook niet zien: het minimaal "
                f"detecteerbare effect is {power.minimum_detectable_effect:.3f} "
                f"SD tegen een verwacht effect van {expected_effect_sd:.2f} SD "
                f"(gemeten AR(1) = {dm.loss_differential_ar1:.3f})"
                + ("" if power_control_passed is not False else
                   ", en de negatieve controle laat zien dat deze toets zelfs "
                   "een forecast met vernietigde timing niet onderscheidt")
                + ". Afwezigheid van bewijs is geen bewijs van afwezigheid."),
            arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
            boundary_ratio=boundary_ratio, dm=dm, power=power,
            proxy_scale_ratio=proxy_scale_ratio,
            forecast_level_ratio=forecast_level_ratio,
            power_control_passed=power_control_passed,
        )

    return H1Verdict(
        status="FALSIFIED", binding=(_NO_IMPROVEMENT,),
        rationale=(
            f"Niet significant (p = {dm.p_value:.4g}) terwijl de toets het "
            f"verwachte effect van {expected_effect_sd:.2f} SD wel kon zien "
            f"(MDE = {power.minimum_detectable_effect:.3f} SD, gemeten AR(1) = "
            f"{dm.loss_differential_ar1:.3f}). EWMA(0.94) blijft de "
            f"productie-estimator."),
        arch_p_value=arch_p_value, convergence_ratio=convergence_ratio,
        boundary_ratio=boundary_ratio, dm=dm, power=power,
        proxy_scale_ratio=proxy_scale_ratio,
        forecast_level_ratio=forecast_level_ratio,
        power_control_passed=power_control_passed,
    )


# --------------------------------------------------------------------------- #
# 6. De verliesreeksen
# --------------------------------------------------------------------------- #
def build_losses(
    rv: np.ndarray,
    forecast: np.ndarray,
    *,
    model: str,
    mask: np.ndarray,
    losses: Sequence[str] = COMPETITION_LOSSES,
) -> dict[str, LossSeries]:
    """Verliesreeksen op de out-of-sample bars, met de volle lengte behouden.

    De MASKER wordt op de PROXY gelegd en niet op de forecast. Dat is geen
    detail van de boekhouding: `LossSeries` splitst uit waarom een bar wegviel,
    en dat onderscheid moet blijven kloppen. Een bar buiten de testvensters is
    voor ELK model gelijk uitgesloten -- dat is een eigenschap van de opzet,
    net als een onbruikbare proxy -- terwijl `n_dropped_forecast` de bars telt
    waarop de proxy er wel was maar het model geen waarde gaf. Dat laatste getal
    is precies wat een lage convergentieratio zichtbaar maakt, en het zou
    onleesbaar worden als de uitgesloten bars erin werden opgeteld.

    De lengte blijft die van de volledige reeks, zodat twee modellen positioneel
    vergelijkbaar blijven en de DM-toets zijn eigen doorsnede kan bepalen.
    """
    rv = np.asarray(rv, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(
        rv.shape == forecast.shape == mask.shape,
        "Proxy, forecast en out-of-sample masker hebben verschillende lengtes.",
        DataContractError,
        n_rv=int(rv.size), n_forecast=int(forecast.size), n_mask=int(mask.size),
    )
    scored = np.where(mask, rv, np.nan)
    return {
        name: loss_series(scored, forecast, loss=name, model=model)
        for name in losses
    }


# --------------------------------------------------------------------------- #
# 7. De negatieve controle op de toets zelf
# --------------------------------------------------------------------------- #
def swap_control_rejection_rate(
    loss_a: LossSeries,
    loss_b: LossSeries,
    *,
    horizon: int,
    n_replicates: int,
    seed: int,
) -> float:
    """Hoe vaak verwerpt DM-HLN wanneer er per constructie NIETS te vinden is?

    Per replicatie wordt per bar met kans een half gewisseld welke van de twee
    verliezen bij welk model hoort. Het verwachte verliesverschil is daarmee
    exact nul, terwijl schaal, staarten en seriële structuur van de echte data
    behouden blijven -- het is dezelfde steekproef, alleen zonder het verschil.

    Dit is de controle die stap 7 eist: *"een toets die ook op ruis significant
    is, meet niets"*. De uitkomst hoort rond `alpha` te liggen. Ligt zij er ver
    boven, dan is elke significante uitslag in dit rapport verdacht -- ook de
    uitslagen die de goede kant op wijzen.
    """
    require(
        n_replicates > 0,
        "Een negatieve controle zonder replicaties.",
        DataContractError, n_replicates=n_replicates,
    )
    a = loss_a.values
    b = loss_b.values
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_replicates):
        swap = rng.random(a.size) < 0.5
        result = diebold_mariano_hln(
            replace(loss_a, values=np.where(swap, b, a)),
            replace(loss_b, values=np.where(swap, a, b)),
            horizon=horizon,
        )
        rejections += int(result.significant)
    return rejections / n_replicates


# --------------------------------------------------------------------------- #
# 9. De twee negatieve controles samen
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NegativeControls:
    """Kan deze toets iets zien, en ziet hij niet te veel?

    Stap 7 vraagt om een negatieve controle op de TOETS, niet op de markt. Zij
    heeft twee kanten, en één ervan alleen is misleidend:

    POWER -- `shuffle_dm`. EWMA tegen een geschudde variant van zichzelf. De
    geschudde reeks heeft dezelfde waarden en alleen zijn timing verloren, en
    dat is precies wat een volatiliteitsmodel te bieden heeft. Ziet DM-HLN dat
    verschil NIET, dan kan hij ook het veel kleinere GARCH-EWMA-verschil niet
    zien en betekent geen enkele niet-significante uitslag in dit rapport iets.

    SIZE -- `swap_rejection_rate`. Per bar wisselen welke van twee echte
    verliesreeksen bij welk model hoort, zodat het verwachte verschil nul is
    terwijl alle andere eigenschappen blijven staan. Verwerpt de toets daar veel
    vaker dan `alpha`, dan verwerpt hij op ruis en is elke significante uitslag
    verdacht.

    Een controle die alleen de eerste kant meet, keurt een toets goed die altijd
    verwerpt. Een die alleen de tweede meet, keurt een toets goed die nooit
    verwerpt. Daarom staan ze hier samen, en eist :attr:`passed` ze allebei.
    """

    model: str
    horizon: int
    n_replicates: int
    #: EWMA tegen zijn geschudde variant. Hoort SIGNIFICANT te zijn.
    shuffle_dm: DieboldMarianoResult
    #: Verwerpingsfractie onder een per constructie nul verschil. Hoort rond
    #: `alpha` te liggen.
    swap_rejection_rate: float

    @property
    def size_ceiling(self) -> float:
        """Hoeveel verwerping op ruis nog acceptabel is: tweemaal `alpha`.

        Met 200 replicaties heeft de gemeten fractie zelf een standaardfout van
        ongeveer 1,5 procentpunt; een grens exact op alpha zou dus op eigen ruis
        afgaan. Tweemaal alpha laat die onzekerheid toe en blijft ver onder de
        orde van grootte waarop een toets structureel te vaak verwerpt.
        """
        return 2.0 * _ECONO.alpha

    @property
    def passed(self) -> bool:
        return self.shuffle_dm.significant and (
            self.swap_rejection_rate <= self.size_ceiling)

    def as_record(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "horizon": self.horizon,
            "n_replicates": self.n_replicates,
            "shuffle_control": self.shuffle_dm.as_record(),
            "swap_rejection_rate": self.swap_rejection_rate,
            "size_ceiling": self.size_ceiling,
            "alpha": _ECONO.alpha,
            "passed": self.passed,
        }


def run_negative_controls(
    *,
    rv: np.ndarray,
    forecast: np.ndarray,
    mask: np.ndarray,
    horizon: int,
    seed: int,
    n_replicates: int,
    model: str,
) -> NegativeControls:
    """Draai beide controles op de verliesreeks van de titelverdediger.

    Op EWMA en niet op een GARCH-variant, en dat is opzet: de controle moet
    onafhankelijk zijn van de uitkomst van de competitie. Zou zij op de winnaar
    draaien, dan hangt haar oordeel af van welk model won.
    """
    real = build_losses(
        rv, forecast, model=model, mask=mask, losses=("qlike",))["qlike"]
    shuffled = build_losses(
        rv, shuffled_forecast(forecast, seed=seed),
        model=f"{model}_geschud", mask=mask, losses=("qlike",))["qlike"]
    return NegativeControls(
        model=model, horizon=horizon, n_replicates=n_replicates,
        shuffle_dm=diebold_mariano_hln(real, shuffled, horizon=horizon),
        swap_rejection_rate=swap_control_rejection_rate(
            real, shuffled, horizon=horizon, n_replicates=n_replicates,
            seed=seed),
    )
