"""H1 — de campagne: elke variant op elk symbool op elke horizon. Stap 7.

Gescheiden van `vol_competition.py` om twee redenen. De praktische: samen zaten
zij op 836 regels, tegen een limiet van 800 (`docs/architecture.md` R-4). De
inhoudelijke, die er al lag: `vol_competition.py` bevat de PRIMITIEVEN en het
OORDEEL — allemaal pure functies op arrays, zonder kennis van folds of van de
GARCH-familie. Dit bestand roept ze aan in de volgorde die de pre-registratie
voorschrijft.

DE POORT STAAT VOOR DE FIT
===========================
Is de ARCH-poort op een reeks dicht, dan wordt er op die reeks NIETS gefit.
Niet omdat het duur zou zijn, maar omdat er anders een QLIKE-getal ontstaat op
een reeks waarop een GARCH-structuur niet gerechtvaardigd is — en zo'n getal
gaat vroeg of laat los van zijn poortoordeel reizen. `_descoped_by_arch`
produceert daarom een oordeel zonder convergentiesamenvatting en zonder
verliesreeksen: er valt niets te citeren.

WAT ÉÉN TRIAL IS
=================
Eén trial is één (variant, symbool, horizon). Vier varianten × zes symbolen ×
twee horizonnen = 48, exact het `planned_trials` van de bevroren
pre-registratie `cef1a3b9a6811d7bde1afc92a2a9503f`. EWMA(0.94) telt niet mee —
er wordt geen ruimte doorzocht, lambda komt uit `conf/model/volatility.yaml` —
en de negatieve controles evenmin: die toetsen de TOETS en niet de markt.

Ref: `Prompts-fases/fase_6_advanced_research.md` stap 7 en §0.9 (de ledger).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..cv.walk_forward import WalkForwardCV
from ..schemas.config import (
    AdequacyConfig,
    adequacy_config,
    econometrics_config,
    volatility_config,
)
from ..utils.failfast import DataContractError, require
from ..volatility.garch import (
    GARCH_FAMILY,
    ConvergenceSummary,
    summarise_convergence,
    walk_forward_variance_forecasts,
)
from ..volatility.realized import PROXY_EFFICIENCY, build_range_proxies
from .vol_competition import (
    H1Verdict,
    NegativeControls,
    build_losses,
    ewma_variance_forecast,
    judge_challenger,
    oos_mask,
    proxy_scale_ratio,
    run_negative_controls,
)
from .vol_metrics import (
    DieboldMarianoResult,
    LossSeries,
    MincerZarnowitzResult,
    diebold_mariano_hln,
    mincer_zarnowitz,
)

__all__ = [
    "REFERENCE_PROXY",
    "CampaignResult",
    "ChallengerOutcome",
    "build_proxy_panel",
    "run_campaign",
    "run_symbol_competition",
]

_ADEQUACY = adequacy_config()
_ECONO = econometrics_config()
_VOL = volatility_config()

#: De proxy waartegen de zuiverheid van de PRIMAIRE proxy wordt gemeten. De
#: gekwadrateerde return is de ruisigste proxy die er is, en tegelijk de enige
#: die per constructie zuiver is voor precies de grootheid die de modellen
#: voorspellen: ``E[r_t^2 | F_{t-1}] = sigma_t^2``. Ruis maakt een schatter niet
#: scheef, en het is de scheefheid die de rangorde van QLIKE breekt.
REFERENCE_PROXY = "squared_return"


# --------------------------------------------------------------------------- #
# 8. De campagne per symbool
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChallengerOutcome:
    """Een uitdager op een symbool en een horizon, met alles wat hem beoordeelt."""

    symbol: str
    spec: str
    horizon: int
    n_oos_bars: int
    #: `mean(primaire proxy) / mean(r^2)` op de gescoorde bars. `None` wanneer
    #: er niets is gefit en er dus geen meetlat in gebruik was.
    proxy_scale: float | None
    #: `mean(forecast) / mean(r^2)` van de UITDAGER, op dezelfde bars. QLIKE
    #: straft een verkeerd niveau even hard als een verkeerde dynamiek; zonder
    #: dit getal is niet te zien of een winst uit de dynamiek komt of uit een
    #: niveau dat toevallig bij een verschoven proxy past.
    forecast_scale: float
    #: Idem voor EWMA(0.94), de titelverdediger.
    baseline_forecast_scale: float
    #: De HOOGSTE forecast op de gescoorde bars, op dezelfde schaal. Een
    #: gemiddelde verbergt een handvol geëxplodeerde bars; de piek is waar een
    #: forecast ophoudt een forecast te zijn.
    forecast_level_ratio: float
    #: `None` wanneer de ARCH-poort dicht stond: er is dan niets gefit.
    convergence: ConvergenceSummary | None
    losses: Mapping[str, LossSeries]
    baseline_losses: Mapping[str, LossSeries]
    dm: DieboldMarianoResult | None
    mz: MincerZarnowitzResult | None
    baseline_mz: MincerZarnowitzResult | None
    verdict: H1Verdict
    #: Dezelfde vergelijking tegen ANDERE variantieproxies, uit dezelfde fits.
    #: QLIKE is proxy-robuust (Patton 2011) zolang de proxy conditioneel zuiver
    #: is; deze tabel maakt van die stelling een meting op deze data. Zij draagt
    #: bewust GEEN oordeel: het verdict staat op de primaire proxy, zoals
    #: vooraf vastgelegd. Zou een tweede proxy het oordeel mogen kantelen, dan
    #: was de proxykeuze achteraf gemaakt.
    robustness: Mapping[str, DieboldMarianoResult]

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spec": self.spec,
            "horizon": self.horizon,
            "n_oos_bars": self.n_oos_bars,
            "proxy_scale": self.proxy_scale,
            "forecast_scale": self.forecast_scale,
            "baseline_forecast_scale": self.baseline_forecast_scale,
            "forecast_level_ratio": self.forecast_level_ratio,
            "convergence": (
                self.convergence.as_record() if self.convergence else None),
            "losses": {k: v.as_record() for k, v in self.losses.items()},
            "baseline_losses": {
                k: v.as_record() for k, v in self.baseline_losses.items()},
            "mincer_zarnowitz": self.mz.as_record() if self.mz else None,
            "baseline_mincer_zarnowitz": (
                self.baseline_mz.as_record() if self.baseline_mz else None),
            "verdict": self.verdict.as_record(),
            "robustness": {
                name: result.as_record()
                for name, result in self.robustness.items()},
        }


def _descoped_by_arch(
    *, symbol: str, spec: str, horizon: int, n_oos_bars: int,
    arch_p_value: float,
) -> ChallengerOutcome:
    """Een de-scope VOORDAT er is gefit.

    Dat de poort vóór de fit staat en niet erna, is de hele betekenis van een
    poort. Zou er eerst worden gefit, dan bestaat er een QLIKE-uitslag op een
    reeks waarop een GARCH-structuur niet gerechtvaardigd is -- en zo'n getal
    wordt vroeg of laat geciteerd, hoe het ook is gelabeld.
    """
    return ChallengerOutcome(
        symbol=symbol, spec=spec, horizon=horizon, n_oos_bars=n_oos_bars,
        proxy_scale=None, forecast_scale=float("nan"),
        baseline_forecast_scale=float("nan"),
        forecast_level_ratio=float("nan"), convergence=None, losses={},
        baseline_losses={}, dm=None, mz=None, baseline_mz=None, robustness={},
        verdict=judge_challenger(
            arch_p_value=arch_p_value, convergence_ratio=float("nan"),
            boundary_ratio=float("nan"), dm=None),
    )


def run_symbol_competition(
    *,
    returns: pd.Series,
    proxy: pd.Series,
    symbol: str,
    cv: WalkForwardCV,
    horizons: Sequence[int],
    arch_p_value: float,
    reference_proxy: pd.Series | None = None,
    power_control: Mapping[int, bool] | None = None,
    specs: Sequence[str] = tuple(GARCH_FAMILY),
    robustness_proxies: Mapping[str, pd.Series] | None = None,
    adequacy: AdequacyConfig | None = None,
    lam: float | None = None,
    burn_in_bars: int | None = None,
) -> list[ChallengerOutcome]:
    """De QLIKE-competitie op één reeks: elke variant tegen EWMA(0.94).

    Beide modellen worden op DEZELFDE bars gescoord -- de testvensters van de
    walk-forward -- en beide forecasts staan op de bar die zij voorspellen. Wat
    daarna verschilt, verschilt dus door het model.

    De ARCH-poort wordt eerst gelezen. Staat hij dicht, dan wordt er op deze
    reeks NIETS gefit en dragen alle varianten de de-scope. Er ontstaat geen
    QLIKE-getal dat later los van zijn poortoordeel kan gaan reizen.
    """
    require(
        len(returns) == len(proxy),
        "Returns en proxy hebben verschillende lengtes.",
        DataContractError, symbol=symbol,
        n_returns=int(len(returns)), n_proxy=int(len(proxy)),
    )
    require(
        len(horizons) > 0 and len(specs) > 0,
        "Een competitie zonder horizonnen of zonder varianten.",
        DataContractError, symbol=symbol,
        n_horizons=len(horizons), n_specs=len(specs),
    )
    cfg = adequacy if adequacy is not None else _ADEQUACY
    decay = lam if lam is not None else _VOL.ewma_lambda
    burn_in = burn_in_bars if burn_in_bars is not None else _VOL.burn_in_bars

    mask = oos_mask(len(returns), cv)
    n_oos = int(mask.sum())
    rv = proxy.to_numpy(dtype=np.float64)
    # De zuiverheid van de meetlat wordt EENMAAL per reeks gemeten en daarna
    # aan elk oordeel meegegeven. Zij hangt niet van de variant of de horizon
    # af: het is een eigenschap van de proxy en van de data, niet van het model
    # dat erop wordt afgerekend.
    reference = (
        None if reference_proxy is None
        else reference_proxy.to_numpy(dtype=np.float64)
    )
    scale = (
        None if reference is None
        else proxy_scale_ratio(rv, reference, mask=mask)
    )

    outcomes: list[ChallengerOutcome] = []
    for spec_name in specs:
        require(
            spec_name in GARCH_FAMILY,
            f"Onbekende GARCH-variant '{spec_name}'. De familie ligt vast in "
            "de pre-registratie; een vijfde variant vereist een nieuwe.",
            DataContractError, known=sorted(GARCH_FAMILY),
        )
        for horizon in horizons:
            if arch_p_value >= _ECONO.alpha:
                outcomes.append(_descoped_by_arch(
                    symbol=symbol, spec=GARCH_FAMILY[spec_name].label,
                    horizon=horizon, n_oos_bars=n_oos,
                    arch_p_value=arch_p_value))
                continue
            outcomes.append(_run_one(
                returns=returns, rv=rv, mask=mask, symbol=symbol,
                spec_name=spec_name, horizon=horizon, cv=cv, cfg=cfg,
                decay=decay, burn_in=burn_in, arch_p_value=arch_p_value,
                n_oos=n_oos, proxy_scale=scale, reference=reference,
                power_control_passed=(
                    None if power_control is None
                    else power_control.get(horizon)),
                robustness_proxies=robustness_proxies or {}))
    return outcomes


def _peak_level(
    forecast: np.ndarray, reference: np.ndarray, mask: np.ndarray,
) -> float:
    """`max(forecast) / mean(r^2)`. De piek, niet het gemiddelde.

    Op deze grootheid staat de poort `degenerate_forecast_level`: een
    gemiddelde verbergt een handvol geexplodeerde bars, en QLIKE verbergt ze
    bijna ook -- hij groeit slechts logaritmisch in een overschatting.
    """
    usable = mask & np.isfinite(forecast) & np.isfinite(reference)
    if not usable.any():
        return float("nan")
    denominator = float(np.mean(reference[usable]))
    if denominator <= 0.0:
        return float("nan")
    return float(np.max(forecast[usable])) / denominator


def _level(forecast: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """`mean(forecast) / mean(r^2)` op de bars waar beide bestaan.

    Dezelfde noemer als :func:`proxy_scale_ratio`, zodat het niveau van een
    model en het niveau van de proxy op dezelfde schaal staan en naast elkaar
    te lezen zijn.
    """
    usable = mask & np.isfinite(forecast) & np.isfinite(reference)
    if not usable.any():
        return float("nan")
    denominator = float(np.mean(reference[usable]))
    if denominator <= 0.0:
        return float("nan")
    return float(np.mean(forecast[usable])) / denominator


def _run_one(
    *,
    returns: pd.Series,
    rv: np.ndarray,
    mask: np.ndarray,
    symbol: str,
    spec_name: str,
    horizon: int,
    cv: WalkForwardCV,
    cfg: AdequacyConfig,
    decay: float,
    burn_in: int,
    arch_p_value: float,
    n_oos: int,
    proxy_scale: float | None,
    reference: np.ndarray | None,
    power_control_passed: bool | None,
    robustness_proxies: Mapping[str, pd.Series],
) -> ChallengerOutcome:
    """Eén variant op één horizon. Fit, scoor, vergelijk, oordeel."""
    spec = GARCH_FAMILY[spec_name]
    forecasts, fits = walk_forward_variance_forecasts(
        returns, spec, cv, cfg, symbol=symbol, horizon=horizon)
    convergence = summarise_convergence(fits)

    baseline = ewma_variance_forecast(
        returns, lam=decay, burn_in_bars=burn_in, horizon=horizon)
    challenger_losses = build_losses(
        rv, forecasts.to_numpy(dtype=np.float64),
        model=spec.label, mask=mask)
    baseline_losses = build_losses(
        rv, baseline.to_numpy(dtype=np.float64),
        model=f"ewma_{decay}", mask=mask)

    # De vergelijking wordt alleen gemaakt wanneer zij eerlijk KAN. Bij een te
    # lage convergentie is de QLIKE-reeks van de uitdager een selectie van de
    # vensters die hij aankon; `diebold_mariano_hln` zou dat op de doorsnede
    # zelf ook weigeren, maar dan als crash halverwege een campagne in plaats
    # van als een oordeel dat in het rapport staat.
    comparable = convergence.comparable(cfg)
    dm = mz = baseline_mz = None
    robustness: dict[str, DieboldMarianoResult] = {}
    if comparable:
        dm = diebold_mariano_hln(
            challenger_losses["qlike"], baseline_losses["qlike"],
            horizon=horizon)
        scored_rv = np.where(mask, rv, np.nan)
        mz = mincer_zarnowitz(
            scored_rv, forecasts.to_numpy(dtype=np.float64), model=spec.label)
        baseline_mz = mincer_zarnowitz(
            scored_rv, baseline.to_numpy(dtype=np.float64),
            model=f"ewma_{decay}")
        # Dezelfde forecasts, andere realisatie. De fits worden NIET herhaald:
        # anders zou de tabel het verschil tussen twee campagnes meten in
        # plaats van tussen twee proxies.
        for name, alternative in robustness_proxies.items():
            alt = np.asarray(alternative, dtype=np.float64)
            robustness[name] = diebold_mariano_hln(
                build_losses(alt, forecasts.to_numpy(dtype=np.float64),
                             model=spec.label, mask=mask,
                             losses=("qlike",))["qlike"],
                build_losses(alt, baseline.to_numpy(dtype=np.float64),
                             model=f"ewma_{decay}", mask=mask,
                             losses=("qlike",))["qlike"],
                horizon=horizon)

    scale_reference = reference if reference is not None else rv
    peak = _peak_level(
        forecasts.to_numpy(dtype=np.float64), scale_reference, mask)
    verdict = judge_challenger(
        arch_p_value=arch_p_value,
        convergence_ratio=convergence.convergence_ratio,
        boundary_ratio=convergence.boundary_ratio,
        dm=dm, proxy_scale_ratio=proxy_scale, forecast_level_ratio=peak,
        power_control_passed=power_control_passed,
    )
    return ChallengerOutcome(
        symbol=symbol, spec=spec.label, horizon=horizon, n_oos_bars=n_oos,
        proxy_scale=proxy_scale,
        forecast_scale=_level(
            forecasts.to_numpy(dtype=np.float64), scale_reference, mask),
        baseline_forecast_scale=_level(
            baseline.to_numpy(dtype=np.float64), scale_reference, mask),
        forecast_level_ratio=peak,
        convergence=convergence, losses=challenger_losses,
        baseline_losses=baseline_losses, dm=dm, mz=mz, baseline_mz=baseline_mz,
        verdict=verdict, robustness=robustness,
    )


# --------------------------------------------------------------------------- #
# De campagne over het hele universum
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CampaignResult:
    """De volledige H1-competitie, met de trial-boekhouding erin.

    `n_planned_trials` is wat de opzet vraagt: varianten × symbolen ×
    horizonnen. `n_trials` is wat er FEITELIJK is gefit. Zij verschillen zodra
    een poort een fit heeft voorkomen, en dan hoort het rapport dat verschil te
    verklaren in plaats van één van beide getallen te noemen.

    Waarom niet gewoon het geplande aantal in `M` boeken? Omdat `M` telt hoeveel
    kansen er zijn geweest om iets te vinden. Een combinatie waarop nooit een
    parameter is geschat, heeft die kans niet gehad — precies de redenering
    waarmee de HAR-RV-trial op nul 5m-dekking buiten `M` blijft.
    """

    outcomes: tuple[ChallengerOutcome, ...]
    controls: Mapping[str, NegativeControls]
    primary_proxy: str
    robustness_proxies: tuple[str, ...]
    n_trials: int
    n_planned_trials: int

    def by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.verdict.status] = (
                counts.get(outcome.verdict.status, 0) + 1)
        return counts

    @property
    def promoted(self) -> tuple[ChallengerOutcome, ...]:
        return tuple(
            o for o in self.outcomes if o.verdict.status == "PROMOTED")

    def as_record(self) -> dict[str, Any]:
        return {
            "primary_proxy": self.primary_proxy,
            "robustness_proxies": list(self.robustness_proxies),
            "n_trials": self.n_trials,
            "n_planned_trials": self.n_planned_trials,
            "status_counts": self.by_status(),
            "n_promoted": len(self.promoted),
            "outcomes": [o.as_record() for o in self.outcomes],
            "controls": {k: v.as_record() for k, v in self.controls.items()},
        }


def run_campaign(
    *,
    returns: pd.DataFrame,
    proxies: Mapping[str, Mapping[str, pd.Series]],
    arch_p_values: Mapping[str, float],
    cv: WalkForwardCV,
    primary_proxy: str,
    horizons: Sequence[int],
    seed: int,
    n_control_replicates: int,
    specs: Sequence[str] = tuple(GARCH_FAMILY),
    adequacy: AdequacyConfig | None = None,
    lam: float | None = None,
    burn_in_bars: int | None = None,
) -> CampaignResult:
    """Draai de competitie over het hele universum en tel de trials eerlijk.

    De primaire proxy wordt MEEGEGEVEN en niet gekozen. Zou deze functie er zelf
    een uitzoeken -- de meest efficiënte, of de gunstigste -- dan zou de meetlat
    van de competitie afhangen van de uitkomst ervan. Elke andere proxy uit
    `proxies` gaat als robuustheidscontrole mee door dezelfde fits.
    """
    decay = lam if lam is not None else _VOL.ewma_lambda
    burn_in = burn_in_bars if burn_in_bars is not None else _VOL.burn_in_bars
    mask = oos_mask(len(returns), cv)

    outcomes: list[ChallengerOutcome] = []
    controls: dict[str, NegativeControls] = {}
    n_trials = 0
    for symbol in returns.columns:
        symbol = str(symbol)
        require(
            symbol in proxies and primary_proxy in proxies[symbol],
            f"Geen proxy '{primary_proxy}' voor {symbol}. De primaire proxy is "
            "de meetlat van deze competitie; er wordt niet stilzwijgend een "
            "andere gekozen.",
            DataContractError, symbol=symbol,
            available=sorted(proxies.get(symbol, {})),
        )
        require(
            symbol in arch_p_values,
            f"Geen ARCH-poortoordeel voor {symbol}. De poort staat VOOR de fit; "
            "zonder oordeel wordt er niet gefit.",
            DataContractError, symbol=symbol,
        )
        require(
            REFERENCE_PROXY in proxies[symbol],
            f"Geen `{REFERENCE_PROXY}` voor {symbol}. Zonder die referentie is "
            "de zuiverheid van de primaire proxy niet te meten, en zou de "
            "campagne een rangorde publiceren waarvan de premisse ongetoetst "
            "blijft.",
            DataContractError, symbol=symbol,
            available=sorted(proxies[symbol]),
        )
        primary = proxies[symbol][primary_proxy]
        extra = {
            name: series for name, series in proxies[symbol].items()
            if name != primary_proxy
        }
        # De controles draaien VOOR het oordeel, want hun uitkomst is er een
        # ingang van: een toets die een forecast met vernietigde timing niet
        # onderscheidt, mag geen falsificatie dragen. Zij draaien per reeks en
        # per horizon; een controle die op het ene symbool slaagt, zegt niets
        # over een toets op het andere.
        passed: dict[int, bool] = {}
        for horizon in horizons:
            control = run_negative_controls(
                rv=primary.to_numpy(dtype=np.float64),
                forecast=ewma_variance_forecast(
                    returns[symbol], lam=decay, burn_in_bars=burn_in,
                    horizon=horizon).to_numpy(dtype=np.float64),
                mask=mask, horizon=horizon, seed=seed,
                n_replicates=n_control_replicates, model=f"ewma_{decay}",
            )
            controls[f"{symbol}|h{horizon}"] = control
            passed[horizon] = control.passed

        symbol_outcomes = run_symbol_competition(
            returns=returns[symbol], proxy=primary, symbol=symbol, cv=cv,
            horizons=horizons, arch_p_value=float(arch_p_values[symbol]),
            reference_proxy=proxies[symbol][REFERENCE_PROXY],
            power_control=passed, specs=specs, robustness_proxies=extra,
            adequacy=adequacy, lam=decay, burn_in_bars=burn_in,
        )
        outcomes.extend(symbol_outcomes)
        n_trials += sum(1 for o in symbol_outcomes if o.convergence is not None)

    return CampaignResult(
        outcomes=tuple(outcomes), controls=controls,
        primary_proxy=primary_proxy,
        robustness_proxies=tuple(sorted({
            name for symbol_proxies in proxies.values()
            for name in symbol_proxies if name != primary_proxy})),
        n_trials=n_trials,
        n_planned_trials=len(specs) * len(returns.columns) * len(horizons),
    )


def build_proxy_panel(
    *,
    ohlc: Mapping[str, pd.DataFrame],
    index: pd.Index,
    names: Sequence[str],
) -> tuple[dict[str, dict[str, pd.Series]], dict[str, dict[str, Any]]]:
    """Bouw de variantieproxies per symbool, plus de tabel voor het rapport.

    De proxies worden op `index` gezet -- de index van de log-returns die de
    modellen voorspellen. Die is één bar korter dan de OHLC-reeks, en dat
    verschil stilzwijgend laten bestaan zou forecast en realisatie een bar uit
    elkaar schuiven: precies de fout die een competitie ongeldig maakt zonder
    dat er iets rood wordt.

    De records tellen over ALLE symbolen. Zou de tabel in het rapport één
    symbool tonen, dan staat er een steekproef in die kleiner is dan de
    steekproef waarop is beslist.
    """
    unknown = sorted(set(names) - set(PROXY_EFFICIENCY))
    require(
        not unknown,
        f"Onbekende proxy {unknown}. De keuze van de proxy is de meetlat van "
        "de competitie; er wordt er niet stilzwijgend een gekozen.",
        DataContractError, known=sorted(PROXY_EFFICIENCY),
    )
    panel: dict[str, dict[str, pd.Series]] = {}
    records: dict[str, dict[str, Any]] = {
        name: {
            "name": name,
            "relative_efficiency_vs_squared_return": PROXY_EFFICIENCY[name],
            "n_symbols": 0, "n_valid_bars": 0, "n_zero_bars": 0,
        }
        for name in names
    }
    for symbol, frame in ohlc.items():
        built = build_range_proxies(frame, name=symbol)
        panel[symbol] = {
            name: built[name].values.reindex(index) for name in names}
        for name in names:
            records[name]["n_symbols"] += 1
            records[name]["n_valid_bars"] += built[name].n_valid_bars
            records[name]["n_zero_bars"] += built[name].n_zero_bars
    return panel, records
