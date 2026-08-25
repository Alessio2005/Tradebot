"""Data Adequacy Gate — Phase 6, §3. Meet vóór de fit of de data de vraag draagt.

WAAROM DEZE POORT BESTAAT
=========================
Het universum is 6 gecertificeerde `ohlcv/1d`-reeksen × 1.743 daily bars,
2021-11-15 t/m 2026-08-23, in één cluster. Op die omvang bestaan er precies twee
manieren om een verkeerd antwoord te geven die er allebei statistisch correct
uitzien:

  * een model **FALSIFICEREN** dat simpelweg te weinig data had. Het verschil
    tussen "bewezen slechter dan zijn baseline" en "niet te onderscheiden van
    zijn baseline" is het hele verschil tussen §6-doctrine en zelfbedrog;
  * een model **PROMOVEREN** dat op tientallen observaties per toestand is
    gefit. Een 2-state HMM waarvan de zeldzame toestand 30 bars per fold beslaat,
    schat zijn variantie met ~26 % relatieve standaardfout en vindt daarom altijd
    wel iets.

De poort werkt daarom als een POORT en niet als een waarschuwing:
:func:`require_adequacy` crasht. Een model dat de eis niet haalt, wordt **niet
gefit**; het krijgt direct het oordeel ``UNPROVEN — insufficient data`` met de
gemeten cijfers erbij, en de trial telt **niet** mee in ``M`` omdat er geen
search heeft plaatsgevonden.

WAT DIT MODULE NIET IS
======================
Dit is geen kwaliteitsscore en geen advies. Elke :class:`AdequacyVerdict` draagt
de gemeten grootheid, de eis waaraan zij is getoetst, en — bij een tekort — wat
er precies te weinig was. Dat is wat een `UNPROVEN`-oordeel onderbouwbaar maakt
met cijfers in plaats van met een vermoeden (exit-criterium 9).

POWER-ANALYSE
=============
De tweede helft van dezelfde vraag. Adequaatheid zegt of het model *fitbaar* is;
power zegt of de TOETS het verwachte effect kan zien. Een toets waarvan het
minimaal detecteerbare effect groter is dan het effect dat je redelijkerwijs
verwacht, is bij voorbaat niet informatief — en dat hoort vóór de run te worden
genoteerd, niet erna (§3, exit-criterium 10).

De cruciale correctie op dit universum is de **design effect** van
:func:`effective_independent_series`. Zes crypto-perpetuals met een gemiddelde
paarsgewijze correlatie rho leveren niet zes onafhankelijke reeksen op maar
``k / (1 + (k-1) * rho)``. Bij rho ~ 0,7 is dat ongeveer 1,3. Wie zijn power
berekent op ``6 * n_bars`` overschat haar met een factor ~2.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md §6, §9.1, §10.1, §12.1, §13.1, §17.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from ..schemas.config import AdequacyConfig, PowerConfig
from ..utils.failfast import DataContractError, require

__all__ = [
    "AdequacyVerdict",
    "INSUFFICIENT_DATA",
    "PowerAnalysis",
    "assess_garch",
    "assess_har_rv",
    "assess_hmm",
    "assess_hrp",
    "assess_meta_labeling",
    "auc_minimum_detectable",
    "effective_independent_series",
    "mean_difference_power",
    "require_adequacy",
    "sharpe_difference_power",
]

#: Het exacte oordeel dat een model krijgt dat de poort niet haalt. Eén string,
#: op één plek, zodat de ledger, het falsificatieregister en de rapporten
#: dezelfde term dragen en een grep hem overal vindt.
INSUFFICIENT_DATA = "UNPROVEN — insufficient data"


@dataclass(frozen=True)
class AdequacyVerdict:
    """Wat er is gemeten, waaraan het is getoetst, en wat er tekortkwam."""

    model_class: str
    adequate: bool
    #: De eis in woorden, zoals hij in `conf/model/adequacy.yaml` staat.
    requirement: str
    #: De gemeten grootheden. Dit is wat een `UNPROVEN`-oordeel onderbouwt.
    measured: Mapping[str, Any]
    #: Leeg wanneer adequaat; anders wat er precies te weinig was.
    shortfall: str = ""

    def __post_init__(self) -> None:
        require(
            bool(self.model_class),
            "AdequacyVerdict zonder modelklasse.",
            DataContractError,
        )
        require(
            self.adequate or bool(self.shortfall),
            "Een niet-adequaat oordeel zonder omschrijving van het tekort. "
            "Dat maakt `UNPROVEN — insufficient data` een vermoeden in plaats "
            "van een meting.",
            DataContractError,
            model_class=self.model_class,
        )
        require(
            not self.adequate or not self.shortfall,
            "Een adequaat oordeel MET een tekort. Eén van beide is fout.",
            DataContractError,
            model_class=self.model_class,
            shortfall=self.shortfall,
        )

    @property
    def verdict(self) -> str:
        """`""` wanneer adequaat, anders het `UNPROVEN`-oordeel."""
        return "" if self.adequate else INSUFFICIENT_DATA

    def as_record(self) -> dict[str, Any]:
        """Platte vorm voor de ledger en de rapporten."""
        return {
            "model_class": self.model_class,
            "adequate": self.adequate,
            "requirement": self.requirement,
            "measured": dict(self.measured),
            "shortfall": self.shortfall,
            "verdict": self.verdict,
        }


def require_adequacy(verdict: AdequacyVerdict) -> None:
    """Crash wanneer het model niet gefit had mogen worden.

    Dit is de poort zelf. Hij staat in elke fit-functie van deze fase, zodat er
    geen pad bestaat waarlangs een model op ontoereikende data wordt geschat —
    ook niet tijdens exploratie, waar zulke fits normaal ontstaan.

    Een aanroeper die het oordeel wil REGISTREREN in plaats van crashen, roept
    deze functie niet aan maar leest `verdict.adequate`. Dat onderscheid is
    bewust: de runner mag een model overslaan, de fit mag hem niet uitvoeren.
    """
    require(
        verdict.adequate,
        f"Data Adequacy Gate: {verdict.model_class} mag NIET worden gefit. "
        f"{verdict.shortfall} Het oordeel is '{INSUFFICIENT_DATA}' en de trial "
        "telt niet mee in M, omdat er geen search heeft plaatsgevonden.",
        DataContractError,
        model_class=verdict.model_class,
        requirement=verdict.requirement,
        **{f"measured_{k}": v for k, v in verdict.measured.items()},
    )


# =========================================================================== #
# Per modelklasse
# =========================================================================== #
def assess_garch(
    n_obs_per_window: Sequence[int],
    cfg: AdequacyConfig,
    *,
    symbol: str = "",
) -> AdequacyVerdict:
    """GARCH-familie: genoeg observaties per fit-venster?

    Parameters
    ----------
    n_obs_per_window : het aantal observaties in elk train-venster van de
        walk-forward-structuur. Niet de totale reekslengte: een GARCH wordt per
        fold gefit en de bindende grootheid is het KLEINSTE venster.

    De convergentieratio en de randoplossingsratio worden hier NIET beoordeeld —
    die zijn pas na de fits bekend en horen in
    :func:`~tradebot.volatility.garch.summarise_convergence`. Deze poort gaat
    over de vraag of de fit überhaupt mag worden geprobeerd.
    """
    windows = np.asarray(list(n_obs_per_window), dtype=np.int64)
    require(
        windows.size > 0,
        "GARCH-adequaatheid zonder vensters: er valt niets te beoordelen.",
        DataContractError,
        symbol=symbol,
    )
    smallest = int(windows.min())
    need = cfg.garch.min_obs_per_fit_window
    ok = smallest >= need
    measured = {
        "n_windows": int(windows.size),
        "min_obs_in_window": smallest,
        "median_obs_in_window": float(np.median(windows)),
        "max_obs_in_window": int(windows.max()),
        "required_min_obs": need,
    }
    return AdequacyVerdict(
        model_class=f"garch:{symbol}" if symbol else "garch",
        adequate=ok,
        requirement=f"elk fit-venster >= {need} observaties",
        measured=measured,
        shortfall="" if ok else (
            f"het kleinste fit-venster telt {smallest} observaties tegen een "
            f"eis van {need}; op minder worden alpha en beta slecht "
            f"geïdentificeerd en loopt beta naar de IGARCH-rand."
        ),
    )


def assess_har_rv(
    n_days_total: int,
    n_days_with_sufficient_bars: int,
    cfg: AdequacyConfig,
    *,
    observed_granularity: str = "",
) -> AdequacyVerdict:
    """HAR-RV: is er genoeg intraday-dekking om realized variance te schatten?

    Dit is de poort die op DIT universum bindt. `docs/DATA_REGISTER.md` en
    `reports/phase1_data_gap.md` leggen vast dat de gecertificeerde store 0 rijen
    1m/5m bevat — een expliciet scope-besluit uit Phase 1, geen omissie. HAR-RV
    op daily bars zou een HAR op een range-proxy zijn en niet op realized
    variance; dat is een ANDER model met een andere signaal-ruisverhouding.
    """
    require(
        n_days_total > 0,
        "HAR-RV-adequaatheid op nul dagen.",
        DataContractError,
    )
    require(
        0 <= n_days_with_sufficient_bars <= n_days_total,
        "Aantal dagen met dekking ligt buiten [0, n_days_total].",
        DataContractError,
        n_days_with_sufficient_bars=n_days_with_sufficient_bars,
        n_days_total=n_days_total,
    )
    pct = 100.0 * n_days_with_sufficient_bars / n_days_total
    need_pct = cfg.har_rv.min_days_with_coverage_pct
    ok = pct >= need_pct
    return AdequacyVerdict(
        model_class="har_rv",
        adequate=ok,
        requirement=(
            f">= {need_pct:.1f} % van de dagen met >= "
            f"{cfg.har_rv.min_bars_per_day} {cfg.har_rv.intraday_granularity}-bars"
        ),
        measured={
            "n_days_total": int(n_days_total),
            "n_days_with_coverage": int(n_days_with_sufficient_bars),
            "coverage_pct": pct,
            "required_coverage_pct": need_pct,
            "required_granularity": cfg.har_rv.intraday_granularity,
            "observed_granularity": observed_granularity or "<geen>",
        },
        shortfall="" if ok else (
            f"{pct:.2f} % van de dagen heeft voldoende "
            f"{cfg.har_rv.intraday_granularity}-dekking tegen een eis van "
            f"{need_pct:.1f} %; realized variance is op deze store niet te "
            f"schatten."
        ),
    )


def assess_hmm(
    n_obs_per_fold: Sequence[int],
    n_states: int,
    cfg: AdequacyConfig,
    *,
    expected_occupancy: Sequence[float] | None = None,
) -> AdequacyVerdict:
    """M2 HMM: genoeg observaties **per toestand per fold**?

    Parameters
    ----------
    expected_occupancy : de verwachte bezettingsfractie per toestand, geschat via
        de **filtered** occupancy op een pilot-fit of via de stationaire
        verdeling van de transitiematrix. ``None`` gebruikt de uniforme aanname
        ``1/k``, die de bezetting van de ZELDZAAMSTE toestand systematisch
        OVERSCHAT — dat is de optimistische kant, en een tekort dat de uniforme
        aanname al vindt, is dus onbetwistbaar.

    De bindende grootheid is de zeldzaamste toestand in de kleinste fold. Een
    regime dat gemiddeld 15 % van de tijd actief is, beslaat in een fold van 500
    bars ~75 observaties; daarop een mean, een variantie en k-1 transitiekansen
    schatten is niet hetzelfde als "het model convergeerde".
    """
    folds = np.asarray(list(n_obs_per_fold), dtype=np.int64)
    require(
        folds.size > 0,
        "HMM-adequaatheid zonder folds.",
        DataContractError,
    )
    require(
        n_states >= 2,
        "Een HMM met minder dan twee toestanden is geen regimemodel.",
        DataContractError,
        n_states=n_states,
    )
    if expected_occupancy is None:
        occupancy = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        occupancy_source = "uniform (1/k) — optimistisch"
    else:
        occupancy = np.asarray(list(expected_occupancy), dtype=np.float64)
        require(
            occupancy.size == n_states,
            "expected_occupancy heeft niet evenveel elementen als toestanden.",
            DataContractError,
            n_states=n_states, n_occupancy=int(occupancy.size),
        )
        require(
            bool(np.all(occupancy > 0.0)) and abs(float(occupancy.sum()) - 1.0) < 1e-6,
            "expected_occupancy is geen kansverdeling.",
            DataContractError,
            occupancy=occupancy.tolist(),
        )
        occupancy_source = "filtered occupancy uit pilot-fit"

    rarest_fraction = float(occupancy.min())
    smallest_fold = int(folds.min())
    rarest_obs = rarest_fraction * smallest_fold

    need_obs = cfg.hmm.min_obs_per_state_per_fold
    need_frac = cfg.hmm.min_state_occupancy_fraction
    ok = rarest_obs >= need_obs and rarest_fraction >= need_frac

    reasons = []
    if rarest_obs < need_obs:
        reasons.append(
            f"de zeldzaamste toestand beslaat naar verwachting {rarest_obs:.1f} "
            f"observaties in de kleinste fold ({smallest_fold} bars) tegen een "
            f"eis van {need_obs}"
        )
    if rarest_fraction < need_frac:
        reasons.append(
            f"haar bezettingsfractie is {rarest_fraction:.3f} tegen een eis van "
            f"{need_frac:.3f}, waarmee zij een uitschieterdetector is en geen regime"
        )
    return AdequacyVerdict(
        model_class=f"hmm_k{n_states}",
        adequate=ok,
        requirement=(
            f">= {need_obs} observaties per toestand per fold én een "
            f"bezettingsfractie >= {need_frac:.2f}"
        ),
        measured={
            "n_states": int(n_states),
            "n_folds": int(folds.size),
            "smallest_fold_bars": smallest_fold,
            "median_fold_bars": float(np.median(folds)),
            "rarest_state_occupancy_fraction": rarest_fraction,
            "rarest_state_obs_in_smallest_fold": rarest_obs,
            "occupancy_source": occupancy_source,
            "required_obs_per_state_per_fold": need_obs,
            "required_occupancy_fraction": need_frac,
        },
        shortfall="" if ok else "; ".join(reasons) + ".",
    )


def assess_meta_labeling(
    n_events_per_fold: Sequence[int],
    positive_ratio_per_fold: Sequence[float],
    cfg: AdequacyConfig,
    *,
    effective_events_per_fold: Sequence[float] | None = None,
) -> AdequacyVerdict:
    """Meta-labeling: genoeg triple-barrier events per fold ná purging én balans?

    Drie eisen die onafhankelijk binden.

    **Nominaal aantal.** Tweehonderd events per fold is de ondergrens waaronder
    een AUC een betrouwbaarheidsinterval krijgt dat de drempel 0,58 omvat.

    **Effectief aantal.** Dit is de eis die er op DEZE opzet echt toe doet. Met
    een event op vrijwel elke bar en een verticale barrière op 10 bars overlapt
    elk label met ongeveer tien buren: het label van bar `t` en dat van bar
    `t+1` delen negen van hun tien toekomstige bars. Het nominale aantal
    overschat de informatie-inhoud dan met ongeveer die factor. De gemiddelde
    uniqueness uit AFML hoofdstuk 4 corrigeert daarvoor; de som ervan is het
    effectieve aantal onafhankelijke labels. Zonder deze eis is "2.900 events
    per fold" een geruststelling die de data niet draagt.

    **Klassebalans.** Duizend events met 96 % één klasse leveren een AUC die
    boven 0,58 kán uitkomen terwijl het model niets nuttigs leert.
    """
    events = np.asarray(list(n_events_per_fold), dtype=np.int64)
    ratios = np.asarray(list(positive_ratio_per_fold), dtype=np.float64)
    require(
        events.size > 0,
        "Meta-label-adequaatheid zonder folds.",
        DataContractError,
    )
    require(
        events.size == ratios.size,
        "Aantal folds in events en klassebalans komt niet overeen.",
        DataContractError,
        n_event_folds=int(events.size), n_ratio_folds=int(ratios.size),
    )
    smallest = int(events.min())
    need = cfg.meta_labeling.min_events_per_fold
    need_eff = cfg.meta_labeling.min_effective_events_per_fold
    lo, hi = (cfg.meta_labeling.min_positive_class_ratio,
              cfg.meta_labeling.max_positive_class_ratio)
    out_of_band = ratios[(ratios < lo) | (ratios > hi)]

    if effective_events_per_fold is None:
        effective = np.array([], dtype=np.float64)
        smallest_effective = float("nan")
        effective_ok = False
        effective_note = (
            "de effectieve steekproefgrootte is NIET gemeten; zonder haar is "
            "het nominale aantal events geen bewijs van informatie-inhoud"
        )
    else:
        effective = np.asarray(list(effective_events_per_fold), dtype=np.float64)
        require(
            effective.size == events.size,
            "Aantal folds in effectieve en nominale events komt niet overeen.",
            DataContractError,
            n_effective=int(effective.size), n_nominal=int(events.size),
        )
        smallest_effective = float(effective.min())
        effective_ok = smallest_effective >= need_eff
        effective_note = (
            f"de kleinste fold houdt {smallest_effective:.1f} EFFECTIEVE labels "
            f"over na uniqueness-correctie tegen een eis van {need_eff}"
        )

    ok = (smallest >= need and effective_ok and out_of_band.size == 0)
    reasons = []
    if smallest < need:
        reasons.append(
            f"de kleinste fold houdt {smallest} events over na purging en "
            f"embargo tegen een eis van {need}"
        )
    if not effective_ok:
        reasons.append(effective_note)
    if out_of_band.size:
        reasons.append(
            f"{out_of_band.size} van de {ratios.size} folds heeft een "
            f"positieve-klasseratio buiten [{lo:.2f}, {hi:.2f}] "
            f"(uiterste: {float(out_of_band[np.argmax(np.abs(out_of_band - 0.5))]):.3f})"
        )
    return AdequacyVerdict(
        model_class="meta_labeling",
        adequate=ok,
        requirement=(
            f">= {need} nominale én >= {need_eff} effectieve events per fold, "
            f"met een positieve-klasseratio in [{lo:.2f}, {hi:.2f}]"
        ),
        measured={
            "n_folds": int(events.size),
            "min_events_per_fold": smallest,
            "median_events_per_fold": float(np.median(events)),
            "total_events": int(events.sum()),
            "min_effective_events_per_fold": smallest_effective,
            "median_effective_events_per_fold": (
                float(np.median(effective)) if effective.size else float("nan")),
            "uniqueness_ratio": (
                float(np.median(effective) / np.median(events))
                if effective.size else float("nan")),
            "min_positive_ratio": float(ratios.min()),
            "max_positive_ratio": float(ratios.max()),
            "required_min_events": need,
            "required_min_effective_events": need_eff,
            "required_ratio_band": [lo, hi],
        },
        shortfall="" if ok else "; ".join(reasons) + ".",
    )


def assess_hrp(
    returns: pd.DataFrame,
    cfg: AdequacyConfig,
    *,
    n_folds: int = 1,
) -> AdequacyVerdict:
    """HRP: is de correlatiematrix stabiel genoeg om een boom op te bouwen?

    Twee grootheden. ``T/N`` bepaalt of de sample-correlatiematrix überhaupt
    goed geschat is; het conditiegetal bepaalt of de recursieve bisectie op
    signaal of op ruis snijdt. Bij ρ ~ 0,73 over zes assets is de matrix bijna
    rang-1 en is de boomordening grotendeels willekeurig — precies het risico
    dat §0.7 vooraf benoemt.
    """
    require(
        isinstance(returns, pd.DataFrame) and returns.shape[1] >= 2,
        "HRP-adequaatheid vereist een returnpaneel met minstens twee assets.",
        DataContractError,
        shape=tuple(returns.shape) if hasattr(returns, "shape") else None,
    )
    clean = returns.dropna(how="any")
    n_obs, n_assets = clean.shape
    require(
        n_obs > n_assets,
        "Minder observaties dan assets: de correlatiematrix is singulier.",
        DataContractError,
        n_obs=int(n_obs), n_assets=int(n_assets),
    )
    obs_per_fold = n_obs // max(n_folds, 1)
    ratio = obs_per_fold / n_assets

    corr = np.corrcoef(clean.to_numpy(dtype=np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    smallest_ev = float(eigenvalues.min())
    condition = float(eigenvalues.max() / smallest_ev) if smallest_ev > 0 else math.inf

    offdiag = corr[~np.eye(n_assets, dtype=bool)]
    mean_rho = float(offdiag.mean())

    need_ratio = cfg.hrp.min_obs_per_asset
    need_cond = cfg.hrp.max_condition_number
    ok = ratio >= need_ratio and condition <= need_cond

    reasons = []
    if ratio < need_ratio:
        reasons.append(
            f"T/N per fold is {ratio:.1f} tegen een eis van {need_ratio}"
        )
    if condition > need_cond:
        reasons.append(
            f"het conditiegetal van de correlatiematrix is {condition:.1f} tegen "
            f"een maximum van {need_cond:.1f}"
        )
    return AdequacyVerdict(
        model_class="hrp",
        adequate=ok,
        requirement=(
            f"T/N >= {need_ratio} per fold én conditiegetal <= {need_cond:.0f}"
        ),
        measured={
            "n_obs": int(n_obs),
            "n_assets": int(n_assets),
            "n_folds": int(n_folds),
            "obs_per_fold": int(obs_per_fold),
            "obs_per_asset_per_fold": ratio,
            "condition_number": condition,
            "smallest_eigenvalue": smallest_ev,
            "mean_pairwise_correlation": mean_rho,
            "required_obs_per_asset": need_ratio,
            "required_max_condition": need_cond,
        },
        shortfall="" if ok else "; ".join(reasons) + ".",
    )


# =========================================================================== #
# Power-analyse
# =========================================================================== #
@dataclass(frozen=True)
class PowerAnalysis:
    """Wat deze opzet kán zien, naast wat je verwacht te vinden."""

    test: str
    #: Het kleinste effect dat bij `alpha` met `target_power` detecteerbaar is.
    minimum_detectable_effect: float
    #: Het effect dat je op grond van de literatuur redelijkerwijs verwacht.
    expected_effect: float
    #: In welke eenheid beide bovenstaande staan.
    unit: str
    n_effective: float
    assumptions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def informative(self) -> bool:
        """True wanneer de toets het verwachte effect kán zien."""
        return abs(self.expected_effect) >= abs(self.minimum_detectable_effect)

    @property
    def power_deficit(self) -> float:
        """Hoeveel keer te klein het verwachte effect is. >1 = onderpowered."""
        if self.expected_effect == 0.0:
            return math.inf
        return abs(self.minimum_detectable_effect) / abs(self.expected_effect)

    def as_record(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "expected_effect": self.expected_effect,
            "unit": self.unit,
            "n_effective": self.n_effective,
            "informative": self.informative,
            "power_deficit": self.power_deficit,
            "assumptions": dict(self.assumptions),
        }


def _z_sum(cfg: PowerConfig) -> float:
    """z_{alpha[/2]} + z_{power} — de constante in elke MDE-formule."""
    alpha = cfg.alpha / 2.0 if cfg.two_sided else cfg.alpha
    return float(_norm.ppf(1.0 - alpha) + _norm.ppf(cfg.target_power))


def effective_independent_series(n_series: int, mean_correlation: float) -> float:
    """Hoeveel ONAFHANKELIJKE reeksen zes gecorreleerde perpetuals waard zijn.

    De variantie van het gemiddelde over ``k`` reeksen met gemiddelde
    paarsgewijze correlatie ``rho`` is opgeblazen met de design effect
    ``(1 + (k-1) * rho) / k``. Het aantal effectief onafhankelijke reeksen is
    daarvan de inverse:

        k_eff = k / (1 + (k - 1) * rho)

    Bij k = 6 en rho = 0,73 is dat **1,29**. Wie zijn power berekent op
    ``6 * n_bars`` in plaats van op ``1,29 * n_bars``, overschat het aantal
    observaties met een factor 4,6 en zijn detecteerbare effect met sqrt(4,6) =
    2,2. Dat is het verschil tussen "dit experiment kan het zien" en "dit
    experiment kan het niet zien".
    """
    require(
        n_series >= 1,
        "effective_independent_series op minder dan één reeks.",
        DataContractError,
        n_series=n_series,
    )
    require(
        -1.0 / max(n_series - 1, 1) < mean_correlation <= 1.0,
        "Gemiddelde correlatie buiten het bereik waarin de design effect "
        "gedefinieerd is.",
        DataContractError,
        mean_correlation=mean_correlation, n_series=n_series,
    )
    denominator = 1.0 + (n_series - 1) * mean_correlation
    return float(n_series / denominator)


def mean_difference_power(
    n_effective: float,
    expected_effect_sd_units: float,
    cfg: PowerConfig,
    *,
    test: str = "Diebold-Mariano (HLN)",
    assumptions: Mapping[str, Any] | None = None,
) -> PowerAnalysis:
    """MDE voor het gemiddelde van een verschilreeks, in eenheden van haar SD.

    Dit is de vorm van elke DM-toets: het verschil in verliesfunctie per bar is
    een reeks ``d_t``, en de nulhypothese is ``E[d_t] = 0``. Het minimaal
    detecteerbare effect in eenheden van ``sd(d)`` is

        MDE = (z_{alpha/2} + z_{power}) / sqrt(n_eff)

    ``n_eff`` is het aantal EFFECTIEVE observaties: gecorrigeerd voor de
    cross-sectionele correlatie tussen symbolen én voor de seriële correlatie
    die de HLN-correctie in de teststatistiek aanbrengt.
    """
    require(
        n_effective > 0,
        "Power-analyse op een niet-positief effectief aantal observaties.",
        DataContractError,
        n_effective=n_effective,
    )
    mde = _z_sum(cfg) / math.sqrt(n_effective)
    return PowerAnalysis(
        test=test,
        minimum_detectable_effect=mde,
        expected_effect=expected_effect_sd_units,
        unit="standaarddeviaties van de per-bar verliesverschilreeks",
        n_effective=float(n_effective),
        assumptions={
            "alpha": cfg.alpha,
            "target_power": cfg.target_power,
            "two_sided": cfg.two_sided,
            **(dict(assumptions) if assumptions else {}),
        },
    )


def sharpe_difference_power(
    n_years: float,
    baseline_sharpe: float,
    correlation_between_strategies: float,
    expected_sharpe_gain: float,
    cfg: PowerConfig,
    *,
    assumptions: Mapping[str, Any] | None = None,
) -> PowerAnalysis:
    """MDE voor het VERSCHIL tussen twee Sharpe-ratio's op dezelfde periode.

    De standaardfout van één geannualiseerde Sharpe over ``T`` jaar is
    (Lo 2002, onder i.i.d. returns):

        SE(SR) = sqrt((1 + SR^2 / 2) / T)

    Twee strategieën die hetzelfde primaire signaal delen, zijn sterk
    gecorreleerd. Voor het verschil geldt (Jobson-Korkie, Memmel-correctie) bij
    benadering:

        SE(dSR) = SE(SR) * sqrt(2 * (1 - rho))

    Die correlatie is precies waarom H2 überhaupt haalbaar zou kunnen zijn: een
    regimefilter op hetzelfde signaal verandert de returnreeks maar deels, en
    het gepaarde verschil is veel preciezer te meten dan elke Sharpe apart.
    Zij is ook waarom het resultaat MOET worden gerapporteerd als een gepaard
    verschil en niet als twee losse Sharpes met overlappende intervallen.
    """
    require(
        n_years > 0,
        "Power-analyse over een niet-positief aantal jaren.",
        DataContractError,
        n_years=n_years,
    )
    require(
        -1.0 <= correlation_between_strategies <= 1.0,
        "Correlatie tussen strategieën buiten [-1, 1].",
        DataContractError,
        correlation=correlation_between_strategies,
    )
    se_single = math.sqrt((1.0 + 0.5 * baseline_sharpe**2) / n_years)
    se_difference = se_single * math.sqrt(
        max(2.0 * (1.0 - correlation_between_strategies), 0.0)
    )
    mde = _z_sum(cfg) * se_difference
    return PowerAnalysis(
        test="Jobson-Korkie / Memmel op het gepaarde Sharpe-verschil",
        minimum_detectable_effect=mde,
        expected_effect=expected_sharpe_gain,
        unit="geannualiseerde Sharpe-eenheden",
        n_effective=float(n_years),
        assumptions={
            "alpha": cfg.alpha,
            "target_power": cfg.target_power,
            "two_sided": cfg.two_sided,
            "n_years": n_years,
            "baseline_sharpe": baseline_sharpe,
            "se_single_sharpe": se_single,
            "correlation_between_strategies": correlation_between_strategies,
            "se_sharpe_difference": se_difference,
            **(dict(assumptions) if assumptions else {}),
        },
    )


def auc_minimum_detectable(
    n_positive: int,
    n_negative: int,
    cfg: PowerConfig,
    *,
    null_auc: float = 0.5,
    expected_auc: float = 0.58,
    assumptions: Mapping[str, Any] | None = None,
) -> PowerAnalysis:
    """MDE voor een AUC tegen de nulwaarde 0,5 (Hanley & McNeil 1982).

    De standaardfout van een AUC hangt af van de AUC zelf:

        Q1 = A / (2 - A)                Q2 = 2 A^2 / (1 + A)
        SE = sqrt([A(1-A) + (n1-1)(Q1 - A^2) + (n0-1)(Q2 - A^2)] / (n1 n0))

    Onder de nulhypothese A = 0,5 vereenvoudigt dit; de MDE wordt uitgedrukt als
    het AUC-OVERSCHOT boven 0,5 dat bij `alpha` met `target_power` detecteerbaar
    is. Dat getal is de eerlijke maat naast de drempel 0,58 uit §24: is de MDE
    groter dan 0,08, dan kan dit experiment het verschil tussen "0,58" en "ruis"
    niet zien, hoe de AUC ook uitvalt.
    """
    require(
        n_positive > 1 and n_negative > 1,
        "AUC-power vereist minstens twee observaties per klasse.",
        DataContractError,
        n_positive=n_positive, n_negative=n_negative,
    )

    def _se(a: float) -> float:
        q1 = a / (2.0 - a)
        q2 = 2.0 * a * a / (1.0 + a)
        return math.sqrt(
            (a * (1.0 - a)
             + (n_positive - 1) * (q1 - a * a)
             + (n_negative - 1) * (q2 - a * a))
            / (n_positive * n_negative)
        )

    # De SE onder de nul (voor de kritieke waarde) en onder het alternatief
    # (voor de power) verschillen; beide meenemen is de eerlijke variant.
    se_null = _se(null_auc)
    se_alt = _se(expected_auc)
    alpha = cfg.alpha / 2.0 if cfg.two_sided else cfg.alpha
    mde = float(
        _norm.ppf(1.0 - alpha) * se_null + _norm.ppf(cfg.target_power) * se_alt
    )
    return PowerAnalysis(
        test="Hanley-McNeil AUC tegen 0,5",
        minimum_detectable_effect=mde,
        expected_effect=expected_auc - null_auc,
        unit="AUC-overschot boven 0,50",
        n_effective=float(min(n_positive, n_negative)),
        assumptions={
            "alpha": cfg.alpha,
            "target_power": cfg.target_power,
            "two_sided": cfg.two_sided,
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
            "se_under_null": se_null,
            "se_under_alternative": se_alt,
            "threshold_auc": expected_auc,
            **(dict(assumptions) if assumptions else {}),
        },
    )
