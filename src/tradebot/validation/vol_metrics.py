"""De weegschaal van de QLIKE-competitie — Phase 6, deliverable 12.

Vier verliesfuncties en twee toetsen. De verliesfuncties zeggen HOEVEEL een
forecast ernaast zat; de toetsen zeggen of dat verschil iets betekent.

    QLIKE      RV/s2 - ln(RV/s2) - 1        robuust tegen proxy-ruis
    MSE-SD     (sqrt(RV) - sqrt(s2))^2      NIET robuust -- zie hieronder
    MAE-SD     |sqrt(RV) - sqrt(s2)|        NIET robuust -- zie hieronder
    MSE-VAR    (RV - s2)^2                  robuust, maar zwaar staartgevoelig

    Mincer-Zarnowitz   is de forecast ZUIVER?      -- H0: alpha = 0, beta = 1
    Diebold-Mariano    is het VERSCHIL echt?       -- H0: E[d_t] = 0

WAAROM QLIKE DE PRIMAIRE MAAT IS, EN MSE-SD NIET
=================================================
De competitie meet niet tegen de ware variantie -- die is onobserveerbaar --
maar tegen een PROXY. Patton (2011) bewees dat de meeste gangbare
verliesfuncties daardoor een VERTEKENDE RANGSCHIKKING opleveren: het model dat
op de proxy wint, hoeft niet het model te zijn dat op de ware variantie wint.
Slechts twee families overleven die vervanging: MSE op de variantieschaal en
QLIKE. MSE-SD en MAE-SD, hoe gebruikelijk ook in de literatuur, doen dat NIET.

Zij staan hier omdat deliverable 12 ze vraagt en omdat een rapport dat alleen
QLIKE toont zich niet laat vergelijken met de literatuur. Maar het oordeel hangt
aan QLIKE, en waar MSE-SD een andere winnaar aanwijst dan QLIKE is dat geen
tegenstrijdigheid om weg te middelen -- het is de vertekening van Patton die
zichtbaar wordt, en zij hoort in het rapport.

QLIKE is bovendien ASYMMETRISCH: onderschatting van de variantie kost meer dan
overschatting. Voor een risicomodel is dat de juiste asymmetrie en niet een
gebrek.

DE NUL-BAR IS EEN QLIKE-BOM
============================
``ln(RV/s2)`` bestaat niet bij ``RV = 0`` en niet bij ``s2 = 0``. Een enkele
zulke bar maakt het gemiddelde verlies ``-inf``, en wie dat met `nanmean`
wegmiddelt, houdt een competitie over waarin de winnaar wordt bepaald door welk
model toevallig op die bar een waarde had. Elke functie hier geeft daarom NaN op
zo'n bar en :class:`LossSeries` TELT ze, zodat het rapport kan tonen hoeveel van
de steekproef eraan opging.

DE VERGELIJKING LOOPT OVER DEZELFDE BARS OF ZIJ LOOPT NIET
===========================================================
Dit is de stilste manier om deze competitie ongeldig te maken. Een
GARCH-variant laat NaN achter op elke fold die niet convergeerde; EWMA heeft
overal een waarde. Wie dan het gemiddelde QLIKE van beide vergelijkt, vergelijkt
EWMA op 1.200 bars met GARCH op 900 ANDERE bars -- en de 300 ontbrekende zijn
precies de moeilijke. :func:`diebold_mariano_hln` snijdt daarom eerst de
doorsnede en rapporteert wat er wegviel; hij weigert te vergelijken wanneer die
doorsnede te ver onder beide reeksen ligt.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md 9.1, 9.2; Patton (2011); Mincer & Zarnowitz
(1969); Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997);
Newey & West (1987).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _stats

from ..utils.failfast import DataContractError, require

__all__ = [
    "LOSS_FUNCTIONS",
    "PROXY_ROBUST_LOSSES",
    "DieboldMarianoResult",
    "LossSeries",
    "MincerZarnowitzResult",
    "diebold_mariano_hln",
    "loss_series",
    "mae_sd",
    "mincer_zarnowitz",
    "mse_sd",
    "mse_variance",
    "qlike",
]

#: De verliesfuncties waarvoor Patton (2011) bewees dat de RANGSCHIKKING onder
#: een zuivere maar ruizige proxy dezelfde blijft als onder de ware variantie.
#: Alleen deze twee mogen een oordeel dragen; de andere zijn rapportagemateriaal.
PROXY_ROBUST_LOSSES: frozenset[str] = frozenset({"qlike", "mse_variance"})


def _aligned(rv: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rv = np.asarray(rv, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    require(
        rv.shape == forecast.shape,
        "Realisatie en forecast hebben verschillende lengtes. Een stille "
        "uitlijning zou de verschuiving tussen forecast en realisatie kunnen "
        "verbergen, en dat is precies de fout die een competitie ongeldig "
        "maakt zonder op te vallen.",
        DataContractError,
        n_realised=int(rv.size), n_forecast=int(forecast.size),
    )
    return rv, forecast


def _usable(rv: np.ndarray, forecast: np.ndarray, *, strict: bool) -> np.ndarray:
    """Welke bars dragen een gedefinieerd verlies?

    `strict` geldt voor de logaritmische verliezen: daar moeten BEIDE waarden
    strikt positief zijn. Voor de kwadratische verliezen volstaat eindigheid,
    want die kennen geen logaritme.
    """
    finite = np.isfinite(rv) & np.isfinite(forecast)
    if not strict:
        return finite
    return finite & (rv > 0.0) & (forecast > 0.0)


def qlike(rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """``RV/s2 - ln(RV/s2) - 1`` per bar. NaN waar hij niet bestaat.

    Nul bij een perfecte forecast, strikt positief daarbuiten -- de functie is
    een Bregman-divergentie en heeft haar enige minimum op ``s2 = RV``. Dat
    laatste is de reden dat een negatieve gemiddelde QLIKE altijd een fout in de
    berekening is en nooit een goed model; :func:`loss_series` controleert het.
    """
    rv, forecast = _aligned(rv, forecast)
    out = np.full(rv.shape, np.nan, dtype=np.float64)
    ok = _usable(rv, forecast, strict=True)
    ratio = rv[ok] / forecast[ok]
    out[ok] = ratio - np.log(ratio) - 1.0
    return out


def mse_variance(rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """``(RV - s2)^2`` per bar. Proxy-robuust, maar zwaar staartgevoelig.

    Op crypto-dagdata wordt het gemiddelde gedomineerd door een handvol bars;
    dat is geen fout maar wel de reden dat QLIKE de primaire maat is.
    """
    rv, forecast = _aligned(rv, forecast)
    out = np.full(rv.shape, np.nan, dtype=np.float64)
    ok = _usable(rv, forecast, strict=False)
    out[ok] = (rv[ok] - forecast[ok]) ** 2
    return out


def mse_sd(rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """``(sqrt(RV) - sqrt(s2))^2`` per bar. **Niet** proxy-robuust."""
    rv, forecast = _aligned(rv, forecast)
    out = np.full(rv.shape, np.nan, dtype=np.float64)
    ok = _usable(rv, forecast, strict=False) & (rv >= 0.0) & (forecast >= 0.0)
    out[ok] = (np.sqrt(rv[ok]) - np.sqrt(forecast[ok])) ** 2
    return out


def mae_sd(rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """``|sqrt(RV) - sqrt(s2)|`` per bar. **Niet** proxy-robuust."""
    rv, forecast = _aligned(rv, forecast)
    out = np.full(rv.shape, np.nan, dtype=np.float64)
    ok = _usable(rv, forecast, strict=False) & (rv >= 0.0) & (forecast >= 0.0)
    out[ok] = np.abs(np.sqrt(rv[ok]) - np.sqrt(forecast[ok]))
    return out


LOSS_FUNCTIONS: Mapping[str, Any] = {
    "qlike": qlike,
    "mse_variance": mse_variance,
    "mse_sd": mse_sd,
    "mae_sd": mae_sd,
}


@dataclass(frozen=True)
class LossSeries:
    """Een verliesreeks met de boekhouding van wat er niet in zit."""

    name: str
    model: str
    #: Per bar, NaN waar het verlies niet bestaat. De volledige lengte blijft
    #: behouden zodat twee modellen positioneel vergelijkbaar blijven.
    values: np.ndarray
    n_total: int
    n_usable: int
    #: Bars die wegvielen omdat de PROXY nul of niet-eindig was. Een eigenschap
    #: van de data, gelijk voor elk model.
    n_dropped_proxy: int
    #: Bars die wegvielen omdat de FORECAST ontbrak. Een eigenschap van het
    #: MODEL -- bij GARCH de niet-geconvergeerde folds.
    n_dropped_forecast: int

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.values)) if self.n_usable else float("nan")

    @property
    def coverage(self) -> float:
        return self.n_usable / self.n_total if self.n_total else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "loss": self.name, "model": self.model, "mean": self.mean,
            "n_total": self.n_total, "n_usable": self.n_usable,
            "coverage": self.coverage,
            "n_dropped_proxy": self.n_dropped_proxy,
            "n_dropped_forecast": self.n_dropped_forecast,
            "proxy_robust": self.name in PROXY_ROBUST_LOSSES,
        }


def loss_series(
    rv: np.ndarray, forecast: np.ndarray, *, loss: str, model: str,
) -> LossSeries:
    """Bereken een verlies en splits uit WAAROM bars wegvielen.

    Het onderscheid tussen `n_dropped_proxy` en `n_dropped_forecast` is niet
    cosmetisch. Het eerste is een eigenschap van de data en treft elk model
    gelijk; het tweede is een eigenschap van het model en is precies wat een
    lage convergentieratio zichtbaar maakt. Wie ze optelt, ziet niet meer of een
    variant weinig bars had omdat de proxy stuk was of omdat hij niet
    convergeerde.
    """
    require(
        loss in LOSS_FUNCTIONS,
        f"Onbekende verliesfunctie '{loss}'.",
        DataContractError, known=sorted(LOSS_FUNCTIONS),
    )
    rv, forecast = _aligned(rv, forecast)
    values = LOSS_FUNCTIONS[loss](rv, forecast)
    strict = loss == "qlike"
    proxy_ok = np.isfinite(rv) & (rv > 0.0 if strict else True)
    forecast_ok = np.isfinite(forecast) & (forecast > 0.0 if strict else True)
    usable = np.isfinite(values)

    if loss == "qlike":
        finite_values = values[usable]
        require(
            bool(np.all(finite_values >= -1e-9)),
            "QLIKE is negatief. Dat kan niet: de functie is een "
            "Bregman-divergentie met haar enige minimum, nul, op s2 = RV. Een "
            "negatieve waarde betekent een fout in de berekening of in de "
            "eenheden van de forecast -- controleer de terugschaling van de "
            "GARCH-variantie.",
            DataContractError,
            model=model, min_value=float(finite_values.min())
            if finite_values.size else float("nan"),
        )

    return LossSeries(
        name=loss, model=model, values=values,
        n_total=int(rv.size), n_usable=int(np.count_nonzero(usable)),
        n_dropped_proxy=int(np.count_nonzero(~proxy_ok)),
        n_dropped_forecast=int(np.count_nonzero(proxy_ok & ~forecast_ok)),
    )


# =========================================================================== #
# Mincer-Zarnowitz
# =========================================================================== #
def _newey_west_lags(n: int) -> int:
    """De automatische bandbreedte van Newey & West (1987): ``4 (n/100)^(2/9)``."""
    return max(1, int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


@dataclass(frozen=True)
class MincerZarnowitzResult:
    """``RV_t = alpha + beta s2_t + e_t`` met de gezamenlijke toets op (0, 1)."""

    model: str
    alpha: float
    beta: float
    se_alpha: float
    se_beta: float
    #: Wald-statistiek van de GEZAMENLIJKE nulhypothese, chi^2 met 2 vrijheden.
    wald_statistic: float
    p_value: float
    r_squared: float
    n_obs: int
    hac_lags: int

    @property
    def unbiased(self) -> bool:
        """Wordt de nulhypothese ``alpha = 0, beta = 1`` NIET verworpen?"""
        return self.p_value >= 0.05

    def as_record(self) -> dict[str, Any]:
        return {
            "model": self.model, "alpha": self.alpha, "beta": self.beta,
            "se_alpha": self.se_alpha, "se_beta": self.se_beta,
            "wald_statistic": self.wald_statistic, "p_value": self.p_value,
            "r_squared": self.r_squared, "n_obs": self.n_obs,
            "hac_lags": self.hac_lags, "unbiased": self.unbiased,
        }


def mincer_zarnowitz(
    rv: np.ndarray, forecast: np.ndarray, *, model: str, min_obs: int = 30,
) -> MincerZarnowitzResult:
    """Zuiverheidsregressie met HAC-standaardfouten.

    WAAROM HAC EN NIET GEWOON OLS
    ------------------------------
    De residuen van deze regressie zijn sterk serieel gecorreleerd, omdat zowel
    RV als de forecast persistent zijn. Onder OLS zijn de standaardfouten
    daardoor te KLEIN, de Wald-statistiek te groot en verwerpt de toets de
    zuiverheid van vrijwel elke forecast -- inclusief een die perfect zuiver is.
    Dat is geen subtiel verschil: de vertekening loopt makkelijk op tot een
    factor twee in de standaardfout.

    Newey-West met de automatische bandbreedte ``4 (n/100)^(2/9)`` corrigeert
    daarvoor. De gekozen bandbreedte staat in het resultaat, zodat het rapport
    hem kan noemen in plaats van hem te verbergen.
    """
    rv, forecast = _aligned(rv, forecast)
    ok = np.isfinite(rv) & np.isfinite(forecast)
    y = rv[ok]
    x = forecast[ok]
    n = int(y.size)
    require(
        n >= min_obs,
        "Mincer-Zarnowitz op te weinig bruikbare bars. De regressie zou "
        "convergeren en een p-waarde geven die niets betekent.",
        DataContractError, model=model, n_obs=n, min_obs=min_obs,
    )
    design = np.column_stack([np.ones(n), x])
    require(
        float(np.std(x)) > 0.0,
        "Mincer-Zarnowitz op een CONSTANTE forecast: beta is dan niet "
        "geïdentificeerd. Een constante forecast is een geldig model (Level 0) "
        "maar deze toets kan er niets over zeggen.",
        DataContractError, model=model,
    )
    xtx_inv = np.linalg.inv(design.T @ design)
    coef = xtx_inv @ design.T @ y
    resid = y - design @ coef

    lags = _newey_west_lags(n)
    scores = design * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)  # Bartlett: garandeert positief-semidefiniet
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    cov = xtx_inv @ meat @ xtx_inv

    deviation = coef - np.array([0.0, 1.0])
    wald = float(deviation @ np.linalg.inv(cov) @ deviation)
    p_value = float(_stats.chi2.sf(wald, df=2))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return MincerZarnowitzResult(
        model=model, alpha=float(coef[0]), beta=float(coef[1]),
        se_alpha=float(math.sqrt(cov[0, 0])), se_beta=float(math.sqrt(cov[1, 1])),
        wald_statistic=wald, p_value=p_value,
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        n_obs=n, hac_lags=lags,
    )


# =========================================================================== #
# Diebold-Mariano met de Harvey-Leybourne-Newbold-correctie
# =========================================================================== #
@dataclass(frozen=True)
class DieboldMarianoResult:
    """Het oordeel over een verschil, met alles wat de pre-registratie vraagt."""

    model_a: str
    model_b: str
    horizon: int
    #: ``mean(loss_a - loss_b)``. Negatief betekent dat A een LAGER verlies had.
    mean_loss_differential: float
    dm_statistic: float
    #: De HLN-gecorrigeerde statistiek, getoetst tegen t met ``n - 1`` vrijheden.
    hln_statistic: float
    p_value: float
    n_obs: int
    #: De gemeten AR(1) van de verliesverschilreeks. Het H1-stopcriterium
    #: `underpowered_test_cannot_falsify` hangt hieraan: boven ~0,3 is een
    #: niet-significante uitkomst `UNPROVEN` en geen falsificatie.
    loss_differential_ar1: float
    n_dropped_to_intersection: int
    alternative: Literal["two-sided", "a-better", "b-better"]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def better(self) -> str:
        """Welk model had het lagere gemiddelde verlies? Los van significantie."""
        if self.mean_loss_differential < 0:
            return self.model_a
        if self.mean_loss_differential > 0:
            return self.model_b
        return "gelijk"

    def as_record(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a, "model_b": self.model_b,
            "horizon": self.horizon,
            "mean_loss_differential": self.mean_loss_differential,
            "dm_statistic": self.dm_statistic,
            "hln_statistic": self.hln_statistic, "p_value": self.p_value,
            "n_obs": self.n_obs,
            "loss_differential_ar1": self.loss_differential_ar1,
            "n_dropped_to_intersection": self.n_dropped_to_intersection,
            "alternative": self.alternative,
            "significant": self.significant, "better": self.better,
        }


def _ar1(d: np.ndarray) -> float:
    """Eerste-orde autocorrelatie van de verliesverschilreeks."""
    centred = d - d.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        return 0.0
    return float((centred[1:] @ centred[:-1]) / denominator)


def diebold_mariano_hln(
    loss_a: LossSeries,
    loss_b: LossSeries,
    *,
    horizon: int,
    alternative: Literal["two-sided", "a-better", "b-better"] = "two-sided",
    min_intersection_ratio: float = 0.80,
) -> DieboldMarianoResult:
    """DM-toets met de kleine-steekproefcorrectie van Harvey-Leybourne-Newbold.

    De ongecorrigeerde DM-statistiek verwerpt te vaak bij eindige steekproeven,
    en dat wordt erger naarmate de horizon groeit. HLN (1997) corrigeert met

        DM* = DM * sqrt( (n + 1 - 2h + h(h-1)/n) / n )

    en toetst tegen een t-verdeling met ``n - 1`` vrijheden in plaats van tegen
    de standaardnormaal. Op de ~1.200 OOS-bars van deze fase is het verschil
    klein maar niet verwaarloosbaar; op de kortere per-symboolreeksen wel
    degelijk merkbaar.

    DE DOORSNEDE
    ------------
    Beide reeksen worden teruggebracht tot de bars waarop ze ALLEBEI een verlies
    hebben. Zonder dat zou een GARCH-variant met niet-geconvergeerde folds
    worden vergeleken met een EWMA die ook op die folds een waarde had -- en de
    ontbrekende bars zijn precies de moeilijke. Valt de doorsnede onder
    `min_intersection_ratio` van de BEST GEDEKTE reeks, dan wordt er niet
    vergeleken: de twee modellen hebben dan feitelijk verschillende
    steekproeven gezien. Dat de referentie de best gedekte reeks is en niet de
    slechtst gedekte, is geen detail -- zie de opmerking bij `reference` in de
    code.
    """
    require(
        loss_a.name == loss_b.name,
        "Diebold-Mariano over twee VERSCHILLENDE verliesfuncties. Het verschil "
        "zou dan geen modelverschil meten maar een functieverschil.",
        DataContractError, loss_a=loss_a.name, loss_b=loss_b.name,
    )
    require(
        loss_a.values.shape == loss_b.values.shape,
        "Diebold-Mariano op reeksen van verschillende lengte.",
        DataContractError,
        n_a=int(loss_a.values.size), n_b=int(loss_b.values.size),
    )
    require(
        horizon >= 1,
        "Diebold-Mariano met een horizon onder 1 bar.",
        DataContractError, horizon=horizon,
    )
    both = np.isfinite(loss_a.values) & np.isfinite(loss_b.values)
    n = int(np.count_nonzero(both))
    # De referentie is de BEST GEDEKTE reeks, niet de slechtst gedekte.
    # DEFECT IN MIJN EIGEN WERK, gevonden door
    # `test_too_little_overlap_is_refused` en hier vastgelegd. De eerste versie
    # toetste tegen `min(n_usable_a, n_usable_b)`, en die poort KAN NIET
    # AANSLAAN wanneer de ene reeks een deelverzameling van de andere is: bij
    # EWMA op 1.000 bars en een GARCH die op 600 folds niet convergeerde, is de
    # doorsnede 400 -- exact 100 % van de kortste reeks, dus altijd boven elke
    # ratio. Precies het geval waarvoor de poort bestaat, glipte er dus
    # gegarandeerd doorheen. Dit is hetzelfde defecttype als de wiring-test met
    # niet-bindende limietwaarden uit `reports/phase5_exit_report.md` §12: een
    # controle die niet rood kán worden, bewijst niets.
    reference = max(loss_a.n_usable, loss_b.n_usable)
    require(
        n > horizon + 1,
        "Diebold-Mariano op te weinig gemeenschappelijke bars.",
        DataContractError, n_common=n, horizon=horizon,
    )
    require(
        n >= min_intersection_ratio * reference,
        "De twee modellen delen te weinig bars om te vergelijken. De reeksen "
        "zijn dan feitelijk verschillende steekproeven, en de bars die "
        "ontbreken zijn juist de vensters waarop het zwakkere model faalde. "
        "Rapporteer het model met zijn convergentieratio en haal het uit de "
        "vergelijking; vergelijk het niet op de vensters die het wel aankon.",
        DataContractError,
        n_common=n, n_usable_a=loss_a.n_usable, n_usable_b=loss_b.n_usable,
        reference_n_usable=reference, required_ratio=min_intersection_ratio,
    )

    d = loss_a.values[both] - loss_b.values[both]
    d_bar = float(d.mean())
    centred = d - d_bar
    gamma0 = float(centred @ centred) / n
    variance = gamma0
    for lag in range(1, horizon):
        gamma = float(centred[lag:] @ centred[:-lag]) / n
        variance += 2.0 * gamma
    require(
        variance > 0.0,
        "De DM-langetermijnvariantie is niet positief. Dat gebeurt bij de "
        "rechthoekig-getrunceerde schatter van Diebold-Mariano wanneer de "
        "autocovarianties sterk negatief zijn. Er wordt NIET stilzwijgend "
        "overgeschakeld op een Bartlett-kernel: dat is een ANDERE toets met "
        "een andere verdeling onder de nulhypothese, en die keuze hoort in een "
        "pre-registratie en niet in een `except`-tak.",
        DataContractError,
        model_a=loss_a.model, model_b=loss_b.model, horizon=horizon,
        long_run_variance=variance, gamma0=gamma0,
    )

    dm = d_bar / math.sqrt(variance / n)
    correction = math.sqrt(
        (n + 1.0 - 2.0 * horizon + horizon * (horizon - 1.0) / n) / n)
    require(
        correction > 0.0,
        "De HLN-correctiefactor is niet positief; de horizon is te groot voor "
        "het aantal observaties.",
        DataContractError, n_obs=n, horizon=horizon,
    )
    hln = dm * correction
    df = n - 1
    if alternative == "two-sided":
        p_value = float(2.0 * _stats.t.sf(abs(hln), df=df))
    elif alternative == "a-better":
        # H1: A heeft een LAGER verlies, dus d_bar < 0 en hln < 0.
        p_value = float(_stats.t.cdf(hln, df=df))
    else:
        p_value = float(_stats.t.sf(hln, df=df))

    return DieboldMarianoResult(
        model_a=loss_a.model, model_b=loss_b.model, horizon=horizon,
        mean_loss_differential=d_bar, dm_statistic=dm, hln_statistic=hln,
        p_value=p_value, n_obs=n, loss_differential_ar1=_ar1(d),
        n_dropped_to_intersection=int(reference - n),
        alternative=alternative,
    )
