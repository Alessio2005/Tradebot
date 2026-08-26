"""L2 - de GARCH-familie als uitdager van EWMA(0.94). Deliverable 10, stap 6.

Vier varianten met Student-t innovaties, elk gefit binnen de walk-forward-
structuur en nooit op de volledige sample:

    GARCH(1,1)   sigma^2_t = omega + alpha e^2_{t-1} + beta sigma^2_{t-1}
    GJR-GARCH    + gamma e^2_{t-1} 1{e_{t-1} < 0}      -- leverage
    EGARCH       ln sigma^2_t = omega + alpha g(z) + beta ln sigma^2_{t-1}
    APARCH       sigma^d_t = omega + alpha (|e|-gamma e)^d + beta sigma^d_{t-1}

WAAROM STUDENT-T EN NIET NORMAAL
=================================
Geen stilistische keuze. Onder een normale verdeling is de QML-schatter van
omega, alpha en beta nog consistent, maar `nu` is precies de parameter die
bepaalt hoeveel van de gemeten kurtosis aan de STAART wordt toegeschreven en
hoeveel aan de VOLATILITEITSDYNAMIEK. Legt men de staart vast op die van een
normaal, dan moet alpha het verschil opvangen en loopt alpha + beta systematisch
naar 1 -- de IGARCH-rand. Op crypto-dagreturns met een kurtosis ver boven 3 is
dat geen randgeval maar de verwachte uitkomst.

De keuze staat hier expliciet in :attr:`GarchSpec.dist` en niet als default in
een functiehandtekening, zodat hij in het rapport verantwoord kan worden.

DE SCHAAL, EN WAAROM HIJ EXPLICIET IS
======================================
`arch` optimaliseert numeriek slecht op reeksen met een standaardafwijking rond
0,03 -- de orde van een crypto-dagreturn. Het pakket lost dat zelf op met
`rescale=True`, en dat is precies wat hier NIET gebeurt: die schaal komt dan in
de geschatte `omega` terecht en de gebruiker moet maar raden in welke eenheid de
forecast staat. Hier wordt met een expliciete :data:`RETURN_SCALE` van 100
gewerkt (returns in procent) en wordt de variantieforecast door
``RETURN_SCALE**2`` teruggedeeld. De QLIKE-competitie vergelijkt de forecast met
een RV-proxy in absolute eenheden; een factor 10.000 verschil zou daar een
QLIKE-verschil van ln(10^4) ~ 9,2 opleveren en de hele competitie beslissen op
een eenheidsfout.

NIET-CONVERGENTIE IS EEN RESULTAAT, GEEN PROBLEEM
==================================================
Twee eisen uit de fase-opdracht die op het eerste gezicht botsen:

    deliverable 10  "harde crash bij niet-convergentie, geen fallback naar EWMA"
    stap 6          "niet-convergentie en randoplossingen zijn resultaten die je
                     registreert, geen problemen die je wegvangt"

Zij botsen niet; ze gaan over verschillende momenten. Een MISLUKTE FIT wordt
geregistreerd -- :class:`GarchFit` draagt `converged=False` met de boodschap van
de optimizer erbij, en de campagne telt hem in de convergentieratio. Maar een
mislukte fit levert GEEN forecast: :meth:`GarchFit.forecast_variance` crasht.
Er bestaat dus geen pad waarlangs een niet-geconvergeerd model stilzwijgend een
getal aan de QLIKE-competitie levert, en al helemaal geen pad waarlangs EWMA dat
getal voor hem invult. Dat laatste zou het ergste van beide werelden zijn: een
competitie waarin de uitdager op de moeilijke vensters het antwoord van de
titelverdediger krijgt aangereikt.

Zakt de convergentieratio van een variant onder `min_convergence_ratio`, dan is
zijn QLIKE-reeks een SELECTIE van de gemakkelijke vensters en wordt hij
gedescopeerd -- zie het stop-criterium `convergence_too_low_to_compare` in
`conf/research/preregistration_h1_garch_vs_ewma.yaml`.

NOOIT OP DE VOLLEDIGE SAMPLE
=============================
Dit is structureel afgedwongen en niet aan de discipline van de aanroeper
overgelaten. :func:`fit_garch_window` eist `train_end < n_total`: een venster dat
tot het einde van de reeks loopt, IS de volledige sample en wordt geweigerd.
Daarnaast schat `arch` dankzij `last_obs` uitsluitend op `y[:train_end]`, terwijl
de variantierecursie daarna wel de FEITELIJK WAARGENOMEN returns gebruikt. Dat
is geen lek maar de bedoeling: de parameters zijn out-of-sample, de conditionele
variantie is gefilterd op waargenomen verleden. Precies zoals het model in
productie zou draaien met bevroren parameters.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md 9.1, 9.2, 19 (L2); Bollerslev (1986);
Glosten, Jagannathan & Runkle (1993); Nelson (1991); Ding, Granger & Engle
(1993); Hansen & Lunde (2005).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..cv.walk_forward import WalkForwardCV
from ..schemas.config import AdequacyConfig
from ..utils.failfast import (
    DataContractError,
    TradebotContractError,
    require,
    require_dependency,
)
from ..validation.data_adequacy import assess_garch, require_adequacy

__all__ = [
    "GARCH_FAMILY",
    "RETURN_SCALE",
    "ConvergenceSummary",
    "GarchFit",
    "GarchSpec",
    "NonConvergenceError",
    "fit_garch_window",
    "summarise_convergence",
    "walk_forward_variance_forecasts",
]

#: Returns worden in PROCENT aan `arch` aangeboden. Zie de moduledocstring: dit
#: staat hier als constante zodat de terugvertaling (`/ RETURN_SCALE**2`) op
#: exact een plek gebeurt en niet als los getal door de code zwerft.
RETURN_SCALE: float = 100.0

#: De uitzonderingen die een QML-optimizer op DEGENERATE DATA werkelijk opwerpt,
#: en die dus "deze fit is niet geconvergeerd" betekenen.
#:
#: DEFECT IN MIJN EIGEN WERK, gevonden door `test_no_silent_fallbacks.py` en
#: hier vastgelegd. De eerste versie ving `Exception`. Dat is niet alleen een
#: schending van de fallback-doctrine; het is inhoudelijk fout. Een
#: `AttributeError` uit een typefout in mijn eigen code, een `MemoryError`, een
#: `KeyError` op een verkeerde parameternaam -- ze zouden alle drie zijn
#: geregistreerd als "het model convergeerde niet op dit venster". Een bug in de
#: code was dan als DATABEVINDING in de convergentieratio beland, en de
#: conclusie "GARCH convergeert slecht op dit universum" zou zijn gestoeld op
#: een gebroken aanroep. Alles buiten deze tuple propageert.
_NUMERICAL_FIT_FAILURES: tuple[type[Exception], ...] = (
    np.linalg.LinAlgError,   # singuliere Hessiaan
    FloatingPointError,
    ZeroDivisionError,
    RuntimeError,            # scipy.optimize geeft het op
    ValueError,              # arch weigert de reeks (bv. nul variantie)
)


class NonConvergenceError(TradebotContractError):
    """Er is een forecast gevraagd aan een fit die niet is geconvergeerd.

    Een eigen klasse en niet `DataContractError`, omdat dit geen datafout is
    maar een aanroepfout: de aanroeper had `GarchFit.converged` moeten lezen.
    Een eigen klasse maakt hem bovendien vindbaar in een `except`-tak zonder dat
    daar per ongeluk echte datacontractfouten in worden gevangen.
    """


@dataclass(frozen=True)
class GarchSpec:
    """Een lid van de familie. De parameterruimte van H1, expliciet."""

    name: str
    vol: Literal["GARCH", "EGARCH", "APARCH"]
    p: int = 1
    o: int = 0
    q: int = 1
    power: float = 2.0
    #: Student-t. Zie de moduledocstring voor waarom dit geen default is dat
    #: iemand ongemerkt op "normal" kan laten staan.
    dist: Literal["t"] = "t"
    #: Hoe de persistentie van deze specificatie wordt gemeten. Zij verschilt
    #: PER VARIANT en `alpha + beta` is voor drie van de vier fout.
    persistence_kind: Literal["sum_ab", "gjr", "beta_only", "aparch"] = "sum_ab"

    @property
    def label(self) -> str:
        return f"{self.name}(p={self.p},o={self.o},q={self.q})-{self.dist}"


#: De vier varianten uit deliverable 10. Vier modellen x 6 symbolen x 2
#: horizonnen = 48 trials, exact het `planned_trials` van de H1-pre-registratie.
#: Een vijfde variant toevoegen vereist een NIEUWE pre-registratie; daarom staat
#: deze mapping hier vast en niet in een config die per run kan verschuiven.
GARCH_FAMILY: Mapping[str, GarchSpec] = {
    "garch": GarchSpec(
        name="garch", vol="GARCH", p=1, o=0, q=1, persistence_kind="sum_ab"),
    "gjr_garch": GarchSpec(
        name="gjr_garch", vol="GARCH", p=1, o=1, q=1, persistence_kind="gjr"),
    "egarch": GarchSpec(
        name="egarch", vol="EGARCH", p=1, o=1, q=1,
        persistence_kind="beta_only"),
    "aparch": GarchSpec(
        name="aparch", vol="APARCH", p=1, o=1, q=1,
        persistence_kind="aparch"),
}


def _abs_moment_student_t(delta: float, nu: float) -> float:
    """``E|z|^delta`` voor een op variantie 1 gestandaardiseerde Student-t.

    Nodig voor de APARCH-persistentie. De formule vraagt ``delta < nu``;
    daarboven bestaat het moment niet en is de persistentie oneindig. Dat wordt
    als ``inf`` teruggegeven en niet als fout: het betekent dat het model niet
    covariantiestationair is, en dat is een meetresultaat dat de
    randoplossingsteller hoort te halen.
    """
    if not delta < nu:
        return math.inf
    log_moment = (
        0.5 * delta * math.log(nu - 2.0)
        + math.lgamma(0.5 * (delta + 1.0))
        + math.lgamma(0.5 * (nu - delta))
        - 0.5 * math.log(math.pi)
        - math.lgamma(0.5 * nu)
    )
    return math.exp(log_moment)


def _persistence(spec: GarchSpec, params: Mapping[str, float]) -> float:
    """De persistentie van deze specificatie, per variant correct gemeten.

    `alpha + beta` klakkeloos toepassen levert voor drie van de vier varianten
    een verkeerd getal op, en dus een verkeerd oordeel over de IGARCH-rand:

    GARCH     ``alpha + beta``.
    GJR       ``alpha + gamma/2 + beta``. De helft omdat ``E[1{z < 0}] = 1/2``
              onder een symmetrische verdeling, en de Student-t is symmetrisch.
              Wie de gamma weglaat, onderschat de persistentie van precies het
              model dat de leverage moet vangen.
    EGARCH    ``beta`` alleen. De recursie loopt in ``ln sigma^2``; alpha
              vermenigvuldigt de geschokte term en zegt niets over hoe lang een
              schok blijft hangen. EGARCH is per constructie stationair voor
              ``|beta| < 1`` ongeacht alpha, dus ``alpha + beta`` zou hier ten
              onrechte alarm slaan.
    APARCH    ``alpha * E[(|z| - gamma z)^delta] + beta``. Onder een
              symmetrische verdeling valt de verwachting uiteen in
              ``0,5 [(1+gamma)^delta + (1-gamma)^delta] * E|z|^delta``.
    """
    alpha = float(params.get("alpha[1]", 0.0))
    beta = float(params.get("beta[1]", 0.0))
    gamma = float(params.get("gamma[1]", 0.0))
    if spec.persistence_kind == "sum_ab":
        return alpha + beta
    if spec.persistence_kind == "gjr":
        return alpha + 0.5 * gamma + beta
    if spec.persistence_kind == "beta_only":
        return beta
    if spec.persistence_kind == "aparch":
        delta = float(params.get("delta", spec.power))
        nu = float(params.get("nu", math.inf))
        kappa = 0.5 * float((1.0 + gamma) ** delta + (1.0 - gamma) ** delta)
        return float(alpha * kappa * _abs_moment_student_t(delta, nu) + beta)
    raise AssertionError(f"onbekende persistence_kind: {spec.persistence_kind}")


@dataclass(frozen=True)
class GarchFit:
    """Een fit op een venster. Draagt zijn eigen convergentiestatus.

    Er bestaat geen variant van dit object die "min of meer" is geconvergeerd.
    `converged` is True of False, en op False levert dit object geen forecast.
    """

    spec: GarchSpec
    symbol: str
    fold_id: int
    n_obs: int
    converged: bool
    #: De boodschap van de optimizer, letterlijk. Dit is wat een
    #: convergentieratio onder de 90 % onderbouwt in het rapport.
    message: str
    params: Mapping[str, float]
    persistence: float
    at_boundary: bool
    loglikelihood: float
    #: Het `arch`-resultaatobject. Buiten de repr en de gelijkheid gehouden: het
    #: is niet zinvol vergelijkbaar en zou een dataclass-repr onleesbaar maken.
    result: Any = field(default=None, repr=False, compare=False)

    def forecast_variance(self, *, horizon: int, start: int) -> pd.DataFrame:
        """Variantieforecasts in ABSOLUTE eenheden, terug van de procentschaal.

        Rij `i` bevat de forecasts die met informatie tot en met bar `i` zijn
        gemaakt; kolom ``h-1`` de `h`-staps forecast, dus voor bar ``i + h``.
        Die verschuiving is de gevaarlijkste plek in dit module en wordt in
        :func:`walk_forward_variance_forecasts` op exact een plaats toegepast.
        """
        require(
            self.converged,
            f"Forecast gevraagd aan een niet-geconvergeerde {self.spec.label} "
            f"op {self.symbol} fold {self.fold_id}. Er wordt NIET teruggevallen "
            "op EWMA of op een eenvoudiger variant: dat zou de uitdager op "
            "precies de moeilijke vensters het antwoord van de titelverdediger "
            "aanreiken. De fit telt als niet-geconvergeerd in de "
            "convergentieratio.",
            NonConvergenceError,
            spec=self.spec.label, symbol=self.symbol, fold_id=self.fold_id,
            optimizer_message=self.message,
        )
        forecast = self.result.forecast(
            horizon=horizon, start=start, reindex=False, method="analytic")
        variance: pd.DataFrame = forecast.variance
        return variance / (RETURN_SCALE ** 2)

    def as_record(self) -> dict[str, Any]:
        return {
            "spec": self.spec.label,
            "symbol": self.symbol,
            "fold_id": self.fold_id,
            "n_obs": self.n_obs,
            "converged": self.converged,
            "message": self.message,
            "persistence": self.persistence,
            "at_boundary": self.at_boundary,
            "loglikelihood": self.loglikelihood,
            **{f"param_{k}": v for k, v in self.params.items()},
        }


def fit_garch_window(
    returns: pd.Series,
    spec: GarchSpec,
    cfg: AdequacyConfig,
    *,
    symbol: str,
    fold_id: int,
    train_end: int,
) -> GarchFit:
    """Fit `spec` op ``returns[:train_end]``. Nooit op de volledige reeks.

    Parameters
    ----------
    returns : de reeks tot en met het einde van het testvenster van deze fold.
        Dat is geen lek: `arch` schat dankzij ``last_obs=train_end`` uitsluitend
        op de bars ervoor, terwijl de variantierecursie erna de feitelijk
        waargenomen returns gebruikt -- precies zoals het model in productie zou
        draaien met bevroren parameters.
    train_end : exclusieve grens van het schattingsvenster. Moet strikt kleiner
        zijn dan de reekslengte; gelijk betekent een fit op de volledige sample,
        en dat is verboden.
    """
    arch_mod = require_dependency(
        "arch",
        needed_for="de Level 2 GARCH-familie (H1)",
        install_hint="pip install arch",
    )
    require(
        0 < train_end < returns.size,
        "GARCH-fit op de volledige sample. `train_end` moet strikt kleiner "
        "zijn dan de reekslengte; anders is er geen out-of-sample venster en "
        "meet de QLIKE-competitie een in-sample fit.",
        DataContractError,
        symbol=symbol, spec=spec.label, train_end=train_end,
        n_total=int(returns.size),
    )
    train = returns.iloc[:train_end].to_numpy(dtype=np.float64)
    require(
        bool(np.all(np.isfinite(train))),
        "GARCH-fit op een venster met niet-eindige returns. Er wordt niet "
        "geimputeerd; een gat is een databevinding.",
        DataContractError,
        symbol=symbol, spec=spec.label, fold_id=fold_id,
        n_non_finite=int(np.count_nonzero(~np.isfinite(train))),
    )
    # De poort. Staat VOOR de fit en niet erna, zodat er geen pad bestaat
    # waarlangs een model op ontoereikende data wordt geschat -- ook niet
    # tijdens exploratie, waar zulke fits normaal ontstaan.
    require_adequacy(assess_garch([train_end], cfg, symbol=symbol))

    model = arch_mod.arch_model(
        returns.to_numpy(dtype=np.float64) * RETURN_SCALE,
        mean="Constant", vol=spec.vol, p=spec.p, o=spec.o, q=spec.q,
        power=spec.power, dist=spec.dist,
        rescale=False,  # Zie RETURN_SCALE: de schaal is expliciet, niet impliciet.
    )
    with warnings.catch_warnings():
        # De convergentiewaarschuwingen van `arch` worden hier tot een RESULTAAT
        # gemaakt in plaats van naar stderr te lekken: `converged` en `message`
        # dragen dezelfde informatie, maar dan telbaar.
        warnings.simplefilter("ignore")
        try:
            res = model.fit(
                last_obs=train_end, disp="off", show_warning=False,
                options={"maxiter": 1000},
            )
        except _NUMERICAL_FIT_FAILURES as exc:
            return GarchFit(
                spec=spec, symbol=symbol, fold_id=fold_id, n_obs=train_end,
                converged=False, message=f"{type(exc).__name__}: {exc}",
                params={}, persistence=float("nan"), at_boundary=True,
                loglikelihood=float("nan"), result=None,
            )

    opt = getattr(res, "optimization_result", None)
    converged = bool(getattr(opt, "success", False))
    message = str(getattr(opt, "message", "geen optimizer-resultaat"))
    params = {k: float(v) for k, v in res.params.items()}
    persistence = _persistence(spec, params)
    # Een niet-eindige persistentie is OOK een randoplossing: de
    # onvoorwaardelijke variantie bestaat dan niet.
    at_boundary = (
        not math.isfinite(persistence)
        or abs(persistence) >= cfg.garch.persistence_boundary
    )
    return GarchFit(
        spec=spec, symbol=symbol, fold_id=fold_id, n_obs=train_end,
        converged=converged, message=message, params=params,
        persistence=persistence, at_boundary=at_boundary,
        loglikelihood=float(res.loglikelihood), result=res,
    )


@dataclass(frozen=True)
class ConvergenceSummary:
    """Wat het stop-criterium `convergence_too_low_to_compare` nodig heeft."""

    spec: str
    symbol: str
    n_fits: int
    n_converged: int
    n_boundary: int
    #: Elke unieke optimizer-boodschap van een mislukte fit, met telling. Zonder
    #: dit is een lage convergentieratio een getal zonder diagnose.
    failure_messages: Mapping[str, int]

    @property
    def convergence_ratio(self) -> float:
        return self.n_converged / self.n_fits if self.n_fits else 0.0

    @property
    def boundary_ratio(self) -> float:
        """Aandeel van de GECONVERGEERDE fits op de rand.

        Noemer is `n_converged` en niet `n_fits`: een randoplossing is een
        eigenschap van een fit die een antwoord gaf. Delen door alle fits zou de
        ratio verdunnen met mislukkingen die al apart worden geteld, en een
        variant die zelden convergeert maar dan altijd op de rand uitkomt, door
        de randpoort laten glippen.
        """
        return self.n_boundary / self.n_converged if self.n_converged else 0.0

    def comparable(self, cfg: AdequacyConfig) -> bool:
        """Mag deze variant meedoen in de QLIKE-competitie?

        Nee wanneer te weinig fits convergeren -- de QLIKE-reeks is dan een
        selectie van de gemakkelijke vensters -- of wanneer te veel
        geconvergeerde fits op de IGARCH-rand staan, want daar bestaat de
        onvoorwaardelijke variantie niet en is de forecast een random walk in
        variantie.
        """
        return (
            self.convergence_ratio >= cfg.garch.min_convergence_ratio
            and self.boundary_ratio <= cfg.garch.max_boundary_solution_ratio
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "spec": self.spec, "symbol": self.symbol, "n_fits": self.n_fits,
            "n_converged": self.n_converged,
            "convergence_ratio": self.convergence_ratio,
            "n_boundary": self.n_boundary,
            "boundary_ratio": self.boundary_ratio,
            "failure_messages": dict(self.failure_messages),
        }


def summarise_convergence(fits: Sequence[GarchFit]) -> ConvergenceSummary:
    """Tel de fits. Aangeroepen door de campagne en door het H1-rapport."""
    require(
        len(fits) > 0,
        "Convergentiesamenvatting zonder fits.",
        DataContractError,
    )
    specs = {f.spec.label for f in fits}
    symbols = {f.symbol for f in fits}
    require(
        len(specs) == 1 and len(symbols) == 1,
        "Convergentie samengevat over meerdere specificaties of symbolen. De "
        "ratio zou dan een gemiddelde over ongelijksoortige fits zijn, en het "
        "stop-criterium zou een variant kunnen doorlaten die op een symbool "
        "structureel faalt.",
        DataContractError,
        specs=sorted(specs), symbols=sorted(symbols),
    )
    failures: dict[str, int] = {}
    for f in fits:
        if not f.converged:
            failures[f.message] = failures.get(f.message, 0) + 1
    return ConvergenceSummary(
        spec=next(iter(specs)),
        symbol=next(iter(symbols)),
        n_fits=len(fits),
        n_converged=sum(1 for f in fits if f.converged),
        n_boundary=sum(1 for f in fits if f.converged and f.at_boundary),
        failure_messages=failures,
    )


def walk_forward_variance_forecasts(
    returns: pd.Series,
    spec: GarchSpec,
    cv: WalkForwardCV,
    cfg: AdequacyConfig,
    *,
    symbol: str,
    horizon: int,
) -> tuple[pd.Series, list[GarchFit]]:
    """OOS variantieforecasts voor `horizon` bars vooruit, per fold gefit.

    Geeft een reeks op de index van `returns` waarin bar `t` de forecast draagt
    die op bar ``t - horizon`` is gemaakt. Bars zonder forecast -- buiten elk
    testvenster, of uit een niet-geconvergeerde fold -- blijven **NaN**. Er wordt
    niet gevuld en niet teruggevallen; QLIKE laat die bars vallen en het rapport
    telt hoeveel het er waren.

    DE INDEXVERSCHUIVING, OP EEN PLAATS
    ------------------------------------
    `arch` geeft in rij `i` de forecasts die met informatie tot en met bar `i`
    zijn gemaakt, en in kolom ``h-1`` de forecast voor bar ``i + h``. De regel
    ``out.iloc[origin + horizon] = variance.iloc[row, horizon - 1]`` is de enige
    plek waar die verschuiving wordt toegepast. Staat zij er een keer te veel of
    te weinig, dan is het verschil tussen een geldige competitie en een lek --
    en dat is precies waarom hij hier staat en niet verspreid over de
    aanroepers.

    DE EMBARGOZONE KRIJGT GEEN FORECAST
    ------------------------------------
    DEFECT IN MIJN EIGEN WERK, gevonden door
    `test_no_forecast_lands_before_the_first_training_window_ends` en hier
    vastgelegd. De eerste versie vulde elke bar tussen `train_end` en `test_end`.
    Met `embargo_bars=5` eindigt het GEEMBARGEERDE trainvenster echter op bar
    495 terwijl het testvenster pas op 500 begint, en die eerste versie schreef
    dus forecasts op de bars 496-499 -- de embargozone, die tot GEEN van beide
    vensters behoort.

    Dat is geen lookahead: de forecast op bar 496 gebruikt informatie tot en met
    495. Het is iets subtielers en daardoor lastiger te vinden: de
    QLIKE-competitie zou zijn beslecht op een steekproef die bars bevat die de
    walk-forward-structuur bewust heeft uitgesloten, en dat aantal groeit
    lineair met het aantal folds. Op 12 folds met een embargo van 5 zijn dat 60
    bars die in geen enkel testvenster staan -- 5 % van de 1.200 OOS-bars.

    De regel is nu `np.isin(targets, fold.test_idx)`: alleen bars die de fold
    zelf als test heeft aangewezen. De oorsprongsbar MAG in de embargozone
    liggen -- die returns zijn waargenomen en de recursie erop is precies wat
    productie zou doen; het embargo beperkt waarop is GEFIT, niet wat er mag
    worden waargenomen.

    GEVOLG VOOR GROTE HORIZONNEN. De forecast begint op oorsprong `train_end`,
    dus een testbar `t` krijgt alleen een waarde als ``t - horizon >=
    train_end``. Met een embargo van `e` bars blijven de eerste
    ``max(0, horizon - e - 1)`` testbars van elke fold NaN. Voor de
    gepre-registreerde horizonnen h = 1 en h = 5 met `embargo_bars = 5` is dat
    nul en is de dekking volledig; wie een langere horizon toevoegt, moet dat
    verlies rapporteren in plaats van het te vullen.
    """
    require(
        horizon >= 1,
        "Een variantieforecast met horizon onder 1 bar bestaat niet.",
        DataContractError, horizon=horizon, symbol=symbol,
    )
    n = int(returns.size)
    # Een numpy-backing en niet direct een Series: positionele toewijzing op een
    # Series via `.iloc` kan bij een Series-rechterkant op de INDEX uitlijnen in
    # plaats van op de positie, en dat zou de indexverschuiving hieronder stil
    # kunnen verschuiven. Op een ndarray bestaat die dubbelzinnigheid niet.
    values = np.full(n, np.nan, dtype=np.float64)
    fits: list[GarchFit] = []

    for fold in cv.split(n):
        train_end = int(fold.train_idx[-1]) + 1 if fold.train_idx.size else 0
        test_end = int(fold.test_idx[-1]) + 1 if fold.test_idx.size else 0
        if train_end == 0 or test_end <= train_end:
            continue
        # Alleen de reeks tot het einde van DIT testvenster. Zo kan er ook bij
        # een programmeerfout verderop geen informatie van latere folds in de
        # variantierecursie van deze fold terechtkomen.
        window = returns.iloc[:test_end]
        fit = fit_garch_window(
            window, spec, cfg, symbol=symbol, fold_id=fold.fold_id,
            train_end=train_end,
        )
        fits.append(fit)
        if not fit.converged:
            # Geregistreerd, niet gerepareerd. De testbars van deze fold blijven
            # NaN en de fit telt mee in de convergentieratio.
            continue
        variance = fit.forecast_variance(horizon=horizon, start=train_end)
        # Kolom h-1 is de h-staps forecast; rij `i` is gemaakt op bar
        # `train_end + i`. De forecast geldt dus voor `train_end + i + horizon`.
        column = variance.to_numpy(dtype=np.float64)[:, horizon - 1]
        targets = train_end + np.arange(column.size) + horizon
        # Uitsluitend bars die de fold ZELF als test heeft aangewezen. Zie
        # "DE EMBARGOZONE KRIJGT GEEN FORECAST" in de functiedocstring.
        inside = np.isin(targets, fold.test_idx)
        values[targets[inside]] = column[inside]

    require(
        len(fits) > 0,
        "De walk-forward-structuur leverde geen enkele fold op. Controleer "
        "`train_size`, `test_size` en de reekslengte; een competitie zonder "
        "folds is geen leeg resultaat maar een configuratiefout.",
        DataContractError,
        symbol=symbol, spec=spec.label, n_obs=n,
    )
    out = pd.Series(values, index=returns.index, name=f"{spec.name}_h{horizon}")
    return out, fits
