# src/tradebot/backtest/metrics.py
"""Portfolio performance metrics — Sharpe, Calmar, MaxDD, Deflated Sharpe.

All functions are pure-NumPy; no pandas dependency.  For DataFrame
convenience use the ``from_series`` class methods on the result objects.

================================================================================
FASE 10, STAP 4A: DE ANNUALISATIE HEEFT GEEN DEFAULT MEER

`docs/MEASUREMENT_CONTRACT.md` §10.1 legde een openstaand defect vast en wees
het aan deze stap toe: dit bestand droeg
``_BARS_PER_YEAR_DEFAULT = 365 * 24`` (= 8760, uurbars). Onder AD-22 is elke bar
een DAGbar en §1 van het contract eist 365. Elke aanroeper die de kwarg wegliet,
verschaalde zijn Sharpe met ``sqrt(8760 / 365) = sqrt(24) ≈ 4,9``.

De reparatie is NIET "zet de default op 365". Dat zou de uitkomst van elke
bestaande aanroeper stil van waarde laten veranderen — precies wat §10.1
"een stille default-flip" noemt en verbiedt. De reparatie is: **er is geen
default meer.** `bars_per_year` is op elke functie hier een VERPLICHT
keyword-argument. Wie hem weglaat, krijgt een `TypeError` en geen getal.

De ene bron van de waarde is `conf/backtest/default.yaml`
(`bars_per_year: 365.0`), bereikbaar via
`tradebot.schemas.config.backtest_config()`. Dit bestand herhaalt haar niet:
een tweede plaats waar 365 staat, is een tweede annualisatie in wording.

Wat hier ook is verdwenen, en waarom:
    * `CALENDAR_DAYS_PER_YEAR = 365.25`, `BARS_PER_DAY_HOURLY`, `BARS_PER_DAY_5M`
      — nul gebruikers in de hele repository, en 365,25 is een VIERDE
      annualisatie naast de drie die §10.1 telt. Constanten die niemand
      gebruikt maar die wel een conventie uitstralen, zijn een uitnodiging.
    * `bootstrap_ci` — nul aanroepplekken, en een DERDE resampling-schema
      (niet-overlappende blokken van vaste lengte) naast de stationaire
      bootstraps in `spa.py`/`evaluation.py` en de circulaire in
      `validation/inference.py`. R-3: één implementatie per statistische
      grootheid. Het Sharpe-interval woont nu in
      `validation.inference.block_bootstrap_ci`.

Conventies die blijven gelden:
    - Annualisatie:  vermenigvuldig de mean/std-verhouding met sqrt(bars_per_year)
    - Zero-padding:  neem nul-returnbars ALTIJD mee (kalendertijd-Sharpe =
                     Sharpe van de live equity-curve).
================================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..utils.failfast import DataContractError, require

__all__ = [
    "DSRResult",
    "annualized_return",
    "annualized_vol",
    "calmar_ratio",
    "deflated_sharpe",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
]


def annualized_return(returns: np.ndarray, *, bars_per_year: float) -> float:
    """Compound annualised return.

    Parameters
    ----------
    returns : 1-D array of bar-level returns (decimal, e.g. 0.002 = 0.2 %).
    bars_per_year :
        VERPLICHT en zonder default (MEASUREMENT_CONTRACT.md §1 en §10.1). De
        waarde komt uit `conf/backtest/default.yaml`.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    compound = float(np.prod(1.0 + r))
    years = max(r.size / bars_per_year, 1e-9)
    return float(compound ** (1.0 / years) - 1.0)


def annualized_vol(returns: np.ndarray, *, bars_per_year: float) -> float:
    """Annualised standard deviation of bar-level returns."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * math.sqrt(bars_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    *,
    risk_free: float = 0.0,
    bars_per_year: float,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : bar-level returns.
    risk_free : annual risk-free rate (decimal).
    bars_per_year :
        VERPLICHT en zonder default (MEASUREMENT_CONTRACT.md §1 en §10.1).

    Notes
    -----
    Dit is de PUNTSCHATTING. Haar standaardfout staat in
    `validation.inference.sharpe_with_se` (Lo 2002 met Newey-West) en nergens
    anders; §3 van het meetcontract verbiedt `1/sqrt(T)` als toets.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    rf_bar = (1.0 + risk_free) ** (1.0 / bars_per_year) - 1.0
    excess = r - rf_bar
    sigma = float(np.std(excess, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(excess) / sigma * math.sqrt(bars_per_year))


def sortino_ratio(
    returns: np.ndarray,
    *,
    risk_free: float = 0.0,
    bars_per_year: float,
) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    rf_bar = (1.0 + risk_free) ** (1.0 / bars_per_year) - 1.0
    excess = r - rf_bar
    downside = excess[excess < 0.0]
    if downside.size < 2:
        return 0.0
    semi_sigma = float(np.std(downside, ddof=1))
    if semi_sigma < 1e-12:
        return 0.0
    return float(np.mean(excess) / semi_sigma * math.sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Maximum drawdown of an equity curve.

    Parameters
    ----------
    equity : 1-D array of equity values (must be > 0).

    Returns
    -------
    (max_dd, peak_idx, trough_idx)
        max_dd      : maximum drawdown as a positive fraction (0.35 = 35 %).
        peak_idx    : index of the equity peak.
        trough_idx  : index of the trough after the peak.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / np.maximum(running_max, 1e-12)
    idx = int(np.argmax(dd))
    peak_idx = int(np.argmax(eq[:idx + 1]))
    return float(dd[idx]), peak_idx, idx


def calmar_ratio(
    equity: np.ndarray,
    *,
    bars_per_year: float,
) -> float:
    """Calmar ratio = annualised return / max drawdown."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    ann_ret = annualized_return(returns, bars_per_year=bars_per_year)
    mdd, _, _ = max_drawdown(eq)
    if mdd < 1e-9:
        return 0.0
    return float(ann_ret / mdd)



# =========================================================================== #
# De Deflated Sharpe Ratio — MEASUREMENT_CONTRACT.md §6
# =========================================================================== #
#: Sentinel voor "niet meegegeven". Een object en geen getal, zodat een
#: ontbrekend moment niet per ongeluk als waarde kan doorrekenen. Dit is de
#: hele reden dat de oude handtekening een defect was: haar defaults
#: (`sr_benchmark=0.0, returns_skew=0.0, returns_kurt=3.0`) rekenden stilzwijgend
#: door met de normale benadering.
_UNSET: Any = object()

#: Euler-Mascheroni.
_GAMMA_EM = 0.5772156649015329

_SIGNATURE_HINT = (
    "deflated_sharpe(sr_hat, *, n_obs, n_trials, sr_variance, skew, kurtosis, "
    "bars_per_year) -> DSRResult"
)


@dataclass(frozen=True)
class DSRResult:
    """Een DSR met alles wat nodig is om hem te herhalen en te beoordelen.

    De DSR was tot fase 10 een kaal getal. Een kaal getal draagt niet welke
    `M` eronder lag, met welke variantie van de trial-Sharpes is gedefleerd, of
    die variantie is GEMETEN dan wel BENADERD. Precies die drie dingen bepalen
    hoe streng de toets was.
    """

    #: `Phi(z)` — de kans dat de waargenomen Sharpe de verwachte maximum-Sharpe
    #: onder de nulhypothese van geen skill overtreft. Slagen is `dsr > 1 - alpha`.
    dsr: float
    #: De waargenomen Sharpe, PER BAR. De afleiding van Bailey-Lopez de Prado
    #: staat op de per-observatie Sharpe; er een geannualiseerd getal in stoppen
    #: schaalt de teller wel en de noemer niet.
    sr_hat: float
    #: `SR_0` — de verwachte maximum-Sharpe onder de nul, in dezelfde
    #: per-bar-eenheden als `sr_hat`.
    sr_zero: float
    #: De gestandaardiseerde afstand waarvan `dsr` de normale CDF is.
    z: float
    n_obs: int
    #: `M`, het aantal beproefde trials. De familiecorrectie loopt UITSLUITEND
    #: hierlangs (§6): geen Benjamini-Hochberg, geen Bonferroni eroverheen.
    n_trials: int
    #: `V[{SR_m}]`, de variantie van de trial-Sharpes.
    sr_variance: float
    skew: float
    kurtosis: float
    bars_per_year: float
    t_years: float
    #: `"empirical"` als `sr_variance` uit de trial-Sharpes is GEMETEN,
    #: `"normal"` als de gedocumenteerde benadering `1/n_obs` is gebruikt. §6
    #: eist dat feit IN HET ARTEFACT. Een benadering die in een JSON staat, is
    #: een keuze; een benadering die in een default staat, is een aanname.
    approximation: str

    def to_dict(self) -> dict[str, Any]:
        """De serialiseerbare vorm, inclusief het drietal uit §10.

        Het record draagt `(n_obs, bars_per_year, t_years)` per constructie en
        wordt daarom geaccepteerd door
        `validation.inference.require_sharpe_triple`. Die poort wordt hier niet
        AANGEROEPEN: `backtest/` importeert nergens uit `validation/`, en deze
        stap voegt die afhankelijkheidsrichting niet toe om één assertie te
        winnen. `tests/unit/test_annualisation_contract.py` legt de koppeling
        wel vast.
        """
        return {
            "dsr": self.dsr,
            "sharpe_per_bar": self.sr_hat,
            "sr_zero": self.sr_zero,
            "z": self.z,
            "n_obs": int(self.n_obs),
            "n_trials": int(self.n_trials),
            "sr_variance": self.sr_variance,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "bars_per_year": self.bars_per_year,
            "t_years": self.t_years,
            "approximation": self.approximation,
        }


def _refuse_legacy_call(
    n_extra_positional: int,
    unknown_keywords: tuple[str, ...],
    missing: tuple[str, ...],
) -> None:
    """Weiger elke aanroepvorm die niet die van §6 is, met de reden erbij.

    MEASUREMENT_CONTRACT.md §6 en stap 4A.4: de oude positionele vorm
    ``deflated_sharpe(sr, n_trials, n_obs, sr_benchmark, returns_skew,
    returns_kurt)`` wordt NIET stilzwijgend verwijderd. Zij gooit, zodat elke
    bestaande aanroepplek zichtbaar wordt in plaats van door te rekenen met de
    normale benadering die in haar defaults verstopt zat.
    """
    problems: list[str] = []
    if n_extra_positional:
        problems.append(
            f"{n_extra_positional + 1} positionele argumenten; alleen `sr_hat` "
            f"is positioneel (de oude vorm was "
            f"`deflated_sharpe(sr, n_trials, n_obs, ...)`)"
        )
    if unknown_keywords:
        problems.append(
            f"onbekende argumenten {list(unknown_keywords)}; de oude namen "
            f"`sr_observed`, `sr_benchmark`, `returns_skew` en `returns_kurt` "
            f"bestaan niet meer"
        )
    if missing:
        problems.append(
            f"ontbrekende verplichte argumenten {list(missing)}; zij hebben "
            f"bewust geen default, want een default is hier een verborgen "
            f"aanname"
        )
    if not problems:
        return
    raise TypeError(
        "deflated_sharpe kreeg een aanroepvorm die niet die van het "
        "meetcontract is: "
        + "; ".join(problems)
        + ". De handtekening is `"
        + _SIGNATURE_HINT
        + "`. Zie docs/MEASUREMENT_CONTRACT.md §6: de DSR heeft V[{SR_m}], de "
        "scheefheid en de kurtosis NODIG, en een aanroep zonder die argumenten "
        "mag niet stilzwijgend de normale benadering pakken. Is de variantie "
        "van de trial-Sharpes niet beschikbaar, geef dan expliciet "
        "sr_variance=1.0/n_obs en approximation='normal' mee, zodat de "
        "benadering in het artefact belandt in plaats van in een default."
    )


def deflated_sharpe(
    sr_hat: float = _UNSET,
    *_legacy_positional: object,
    n_obs: int = _UNSET,
    n_trials: int = _UNSET,
    sr_variance: float = _UNSET,
    skew: float = _UNSET,
    kurtosis: float = _UNSET,
    bars_per_year: float = _UNSET,
    approximation: str | None = None,
    **_legacy_keyword: object,
) -> DSRResult:
    """Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014), §6 van het meetcontract.

    ::

        SR_0 = sqrt(sr_variance) * [ (1 - gamma) Phi^-1(1 - 1/M)
                                     + gamma Phi^-1(1 - 1/(M e)) ]

        DSR  = Phi( (SR_hat - SR_0) sqrt(T - 1)
                    / sqrt(1 - g3 SR_hat + ((g4 - 1)/4) SR_hat^2) )

    Parameters
    ----------
    sr_hat
        De waargenomen Sharpe, PER BAR en niet geannualiseerd.
    n_obs
        `T`, het aantal observaties waarop `sr_hat` is gemeten.
    n_trials
        `M`, het EERLIJKE aantal beproefde varianten. De familiecorrectie loopt
        uitsluitend hierlangs; wie strenger wil zijn, verhoogt `M` (§6).
    sr_variance
        `V[{SR_m}]`, de EMPIRISCHE variantie van de trial-Sharpes wanneer die
        beschikbaar is. Is zij dat niet, geef dan de gedocumenteerde benadering
        `1/n_obs` mee, met `approximation="normal"`.
    skew, kurtosis
        De derde en vierde gestandaardiseerde momenten van de returnreeks.
        GEMETEN, niet aangenomen: crypto-returns zijn scheef en dikstaartig, en
        de Gaussische aanname (0 en 3) maakt de noemer te klein en de toets dus
        te soepel.
    bars_per_year
        Draagt het drietal `(n_obs, bars_per_year, t_years)` van §10 het
        resultaat in. De DSR-formule zelf gebruikt hem NIET: zij staat op de
        per-bar Sharpe. Hij is verplicht omdat een DSR zonder venster niet te
        vergelijken is met de t-drempel die bij dat venster hoort.
    approximation
        `"normal"` of `"empirical"`. `None` = afleiden uit `sr_variance`: is zij
        exact `1/n_obs`, dan IS dat de normale benadering. Zo kan de vlag niet
        worden vergeten. Wordt hij WEL expliciet meegegeven, dan moet hij het
        met die afleiding EENS zijn (fixronde 1, item 7): `approximation="normal"`
        bij een `sr_variance` die niet `1/n_obs` is, raist -- anders zou het
        label in het artefact een empirische variantie als de gedocumenteerde
        benadering vermommen, en dat is precies de mislabeling die §6 verbiedt.

    Raises
    ------
    TypeError
        Bij de oude positionele vorm, bij de oude argumentnamen, en bij elke
        aanroep waarin een van de zeven verplichte argumenten ontbreekt. Dat is
        opzet: stap 4A.4 maakt elke bestaande aanroepplek zichtbaar.
    """
    _refuse_legacy_call(
        len(_legacy_positional),
        tuple(sorted(_legacy_keyword)),
        tuple(
            name
            for name, value in (
                ("sr_hat", sr_hat),
                ("n_obs", n_obs),
                ("n_trials", n_trials),
                ("sr_variance", sr_variance),
                ("skew", skew),
                ("kurtosis", kurtosis),
                ("bars_per_year", bars_per_year),
            )
            if value is _UNSET
        ),
    )

    from scipy import stats  # lazy import — scipy is in core deps

    trials = int(n_trials)
    obs = int(n_obs)
    variance = float(sr_variance)
    sr = float(sr_hat)
    require(
        trials >= 2,
        f"De DSR is niet gedefinieerd bij M={trials}. Het verwachte maximum van "
        f"één trial is die trial zelf, en Phi^-1(1 - 1/M) loopt bij M = 1 naar "
        f"-oneindig. Een deflatie over één hypothese is geen deflatie; zie "
        f"docs/MEASUREMENT_CONTRACT.md §6.",
        DataContractError,
        n_trials=trials,
    )
    require(
        obs >= 2,
        f"De DSR vereist minstens twee observaties, kreeg {obs}: de factor "
        f"sqrt(T - 1) is anders nul of imaginair.",
        DataContractError,
        n_obs=obs,
    )
    require(
        variance > 0.0,
        f"sr_variance={variance} is niet positief. V[SR_m] is de spreiding van "
        f"de trial-Sharpes; nul spreiding betekent dat elke trial dezelfde "
        f"Sharpe gaf, en dan is er niets om voor te defleren.",
        DataContractError,
        sr_variance=variance,
    )
    require(
        float(bars_per_year) > 0.0,
        "bars_per_year moet positief zijn; annualiseren met nul bestaat niet.",
        DataContractError,
        bars_per_year=float(bars_per_year),
    )

    normal_approximation = abs(variance - 1.0 / obs) <= 1e-12 / obs
    inferred_label = "normal" if normal_approximation else "empirical"
    if approximation is None:
        label = inferred_label
    else:
        label = str(approximation)
        require(
            label in ("normal", "empirical"),
            f"approximation={label!r} bestaat niet; toegestaan zijn 'normal' "
            f"(de gedocumenteerde benadering V[SR_m] = 1/n_obs) en 'empirical' "
            f"(gemeten uit de trial-Sharpes). Zie "
            f"docs/MEASUREMENT_CONTRACT.md §6.",
            DataContractError,
            approximation=label,
        )
        # FIXRONDE 1, ITEM 7 (RULING T4A-G): dezelfde tegenspraak-toets die de
        # AFGELEIDE vlag hierboven al draait (`approximation=None`), nu ook op
        # het EXPLICIETE pad. Zonder deze toets accepteerde
        # `deflated_sharpe(..., sr_variance=4.0/n, approximation="normal")` een
        # empirische variantie onder het label van de gedocumenteerde
        # benadering -- precies het label-defect in een meetrecord dat §6
        # verbiedt.
        require(
            label == inferred_label,
            f"approximation={label!r} is in tegenspraak met sr_variance={variance!r}: "
            f"bij n_obs={obs} is de gedocumenteerde normale benadering exact "
            f"1/n_obs={1.0 / obs!r}, en sr_variance ligt daar hier "
            f"{'wel' if inferred_label == 'normal' else 'niet'} op. Een label dat "
            f"niet bij de meegegeven variantie hoort, is de mislabeling die §6 "
            f"van docs/MEASUREMENT_CONTRACT.md juist wil voorkomen -- geef "
            f"approximation={inferred_label!r} mee, of geef de variantie mee die "
            f"bij {label!r} hoort.",
            DataContractError,
            approximation=label, inferred_label=inferred_label,
            sr_variance=variance, n_obs=obs,
        )

    # Het verwachte maximum van M standaardnormalen (Bailey-Lopez de Prado 2014,
    # eq. 8), in Z-EENHEDEN. `sqrt(sr_variance)` zet het om naar Sharpe-eenheden;
    # dat is de dimensionaliteitsfix van 2026-06-08, die hier bewaard blijft
    # doordat §6 hem in de formule zelf heeft opgenomen.
    e_max_z = (1.0 - _GAMMA_EM) * stats.norm.ppf(1.0 - 1.0 / trials) + _GAMMA_EM * (
        stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
    )
    sr_zero = math.sqrt(variance) * float(e_max_z)

    denominator_squared = (
        1.0 - float(skew) * sr + ((float(kurtosis) - 1.0) / 4.0) * sr * sr
    )
    require(
        denominator_squared > 0.0,
        f"De variantie van de Sharpe-schatter is niet positief "
        f"(1 - g3 SR + ((g4-1)/4) SR^2 = {denominator_squared:.6g} bij "
        f"SR={sr:.6g}, skew={float(skew):.4g}, kurtosis={float(kurtosis):.4g}). "
        f"Dat gebeurt bij een extreem scheve reeks met een hoge Sharpe; de "
        f"asymptotiek van Bailey-Lopez de Prado geldt daar niet en een DSR die "
        f"er tóch uitrolt, is een verzinsel.",
        DataContractError,
        skew=float(skew), kurtosis=float(kurtosis), sharpe_per_bar=sr,
    )
    z = (sr - sr_zero) * math.sqrt(obs - 1) / math.sqrt(denominator_squared)
    return DSRResult(
        dsr=float(stats.norm.cdf(z)),
        sr_hat=sr,
        sr_zero=float(sr_zero),
        z=float(z),
        n_obs=obs,
        n_trials=trials,
        sr_variance=variance,
        skew=float(skew),
        kurtosis=float(kurtosis),
        bars_per_year=float(bars_per_year),
        t_years=obs / float(bars_per_year),
        approximation=label,
    )
