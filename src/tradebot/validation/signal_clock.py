# src/tradebot/validation/signal_clock.py
"""De signaalklok — hoe vaak een besluitpaneel werkelijk iets nieuws zegt. Fase 11.

TEMPORELE BREEDTE
=================
De breedte uit de fundamentele wet is het aantal ONAFHANKELIJKE besluiten per
jaar. Een paneel dat elke bar herschikt maar waarvan de gewichten over dertig
bars nauwelijks veranderen, neemt geen 365 besluiten per jaar maar ongeveer
twaalf. Die klok wordt hier gemeten op het besluitpaneel zelf, zonder één
rendement: de geïntegreerde autocorrelatietijd `tau_int` van de gewichten, en
`bars_per_year / tau_int` onafhankelijke besluiten per jaar.

DE SCHATTER
===========
`tau_int = 1 + 2 * sum_{h=1..M} rho(h)`, met het automatische venster van Sokal:
M is de kleinste lag waarvoor `M >= c * tau_int(M)`. Wordt dat venster binnen
`max_lag` niet bereikt, dan is de som afgekapt terwijl de autocorrelaties nog
positief zijn; het getal is dan een ONDERGRENS, en het resultaat zegt dat
(`window_reached = False`). Beide parameters komen uit
`conf/research/breadth.yaml` en zijn gezet vóór de eerste meting.

Een reeks zonder variatie heeft geen eindige autocorrelatietijd: zij zegt nooit
iets nieuws. De schatter geeft dan `inf` en deelt niet door nul. Dat is precies
de primaire cel van H-10.1: één unieke gewichtsrij, temporele breedte nul.

OMZET
=====
Omzet wordt hier niet opnieuw uitgerekend. De aanroeper geeft de reeks mee uit
`backtest/vectorized.py::run_vectorized` — de enige definitie (R-3).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = [
    "ClockRow",
    "IactResult",
    "decision_panel_clock",
    "integrated_autocorrelation_time",
    "median_is_determined",
]


@dataclass(frozen=True)
class IactResult:
    """Een geïntegreerde autocorrelatietijd, met de lag waarop is gestopt."""

    tau: float
    lags_used: int
    #: False: het Sokal-venster is binnen `max_lag` niet bereikt; `tau` is dan
    #: een ondergrens.
    window_reached: bool


def _autocorrelations(x: np.ndarray[Any, Any], max_lag: int) -> np.ndarray[Any, Any]:
    centred = x - x.mean()
    n = centred.size
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, n=size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[: max_lag + 1]
    return np.asarray(acov / acov[0], dtype=np.float64)


def integrated_autocorrelation_time(
    series: np.ndarray[Any, Any], *, window_c: float, max_lag: int
) -> IactResult:
    """`tau_int` van één reeks met het automatische venster van Sokal."""
    x = np.asarray(series, dtype=np.float64)
    require(
        window_c > 0 and max_lag >= 1,
        "De vensterconstante en de maximale lag moeten positief zijn; zij komen "
        "uit conf/research/breadth.yaml.",
        DataContractError, window_c=window_c, max_lag=max_lag,
    )
    require(
        x.ndim == 1 and x.size > max_lag and bool(np.isfinite(x).all()),
        "De reeks moet eindig zijn en langer dan de maximale lag. Een ontbrekend "
        "gewicht is geen nul en wordt hier niet ingevuld.",
        DataContractError, size=int(x.size), max_lag=int(max_lag),
    )
    if float(np.ptp(x)) == 0.0:  # exact constant; np.var geeft floatruis (1/6 * n)
        return IactResult(tau=math.inf, lags_used=0, window_reached=True)
    rho = _autocorrelations(x, int(max_lag))
    tau = 1.0
    for lag in range(1, int(max_lag) + 1):
        tau += 2.0 * float(rho[lag])
        if lag >= window_c * tau:
            return IactResult(tau=tau, lags_used=lag, window_reached=True)
    return IactResult(tau=tau, lags_used=int(max_lag), window_reached=False)


@dataclass(frozen=True)
class ClockRow:
    """De klok van één besluitpaneel."""

    n_bars: int
    unique_rows: int
    bars_with_change: int
    mean_turnover: float
    turnover_per_year: float
    ac1_median: float | None
    tau_int_median: float
    #: False wanneer bij minstens één naam het venster niet is bereikt.
    tau_window_reached_all: bool
    #: True wanneer de mediaan niet van een ondergrens afhangt (zie
    #: `median_is_determined`). Dit, en niet `tau_window_reached_all`, beslist
    #: of de mediaan een meting is.
    tau_median_determined: bool
    tau_int_by_name: dict[str, float]
    window_reached_by_name: dict[str, bool]
    independent_decisions_per_year: float
    sign_flips_per_name_per_year: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def median_is_determined(results: list[IactResult]) -> bool:
    """Hangt de mediaan van `tau` af van een naam waarvan `tau` een ondergrens is?

    Een afgekapte `tau` is een ondergrens: de werkelijke waarde ligt er op of
    boven. Ligt zo'n ondergrens STRIKT boven de bovenste naam die de mediaan
    bepaalt (bij een even aantal de hoogste van de middelste twee), dan ligt de
    werkelijke waarde daar ook boven, en schuift de mediaan niet. In elk ander
    geval kan zij schuiven, en is de mediaan zelf een ondergrens.
    """
    if not results:
        return True
    upper_median = float(np.sort([r.tau for r in results])[len(results) // 2])
    return all(r.window_reached or r.tau > upper_median for r in results)


def decision_panel_clock(
    weights: pd.DataFrame,
    *,
    turnover: pd.Series,
    window_c: float,
    max_lag: int,
    bars_per_year: float,
) -> ClockRow:
    """De klok van een gewichtspaneel. `turnover` komt van `run_vectorized` (R-3)."""
    values = weights.to_numpy(dtype=np.float64)
    require(
        bool(np.isfinite(values).all()) and turnover.index.equals(weights.index),
        "Het gewichtspaneel moet volledig zijn en de omzetreeks moet op dezelfde "
        "bars liggen.",
        DataContractError,
    )
    moving = [c for c in weights.columns if float(np.ptp(weights[c].to_numpy())) > 0.0]
    taus = [integrated_autocorrelation_time(weights[c].to_numpy(), window_c=window_c,
                                            max_lag=max_lag) for c in moving]
    tau_median = float(np.median([t.tau for t in taus])) if taus else math.inf
    years = len(weights) / float(bars_per_year)
    signs = np.sign(values)
    flips = int((signs[1:] != signs[:-1]).sum())
    changed = (np.abs(np.diff(values, axis=0)).sum(axis=1) > 0.0)
    return ClockRow(
        n_bars=int(len(weights)),
        unique_rows=int(len(np.unique(values, axis=0))),
        bars_with_change=int(changed.sum()),
        mean_turnover=float(turnover.mean()),
        turnover_per_year=float(turnover.mean() * float(bars_per_year)),
        ac1_median=float(np.median([weights[c].autocorr(1) for c in moving]))
        if moving else None,
        tau_int_median=tau_median,
        tau_window_reached_all=all(t.window_reached for t in taus),
        tau_median_determined=median_is_determined(taus),
        tau_int_by_name={str(c): float(t.tau) for c, t in zip(moving, taus, strict=True)},
        window_reached_by_name={str(c): bool(t.window_reached)
                                for c, t in zip(moving, taus, strict=True)},
        independent_decisions_per_year=0.0 if math.isinf(tau_median)
        else float(bars_per_year) / tau_median,
        sign_flips_per_name_per_year=float(flips / weights.shape[1] / years),
    )
