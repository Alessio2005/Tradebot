"""De signaalklok: hoe vaak een besluitpaneel werkelijk iets nieuws zegt. Stap 3.

De grondwaarheid is analytisch. Een AR(1) met coëfficiënt phi heeft een
geïntegreerde autocorrelatietijd `(1 + phi) / (1 - phi)`: 19 bij phi = 0,9. Een
paneel dat per blok van k bars constant is, heeft er ongeveer k. Een paneel dat
nooit verandert, heeft er geen eindige: dat is de toestand van de primaire cel
van H-10.1, en de schatter moet dat zeggen in plaats van een getal te verzinnen.

Negatieve controle, met de hand gedraaid en niet gecommit: een schatter die
alleen lag 1 meeneemt, geeft bij phi = 0,9 ongeveer 2,8 en maakt
`test_an_ar1_recovers_its_analytic_time` rood.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.signal_clock import (
    IactResult,
    decision_panel_clock,
    integrated_autocorrelation_time,
)

WINDOW_C = 5.0
MAX_LAG = 400


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.standard_normal()
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    return x


def test_an_ar1_recovers_its_analytic_time() -> None:
    tau = integrated_autocorrelation_time(_ar1(0.9, 200_000, 1), window_c=WINDOW_C,
                                          max_lag=MAX_LAG)
    assert tau.tau == pytest.approx(19.0, rel=0.10)
    assert tau.window_reached


def test_white_noise_is_one() -> None:
    rng = np.random.default_rng(2)
    tau = integrated_autocorrelation_time(rng.standard_normal(100_000),
                                          window_c=WINDOW_C, max_lag=MAX_LAG)
    assert tau.tau == pytest.approx(1.0, abs=0.1)


def test_a_constant_series_has_no_finite_time() -> None:
    tau = integrated_autocorrelation_time(np.full(500, 1 / 6), window_c=WINDOW_C,
                                          max_lag=MAX_LAG)
    assert math.isinf(tau.tau)


def test_a_window_that_is_not_reached_is_reported_as_a_lower_bound() -> None:
    tau = integrated_autocorrelation_time(_ar1(0.99, 50_000, 3), window_c=WINDOW_C,
                                          max_lag=50)
    assert not tau.window_reached


def test_bad_parameters_are_refused() -> None:
    with pytest.raises(DataContractError):
        integrated_autocorrelation_time(np.arange(10.0), window_c=0.0, max_lag=5)
    with pytest.raises(DataContractError):
        integrated_autocorrelation_time(np.array([1.0, np.nan, 2.0]), window_c=5.0,
                                        max_lag=5)


def _blocks(k: int, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    levels = rng.uniform(-1, 1, size=(n // k + 1, 3))
    values = np.repeat(levels, k, axis=0)[:n]
    return pd.DataFrame(values, index=pd.date_range("2022-01-01", periods=n,
                                                    freq="D", tz="UTC"),
                        columns=list("ABC"))


def test_a_constant_panel_makes_zero_independent_decisions() -> None:
    panel = pd.DataFrame(np.full((300, 6), 1 / 6), columns=list("ABCDEF"),
                         index=pd.date_range("2022-01-01", periods=300, freq="D",
                                             tz="UTC"))
    turnover = panel.diff().abs().sum(axis=1).fillna(1.0)
    clock = decision_panel_clock(panel, turnover=turnover, window_c=WINDOW_C,
                                 max_lag=MAX_LAG, bars_per_year=365.0)
    assert clock.unique_rows == 1
    assert clock.bars_with_change == 0
    assert math.isinf(clock.tau_int_median)
    assert clock.independent_decisions_per_year == 0.0


def test_a_blockwise_panel_has_a_clock_near_its_block() -> None:
    panel = _blocks(20, 20_000, 4)
    turnover = panel.diff().abs().sum(axis=1).fillna(0.0)
    clock = decision_panel_clock(panel, turnover=turnover, window_c=WINDOW_C,
                                 max_lag=MAX_LAG, bars_per_year=365.0)
    assert 14.0 <= clock.tau_int_median <= 26.0
    assert clock.bars_with_change == pytest.approx(20_000 / 20, rel=0.02)
    assert clock.independent_decisions_per_year == pytest.approx(
        365.0 / clock.tau_int_median)


def _results(taus: list[float], truncated: set[int]) -> list[IactResult]:
    return [IactResult(tau=t, lags_used=200, window_reached=i not in truncated)
            for i, t in enumerate(taus)]


@pytest.mark.parametrize(("truncated", "determined"), [
    (set(), True),     # niets afgekapt
    ({5}, True),       # de hoogste is een ondergrens: de mediaan hangt er niet van af
    ({4, 5}, True),
    ({3}, False),      # de bovenste mediaannaam zelf is een ondergrens
    ({0}, False),      # een ondergrens onder de mediaan kan haar verschuiven
])
def test_the_median_is_determined_only_when_every_lower_bound_lies_above_it(
    truncated: set[int], determined: bool,
) -> None:
    from tradebot.validation.signal_clock import median_is_determined

    results = _results([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], truncated)
    assert median_is_determined(results) is determined


def test_the_panel_clock_reports_every_name_and_whether_its_median_holds() -> None:
    panel = _blocks(20, 5_000, 4)
    turnover = panel.diff().abs().sum(axis=1).fillna(0.0)
    clock = decision_panel_clock(panel, turnover=turnover, window_c=WINDOW_C,
                                 max_lag=MAX_LAG, bars_per_year=365.0)
    assert sorted(clock.tau_int_by_name) == sorted(panel.columns)
    assert clock.tau_median_determined is True
    assert clock.tau_int_median == pytest.approx(
        float(np.median(list(clock.tau_int_by_name.values()))))
