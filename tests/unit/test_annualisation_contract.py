"""De annualisatie heeft één bron, en de consolidatie van fase 10 stap 4A.

`docs/MEASUREMENT_CONTRACT.md` §10.1 legde een openstaand defect vast en wees het
aan deze stap toe: er stonden DRIE annualisaties naast elkaar in de repository.

    365    `conf/backtest/default.yaml` en §1 van het meetcontract
    8760   `backtest/metrics.py::_BARS_PER_YEAR_DEFAULT` (= 365 * 24, uurbars)
    252    `backtest/pbo.py::_sharpe_ratio(annualization=252.0)` (handelsdagen)

Onder AD-22 is elke bar een dagbar. Een aanroeper die de kwarg wegliet, blies
zijn Sharpe met `sqrt(8760/365) = sqrt(24) ≈ 4,9` op.

Dit bestand legt de reparatie vast. Zij is opzettelijk GEEN default-flip van 8760
naar 365: dat zou elke bestaande aanroeper stil van waarde laten veranderen. De
default is verdwenen. Wie `bars_per_year` weglaat, krijgt een `TypeError`.

Hier staan ook de drie andere consolidaties van deze stap (R-3: één
implementatie per statistische grootheid), elk met de eigenschap die zij bewaakt.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest import metrics, pbo
from tradebot.schemas.config import backtest_config
from tradebot.validation import inference, sharpe_difference, vol_metrics

_RETURNS = np.random.default_rng(11).normal(0.0005, 0.01, 500)


# =========================================================================== #
# 1. De annualisatie
# =========================================================================== #
def test_the_single_source_of_bars_per_year_is_the_config() -> None:
    """§1: `bars_per_year = 365`, en dat getal woont in `conf/`."""
    assert backtest_config().bars_per_year == 365.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.sharpe_ratio(_RETURNS),
        lambda: metrics.sortino_ratio(_RETURNS),
        lambda: metrics.annualized_return(_RETURNS),
        lambda: metrics.annualized_vol(_RETURNS),
        lambda: metrics.calmar_ratio(np.cumprod(1.0 + _RETURNS)),
    ],
)
def test_no_metric_annualises_without_being_told_how(call) -> None:
    """§10.1: de default is niet omgezet maar VERWIJDERD.

    Een stille default-flip van 8760 naar 365 zou elke bestaande uitkomst met
    een factor 4,9 verschuiven zonder dat één aanroepplek zichtbaar werd.
    """
    with pytest.raises(TypeError):
        call()


def test_the_defect_was_a_factor_of_almost_five() -> None:
    """Het getal uit §10.1, gemeten in plaats van geciteerd.

    Deze test bestaat om de omvang van het defect vast te leggen: wie vóór deze
    stap `bars_per_year` wegliet, rapporteerde een Sharpe die `sqrt(24)` keer te
    hoog was. Dat is geen afrondingsverschil.
    """
    hourly = metrics.sharpe_ratio(_RETURNS, bars_per_year=365 * 24)
    daily = metrics.sharpe_ratio(_RETURNS, bars_per_year=365)
    assert abs(hourly / daily - math.sqrt(24.0)) < 1e-12
    assert math.sqrt(24.0) > 4.8


def test_the_hourly_and_the_365_25_constants_are_gone() -> None:
    """`CALENDAR_DAYS_PER_YEAR = 365.25` was een VIERDE annualisatie.

    Zij had nul gebruikers maar straalde wel een conventie uit, en 365,25 is
    niet 365. Constanten die een conventie suggereren die het contract niet
    kent, zijn een uitnodiging tot precies dit defect.
    """
    for gone in ("_BARS_PER_YEAR_DEFAULT", "CALENDAR_DAYS_PER_YEAR",
                 "BARS_PER_DAY_HOURLY", "BARS_PER_DAY_5M"):
        assert not hasattr(metrics, gone), gone


# =========================================================================== #
# 2. PBO gebruikt de ene Sharpe, en dat verschuift geen enkel gemeten getal
# =========================================================================== #
def test_pbo_uses_the_one_sharpe_implementation() -> None:
    """`pbo.py` had een eigen `_sharpe_ratio` met 252 handelsdagen (§10.1)."""
    assert not hasattr(pbo, "_sharpe_ratio")
    expected = metrics.sharpe_ratio(_RETURNS, bars_per_year=365.0)
    assert pbo._default_metric(_RETURNS) == pytest.approx(expected)


def test_the_pbo_repair_moves_no_measured_number() -> None:
    """CSCV rangschikt; een positieve monotone herschaling laat elke rang staan.

    Dat is waarom 252 zo lang onopgemerkt kon blijven — en waarom hem
    vervangen door 365 veilig is. Zonder deze test zou de reparatie een
    onbewezen bewering zijn.
    """
    matrix = np.random.default_rng(3).normal(0.0, 0.01, (240, 60))
    with_252 = pbo.compute_pbo(
        matrix, n_subsets=8,
        metric_fn=lambda r: metrics.sharpe_ratio(r, bars_per_year=252.0),
    )
    with_365 = pbo.compute_pbo(matrix, n_subsets=8)
    assert with_252 == with_365


# =========================================================================== #
# 3. Eén Newey-West bandbreedte (R-3)
# =========================================================================== #
def test_vol_metrics_imports_the_bandwidth_instead_of_redefining_it() -> None:
    """§3 plaatst `4 (T/100)^(2/9)` in `inference.py` en nergens anders."""
    assert vol_metrics.newey_west_lags is inference.newey_west_lags
    assert not hasattr(vol_metrics, "_newey_west_lags")
    # De waarde die §3 noemt voor beide vensters van §2.
    assert inference.newey_west_lags(1743) == 7
    assert inference.newey_west_lags(1615) == 7


def test_champion_challenger_imports_the_same_bandwidth() -> None:
    """Fixronde 1, ruling T4A-D: `champion_challenger._dm_test` had een tweede,
    byte-voor-byte identieke `4 (T/100)^(2/9)`-uitdrukking naast (1). Zij is nu
    geïmporteerd in plaats van herhaald -- dezelfde eigenschap als hierboven,
    gepind op het TWEEDE aanroeppunt zodat de regel niet stilletjes weer een
    eigen kopie kan krijgen."""
    from tradebot.compliance import champion_challenger

    assert champion_challenger.newey_west_lags is inference.newey_west_lags
    assert not hasattr(champion_challenger, "_newey_west_lags")


# =========================================================================== #
# 4. Eén Sharpe-verschiltoets, en één bevroren voorganger
# =========================================================================== #
def test_the_frozen_h2_method_says_so_in_its_own_header() -> None:
    """P15: `sharpe_difference.py` wordt niet herschreven maar wel gemarkeerd.

    Zij draagt de methode van een AFGESLOTEN, vooraf geregistreerd experiment.
    Zonder de markering bindt de volgende meting er per ongeluk aan, en dan
    bestaan er twee verschiltoetsen zonder dat iemand koos.
    """
    doc = sharpe_difference.__doc__ or ""
    assert "SUPERSEDED" in doc
    assert "inference.py" in doc
    assert "3d3af28730a6c7f9da48d13139522a05" in doc


def test_the_sharpe_interval_no_longer_has_two_homes() -> None:
    """`metrics.bootstrap_ci` was een DERDE resampling-schema met nul aanroepers."""
    import tradebot.backtest as backtest_pkg

    assert not hasattr(metrics, "bootstrap_ci")
    assert not hasattr(backtest_pkg, "bootstrap_ci")
    assert callable(inference.block_bootstrap_ci)


# =========================================================================== #
# 5. De DSR-handtekening en het drietal van §10
# =========================================================================== #
def test_the_dsr_signature_is_literally_the_one_in_the_contract() -> None:
    """§6, argument voor argument. Ruling P4: de naam is `sr_variance`."""
    parameters = inspect.signature(metrics.deflated_sharpe).parameters
    keyword_only = [
        name
        for name, p in parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY and not name.startswith("_")
    ]
    assert next(iter(parameters)) == "sr_hat"
    assert keyword_only[:6] == [
        "n_obs", "n_trials", "sr_variance", "skew", "kurtosis", "bars_per_year",
    ]


def test_a_serialised_dsr_carries_the_triple_that_section_10_demands() -> None:
    """§10: een Sharpe zonder `(n_obs, bars_per_year, t_years)` is een gerucht.

    `backtest/` importeert niets uit `validation/`, dus `DSRResult.to_dict()`
    roept de poort niet zelf aan. Deze test legt de koppeling wel vast: het
    record dat zij oplevert, moet door de poort komen.
    """
    record = metrics.deflated_sharpe(
        0.0658, n_obs=1815, n_trials=12, sr_variance=1.0 / 1815,
        skew=0.0, kurtosis=3.0, bars_per_year=365.0, approximation="normal",
    ).to_dict()
    inference.require_sharpe_triple(record, where="test")
    assert record["t_years"] == pytest.approx(1815 / 365.0)
    assert record["approximation"] == "normal"


# =========================================================================== #
# 6. `clustered_mean`'s optionele drietal (fixronde 1, item 1, ruling T4A-B)
# =========================================================================== #
def test_clustered_mean_carries_no_triple_when_bars_per_year_is_omitted() -> None:
    """De brief mandateert zelf `clustered_mean(panel)` zonder annualisatie
    (`tests/unit/test_inference.py:108`); dat gedrag moet exact blijven staan."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    panel = pd.DataFrame(
        {f"S{i}": rng.normal(0.001, 0.01, 40) for i in range(4)}, index=idx
    )
    record = inference.clustered_mean(panel).to_dict()
    assert "bars_per_year" not in record
    assert "t_years" not in record


def test_clustered_mean_carries_the_triple_when_bars_per_year_is_given() -> None:
    """Plan §9 scoopt het verplichte drietal tot Sharpes; `clustered_mean` is
    er geen, dus het drietal is hier OPTIONEEL. Meegegeven, dan draagt
    `to_dict()` het volledig, inclusief het afgeleide `t_years`."""
    rng = np.random.default_rng(43)
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    panel = pd.DataFrame(
        {f"S{i}": rng.normal(0.001, 0.01, 40) for i in range(4)}, index=idx
    )
    result = inference.clustered_mean(panel, bars_per_year=365.0)
    record = result.to_dict()
    assert record["bars_per_year"] == 365.0
    assert record["n_obs"] == result.n_obs
    assert record["t_years"] == pytest.approx(result.n_obs / 365.0)
