# tests/unit/test_vol_forecast_monitor.py
"""De live vol-forecastmonitor. Stage D-3.

De twee eigenschappen die er hier toe doen:

* **te weinig data is geen groen licht.** Een monitor die zwijgt omdat hij niets
  weet, mag niet worden gelezen als een monitor die in orde staat. Vandaar
  `is_conclusive` naast `is_breach`;
* **QLIKE en Mincer-Zarnowitz vangen verschillende dingen.** Een estimator die
  structureel te laag voorspelt kan een acceptabele QLIKE houden en toch elke
  positie te groot maken, want vol-targeting deelt door precies dat getal. De
  MZ-toets moet die vertekening zien terwijl QLIKE hem doorlaat.

Ref: fase-opdracht Stage D-3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.monitoring.vol_forecast_monitor import (
    MZ_BIASED,
    QLIKE_DEGRADED,
    VolForecastMonitor,
)
from tradebot.schemas.config import monitoring_config
from tradebot.utils.failfast import DataContractError

CFG = monitoring_config()
N = CFG.vol_forecast_min_obs + 50


def _series(seed: int = 7, n: int = N) -> np.ndarray:
    """Een plausibele reeks gerealiseerde varianties."""
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(-9.0, 0.5, n))


def _fed(monitor: VolForecastMonitor, rv: np.ndarray,
         forecast: np.ndarray) -> VolForecastMonitor:
    for r, f in zip(rv, forecast, strict=True):
        monitor.record(r, f)
    return monitor


class TestSilenceIsNotGreen:
    def test_below_the_minimum_there_is_no_verdict(self) -> None:
        monitor = VolForecastMonitor(baseline_qlike=0.5)
        rv = _series(n=CFG.vol_forecast_min_obs - 1)
        verdict = _fed(monitor, rv, rv).verdict()
        assert not verdict.is_conclusive
        assert not verdict.is_breach
        assert verdict.n_obs == CFG.vol_forecast_min_obs - 1

    def test_above_the_minimum_there_is_one(self) -> None:
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.5), rv, rv).verdict()
        assert verdict.is_conclusive

    def test_a_degenerate_forecast_is_not_a_clean_bill_of_health(self) -> None:
        """Genoeg BARS, maar geen enkele bruikbare QLIKE-term.

        `qlike` geeft NaN zodra de forecast niet strikt positief is. Een
        estimator die live naar nul ontspoort, leverde daarmee een venster op
        waarin `nanmean` NaN werd, `NaN > drempel` False was en `breached` leeg
        bleef -- en dat ging als CONCLUSIEF naar buiten. Volledig kapot, en
        toch groen.
        """
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.5),
                       rv, np.zeros_like(rv)).verdict()
        assert not verdict.is_conclusive
        assert not verdict.is_breach
        assert verdict.n_obs == len(rv)

    def test_a_negative_forecast_is_equally_inconclusive(self) -> None:
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.5),
                       rv, -rv).verdict()
        assert not verdict.is_conclusive

    def test_too_few_usable_terms_is_no_verdict_even_with_enough_bars(
            self) -> None:
        """Het venster is vol, maar bijna alles is onbruikbaar.

        Zonder deze poort baseerde het oordeel zich op een handvol punten en
        heette het toch conclusief.
        """
        rv = _series()
        forecast = np.zeros_like(rv)
        forecast[:5] = rv[:5]          # vijf bruikbare paren, verder niets
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.5),
                       rv, forecast).verdict()
        assert not verdict.is_conclusive

    def test_enough_usable_terms_still_yields_a_verdict(self) -> None:
        """De poort mag een gezond venster niet stilleggen."""
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.5), rv, rv).verdict()
        assert verdict.is_conclusive
        assert np.isfinite(verdict.qlike_ratio)


class TestTheBaselineIsRequiredAndMeasured:
    def test_a_monitor_without_a_baseline_cannot_be_built(self) -> None:
        with pytest.raises(TypeError):
            VolForecastMonitor()  # type: ignore[call-arg]

    def test_a_non_positive_baseline_crashes(self) -> None:
        with pytest.raises(DataContractError):
            VolForecastMonitor(baseline_qlike=0.0)


class TestQlikeDegradation:
    def test_a_perfect_forecast_does_not_breach(self) -> None:
        """Negatieve controle: QLIKE is nul bij een perfecte forecast."""
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.5), rv, rv).verdict()
        assert verdict.live_qlike == pytest.approx(0.0, abs=1e-12)
        assert QLIKE_DEGRADED not in verdict.breached

    def test_a_wildly_wrong_forecast_breaches(self) -> None:
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=0.01),
                       rv, rv * 10.0).verdict()
        assert QLIKE_DEGRADED in verdict.breached
        assert verdict.qlike_ratio > CFG.vol_qlike_degradation_ratio

    def test_the_ratio_is_against_the_supplied_baseline(self) -> None:
        """Dezelfde forecast, een andere baseline, een ander oordeel.

        Dat is het punt van een RELATIEVE poort: hij vergelijkt met wat de
        backtest haalde en niet met een verzonnen absolute grens.
        """
        rv = _series()
        forecast = rv * 1.6
        strict = _fed(VolForecastMonitor(baseline_qlike=0.001),
                      rv, forecast).verdict()
        lenient = _fed(VolForecastMonitor(baseline_qlike=10.0),
                       rv, forecast).verdict()
        assert strict.live_qlike == pytest.approx(lenient.live_qlike)
        assert QLIKE_DEGRADED in strict.breached
        assert QLIKE_DEGRADED not in lenient.breached


class TestBiasIsCaughtEvenWhenTheSizeIsAcceptable:
    def test_a_systematically_low_forecast_is_flagged_by_mz(self) -> None:
        """De vertekening die vol-targeting rechtstreeks doorgeeft aan de
        positiegrootte."""
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=10.0),
                       rv, rv * 0.7).verdict()
        assert MZ_BIASED in verdict.breached, (
            "Een forecast die structureel 30 % te laag is, hoort door de "
            "Mincer-Zarnowitz-toets te worden gezien.")

    def test_a_noisy_but_unbiased_forecast_is_not_flagged(self) -> None:
        """Een REALISTISCHE zuivere forecast: onnauwkeurig maar niet scheef.

        De constructie doet ertoe en is bij het schrijven een keer misgegaan.
        `forecast = rv * ruis` is NIET zuiver in Mincer-Zarnowitz-zin: de ruis
        zit dan in de REGRESSOR, en dat geeft attenuatie (errors-in-variables),
        met een beta systematisch onder 1. Gemeten met sigma = 0,15: beta =
        0,892, p = 7,2e-5 -- terecht als vertekend gemeld.

        Zuiverheid betekent `E[RV | forecast] = forecast`, dus de ruis hoort in
        de REALISATIE gegeven de forecast. Vandaar deze volgorde: eerst de
        forecast, dan de realisatie eromheen, met een multiplicatieve ruis van
        gemiddeld 1.
        """
        rng = np.random.default_rng(11)
        forecast = _series()
        noise = rng.lognormal(mean=-0.5 * 0.15**2, sigma=0.15, size=forecast.size)
        rv = forecast * noise
        verdict = _fed(VolForecastMonitor(baseline_qlike=10.0),
                       rv, forecast).verdict()
        assert verdict.mz_beta == pytest.approx(1.0, abs=0.15)
        assert MZ_BIASED not in verdict.breached

    def test_a_perfect_forecast_does_not_trigger_a_false_halt(self) -> None:
        """De numerieke degeneratie die een vals alarm zou geven.

        Op een exacte forecast gaan de HAC-standaardfouten naar nul en explodeert
        de Wald-statistiek; `mincer_zarnowitz` meldt dan p ~ 7e-9 op een
        regressie met alpha ~ 0 en beta ~ 1. Voor een backtest is dat een
        randgeval; voor een monitor die mag halteren is het een boek dat
        onterecht dichtgaat.
        """
        rv = _series()
        verdict = _fed(VolForecastMonitor(baseline_qlike=10.0), rv, rv).verdict()
        assert verdict.mz_beta == pytest.approx(1.0, abs=1e-9)
        assert MZ_BIASED not in verdict.breached


class TestTheWindowRolls:
    def test_only_the_last_window_counts(self) -> None:
        """Een ontsporing van vandaag mag niet worden uitgemiddeld tegen een
        jaar schone historie."""
        monitor = VolForecastMonitor(baseline_qlike=0.01, window=N)
        clean = _series(seed=1)
        _fed(monitor, clean, clean)
        assert QLIKE_DEGRADED not in monitor.verdict().breached
        broken = _series(seed=2)
        _fed(monitor, broken, broken * 10.0)
        assert QLIKE_DEGRADED in monitor.verdict().breached
