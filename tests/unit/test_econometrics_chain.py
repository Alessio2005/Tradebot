"""De econometrische toetsingsketen doet wat zij beweert — Phase 6, deliverable 7.

Een statistische toets is de makkelijkste soort code om verkeerd te gebruiken en
groen te houden: hij geeft altijd een p-waarde terug. De vraag is of die p-waarde
over de juiste nulhypothese gaat.

Elke toets hieronder wordt daarom op een proces gedraaid waarvan de uitkomst
VOORAF bekend is uit de constructie, en op zijn tegenvoorbeeld. Dat is de
negatieve controle uit §0.10 toegepast op de toetsen zelf: een ARCH-toets die
ook verwerpt op i.i.d. ruis, meet niets.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.econometrics import (
    DiagnosticOutcome,
    adf_test,
    arch_gate_verdict,
    cusum_test,
    diagnose_series,
    engle_arch_test,
    kpss_test,
    ljung_box_test,
)

SEED = 20260825


def _white_noise(n: int = 1500) -> np.ndarray:
    return np.random.default_rng(SEED).normal(0.0, 1.0, n)


def _random_walk(n: int = 1500) -> np.ndarray:
    return np.cumsum(np.random.default_rng(SEED + 1).normal(0.0, 1.0, n))


def _garch_process(n: int = 2500, alpha: float = 0.10,
                   beta: float = 0.85) -> np.ndarray:
    """Een GARCH(1,1) met bekende parameters. ARCH-effecten per constructie."""
    rng = np.random.default_rng(SEED + 2)
    z = rng.normal(0.0, 1.0, n)
    h = np.empty(n)
    e = np.zeros(n)
    h[0] = 1.0
    for t in range(1, n):
        h[t] = 0.05 + alpha * e[t - 1] ** 2 + beta * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return e


def _ar1(n: int = 1500, phi: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(SEED + 3)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, 1.0)
    return x


def _mean_shift(n: int = 1500, shift: float = 1.5) -> np.ndarray:
    rng = np.random.default_rng(SEED + 4)
    x = rng.normal(0.0, 1.0, n)
    x[n // 2:] += shift
    return x


# --------------------------------------------------------------------------- #
# ADF en KPSS — tegengestelde nulhypotheses
# --------------------------------------------------------------------------- #
def test_adf_rejects_on_white_noise_and_not_on_a_random_walk() -> None:
    assert adf_test(_white_noise()).rejected
    assert not adf_test(_random_walk()).rejected


def test_kpss_does_the_opposite_of_adf() -> None:
    """De hele reden dat beide toetsen draaien.

    Zonder deze test zou iemand de twee kunnen verwisselen en jarenlang
    "KPSS verwerpt, dus stationair" rapporteren — een omkering die in een
    tabel volstrekt normaal oogt.
    """
    assert not kpss_test(_white_noise()).rejected
    assert kpss_test(_random_walk()).rejected


def test_the_null_hypotheses_are_stated_and_are_opposites() -> None:
    adf = adf_test(_white_noise())
    kp = kpss_test(_white_noise())
    assert "eenheidswortel" in adf.null_hypothesis
    assert "IS stationair" in kp.null_hypothesis
    assert adf.null_hypothesis != kp.null_hypothesis


def test_a_test_outcome_without_a_null_hypothesis_is_rejected() -> None:
    with pytest.raises(DataContractError, match="nulhypothese"):
        DiagnosticOutcome(name="x", statistic=1.0, p_value=0.5,
                          null_hypothesis="", rejected=False, alpha=0.05,
                          detail={})


# --------------------------------------------------------------------------- #
# Ljung-Box
# --------------------------------------------------------------------------- #
def test_ljung_box_finds_ar1_autocorrelation_and_not_white_noise() -> None:
    assert ljung_box_test(_ar1()).rejected
    assert not ljung_box_test(_white_noise()).rejected


# --------------------------------------------------------------------------- #
# Engle ARCH — de poortwachter
# --------------------------------------------------------------------------- #
def test_engle_arch_finds_a_garch_process() -> None:
    outcome = engle_arch_test(_garch_process())
    assert outcome.rejected
    assert outcome.p_value < 1e-6


def test_engle_arch_does_not_fire_on_iid_noise() -> None:
    """De negatieve controle op de poortwachter zelf.

    Zonder deze test bewijst "6 van 6 reeksen passeren de ARCH-poort" niets:
    een toets die altijd verwerpt, laat per definitie alles door.
    """
    assert not engle_arch_test(_white_noise()).rejected


def test_the_arch_gate_verdict_states_the_consequence_not_just_the_number() -> None:
    open_gate = arch_gate_verdict(
        diagnose_series(_garch_process(), name="garch"))
    shut_gate = arch_gate_verdict(
        diagnose_series(_white_noise(), name="noise"))
    assert open_gate.startswith("OPEN")
    assert "gerechtvaardigd" in open_gate
    assert shut_gate.startswith("DICHT")
    assert "GEEN falsificatie" in shut_gate, (
        "het oordeel moet expliciet zeggen dat een dichte poort geen "
        "falsificatie is; §6 verbiedt het falsificeren van een model dat niet "
        "van toepassing was"
    )


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #
def test_cusum_finds_a_mean_shift_and_not_a_stable_mean() -> None:
    assert cusum_test(_mean_shift()).rejected
    assert not cusum_test(_white_noise()).rejected


def test_cusum_refuses_an_alpha_it_has_no_critical_value_for() -> None:
    """Een geïnterpoleerde kritieke waarde zou een verzonnen drempel zijn."""
    with pytest.raises(DataContractError, match="kritieke waarden"):
        cusum_test(_white_noise(), alpha=0.025)


# --------------------------------------------------------------------------- #
# Degenerate invoer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "test", [adf_test, kpss_test, ljung_box_test, engle_arch_test, cusum_test])
def test_every_test_refuses_a_constant_series(test) -> None:
    """Geen enkele toets mag een p-waarde produceren op een puntmassa."""
    with pytest.raises(DataContractError, match="zonder variatie"):
        test(np.full(500, 3.14))


@pytest.mark.parametrize(
    "test", [adf_test, kpss_test, ljung_box_test, engle_arch_test, cusum_test])
def test_every_test_refuses_a_series_that_is_too_short(test) -> None:
    with pytest.raises(DataContractError, match="te weinig eindige observaties"):
        test(np.random.default_rng(0).normal(0, 1, 10))


@pytest.mark.parametrize(
    "test", [adf_test, kpss_test, ljung_box_test, engle_arch_test, cusum_test])
def test_no_test_silently_imputes_a_gap(test) -> None:
    """Niet-eindige waarden worden verwijderd, niet ingevuld.

    Een reeks met 400 geldige en 1.100 ontbrekende waarden moet crashen op de
    minimumeis en niet stilzwijgend op 400 punten worden getoetst alsof het er
    1.500 waren.
    """
    arr = np.full(1500, np.nan)
    arr[:10] = np.random.default_rng(0).normal(0, 1, 10)
    with pytest.raises(DataContractError):
        test(arr)


# --------------------------------------------------------------------------- #
# De gecombineerde diagnose
# --------------------------------------------------------------------------- #
def test_the_four_stationarity_combinations_are_distinguished() -> None:
    stationary = diagnose_series(_white_noise(), name="wn")
    unit_root = diagnose_series(_random_walk(), name="rw")
    assert stationary.stationarity_verdict.startswith("stationair")
    assert unit_root.stationarity_verdict.startswith("eenheidswortel")
    assert stationary.stationarity_verdict != unit_root.stationarity_verdict


def test_the_diagnostics_record_carries_every_test() -> None:
    record = diagnose_series(_garch_process(), name="garch").as_record()
    assert set(record["tests"]) == {
        "adf", "kpss", "ljung_box", "engle_arch", "cusum"}
    assert record["arch_gate_open"] is True
    for test in record["tests"].values():
        assert test["null_hypothesis"]
