"""De diagnose moet BEIDE assen rapporteren, de richtingsas met haar
onzekerheid, en die onzekerheid moet op datum geclusterd zijn -- anders leest
iemand een gepoolde 1,16 als een bevinding terwijl de gedefleerde waarde 0,53
is."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from tradebot.regime.state import StateAssignment, VolState, assign_by_variance
from tradebot.regime.state_diagnostics import diagnose

IDX = pd.date_range("2022-01-01", periods=600, freq="D", tz="UTC")


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 600))}, index=IDX)
    returns = pd.DataFrame(
        {"A": rng.normal(0.0, 1.0, 600) * sigma["A"].to_numpy() / np.sqrt(365)},
        index=IDX,
    )
    return sigma, returns


def _diagnose():
    sigma, returns = _fixture()
    return diagnose(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100),
        forward_returns=returns,
        bars_per_year=365,
    )


def test_reports_both_axes() -> None:
    result = _diagnose()
    assert set(result.separation_vol) == {"LOW", "NORMAL", "HIGH"}
    assert set(result.separation_return) == {"LOW", "NORMAL", "HIGH"}


def test_the_direction_axis_carries_clustered_uncertainty() -> None:
    """Zonder t-statistiek is een conditioneel gemiddelde geen bevinding maar
    een getal. En zonder clustering is de t-statistiek zelf geen bevinding."""
    result = _diagnose()
    for state in ("LOW", "NORMAL", "HIGH"):
        cell = result.separation_return[state]
        for field in ("n_bars", "n_episodes", "t_stat_pooled",
                      "t_stat_clustered", "t_stat_neff_deflated", "se"):
            assert field in cell


def test_the_deflated_t_is_never_larger_than_the_pooled_t() -> None:
    """De deflatie moet de goede kant op werken. Een 'correctie' die de t
    vergroot, ziet eruit als zorgvuldigheid en is het tegendeel."""
    result = _diagnose()
    for state in ("LOW", "NORMAL", "HIGH"):
        cell = result.separation_return[state]
        assert abs(cell["t_stat_neff_deflated"]) <= abs(cell["t_stat_pooled"]) + 1e-9


def test_persistence_is_reported_as_a_transition_matrix() -> None:
    """Een toestand waarop je handelt, moet lang genoeg duren om erop te
    handelen. De diagonaal van de overgangsmatrix is die eigenschap."""
    result = _diagnose()
    assert result.transition_matrix.shape == (3, 3)
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)


def test_synthetic_returns_with_no_state_dependence_show_no_separation() -> None:
    """Negatieve controle. Op returns die per constructie NIET van de toestand
    afhangen, mag geen enkele t boven 2 uitkomen."""
    result = _diagnose()
    assert all(
        abs(result.separation_return[s]["t_stat_clustered"]) < 2.0
        for s in ("LOW", "NORMAL", "HIGH")
    )


def test_per_symbol_results_are_not_hidden_behind_the_pool() -> None:
    """De nulmeting toont tegenstrijdige tekens per symbool. Een gepoolde tabel
    verbergt dat, en dat is precies de informatie die telt."""
    result = _diagnose()
    assert "A" in result.per_symbol


# --------------------------------------------------------------------------- #
# RULING P31 -- `StateAssignment.mean_duration()` had bij oplevering van stap 5
# NUL testdekking, en stap 6 rapporteert die grootheid. R-3 verbiedt een tweede
# implementatie, dus de diagnose ROEPT de bestaande methode aan; dan hoort de
# dekking die zij mist hier te staan. Een diagnose die op een ongetoetste
# methode leunt, is een diagnose die op niets rust.
# --------------------------------------------------------------------------- #
def _handmade() -> StateAssignment:
    """Acht bars met een met de hand geteld antwoord: LOW in twee episodes van
    vier en twee bars, NORMAAL in een van twee, HOOG nooit."""
    index = pd.date_range("2022-01-01", periods=8, freq="D", tz="UTC")
    states = pd.DataFrame(
        {"A": [float(VolState.LOW)] * 4
              + [float(VolState.NORMAL)] * 2
              + [float(VolState.LOW)] * 2},
        index=index,
    )
    return StateAssignment(
        states=states, source="handmade", ordering="variance",
        params={"low_q": 0.25, "high_q": 0.75, "min_periods": 2.0, "lag": 1.0},
    )


def test_mean_duration_counts_episodes_and_not_bars() -> None:
    """Zes LOW-bars in twee episodes is een gemiddelde episodelengte van DRIE,
    niet van zes. Dat onderscheid is de hele reden dat het veld bestaat: het is
    de effectieve steekproefomvang van een persistente toestand (Q7)."""
    duration = _handmade().mean_duration()
    assert duration.loc["A", "LOW"] == 3.0
    assert duration.loc["A", "NORMAL"] == 2.0


def test_mean_duration_of_an_unobserved_state_is_nan_not_zero() -> None:
    """Nul episodes geeft geen episodelengte van nul -- nul zou lezen als 'de
    toestand duurt geen enkele bar', terwijl hij simpelweg niet voorkomt."""
    assert np.isnan(_handmade().mean_duration().loc["A", "HIGH"])


def test_the_diagnosis_reuses_the_assignment_mean_duration() -> None:
    """R-3 in de vorm van een test: de diagnose mag `mean_duration` niet
    naast `StateAssignment` opnieuw uitrekenen, ook niet als zij hetzelfde
    getal zou geven."""
    sigma, returns = _fixture()
    assignment = assign_by_variance(
        sigma, low_q=0.25, high_q=0.75, min_periods=100)
    result = diagnose(assignment, forward_returns=returns, bars_per_year=365)
    pd.testing.assert_frame_equal(
        result.mean_duration, assignment.mean_duration(), check_like=True)


# --------------------------------------------------------------------------- #
# FIXRONDE 1, BEVINDING I-2 -- de zes verplichte tests hierboven staan er
# letterlijk zoals de faseopdracht ze schrijft en pinnen de INTERFACE. Zij
# bijten niet op de METING: de review muteerde `_ann_vol` naar een constante,
# de overgangsmatrix naar uniform 1/3, `_per_symbol` naar lege cellen en
# occupancy/episodes naar 1,0/0 -- en alle negen bleven groen.
#
# De oorzaak is het paneel van EEN kolom. Bij `n_units = 1` geeft
# `clustered_mean` per constructie `rho_bar = 0` en `neff_factor = 1,0`, dus
# `t_gepoold == t_geclusterd == t_gedefleerd` exact. De clustering en de
# N_eff-deflatie -- de hele reden dat `separation_return` deze vorm heeft, en
# het onderwerp van de docstring bovenaan dit bestand -- werden nooit
# uitgevoerd.
#
# Hieronder staat die ontbrekende dekking, op een paneel dat per constructie
# draagt wat de meting hoort te vinden: zes namen met een GEMEENSCHAPPELIJKE
# factor (dus `rho_bar > 0` en een deflatie die echt defleert), sterk
# uiteenlopende vol-niveaus (dus de pool verbergt de doorsnede) en een
# persistent vol-proces (dus de toestand duurt lang genoeg om een diagonaal te
# hebben).
# --------------------------------------------------------------------------- #
PANEL_IDX = pd.date_range("2021-01-01", periods=700, freq="D", tz="UTC")
PANEL_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")
#: Vol-niveaus die een factor ruim tien uit elkaar liggen. Dat is geen
#: versiering: het is de eigenschap waardoor een gepoolde cel aantoonbaar iets
#: anders zegt dan de zes cellen waaruit zij is opgebouwd (§2 van het rapport).
PANEL_SCALE = (0.2, 0.4, 0.8, 1.2, 1.8, 2.5)
#: Het aandeel gemeenschappelijke variantie in het rendement. Bij 0,85 ligt
#: `rho_bar` rond 0,82 en `neff_factor` rond 0,44 -- vergelijkbaar met het
#: gemeten paneel (0,74 resp. 0,46), dus de test draait op dezelfde orde van
#: afhankelijkheid als de meting die hij bewaakt.
PANEL_COMMON_SHARE = 0.85
#: AR(1)-coefficient van de log-vol. Hoog genoeg dat een toestand tientallen
#: bars duurt, want zonder persistentie heeft de overgangsmatrix geen diagonaal
#: om te toetsen.
PANEL_VOL_PERSISTENCE = 0.97


def _correlated_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Zes namen met een gemeenschappelijke factor en uiteenlopende vol-niveaus.

    De volatiliteit is een persistent AR(1)-proces in de log, zodat de toestand
    lang genoeg duurt om een diagonaal te hebben, en het rendement is per
    constructie `sigma / sqrt(bars_per_year)` maal een geschokte standaardnormaal.
    De vol-separatie is dus GECONSTRUEERD en bekend: HOOG hoort meetbaar boven
    NORMAAL te liggen en NORMAAL boven LAAG.
    """
    rng = np.random.default_rng(4242)
    n = len(PANEL_IDX)
    common_vol_shock = rng.normal(0.0, 1.0, n)
    common_return_factor = rng.normal(0.0, 1.0, n)
    sigma: dict[str, np.ndarray] = {}
    returns: dict[str, np.ndarray] = {}
    for symbol, scale in zip(PANEL_SYMBOLS, PANEL_SCALE, strict=True):
        innovation = (np.sqrt(0.7) * common_vol_shock
                      + np.sqrt(0.3) * rng.normal(0.0, 1.0, n))
        log_vol = np.zeros(n)
        for t in range(1, n):
            log_vol[t] = PANEL_VOL_PERSISTENCE * log_vol[t - 1] + 0.12 * innovation[t]
        shock = (np.sqrt(PANEL_COMMON_SHARE) * common_return_factor
                 + np.sqrt(1.0 - PANEL_COMMON_SHARE) * rng.normal(0.0, 1.0, n))
        sigma[symbol] = scale * np.exp(log_vol)
        # De +0,12 drift zet de gepoolde t ver genoeg van nul dat de ketting
        # t_gepoold > t_geclusterd > t_gedefleerd over ruis heen meetbaar is.
        returns[symbol] = (sigma[symbol] / np.sqrt(365.0)) * (shock + 0.12)
    return (pd.DataFrame(sigma, index=PANEL_IDX),
            pd.DataFrame(returns, index=PANEL_IDX))


@lru_cache(maxsize=1)
def _diagnose_panel():
    sigma, returns = _correlated_panel()
    return diagnose(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100,
                           source="correlated_panel"),
        forward_returns=returns,
        bars_per_year=365,
    )


def test_clustering_and_deflation_strictly_shrink_the_t_on_a_correlated_panel() -> None:
    """Op een echt paneel moet elke stap in de ketting KLEINER worden.

    Met een enkele kolom vallen de drie lezingen samen en toetst `<=` een getal
    tegen zichzelf. Hier is `rho_bar > 0`, dus de geclusterde standaardfout is
    strikt groter dan de naieve en de deflatie is strikt actief. Wie de
    clustering of de deflatie stilzwijgend uitzet, krijgt gelijkheid terug en
    deze test wordt rood.
    """
    result = _diagnose_panel()
    for state in ("LOW", "NORMAL", "HIGH"):
        cell = result.separation_return[state]
        assert cell["n_units"] == len(PANEL_SYMBOLS)
        assert cell["rho_bar"] > 0.5
        assert cell["neff_factor"] < 1.0
        assert abs(cell["t_stat_pooled"]) > 1.0
        assert abs(cell["t_stat_clustered"]) < abs(cell["t_stat_pooled"])
        assert abs(cell["t_stat_neff_deflated"]) < abs(cell["t_stat_clustered"])


def test_a_constructed_vol_separation_comes_out_ordered_and_by_a_margin() -> None:
    """De as waarop de toestand hoort te scheiden, met een marge in plaats van
    alleen een sleutel. Een `ann_vol` die de spreiding weggooit -- een
    constante, een vergeten annualisatie, een vergeten wortel -- haalt deze
    ordening niet."""
    result = _diagnose_panel()
    vol = {state: result.separation_vol[state]["ann_vol"]
           for state in ("LOW", "NORMAL", "HIGH")}
    assert vol["NORMAL"] > vol["LOW"] * 1.5
    assert vol["HIGH"] > vol["NORMAL"] * 1.2
    for state in ("LOW", "NORMAL", "HIGH"):
        cell = result.separation_vol[state]
        assert cell["ann_vol_ci_low"] < cell["ann_vol"] < cell["ann_vol_ci_high"]
        assert 0.0 < cell["occupancy"] < 1.0
        assert cell["n_episodes"] > 0


def test_a_persistent_state_shows_a_dominant_transition_diagonal() -> None:
    """Rijen die tot 1 sommeren is een normalisatie, geen persistentie. Op een
    per constructie persistent paneel moet de diagonaal de rij DOMINEREN; een
    uniforme 1/3-matrix sommeert keurig tot 1 en is precies de matrix die
    'deze toestand is niet handelbaar' betekent."""
    result = _diagnose_panel()
    for state in VolState:
        row = result.transition_matrix[int(state)]
        off_diagonal = np.delete(row, int(state))
        assert row[int(state)] > 0.75
        assert row[int(state)] > off_diagonal.max() + 0.5


def test_the_pool_hides_more_than_a_factor_five_between_symbols() -> None:
    """De per-symbool-tabel bestaat omdat de pool een doorsnede is van namen die
    het oneens zijn. Op dit paneel ligt de stilste naam een factor tien onder de
    luidruchtigste; een gepoolde cel noemt geen van beide."""
    result = _diagnose_panel()
    assert set(result.per_symbol) == set(PANEL_SYMBOLS)
    pooled_high = result.separation_vol["HIGH"]["ann_vol"]
    quietest = result.per_symbol[PANEL_SYMBOLS[0]]["HIGH"]["ann_vol"]
    loudest = result.per_symbol[PANEL_SYMBOLS[-1]]["HIGH"]["ann_vol"]
    assert quietest < pooled_high * 0.5
    assert loudest > pooled_high * 1.2
    assert loudest > quietest * 5.0
    for symbol in PANEL_SYMBOLS:
        for state in ("LOW", "NORMAL", "HIGH"):
            cell = result.per_symbol[symbol][state]
            for field in ("n_bars", "occupancy", "n_episodes", "ann_vol",
                          "ann_return", "se", "t_stat_clustered"):
                assert field in cell
            assert cell["n_bars"] > 0
