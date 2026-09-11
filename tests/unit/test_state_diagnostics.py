"""De diagnose moet BEIDE assen rapporteren, de richtingsas met haar
onzekerheid, en die onzekerheid moet op datum geclusterd zijn -- anders leest
iemand een gepoolde 1,16 als een bevinding terwijl de gedefleerde waarde 0,53
is."""
from __future__ import annotations

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
