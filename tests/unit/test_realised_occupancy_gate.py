"""De poort die H2 had moeten tegenhouden vóór de fit in plaats van erna.

De a-priori-poort rekent met een uniforme bezetting van 1/k en noemt dat in het
artefact zelf 'optimistisch'. Voor hmm_k3 gaf dat 165,0 observaties per fold en
`adequate: true`; gerealiseerd waren het er 5,0 tot 28,0 tegen een eis van 100.

Revisie 2 voegt de episodetelling toe: bars zijn niet de effectieve
steekproefomvang van een persistente toestand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import StateAssignment, assign_by_variance
from tradebot.utils.failfast import DataContractError
from tradebot.validation.data_adequacy import assert_realised_occupancy

IDX = pd.date_range("2020-01-01", periods=1200, freq="D", tz="UTC")
LIMITS = dict(min_obs_per_state_per_fold=50,
              min_episodes_per_state_per_fold=10,
              min_occupancy_fraction=0.10)


def _balanced():
    rng = np.random.default_rng(2)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 1200))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.30, high_q=0.70, min_periods=100)


def _starved():
    """Een sigma-reeks waarin HOOG bijna nooit voorkomt -- de H2-situatie."""
    rng = np.random.default_rng(2)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.02, 1200))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.005, high_q=0.995, min_periods=100)


def _persistent_but_clustered():
    """Genoeg BARS, te weinig EPISODES: een lange trage sigma-golf levert een
    HOOG-toestand met honderden bars in een handvol blokken. Dit is het geval
    dat een bar-poort doorlaat en dat statistisch niets is."""
    wave = 0.5 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 1200))
    sigma = pd.DataFrame({"A": wave}, index=IDX)
    return assign_by_variance(sigma, low_q=0.30, high_q=0.70, min_periods=100)


def test_a_balanced_assignment_passes() -> None:
    verdict = assert_realised_occupancy(_balanced(), n_folds=4, **LIMITS)
    assert verdict.adequate


def test_a_starved_state_is_refused_on_the_MINIMUM_not_the_median() -> None:
    """AD-18: het oordeel gaat over het minimum. Een mediaan van 165 met een
    minimum van 5 is precies de meting die H2 groen liet lijken."""
    with pytest.raises(DataContractError):
        assert_realised_occupancy(
            _starved(), n_folds=4,
            min_obs_per_state_per_fold=100,
            min_episodes_per_state_per_fold=10,
            min_occupancy_fraction=0.10,
        )


def test_enough_bars_but_too_few_episodes_is_refused() -> None:
    """Q7. De test die revisie 1 niet had, en het geval dat er in de praktijk
    het vaakst is."""
    with pytest.raises(DataContractError):
        assert_realised_occupancy(
            _persistent_but_clustered(), n_folds=4,
            min_obs_per_state_per_fold=50,
            min_episodes_per_state_per_fold=25,
            min_occupancy_fraction=0.10,
        )


def test_the_verdict_reports_both_measured_minima() -> None:
    verdict = assert_realised_occupancy(
        _balanced(), n_folds=4, raise_on_failure=False, **LIMITS
    )
    assert "rarest_state_obs_in_smallest_fold" in verdict.measured
    assert "rarest_state_episodes_in_smallest_fold" in verdict.measured
    assert verdict.measured["occupancy_source"] == "realised"


# --------------------------------------------------------------------------- #
# Toegevoegd bij de implementatie. `_persistent_but_clustered` hierboven bewijst
# de episode-eis NIET geisoleerd: op die sinusgolf valt LOW in de eerste fold
# volledig weg, dus de weigering kan ook op de BARS staan. Zonder de twee tests
# hieronder kan de episodetelling stuk zijn zonder dat een test rood wordt.
# --------------------------------------------------------------------------- #
LONG_IDX = pd.date_range("2015-01-01", periods=2400, freq="D", tz="UTC")


def _long_runs():
    """Ruim genoeg BARS per fold en per toestand, maar weinig EPISODES.

    Een zaagtand met een periode van 100 bars: sigma loopt lineair op en valt
    dan terug. De verdeling is daardoor vlak -- elke toestand krijgt ongeveer
    zijn kwantielaandeel van de bars in elke fold -- terwijl elke toestand per
    cyclus maar EEN aaneengesloten blok bezet. Vier folds van 600 bars bevatten
    zes cycli, dus zes episodes per toestand tegen 150 bars per toestand.
    """
    cycle = np.linspace(0.2, 0.8, 100, endpoint=False)
    sigma = pd.DataFrame({"A": np.tile(cycle, 24)}, index=LONG_IDX)
    return assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)


def test_the_episode_shortfall_stands_on_its_own() -> None:
    """De bars halen de eis ruim; alleen de episodes niet. Dit is het verschil
    tussen revisie 1 en revisie 2, en het is hier het ENIGE dat faalt."""
    verdict = assert_realised_occupancy(
        _long_runs(), n_folds=4, raise_on_failure=False,
        min_obs_per_state_per_fold=50,
        min_episodes_per_state_per_fold=20,
        min_occupancy_fraction=0.10,
    )
    assert verdict.measured["rarest_state_obs_in_smallest_fold"] >= 50
    assert verdict.measured["rarest_state_occupancy_fraction"] >= 0.10
    assert verdict.measured["rarest_state_episodes_in_smallest_fold"] < 20
    assert not verdict.adequate
    assert "EPISODES" in verdict.shortfall


def test_the_refusal_names_the_binding_cell_and_the_threshold() -> None:
    """R-5. Een poort die alleen 'niet adequaat' zegt, dwingt de lezer terug de
    code in; deze poort noemt de toestand, de fold en beide getallen."""
    with pytest.raises(DataContractError) as raised:
        assert_realised_occupancy(
            _long_runs(), n_folds=4,
            min_obs_per_state_per_fold=50,
            min_episodes_per_state_per_fold=20,
            min_occupancy_fraction=0.10,
        )
    reported = assert_realised_occupancy(
        _long_runs(), n_folds=4, raise_on_failure=False,
        min_obs_per_state_per_fold=50,
        min_episodes_per_state_per_fold=20,
        min_occupancy_fraction=0.10,
    )
    message = str(raised.value)
    assert reported.measured["rarest_state_episodes_at"] in message
    assert "20" in message


def test_a_state_that_exactly_meets_the_bar_requirement_is_not_refused() -> None:
    """De bar-telling is GEHEEL, en `(k/n) * n` is dat in IEEE754 niet altijd.

    Bij 90 toegewezen bars levert een toestand van 13 bars de drijvendekomma-
    waarde 12,999999999999998 op. Zonder afronding weigert de poort daarmee een
    toestand die zijn eis EXACT haalt -- een fout in de verkeerde richting: de
    poort hoort streng te zijn op de data, niet op de representatie.
    """
    idx = pd.date_range("2020-01-01", periods=90, freq="D", tz="UTC")
    states = pd.DataFrame({"A": [0.0] * 13 + [1.0] * 38 + [2.0] * 39}, index=idx)
    assignment = StateAssignment(
        states=states, source="test", ordering="variance",
        params={"low_q": 0.1, "high_q": 0.9, "min_periods": 2.0, "lag": 1.0})

    verdict = assert_realised_occupancy(
        assignment, n_folds=1, raise_on_failure=False,
        min_obs_per_state_per_fold=13,
        min_episodes_per_state_per_fold=1,
        min_occupancy_fraction=0.10,
    )
    assert verdict.measured["rarest_state_obs_in_smallest_fold"] == 13
    assert verdict.adequate, verdict.shortfall
