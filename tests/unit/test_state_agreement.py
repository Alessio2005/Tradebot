"""De proxyvrije vergelijking. Een toestandstoewijzer wordt vergeleken met een
andere toestandstoewijzer, en dat is een telling in plaats van een toets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import StateAssignment, VolState, assign_by_variance
from tradebot.regime.state_agreement import agreement
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")

NAN = float("nan")
LOW, NORMAL, HIGH = float(VolState.LOW), float(VolState.NORMAL), float(VolState.HIGH)


def _assignment(seed: int):
    rng = np.random.default_rng(seed)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 400))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)


def test_a_model_agrees_perfectly_with_itself() -> None:
    a = _assignment(5)
    report = agreement(a, a)
    assert report.fraction_identical == 1.0
    assert report.n_decision_changes == 0
    assert report.cohen_kappa == 1.0


def test_different_sigma_paths_disagree_somewhere() -> None:
    report = agreement(_assignment(5), _assignment(6))
    assert report.fraction_identical < 1.0


def test_agreement_is_reported_against_chance_not_only_in_raw_percent() -> None:
    """Bij 79 % NORMAAL is 80 % ruwe overeenstemming bijna niets. Cohens kappa
    corrigeert voor de bezetting en is daarom de maat die telt."""
    report = agreement(_assignment(5), _assignment(6))
    assert report.cohen_kappa < report.fraction_identical


def test_confusion_matrix_is_three_by_three_per_symbol() -> None:
    report = agreement(_assignment(5), _assignment(6))
    assert report.confusion["A"].shape == (3, 3)


# --------------------------------------------------------------------------- #
# Wat de vier hierboven niet vastpinnen. Elke test hieronder faalt zodra het
# gedrag dat hij beschrijft verdwijnt -- dat is de enige reden dat hij bestaat.
# --------------------------------------------------------------------------- #
def _hand(source: str, **columns: list[float]) -> StateAssignment:
    """Een met de hand gezette toewijzing.

    De enige manier om een GAT op een bekende plek te leggen en de noemer er
    daarna op af te rekenen; uit `assign_by_variance` komt een gat alleen waar
    de opstartfase het toevallig legt.
    """
    values = list(columns.values())
    frame = pd.DataFrame(columns, index=IDX[: len(values[0])], dtype="float64")
    return StateAssignment(
        states=frame, source=source, ordering="variance",
        params={"low_q": 0.25, "high_q": 0.75, "min_periods": 2.0, "lag": 1.0},
    )


def test_the_comparison_is_invariant_under_a_monotone_rescaling() -> None:
    """RULING P35. De toewijzing loopt over kwantielen van de reeks zelf, dus
    elke monotone herschaling van sigma-dak levert dezelfde toestanden op. Dat
    is precies waarom een VARIANTIE-forecast (GARCH) en een VOLATILITEIT-panel
    (EWMA) zonder schaalstap naast elkaar mogen: het verschil in eenheid KAN
    deze telling niet raken. Het is tegelijk de grens van de claim -- "GARCH
    staat systematisch hoger dan EWMA" is hier per constructie onzichtbaar.
    """
    rng = np.random.default_rng(11)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 400))}, index=IDX)
    report = agreement(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100),
        assign_by_variance(sigma**2, low_q=0.25, high_q=0.75, min_periods=100),
    )
    assert report.fraction_identical == 1.0
    assert report.cohen_kappa == 1.0


def test_a_hole_is_dropped_from_the_denominator_instead_of_filled() -> None:
    """RULING P37. Een cel die een van beide modellen niet kent, telt in GEEN
    van beide richtingen mee -- niet als overeenstemming en niet als verschil.
    """
    report = agreement(
        _hand("a", A=[LOW, NORMAL, HIGH, NORMAL]),
        _hand("b", A=[LOW, NORMAL, HIGH, NAN]),
    )
    assert report.n_cells_compared == 3
    assert report.n_cells_dropped == 1
    assert report.n_identical == 3
    assert report.fraction_identical == 1.0


def test_the_bar_denominator_requires_every_symbol_to_be_known() -> None:
    """RULING P37. Een bar met een onbekende naam kan niet "ongewijzigd" heten
    zonder de toestand van die naam te verzinnen, dus hij valt uit de
    bar-noemer -- ook wanneer de namen die wel bekend zijn het eens zijn.
    """
    report = agreement(
        _hand("a", A=[LOW, NORMAL, HIGH], B=[LOW, NORMAL, HIGH]),
        _hand("b", A=[LOW, NAN, HIGH], B=[LOW, NORMAL, NORMAL]),
    )
    assert report.n_bars_compared == 2
    assert report.n_bars_dropped == 1
    assert report.n_any_state_changes == 1
    assert report.fraction_bars_with_any_state_change == 0.5
    assert report.n_cells_compared == 5


def test_the_bar_denominator_is_reported_as_a_window_and_not_only_as_a_count() -> None:
    """RULING P37. De bar-noemer is een VENSTER. Op een walk-forward-paneel ligt
    dat venster MIDDEN in de sample -- het paneel is leeg voor de eerste
    trainperiode en na de laatste volle testperiode -- dus een lezer die alleen
    "n van N" krijgt, gokt op de staart en gokt verkeerd. Het gat staat hier op
    bar 1 zodat het venster aantoonbaar niet op bar 0 begint.
    """
    report = agreement(
        _hand("a", A=[LOW, NORMAL, HIGH, NORMAL]),
        _hand("b", A=[NAN, NORMAL, HIGH, NORMAL]),
    )
    assert list(report.bars_compared) == list(IDX[1:4])
    # De telling kan niet van het venster afwijken; zij wordt eruit afgeleid.
    assert report.n_bars_compared == len(report.bars_compared) == 3


def test_cell_agreement_and_bar_change_are_two_numbers_not_one() -> None:
    """RULING P36. De beslisregel leest in clausule 1 de CEL-overeenstemming en
    in clausule 2 het BAR-paar. Vallen die twee samen, dan staat er een getal in
    twee kostuums en toetst de regel maar de helft van wat zij beweert.
    """
    report = agreement(
        _hand("a", A=[LOW, NORMAL, HIGH, NORMAL], B=[LOW, NORMAL, HIGH, NORMAL]),
        _hand("b", A=[HIGH, NORMAL, HIGH, NORMAL], B=[LOW, HIGH, HIGH, NORMAL]),
    )
    assert report.fraction_identical == 0.75
    assert report.n_any_state_changes == 2
    assert report.fraction_bars_with_any_state_change == 0.5


def test_the_gate_ignores_a_swap_that_it_merges_and_the_bound_does_not() -> None:
    """RULING P36 (geamendeerd). De vooraf geregistreerde poort is
    `flat_states={HIGH}` (stap 13.3): een LAAG/NORMAAL-wissel verandert het
    besluit NIET, want de poort voegt die twee samen. De poortvrije telling ziet
    hem wel, en is daarom een BOVENGRENS -- op elke denkbare toestandspoort.
    """
    report = agreement(
        _hand("a", A=[LOW, NORMAL, HIGH]),
        _hand("b", A=[NORMAL, LOW, NORMAL]),
    )
    assert report.n_any_state_changes == 3
    assert report.n_decision_changes == 1
    assert report.fraction_bars_with_decision_change < (
        report.fraction_bars_with_any_state_change)


def test_the_default_gate_is_the_one_the_phase_pre_registered() -> None:
    """Een expliciete `{HIGH}` en de default moeten hetzelfde getal geven;
    anders komt de beslisregel op een andere poort uit dan stap 13.3 draait."""
    a = _hand("a", A=[LOW, NORMAL, HIGH])
    b = _hand("b", A=[NORMAL, LOW, NORMAL])
    assert (agreement(a, b, flat_states=frozenset({VolState.HIGH}))
            .n_decision_changes == agreement(a, b).n_decision_changes)


def test_gating_every_state_is_refused_by_the_gate_itself() -> None:
    """Het poortpredikaat komt uit `state_mapping.gate_by_state` en wordt hier
    niet nagebouwd (R-3), dus de guard van die poort geldt ook hier."""
    a = _hand("a", A=[LOW, NORMAL, HIGH])
    with pytest.raises(DataContractError, match="uit-knop"):
        agreement(a, a, flat_states=frozenset(VolState))


def test_an_empty_gate_is_refused_instead_of_counting_zero_changes() -> None:
    a = _hand("a", A=[LOW, NORMAL, HIGH])
    with pytest.raises(DataContractError, match="flat_states"):
        agreement(a, a, flat_states=frozenset())


def test_two_different_symbol_sets_are_refused_by_name() -> None:
    with pytest.raises(DataContractError, match="C"):
        agreement(
            _hand("a", A=[LOW, NORMAL], B=[LOW, NORMAL]),
            _hand("b", A=[LOW, NORMAL], C=[LOW, NORMAL]),
        )


def test_two_different_windows_are_refused_instead_of_intersected() -> None:
    """Een stilzwijgende doorsnede zou een noemer opleveren die in geen enkel
    veld van het rapport staat -- precies wat P37 verbiedt."""
    short = _hand("a", A=[LOW, NORMAL])
    long = _hand("b", A=[LOW, NORMAL, HIGH])
    with pytest.raises(DataContractError, match="n_bars"):
        agreement(short, long)


def test_kappa_without_variation_is_refused_instead_of_reported_as_nan() -> None:
    """Bij een toestand op het hele venster is de toevalsovereenstemming 1 en
    is kappa 0/0. Een NaN in dat veld zou als meting worden gelezen."""
    a = _hand("a", A=[NORMAL, NORMAL, NORMAL])
    with pytest.raises(DataContractError, match="NORMAL"):
        agreement(a, a)
