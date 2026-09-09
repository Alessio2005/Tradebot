"""REGEL V, afdwingbaar gemaakt.

AD-16 meet dat een cross-sectioneel UNIFORME schaal onzichtbaar is: L7
herschaalt naar het vol-target en deelt elke constante c er weer uit. Een
toestandsafbeelding die alleen de OMVANG raakt, is daarmee per constructie een
lege operatie. Deze tests bewijzen dat `gate_by_state` de SAMENSTELLING raakt.

Nieuw in revisie 2: de poort zet namen op nul, waarna L7 de resterende namen
OPHOOGT om het vol-target te halen. Die concentratie is een risico dat de poort
introduceert, en zij hoort gemeten en begrensd te worden (Q9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import VolState, assign_by_variance
from tradebot.regime.state_mapping import concentration_report, gate_by_state
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")


def _fixture():
    rng = np.random.default_rng(3)
    sigma = pd.DataFrame(
        {"A": np.abs(rng.normal(0.4, 0.1, 400)),
         "B": np.abs(rng.normal(0.9, 0.3, 400))},
        index=IDX,
    )
    exposures = pd.DataFrame(
        {"A": rng.uniform(-1, 1, 400), "B": rng.uniform(-1, 1, 400)}, index=IDX,
    )
    assignment = assign_by_variance(
        sigma, low_q=0.25, high_q=0.75, min_periods=100,
    )
    return exposures, assignment


def _gated():
    exposures, assignment = _fixture()
    return exposures, assignment, gate_by_state(
        exposures, assignment, flat_states=frozenset({VolState.HIGH}),
    )


def test_the_mapping_survives_vol_targeting() -> None:
    """De kerntest. Normaliseer beide boeken op gelijke bruto exposure -- de
    operatie die L7 uitvoert -- en zij MOETEN dan nog verschillen. Een
    schaalafbeelding zou hier identiek worden."""
    exposures, _, gated = _gated()

    def gross_normalised(frame: pd.DataFrame) -> pd.DataFrame:
        gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
        return frame.div(gross, axis=0)

    base = gross_normalised(exposures)
    conditioned = gross_normalised(gated)
    common = base.dropna().index.intersection(conditioned.dropna().index)
    assert not np.allclose(
        base.loc[common].to_numpy(), conditioned.loc[common].to_numpy(),
        equal_nan=True,
    )


def test_a_uniform_multiplier_would_fail_that_same_test() -> None:
    """De negatieve controle op REGEL V zelf. Zonder deze test bewijst de
    vorige test niets over de POORT -- alleen dat er iets is veranderd."""
    exposures, assignment, _ = _gated()
    scaled = exposures.mul(0.5)

    def gross_normalised(frame: pd.DataFrame) -> pd.DataFrame:
        gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
        return frame.div(gross, axis=0)

    base = gross_normalised(exposures).dropna()
    uniform = gross_normalised(scaled).dropna()
    common = base.index.intersection(uniform.index)
    assert np.allclose(
        base.loc[common].to_numpy(), uniform.loc[common].to_numpy(),
    )


def test_a_gated_state_produces_exactly_zero_exposure() -> None:
    exposures, assignment, gated = _gated()
    for column in exposures.columns:
        high = assignment.states[column] == float(VolState.HIGH)
        assert (gated[column][high].fillna(0.0) == 0.0).all()


def test_an_ungated_state_is_passed_through_untouched() -> None:
    """Geen dempingsfactor, geen herschaling: de view of geen view."""
    exposures, assignment, gated = _gated()
    for column in exposures.columns:
        keep = assignment.states[column].isin(
            [float(VolState.LOW), float(VolState.NORMAL)]
        )
        pd.testing.assert_series_equal(
            gated[column][keep], exposures[column][keep], check_names=False,
        )


def test_gating_every_state_is_refused() -> None:
    """Een afbeelding die alles dichtzet, is geen conditioneerder maar een
    uit-knop, en zij zou een lege reeks als 'resultaat' opleveren."""
    exposures, assignment = _fixture()
    with pytest.raises(DataContractError):
        gate_by_state(
            exposures, assignment,
            flat_states=frozenset({VolState.LOW, VolState.NORMAL, VolState.HIGH}),
        )


def test_burn_in_stays_nan_and_is_not_gated_to_zero() -> None:
    """Geen toestand is geen besluit. Nul zou 'wij kiezen vlak' betekenen."""
    exposures, assignment, gated = _gated()
    unknown = assignment.states.isna()
    assert gated.where(unknown).isna().all().all()


def test_the_gate_increases_concentration_and_that_is_measured() -> None:
    """Q9. De poort zet namen op nul; L7 hoogt de rest op. Die concentratie is
    een NIEUW risico dat de poort introduceert en het hoort in het artefact."""
    exposures, _, gated = _gated()
    report = concentration_report(exposures, gated)
    assert report.max_weight_gated >= report.max_weight_base - 1e-12
    assert report.effective_names_gated <= report.effective_names_base + 1e-12


def test_a_fully_gated_bar_is_flat_and_not_infinitely_levered() -> None:
    """De randgeval-test. Zijn ALLE namen op een bar gepoort, dan is het boek
    vlak. L7 mag daar niet op reageren met een deling door nul."""
    exposures, assignment = _fixture()
    all_high = assignment.states.copy()
    all_high.iloc[200:] = float(VolState.HIGH)
    from dataclasses import replace
    gated = gate_by_state(
        exposures, replace(assignment, states=all_high),
        flat_states=frozenset({VolState.HIGH}),
    )
    assert (gated.iloc[200:].abs().sum(axis=1) == 0.0).all()


def test_the_comparison_uses_one_shared_bar_set() -> None:
    """Fix ronde 1, bevinding P33. `_per_bar_metrics` blanked een hele bar
    zodra een symbool een onbekende toestand draagt (de conservatieve regel
    blijft staan). Zonder verdere maatregel zou het basisboek dan over MEER
    bars zijn gemeten dan het gepoorte boek -- een before/after die twee
    verschillende steekproeven naast elkaar zet. `concentration_report`
    moet daarom BEIDE boeken op het venster van het gepoorte boek meten, en
    dat venster moet zichtbaar zijn op het rapport."""
    exposures, _, gated = _gated()
    report = concentration_report(exposures, gated)

    expected_compared = int((~gated.isna().any(axis=1)).sum())
    assert report.n_bars_total == len(exposures)
    assert report.n_bars_compared == expected_compared
    # De opstartfase bestaat in deze fixture (min_periods=100): het venster
    # is dus een ECHTE deelverzameling en geen no-op-restrictie.
    assert report.n_bars_compared < report.n_bars_total


def test_gross_after_vol_target_is_an_indicator_not_a_measurement() -> None:
    """Fix ronde 1, bevinding P34. Met een CONSTANTE vol-target is de
    bruto-verhouding per constructie 1 zodra het boek niet leeg en niet
    onbekend is -- deze functie kent geen sigma-dak (haar handtekening is
    `(exposures, gated)`) en kan dus geen echte na-targeting-waarde
    uitrekenen. Op deze fixture is het gepoorte boek in 20 van de 300
    vergeleken bars volledig plat (zie de volgende test) en toch is elk van
    de 50/95/99-kwantielen nog exact 1 -- want 20/300 is minder dan de 5% en
    1% die respectievelijk p95 en p99 zouden kunnen buigen. Dat de waarde op
    GEEN van de drie kwantielen ooit tussen 0 en 1 in ligt, is precies het
    bewijs dat dit een binaire indicator is en geen gemeten grootheid."""
    exposures, _, gated = _gated()
    report = concentration_report(exposures, gated)

    assert report.gross_after_vol_target_base == 1.0
    assert report.gross_after_vol_target_gated == 1.0
    for quantile in (0.5, 0.95, 0.99):
        assert report.quantiles.loc[quantile, "gross_after_vol_target_base"] == 1.0
        assert report.quantiles.loc[quantile, "gross_after_vol_target_gated"] == 1.0


def test_a_fully_gated_run_reports_zero_gross_not_nan_or_one() -> None:
    """Fix ronde 1, bevinding 3. De `is_flat`-tak in `_per_bar_metrics`
    (bruto exact 0 -> 0,0, niet 1,0 en niet oneindig) wordt hier
    daadwerkelijk uitgeoefend via de publieke `concentration_report`, en niet
    alleen via het interne masker. De tweede helft van de reeks is voor
    BEIDE symbolen tegelijk HOOG, dus het gepoorte boek is daar op elke bar
    vlak -- de mediaan van `gross_after_vol_target_gated` moet dan 0,0 zijn:
    0,0 en niet 1,0 (het target zou ten onrechte 'gehaald' lijken) en niet
    NaN (een vlakke bar is een BESLOTEN bar, geen onbekende -- dat verschil
    is precies wat `test_burn_in_stays_nan_and_is_not_gated_to_zero` op
    `gate_by_state` bewaakt en wat dit op `concentration_report` bewaakt)."""
    exposures, assignment = _fixture()
    all_high = assignment.states.copy()
    all_high.iloc[200:] = float(VolState.HIGH)
    from dataclasses import replace
    gated = gate_by_state(
        exposures, replace(assignment, states=all_high),
        flat_states=frozenset({VolState.HIGH}),
    )

    report = concentration_report(exposures, gated)

    assert report.gross_after_vol_target_gated == 0.0
    assert report.gross_after_vol_target_base == 1.0
    assert np.isfinite(report.quantiles.to_numpy()).all()
