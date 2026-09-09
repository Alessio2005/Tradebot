"""Het toestandscontract. Vijf eigenschappen, elk met een eigen falsificatie:
de ordening is op variantie en niet op gemiddelde; de toestand is discreet; de
opstartfase is NaN en geen ingevulde toestand; de toewijzing draagt haar
herkomst mee; en de kwantielen moeten oplopen."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import StateAssignment, VolState, assign_by_variance
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")


def _sigma(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"A": np.abs(rng.normal(0.5, 0.15, 400)),
         "B": np.abs(rng.normal(0.8, 0.25, 400))},
        index=IDX,
    )


def test_states_are_ordered_by_variance_not_by_mean() -> None:
    """HIGH is de toestand met de HOOGSTE sigma. Ordenen op gemiddeld rendement
    zou een richtingsclaim zijn, en die wordt op dit paneel niet gedragen: na
    datumclustering en N_eff-deflatie haalt geen enkele richtings-t 0,8."""
    sigma = _sigma()
    result = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    known = sigma.shift(1)
    for column in sigma.columns:
        states = result.states[column]
        mask = states.notna()
        high = known[column][mask][states[mask] == VolState.HIGH]
        low = known[column][mask][states[mask] == VolState.LOW]
        assert high.mean() > low.mean()
    assert result.ordering == "variance"


def test_states_are_discrete() -> None:
    result = assign_by_variance(_sigma(), low_q=0.25, high_q=0.75, min_periods=100)
    values = result.states.to_numpy()
    finite = values[np.isfinite(values)]
    assert set(np.unique(finite)) <= {0.0, 1.0, 2.0}


def test_burn_in_is_nan_never_a_default_state() -> None:
    """Een toestand tijdens de opstartfase is een INGEVULDE toestand, en die
    zou als NORMAAL worden gelezen -- een besluit dat niemand heeft genomen."""
    result = assign_by_variance(_sigma(), low_q=0.25, high_q=0.75, min_periods=100)
    head = result.states.iloc[:100]
    assert head.isna().all().all()


def test_assignment_carries_its_source() -> None:
    result = assign_by_variance(
        _sigma(), low_q=0.25, high_q=0.75, min_periods=100, source="ewma_0.94",
    )
    assert result.source == "ewma_0.94"
    assert isinstance(result, StateAssignment)


def test_quantiles_must_be_ordered() -> None:
    with pytest.raises(DataContractError):
        assign_by_variance(_sigma(), low_q=0.75, high_q=0.25, min_periods=100)


def test_occupancy_and_episodes_are_both_reported() -> None:
    """Q7. Bars en episodes zijn verschillende grootheden, en de tweede is de
    effectieve steekproefomvang van een persistente toestand."""
    result = assign_by_variance(_sigma(), low_q=0.25, high_q=0.75, min_periods=100)
    occ, epi = result.occupancy(), result.episodes()
    assert set(occ.columns) == {"LOW", "NORMAL", "HIGH"}
    assert set(epi.columns) == {"LOW", "NORMAL", "HIGH"}
    assert (epi.to_numpy() <= occ.to_numpy() * len(result.states)).all()
