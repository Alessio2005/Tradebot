"""R-1 op de toestand, en dit is de test die revisie 1 miste.

Truncatie-invariantie is NIET voldoende: een gelijktijdige toewijzing is
truncatie-invariant, want het afkappen van de reeks na bar t verandert de
toestand op bar t niet. De eigenschap die je wilt, is dat een schok in sigma op
bar t de toestand op bar t NIET raakt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.regime.state import VolState, assign_by_variance

IDX = pd.date_range("2022-01-01", periods=500, freq="D", tz="UTC")


def test_truncation_cannot_change_an_earlier_state() -> None:
    rng = np.random.default_rng(7)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 500))}, index=IDX)
    full = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    for cut in (200, 300, 400):
        truncated = assign_by_variance(
            sigma.iloc[:cut], low_q=0.25, high_q=0.75, min_periods=100,
        )
        pd.testing.assert_frame_equal(
            truncated.states, full.states.iloc[:cut], check_freq=False,
        )


def test_a_spike_in_sigma_cannot_change_the_state_of_its_own_bar() -> None:
    """De harde causaliteitstest. Zet op bar 300 een extreme sigma en toon aan
    dat de toestand op bar 300 ONVERANDERD blijft en dat de HIGH pas op bar 301
    verschijnt. Een gelijktijdige toewijzing faalt hier onmiddellijk."""
    rng = np.random.default_rng(8)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.05, 500))}, index=IDX)
    spiked = sigma.copy()
    spiked.iloc[300, 0] = 50.0

    base = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    after = assign_by_variance(spiked, low_q=0.25, high_q=0.75, min_periods=100)

    assert after.states.iloc[300, 0] == base.states.iloc[300, 0]
    assert after.states.iloc[301, 0] == float(VolState.HIGH)


def test_the_quantile_boundaries_use_only_past_bars() -> None:
    """Tweede helft van Q1. Niet alleen sigma moet gelagged zijn, ook de
    DREMPEL. Een reeks die na bar 400 explodeert, mag de toestanden vóór 400
    niet verschuiven -- ook niet via het kwantiel."""
    rng = np.random.default_rng(9)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.05, 500))}, index=IDX)
    exploded = sigma.copy()
    exploded.iloc[400:, 0] *= 20.0

    base = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    after = assign_by_variance(exploded, low_q=0.25, high_q=0.75, min_periods=100)

    pd.testing.assert_frame_equal(
        after.states.iloc[:400], base.states.iloc[:400], check_freq=False,
    )


def test_lag_zero_reintroduces_the_leak_and_the_test_catches_it() -> None:
    """De negatieve controle op de causaliteitstest zelf."""
    rng = np.random.default_rng(8)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.05, 500))}, index=IDX)
    spiked = sigma.copy()
    spiked.iloc[300, 0] = 50.0
    leaky = assign_by_variance(
        spiked, low_q=0.25, high_q=0.75, min_periods=100, lag=0,
    )
    assert leaky.states.iloc[300, 0] == float(VolState.HIGH)
