# tests/lookahead/test_decision_frequency_causality.py
"""R-1 op de vasthoudoperatie -- de test die de resample-valkuil vangt."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.portfolio.decision_frequency import hold_decision

IDX = pd.date_range("2022-01-01", periods=200, freq="D", tz="UTC")


def test_a_future_change_cannot_move_an_earlier_held_value() -> None:
    rng = np.random.default_rng(7)
    exposures = pd.DataFrame({"A": rng.uniform(-1, 1, 200)}, index=IDX)
    for k in (2, 5, 10):
        base = hold_decision(exposures, k=k)
        for cut in (50, 100, 150):
            truncated = hold_decision(exposures.iloc[:cut], k=k)
            pd.testing.assert_frame_equal(
                truncated, base.iloc[:cut], check_freq=False,
            )


def test_a_perturbation_at_bar_t_cannot_move_bars_before_t() -> None:
    rng = np.random.default_rng(8)
    exposures = pd.DataFrame({"A": rng.uniform(-1, 1, 200)}, index=IDX)
    perturbed = exposures.copy()
    perturbed.iloc[123, 0] = 99.0
    for k in (2, 5, 10):
        pd.testing.assert_frame_equal(
            hold_decision(perturbed, k=k).iloc[:123],
            hold_decision(exposures, k=k).iloc[:123],
            check_freq=False,
        )


def _end_anchored_hold(exposures: pd.DataFrame, k: int) -> pd.DataFrame:
    """De valkuil, expliciet gemaakt: blokken vanaf de LAATSTE bar terugleggen.

    Dit is `resample(origin="end")` in de kern. Het is de variant die intuitief
    net zo onschuldig lijkt als `anchor="first_bar"` -- elke bar krijgt de
    waarde van het begin van zijn eigen blok -- maar de blokGRENZEN hangen af
    van waar de steekproef EINDIGT, en dat is de toekomst. Alleen in productie
    zou dit een lookahead zijn; hier is het de negatieve controle.
    """
    n = len(exposures)
    offset = (-n) % k  # waar het eerste (afgeknotte) blok begint
    starts = np.maximum(((np.arange(n) + offset) // k) * k - offset, 0)
    return pd.DataFrame(
        exposures.to_numpy()[starts], index=exposures.index,
        columns=exposures.columns,
    )


def test_the_truncation_guard_goes_red_on_an_end_anchored_hold() -> None:
    """De negatieve controle op de causaliteitstest zelf.

    Zonder dit bewijs is de test hierboven niet onderscheidend: een test die
    altijd groen is, bewijst niet dat `hold_decision` causaal is maar dat de
    vergelijking niets meet.

    De keuze van k en cut is hier GEEN detail. De eindankering verschuift de
    blokgrenzen alleen wanneer `(-n) % k` verandert door het afknotten, en bij
    k=3 met cut=50 gebeurt dat NIET: 200 % 3 = 2 en 50 % 3 = 2, dus beide
    offsets zijn 1 en het rooster is over de eerste 50 bars identiek. Die
    combinatie stond hier eerst, met de onjuiste onderbouwing dat de resten 2
    en 1 zouden zijn -- en zij liet deze negatieve controle groen worden, wat
    precies de fout is die een negatieve controle hoort te vangen. Bij k=4 is
    200 % 4 = 0 tegen 50 % 4 = 2, dus de offsets zijn 0 en 2 en het rooster
    schuift werkelijk op.
    """
    rng = np.random.default_rng(9)
    exposures = pd.DataFrame({"A": rng.uniform(-1, 1, 200)}, index=IDX)
    base = _end_anchored_hold(exposures, 4)
    truncated = _end_anchored_hold(exposures.iloc[:50], 4)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            truncated, base.iloc[:50], check_freq=False,
        )
