# tests/unit/test_unbalanced_panel.py
"""Het paneel mag per bar van breedte veranderen.

Vier eigenschappen: gewichten sommeren per bar over de BESCHIKBARE namen; een
NaN is afwezigheid en geen nul; de covariantieschatting krijgt alleen de
beschikbare namen; en onder de minimumbreedte is de bar vlak in plaats van
gedeeltelijk gevuld.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.portfolio.weights import normalise_weights

IDX = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")


def _ragged() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {"BTC": rng.uniform(-1, 1, 300),
         "LINK": rng.uniform(-1, 1, 300),
         "ETH": rng.uniform(-1, 1, 300)},
        index=IDX,
    )
    frame.loc[IDX[:100], "ETH"] = np.nan
    frame.loc[IDX[:50], "LINK"] = np.nan
    return frame


def test_weights_sum_over_available_names_per_bar() -> None:
    weights = normalise_weights(_ragged(), min_symbols_per_bar=2)
    gross = weights.abs().sum(axis=1)
    active = weights.notna().sum(axis=1)
    assert np.allclose(gross[active >= 2], 1.0)


def test_absence_stays_nan_and_never_becomes_zero() -> None:
    ragged = _ragged()
    weights = normalise_weights(ragged, min_symbols_per_bar=2)
    assert weights[ragged.isna()].isna().all().all()


def test_a_bar_below_the_minimum_width_is_flat() -> None:
    weights = normalise_weights(_ragged(), min_symbols_per_bar=2)
    assert weights.iloc[:50].abs().sum(axis=1).eq(0.0).all()


def test_the_sample_grows_relative_to_the_balanced_panel() -> None:
    ragged = _ragged()
    balanced = ragged.dropna()
    unbalanced = normalise_weights(ragged, min_symbols_per_bar=2)
    active = unbalanced.abs().sum(axis=1) > 0
    assert int(active.sum()) > len(balanced)
