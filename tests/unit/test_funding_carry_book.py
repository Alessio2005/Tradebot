# tests/unit/test_funding_carry_book.py
"""Het carryboek van H-11.1: cross-sectioneel, dollar-neutraal, parameterloos. Stap 7.

De constructie is vastgelegd in de bevroren pre-registratie
`7b602047ebffbab5f6ea155303d69f98`: op elke herbalanceringsbar aflopend
gerangschikt op de BEKENDE carry, de bovenste helft short, de onderste helft
long, allemaal met dezelfde absolute exposure, vastgehouden tot de volgende
herbalancering. Geen drempel, geen venster, geen z-score: elke drempel is een
parameter en elke parameter een trial (R-2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha.funding_carry_book import (
    carry_weights,
    known_carry,
    permuted_known_carry,
)
from tradebot.utils.failfast import DataContractError

SYMBOLS = ["AVAXUSDT", "BTCUSDT", "DOTUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT"]


def _panel(n: int = 30, seed: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC", name="asof_ts")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(2e-4, 3e-4, (n, len(SYMBOLS))),
                        index=idx, columns=SYMBOLS)


class TestKnownCarry:
    def test_it_is_the_panel_one_bar_later(self) -> None:
        panel = _panel()
        known = known_carry(panel, lag_bars=1)
        assert known.iloc[0].isna().all()
        pd.testing.assert_frame_equal(known.iloc[1:], panel.iloc[:-1].set_axis(
            panel.index[1:]))

    def test_a_negative_lag_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="lag"):
            known_carry(_panel(), lag_bars=-1)


class TestTheBook:
    def test_the_highest_carry_is_short_and_the_lowest_is_long(self) -> None:
        known = _panel()
        w = carry_weights(known, holding_period=1)
        first = known.index[0]
        order = known.loc[first].sort_values(ascending=False).index
        assert (w.loc[first, order[:3]] == -1.0).all()
        assert (w.loc[first, order[3:]] == 1.0).all()

    def test_it_is_dollar_neutral_on_every_bar(self) -> None:
        w = carry_weights(_panel(), holding_period=3)
        assert (w.sum(axis=1) == 0.0).all()
        assert set(np.unique(w.to_numpy())) <= {-1.0, 0.0, 1.0}

    def test_it_holds_between_rebalances(self) -> None:
        """Elke h bars een besluit; daartussen verandert het boek niet."""
        w = carry_weights(_panel(), holding_period=5)
        changes = (w.diff().abs().sum(axis=1) > 0).to_numpy()
        assert not changes[1:5].any() and not changes[6:10].any()

    def test_it_starts_on_the_first_bar_where_every_name_has_a_carry(self) -> None:
        known = known_carry(_panel(), lag_bars=1)
        w = carry_weights(known, holding_period=2)
        assert (w.iloc[0] == 0.0).all()
        assert w.iloc[1].abs().sum() == len(SYMBOLS)

    def test_ties_are_broken_on_the_symbol_name_not_on_order(self) -> None:
        """Gelijke carry: de volgorde volgt de naam, niet de kolomvolgorde."""
        idx = pd.date_range("2022-01-01", periods=2, freq="D", tz="UTC")
        flat = pd.DataFrame(1e-4, index=idx, columns=SYMBOLS)
        a = carry_weights(flat, holding_period=1)
        b = carry_weights(flat[SYMBOLS[::-1]], holding_period=1)[SYMBOLS]
        pd.testing.assert_frame_equal(a, b)

    def test_an_odd_universe_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="even"):
            carry_weights(_panel()[SYMBOLS[:5]], holding_period=1)

    def test_a_non_positive_holding_period_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="houdduur"):
            carry_weights(_panel(), holding_period=0)


class TestThePermutation:
    """Negatieve controle 2: de bekende carry per symbool geschud in de tijd."""

    def test_it_keeps_each_symbols_values_and_moves_them(self) -> None:
        known = known_carry(_panel(), lag_bars=1)
        shuffled = permuted_known_carry(known, seed=20260923)
        for s in SYMBOLS:
            assert sorted(shuffled[s].dropna()) == sorted(known[s].dropna())
        assert not shuffled.equals(known)
        assert shuffled.iloc[0].isna().all()

    def test_it_is_reproducible_from_its_seed(self) -> None:
        known = known_carry(_panel(), lag_bars=1)
        pd.testing.assert_frame_equal(permuted_known_carry(known, seed=7),
                                      permuted_known_carry(known, seed=7))
