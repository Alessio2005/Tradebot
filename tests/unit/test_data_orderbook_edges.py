# tests/unit/test_data_orderbook_edges.py
"""Restgaten van `data/orderbook.py`: lege boekkanten, nulvolume en een lege snapshot-reeks.

De happy path (best bid/ask, spread, imbalance-bereik, rolling features) staat in `test_wave6.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.data.orderbook import (
    OrderBookSnapshot,
    compute_book_imbalance,
    compute_depth_weighted_midprice,
    rolling_book_features,
)

TS = pd.Timestamp("2024-01-01", tz="UTC")
NO_LEVELS = np.empty((0, 2))


def _snap(bids, asks, ts: pd.Timestamp = TS) -> OrderBookSnapshot:
    return OrderBookSnapshot(ts, np.asarray(bids, dtype=float).reshape(-1, 2), np.asarray(asks, dtype=float).reshape(-1, 2))


class TestOneSidedBooks:
    def test_an_empty_side_has_no_best_price_mid_or_spread(self) -> None:
        one_sided = _snap([[100.0, 1.0]], NO_LEVELS)
        assert one_sided.best_bid() == 100.0 and np.isnan(one_sided.best_ask())
        assert np.isnan(one_sided.mid_price()) and np.isnan(one_sided.spread())
        assert np.isnan(_snap(NO_LEVELS, [[101.0, 1.0]]).mid_price())

    def test_imbalance_of_an_empty_side_is_fully_one_sided(self) -> None:
        assert compute_book_imbalance(_snap([[100.0, 4.0]], NO_LEVELS)) == pytest.approx(1.0)
        assert compute_book_imbalance(_snap(NO_LEVELS, [[101.0, 4.0]])) == pytest.approx(-1.0)
        assert compute_book_imbalance(_snap(NO_LEVELS, NO_LEVELS)) == 0.0

    def test_imbalance_only_counts_the_requested_depth(self) -> None:
        snap = _snap([[100.0, 1.0], [99.0, 100.0]], [[101.0, 1.0], [102.0, 1.0]])
        assert compute_book_imbalance(snap, depth_levels=1) == pytest.approx(0.0)
        assert compute_book_imbalance(snap, depth_levels=2) > 0.9


class TestDepthWeightedMid:
    def test_an_empty_side_falls_back_to_the_plain_mid(self) -> None:
        assert np.isnan(compute_depth_weighted_midprice(_snap([[100.0, 1.0]], NO_LEVELS)))

    def test_prices_are_weighted_by_the_quantity_on_each_level(self) -> None:
        snap = _snap([[100.0, 3.0], [99.0, 1.0]], [[101.0, 1.0], [103.0, 1.0]])
        assert compute_depth_weighted_midprice(snap) == pytest.approx((99.75 + 102.0) / 2)

    def test_levels_without_quantity_fall_back_to_the_best_price(self) -> None:
        snap = _snap([[100.0, 0.0], [99.0, 0.0]], [[101.0, 0.0]])
        assert compute_depth_weighted_midprice(snap) == pytest.approx(100.5)


class TestRollingFeatures:
    def test_an_empty_sequence_gives_an_empty_frame_with_the_documented_columns(self) -> None:
        df = rolling_book_features([])
        assert df.empty and list(df.columns) == ["depth_mid_price", "spread_bps", "book_imbalance", "obi_momentum"]

    def test_the_first_windows_are_nan_until_enough_snapshots_exist(self) -> None:
        snaps = [_snap([[100.0, 1.0 + i]], [[101.0, 1.0]], TS + pd.Timedelta(minutes=i)) for i in range(12)]
        df = rolling_book_features(snaps, window=20)                    # min_periods = max(5, 20 // 4) = 5
        assert df["book_imbalance"].iloc[:4].isna().all() and df["book_imbalance"].iloc[4:].notna().all()
        assert df["spread_bps"].iloc[0] == pytest.approx(1.0 / 100.5 * 10_000)

    def test_a_book_without_asks_has_no_spread(self) -> None:
        snaps = [_snap([[100.0, 1.0]], NO_LEVELS, TS + pd.Timedelta(minutes=i)) for i in range(6)]
        assert rolling_book_features(snaps, window=5)["spread_bps"].isna().all()
