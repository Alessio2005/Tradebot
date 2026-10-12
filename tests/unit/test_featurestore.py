# tests/unit/test_featurestore.py
"""De feature-cache: upsert per maandpartitie, lezen over een venster, backfill.

De randgevallen die hier staan zijn de stille: een lege append, een partitie
zonder bestand, een leeg partitiebestand, en een upsert waarvan de dubbelen
binnen de NIEUWE rijen zitten in plaats van tussen oud en nieuw.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.featurestore.backfill import (
    backfill_from_dataframe,
    backfill_from_directory,
)
from tradebot.featurestore.store import FeatureStore


def _frame(start: str, periods: int, value: float = 1.0) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    return pd.DataFrame({"f1": np.full(periods, value)}, index=index)


class TestStore:
    def test_an_empty_append_writes_nothing(self, tmp_path: Path) -> None:
        store = FeatureStore(tmp_path)
        store.append(_frame("2024-01-01", 0), "BTC")
        assert not (tmp_path / "BTC").exists()
        assert store.read("BTC", pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2024-12-31", tz="UTC")).empty
        assert store.latest_timestamp("BTC") is None

    def test_duplicates_inside_the_new_rows_are_reported_without_a_range(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Overlap die niet uit de bestaande partitie komt: geen eerste en
        laatste overschreven bar, maar de waarschuwing blijft."""
        store = FeatureStore(tmp_path)
        store.append(_frame("2024-01-01", 3), "BTC")
        new = _frame("2024-01-10", 2, value=2.0)
        new = pd.concat([new, new.iloc[[0]]])
        with caplog.at_level(logging.WARNING):
            store.append(new, "BTC")
        assert "first=None, last=None" in caplog.text
        stored = store.read("BTC", pd.Timestamp("2024-01-01", tz="UTC"),
                            pd.Timestamp("2024-01-31", tz="UTC"))
        assert len(stored) == 5

    def test_reading_skips_missing_years_and_months_without_a_file(
        self, tmp_path: Path,
    ) -> None:
        store = FeatureStore(tmp_path)
        store.append(_frame("2024-03-01", 2), "BTC")
        (tmp_path / "BTC" / "year=2024" / "month=4").mkdir()
        window = store.read("BTC", pd.Timestamp("2023-01-01", tz="UTC"),
                            pd.Timestamp("2024-12-31", tz="UTC"))
        assert len(window) == 2

    def test_a_window_with_only_empty_month_directories_is_empty(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "BTC" / "year=2024" / "month=1").mkdir(parents=True)
        store = FeatureStore(tmp_path)
        assert store.read("BTC", pd.Timestamp("2024-01-01", tz="UTC"),
                          pd.Timestamp("2024-12-31", tz="UTC")).empty

    def test_an_empty_partition_does_not_set_the_latest_timestamp(
        self, tmp_path: Path,
    ) -> None:
        store = FeatureStore(tmp_path)
        store.append(_frame("2024-01-01", 3), "BTC")
        empty_dir = tmp_path / "BTC" / "year=2024" / "month=9"
        empty_dir.mkdir(parents=True)
        _frame("2024-09-01", 0).to_parquet(empty_dir / "data.parquet")
        assert store.latest_timestamp("BTC") == pd.Timestamp(
            "2024-01-03", tz="UTC")


class TestBackfill:
    def test_a_dataframe_is_appended_and_counted(self, tmp_path: Path) -> None:
        store = FeatureStore(tmp_path)
        assert backfill_from_dataframe(_frame("2024-01-01", 4), "ETH", store) == 4
        assert store.latest_timestamp("ETH") == pd.Timestamp(
            "2024-01-04", tz="UTC")

    @pytest.mark.parametrize("from_stem", [True, False])
    def test_a_directory_is_loaded_one_file_per_symbol(
        self, tmp_path: Path, from_stem: bool,
    ) -> None:
        source = tmp_path / "features"
        source.mkdir()
        _frame("2024-01-01", 3).to_parquet(source / "BTC.parquet")
        _frame("2024-01-01", 2).to_parquet(source / "ETH.parquet")
        written = backfill_from_directory(
            source, tmp_path / "store", symbol_from_stem=from_stem)
        if from_stem:
            assert written == {"BTC": 3, "ETH": 2}
        else:
            assert written == {"BTC.parquet": 3, "ETH.parquet": 2}
