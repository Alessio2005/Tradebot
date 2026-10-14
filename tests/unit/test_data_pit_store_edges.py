# tests/unit/test_data_pit_store_edges.py
"""Restgaten van `data/pit_store.py`: NaN-tijdstempels, scan-filters, vreemde mappen, metadata en opruimen.

Het append-only contract zelf staat in `test_pit_store_immutability.py`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tradebot.data.pit_store import PartitionRef, PitStore
from tradebot.utils.failfast import DataContractError

DAY = 86_400_000_000_000
REF = PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", "2024-01-01")


def _frame(n: int = 4, offset: int = 0) -> pd.DataFrame:
    ev = (np.arange(n, dtype=np.int64) + offset) * DAY
    return pd.DataFrame({"event_ts_ns": ev, "asof_ts_ns": ev + DAY, "close": np.linspace(100.0, 101.0, n)})


@pytest.fixture()
def store(tmp_path) -> PitStore:
    return PitStore(tmp_path / "pit")


class TestValidationEdges:
    @pytest.mark.parametrize("col", ["event_ts_ns", "asof_ts_ns"])
    def test_a_missing_timestamp_inside_an_integer_column_is_refused(self, store: PitStore, col: str) -> None:
        df = _frame().astype({"event_ts_ns": "Int64", "asof_ts_ns": "Int64"})
        df.loc[1, col] = pd.NA
        with pytest.raises(DataContractError, match=f"{col} bevat NaN"):
            store.write(df, REF, source="t")
        assert not store.exists(REF)


class TestWriteMetadata:
    def test_extra_meta_is_stored_next_to_the_provenance(self, store: PitStore) -> None:
        store.write(_frame(), REF, source="bybit", extra_meta={"vendor_version": "v5", "rows": 999})
        meta = store.read_meta(REF)
        assert meta["vendor_version"] == "v5" and meta["source"] == "bybit"
        assert meta["event_ts_ns_min"] == 0 and meta["event_ts_ns_max"] == 3 * DAY and meta["written_at_ns"] > 0
        assert meta["columns"] == ["event_ts_ns", "asof_ts_ns", "close"]

    def test_the_temporary_file_never_survives_a_write(self, store: PitStore) -> None:
        store.write(_frame(), REF, source="t")
        assert sorted(p.name for p in store.partition_dir(REF).iterdir()) == ["_meta.json", "data.parquet"]

    def test_a_partition_without_metadata_has_no_provenance(self, store: PitStore) -> None:
        store.write(_frame(), REF, source="t")
        store.meta_path(REF).unlink()
        with pytest.raises(DataContractError, match="geen metadata"):
            store.read_meta(REF)


class TestScan:
    def test_a_missing_root_has_no_partitions(self, tmp_path) -> None:
        assert PitStore(tmp_path / "does-not-exist").partitions() == []

    def test_every_filter_narrows_the_scan(self, store: PitStore) -> None:
        refs = [PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", "2024-01-01"),
                PartitionRef("crypto", "ohlcv", "BTCUSDT", "1h", "2024-01-01"),
                PartitionRef("crypto", "funding", "BTCUSDT", "8h", "2024-01-01"),
                PartitionRef("equities", "ohlcv", "SPY", "1d", "2024-01-01")]
        for r in refs:
            store.write(_frame(), r, source="t")
        assert len(store.partitions()) == 4
        assert len(store.partitions(asset_class="equities")) == 1
        assert len(store.partitions(dataset="funding")) == 1
        assert len(store.partitions(symbol="SPY")) == 1
        assert [r.granularity for r in store.partitions(granularity="1h")] == ["1h"]
        assert store.partitions(asset_class="crypto", dataset="ohlcv", symbol="BTCUSDT", granularity="4h") == []

    def test_files_that_do_not_follow_the_layout_are_ignored(self, store: PitStore) -> None:
        store.write(_frame(), REF, source="t")
        (store.root / "data.parquet").write_bytes(b"x")                          # te ondiep
        stray = store.root / "a=1" / "b=2" / "c=3" / "d=4" / "e=5"
        stray.mkdir(parents=True)
        (stray / "data.parquet").write_bytes(b"x")                               # verkeerde sleutels
        assert store.partitions() == [REF]


class TestLoadAndHash:
    def test_a_series_is_the_concatenation_of_its_partitions_in_time_order(self, store: PitStore) -> None:
        store.write(_frame(offset=4), PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", "2024-02-01"), source="t")
        store.write(_frame(offset=0), REF, source="t")
        out = store.load("crypto", "ohlcv", "BTCUSDT", "1d")
        assert out["event_ts_ns"].tolist() == [i * DAY for i in range(8)]

    def test_the_series_hash_is_stable_and_changes_with_the_content(self, tmp_path) -> None:
        a, b = PitStore(tmp_path / "a"), PitStore(tmp_path / "b")
        a.write(_frame(), REF, source="t")
        b.write(_frame(), REF, source="t")
        assert a.series_hash("crypto", "ohlcv", "BTCUSDT", "1d") == b.series_hash("crypto", "ohlcv", "BTCUSDT", "1d")
        b.write(_frame(offset=4), PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", "2024-02-01"), source="t")
        assert a.series_hash("crypto", "ohlcv", "BTCUSDT", "1d") != b.series_hash("crypto", "ohlcv", "BTCUSDT", "1d")


class TestDropPartition:
    def test_an_acknowledged_drop_removes_the_partition_and_a_second_one_is_a_noop(self, store: PitStore) -> None:
        store.write(_frame(), REF, source="t")
        store.drop_partition(REF, i_understand=True)
        assert not store.exists(REF)
        store.drop_partition(REF, i_understand=True)

    def test_the_metadata_file_is_valid_json(self, store: PitStore) -> None:
        store.write(_frame(), REF, source="t")
        assert json.loads(store.meta_path(REF).read_text(encoding="utf-8"))["data_hash"]
