"""Phase 1, deliverable 13 - onveranderlijkheid van de PIT-store.

Geschreven en groen VOORDAT er ook maar één byte data in de store is geschreven,
conform stap 3: *"Test de onveranderlijkheid vóór je er data in schrijft."*
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.data.pit_store import PartitionRef, PitStore
from tradebot.utils.failfast import DataContractError
from tradebot.utils.hashing import dataframe_content_hash

REF = PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", "2024-01-01")


def frame(n: int = 5, seed: int = 0, close_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ev = np.arange(n, dtype=np.int64) * 86_400_000_000_000
    return pd.DataFrame({
        "event_ts_ns": ev,
        "asof_ts_ns": ev + 86_400_000_000_000,   # bekend één dag later
        "open": rng.normal(100, 1, n),
        "high": rng.normal(101, 1, n),
        "low": rng.normal(99, 1, n),
        "close": rng.normal(100, 1, n) + close_shift,
        "volume": rng.uniform(1, 10, n),
    })


@pytest.fixture()
def store(tmp_path):
    return PitStore(tmp_path / "pit_store")


class TestImmutability:
    def test_first_write_succeeds(self, store: PitStore) -> None:
        h = store.write(frame(), REF, source="test")
        assert h and store.exists(REF)

    def test_rewrite_with_identical_content_is_a_noop(self, store: PitStore) -> None:
        """Een herstarte ingestion-run moet veilig zijn."""
        h1 = store.write(frame(), REF, source="test")
        h2 = store.write(frame(), REF, source="test")
        assert h1 == h2

    def test_rewrite_with_different_content_crashes(self, store: PitStore) -> None:
        """Exit criterium 5: overschrijven crasht met DataContractError."""
        store.write(frame(), REF, source="test")
        with pytest.raises(DataContractError, match="APPEND-ONLY"):
            store.write(frame(close_shift=1.0), REF, source="test")

    def test_rewrite_with_more_rows_crashes(self, store: PitStore) -> None:
        store.write(frame(n=5), REF, source="test")
        with pytest.raises(DataContractError, match="APPEND-ONLY"):
            store.write(frame(n=6), REF, source="test")

    def test_crash_message_names_both_hashes(self, store: PitStore) -> None:
        """De crash moet zonder debugger diagnosticeerbaar zijn."""
        store.write(frame(), REF, source="test")
        with pytest.raises(DataContractError) as ei:
            store.write(frame(close_shift=2.0), REF, source="test")
        msg = str(ei.value)
        assert "existing_hash=" in msg and "new_hash=" in msg
        assert str(REF) in msg

    def test_original_content_survives_a_rejected_overwrite(self, store: PitStore) -> None:
        """Een geweigerde schrijfactie mag de bestaande partitie niet aantasten."""
        store.write(frame(), REF, source="test")
        before = store.read(REF)
        with pytest.raises(DataContractError):
            store.write(frame(close_shift=3.0), REF, source="test")
        pd.testing.assert_frame_equal(before, store.read(REF))

    def test_there_is_no_overwrite_escape_hatch(self) -> None:
        """`write` mag geen force/overwrite-parameter kennen."""
        import inspect

        params = set(inspect.signature(PitStore.write).parameters)
        assert not (params & {"overwrite", "force", "replace", "mode"})

    def test_drop_partition_requires_explicit_acknowledgement(self, store: PitStore) -> None:
        store.write(frame(), REF, source="test")
        with pytest.raises(DataContractError, match="append-only"):
            store.drop_partition(REF)
        store.drop_partition(REF, i_understand=True)
        assert not store.exists(REF)


class TestContractValidation:
    def test_empty_frame_is_rejected(self, store: PitStore) -> None:
        """Een lege dataset is geen resultaat maar een gefaalde fetch."""
        with pytest.raises(DataContractError, match="LEGE partitie"):
            store.write(frame().iloc[0:0], REF, source="test")

    def test_missing_time_columns_are_rejected(self, store: PitStore) -> None:
        with pytest.raises(DataContractError, match="verplichte tijdkolommen"):
            store.write(frame().drop(columns=["asof_ts_ns"]), REF, source="test")

    def test_datetime_instead_of_int64_is_rejected(self, store: PitStore) -> None:
        """Sectie 7.2: UTC Unix nanoseconden, int64 — geen datetime."""
        df = frame()
        df["event_ts_ns"] = pd.to_datetime(df["event_ts_ns"], unit="ns", utc=True)
        with pytest.raises(DataContractError, match="int64"):
            store.write(df, REF, source="test")

    def test_float_timestamps_are_rejected(self, store: PitStore) -> None:
        df = frame()
        df["event_ts_ns"] = df["event_ts_ns"].astype(float)
        with pytest.raises(DataContractError, match="int64"):
            store.write(df, REF, source="test")

    def test_unsorted_event_ts_is_rejected(self, store: PitStore) -> None:
        df = frame().iloc[::-1].reset_index(drop=True)
        with pytest.raises(DataContractError, match="niet oplopend gesorteerd"):
            store.write(df, REF, source="test")

    def test_asof_before_event_is_rejected(self, store: PitStore) -> None:
        """asof < event betekent: bekend voordat het gebeurde. Lookahead."""
        df = frame()
        df.loc[2, "asof_ts_ns"] = int(df.loc[2, "event_ts_ns"]) - 1
        with pytest.raises(DataContractError, match="lookahead"):
            store.write(df, REF, source="test")


class TestRoundTripAndHash:
    def test_roundtrip_preserves_content_hash(self, store: PitStore) -> None:
        """Deliverable 14: ingest -> hash -> persist -> reload -> identieke hash."""
        df = frame()
        written = store.write(df, REF, source="test")
        reloaded = store.read(REF)
        assert dataframe_content_hash(reloaded.reset_index(drop=True)) == written

    def test_meta_records_provenance(self, store: PitStore) -> None:
        h = store.write(frame(), REF, source="bybit-v5")
        meta = store.read_meta(REF)
        assert meta["data_hash"] == h
        assert meta["source"] == "bybit-v5"
        assert meta["rows"] == 5
        assert meta["symbol"] == "BTCUSDT"

    def test_two_independent_writes_give_the_same_hash(self, tmp_path) -> None:
        """Stap 9: twee onafhankelijke runs op dezelfde bronperiode -> zelfde hash."""
        a = PitStore(tmp_path / "a")
        b = PitStore(tmp_path / "b")
        assert a.write(frame(), REF, source="run-1") == \
               b.write(frame(), REF, source="run-2")

    def test_load_crashes_instead_of_returning_empty(self, store: PitStore) -> None:
        """Nooit een lege DataFrame teruggeven en doorgaan."""
        with pytest.raises(DataContractError, match="GEEN lege"):
            store.load("crypto", "ohlcv", "NOPEUSDT", "1d")

    def test_read_missing_partition_crashes(self, store: PitStore) -> None:
        with pytest.raises(DataContractError, match="bestaat niet"):
            store.read(REF)


class TestPartitioning:
    def test_partition_path_encodes_the_key(self) -> None:
        assert str(REF) == (
            "asset_class=crypto/dataset=ohlcv/symbol=BTCUSDT/"
            "granularity=1d/date=2024-01-01")

    def test_scan_finds_written_partitions(self, store: PitStore) -> None:
        for d in ("2024-01-01", "2024-01-02"):
            store.write(frame(), PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", d),
                        source="test")
        store.write(frame(), PartitionRef("crypto", "ohlcv", "ETHUSDT", "1d", "2024-01-01"),
                    source="test")
        assert len(store.partitions("crypto")) == 3
        assert len(store.partitions("crypto", "ohlcv")) == 3
        assert len(store.partitions("crypto", "ohlcv", "BTCUSDT")) == 2
        assert len(store.partitions("crypto", "ohlcv", "ETHUSDT")) == 1

    def test_load_concatenates_in_time_order(self, store: PitStore) -> None:
        for i, d in enumerate(("2024-01-02", "2024-01-01")):  # bewust omgekeerd
            df = frame()
            df["event_ts_ns"] += i * 10**15
            df["asof_ts_ns"] += i * 10**15
            store.write(df, PartitionRef("crypto", "ohlcv", "BTCUSDT", "1d", d), source="test")
        out = store.load("crypto", "ohlcv", "BTCUSDT", "1d")
        assert out["event_ts_ns"].is_monotonic_increasing
        assert len(out) == 10

    def test_partition_ref_is_frozen(self) -> None:
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            REF.symbol = "ETHUSDT"  # type: ignore[misc]
