"""Phase 1, deliverable 14 — ingest → hash → persist → reload → hash.

Bewijst dat de `data_hash` een roundtrip door Parquet overleeft, en dat twee
onafhankelijke ingestion-runs op dezelfde bronperiode dezelfde hash opleveren
(stap 9). Draait op een SYNTHETISCHE bron, zodat de test geen netwerk nodig
heeft en deterministisch is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data.ingestion.contract import (
    IngestionSource,
    IngestionSpec,
    run_ingestion,
)
from tradebot.data.pit_store import PitStore
from tradebot.data.validation import OHLCV_SPEC, GapLedger
from tradebot.utils.failfast import DataContractError
from tradebot.utils.hashing import dataframe_content_hash

pytestmark = pytest.mark.integration

DAY_NS = 86_400_000_000_000


class SyntheticOhlcv(IngestionSource):
    """Deterministische bron: dezelfde seed levert exact dezelfde bars."""

    def __init__(self, n: int = 400, seed: int = 7, shift: float = 0.0) -> None:
        self.n, self.seed, self.shift = n, seed, shift
        self.fetch_calls = 0

    def fetch(self, spec: IngestionSpec) -> pd.DataFrame:
        self.fetch_calls += 1
        rng = np.random.default_rng(self.seed)
        # 2021-01-01 als startpunt, ruim in het verleden zodat geen enkele bar
        # door de "nog niet gesloten"-guard wordt verwijderd.
        start = int(pd.Timestamp("2021-01-01", tz="UTC").value)
        ev = start + np.arange(self.n, dtype=np.int64) * DAY_NS
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, self.n))) + self.shift
        return pd.DataFrame({
            "start_ns": ev, "close": close,
            "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
            "volume": rng.uniform(1, 10, self.n),
        })

    def normalise(self, raw: pd.DataFrame, spec: IngestionSpec) -> pd.DataFrame:
        return pd.DataFrame({
            "event_ts_ns": raw["start_ns"].astype("int64"),
            "asof_ts_ns": (raw["start_ns"] + DAY_NS).astype("int64"),
            "open": raw["open"], "high": raw["high"], "low": raw["low"],
            "close": raw["close"], "volume": raw["volume"],
            "symbol": spec.symbol,
        })


def make_spec(**over) -> IngestionSpec:
    base = dict(
        asset_class="crypto", dataset="ohlcv", symbol="TESTUSDT",
        granularity="1d", series_spec=OHLCV_SPEC, gap_policy="reject",
        max_abs_log_return=0.35, allow_price_jumps=True,
        source_name="synthetic",
    )
    base.update(over)
    return IngestionSpec(**base)  # type: ignore[arg-type]


@pytest.fixture()
def env(tmp_path: Path):
    return PitStore(tmp_path / "pit"), GapLedger(tmp_path / "gaps.jsonl")


class TestRoundTrip:
    def test_hash_survives_persist_and_reload(self, env) -> None:
        store, ledger = env
        res = run_ingestion(SyntheticOhlcv(), make_spec(), store, ledger)
        reloaded = store.load("crypto", "ohlcv", "TESTUSDT", "1d")
        assert dataframe_content_hash(reloaded) == res.data_hash

    def test_two_independent_runs_agree(self, tmp_path: Path) -> None:
        """Stap 9: non-determinisme in de pipeline moet hier zichtbaar worden."""
        a = run_ingestion(SyntheticOhlcv(), make_spec(),
                          PitStore(tmp_path / "a"), GapLedger(tmp_path / "ga"))
        b = run_ingestion(SyntheticOhlcv(), make_spec(),
                          PitStore(tmp_path / "b"), GapLedger(tmp_path / "gb"))
        assert a.data_hash == b.data_hash
        assert a.n_rows == b.n_rows

    def test_rerun_into_the_same_store_is_a_noop(self, env) -> None:
        store, ledger = env
        a = run_ingestion(SyntheticOhlcv(), make_spec(), store, ledger)
        b = run_ingestion(SyntheticOhlcv(), make_spec(), store, ledger)
        assert a.data_hash == b.data_hash

    def test_changed_content_is_rejected_by_the_store(self, env) -> None:
        """Append-only: dezelfde partitie met andere inhoud crasht."""
        store, ledger = env
        run_ingestion(SyntheticOhlcv(), make_spec(), store, ledger)
        with pytest.raises(DataContractError, match="APPEND-ONLY"):
            run_ingestion(SyntheticOhlcv(shift=5.0), make_spec(), store, ledger)

    def test_different_content_gives_a_different_hash(self, tmp_path: Path) -> None:
        a = run_ingestion(SyntheticOhlcv(), make_spec(),
                          PitStore(tmp_path / "a"), GapLedger(tmp_path / "ga"))
        b = run_ingestion(SyntheticOhlcv(shift=1.0), make_spec(),
                          PitStore(tmp_path / "b"), GapLedger(tmp_path / "gb"))
        assert a.data_hash != b.data_hash


class TestContractOrderIsEnforced:
    def test_empty_fetch_writes_nothing(self, env) -> None:
        store, ledger = env

        class Empty(SyntheticOhlcv):
            def fetch(self, spec):  # type: ignore[override]
                return pd.DataFrame()

        with pytest.raises(DataContractError, match="gefaalde fetch"):
            run_ingestion(Empty(), make_spec(), store, ledger)
        assert store.partitions() == []

    def test_schema_violation_writes_nothing(self, env) -> None:
        """Validatie komt VOOR persist; een kapotte frame bereikt de store niet."""
        store, ledger = env

        class Broken(SyntheticOhlcv):
            def normalise(self, raw, spec):  # type: ignore[override]
                df = super().normalise(raw, spec)
                df.loc[10, "high"] = df.loc[10, "low"] * 0.5
                return df

        with pytest.raises(DataContractError, match="OHLC-ordening"):
            run_ingestion(Broken(), make_spec(), store, ledger)
        assert store.partitions() == []

    def test_gap_rejection_writes_nothing(self, env) -> None:
        store, ledger = env

        class Holed(SyntheticOhlcv):
            def normalise(self, raw, spec):  # type: ignore[override]
                df = super().normalise(raw, spec)
                return df.drop(index=[50, 51, 52]).reset_index(drop=True)

        with pytest.raises(DataContractError, match="ontbrekende bar"):
            run_ingestion(Holed(), make_spec(), store, ledger)
        assert store.partitions() == []
        assert len(ledger.read_all()) == 1  # het gat is wel geregistreerd

    def test_gap_register_policy_persists(self, env) -> None:
        store, ledger = env

        class Holed(SyntheticOhlcv):
            def normalise(self, raw, spec):  # type: ignore[override]
                df = super().normalise(raw, spec)
                return df.drop(index=[50, 51, 52]).reset_index(drop=True)

        res = run_ingestion(Holed(), make_spec(gap_policy="register"),
                            store, ledger)
        assert res.n_missing_bars == 3
        assert store.partitions()

    def test_future_rows_are_dropped_before_persist(self, env) -> None:
        """Wat nog niet kenbaar is, is geen observatie."""
        store, ledger = env

        class Future(SyntheticOhlcv):
            def normalise(self, raw, spec):  # type: ignore[override]
                df = super().normalise(raw, spec)
                tomorrow = int(pd.Timestamp.now(tz="UTC").value) + DAY_NS
                extra = df.tail(1).copy()
                extra["event_ts_ns"] = tomorrow
                extra["asof_ts_ns"] = tomorrow + DAY_NS
                return pd.concat([df, extra], ignore_index=True)

        res = run_ingestion(Future(), make_spec(gap_policy="register"),
                            store, ledger)
        now_ns = int(pd.Timestamp.now(tz="UTC").value)
        stored = store.load("crypto", "ohlcv", "TESTUSDT", "1d")
        assert int(stored["asof_ts_ns"].max()) <= now_ns
        assert res.n_rows == 400
