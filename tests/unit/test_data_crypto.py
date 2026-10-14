# tests/unit/test_data_crypto.py
"""`data/crypto.py`: Bybit-archief → 5s-bars, de maandelijkse parquet-opslag en de ingestion-engine.

Geen netwerk: `aiohttp` krijgt een nep-sessie met vooraf bepaalde antwoorden, en
de CSV's worden ter plekke gegzipt.
"""
from __future__ import annotations

import asyncio
import gzip
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data import crypto as cr
from tradebot.data.crypto import BybitPublicClient, CryptoIngestionEngine, ParquetStorage

UTC = timezone.utc
HEADER = "timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n"


def _run(coro):
    return asyncio.run(coro)


def _csv(rows: list[tuple[float, str, float, float]], header: str = HEADER) -> bytes:
    lines = [f"{t},BTCUSDT,{side},{size},{price},PlusTick,id,0,0,0" for t, side, size, price in rows]
    return gzip.compress((header + "\n".join(lines) + "\n").encode())


BASE = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
TRADES = [(BASE + 1.5, "Buy", 1.0, 100.0), (BASE + 3.0, "Sell", 2.0, 102.0), (BASE + 4.9, "Buy", 0.5, 99.0),
          (BASE + 5.0, "Sell", 1.0, 101.0), (BASE + 5.5, "Buy", 3.0, 105.0)]


def _client(tmp_path: Path, **cfg) -> BybitPublicClient:
    return BybitPublicClient({"machine": {"data_dir": str(tmp_path)}, **cfg})


# --------------------------------------------------------------------------- #
# parsing the daily tape
# --------------------------------------------------------------------------- #
class TestClientSetup:
    def test_the_symbol_comes_from_the_override_then_the_config_then_btc(self, tmp_path: Path) -> None:
        assert _client(tmp_path, data={"bybit_symbol": "ETHUSDT"}).symbol == "ETHUSDT"
        assert BybitPublicClient({"machine": {"data_dir": str(tmp_path)}}, symbol_override="SOLUSDT").symbol == "SOLUSDT"
        assert _client(tmp_path, data=None).symbol == "BTCUSDT" and (tmp_path / "temp").is_dir()


class TestParseTape:
    def test_trades_become_right_labelled_five_second_bars_with_taker_volumes(self, tmp_path: Path) -> None:
        path = tmp_path / "day.csv.gz"
        path.write_bytes(_csv(TRADES))
        bars = _client(tmp_path)._process_gz_chunked(path)
        assert bars["timestamp"].tolist() == [pd.Timestamp("2024-01-01 00:00:05", tz="UTC"),
                                              pd.Timestamp("2024-01-01 00:00:10", tz="UTC")]
        first, second = bars.iloc[0], bars.iloc[1]
        assert (first["open"], first["high"], first["low"], first["close"]) == (100.0, 102.0, 99.0, 101.0)
        assert first["tick_volume"] == 4 and first["real_volume"] == pytest.approx(4.5)
        assert first["taker_buy_volume"] == pytest.approx(1.5) and first["taker_sell_volume"] == pytest.approx(3.0)
        assert (second["open"], second["close"], second["taker_buy_volume"]) == (105.0, 105.0, 3.0)

    def test_millisecond_timestamps_are_recognised(self, tmp_path: Path) -> None:
        path = tmp_path / "ms.csv.gz"
        path.write_bytes(_csv([(t * 1000, s, q, p) for t, s, q, p in TRADES]))
        bars = _client(tmp_path)._process_gz_chunked(path)
        assert bars["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00:05", tz="UTC")

    def test_bars_that_span_chunk_boundaries_are_merged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "chunks.csv.gz"
        path.write_bytes(_csv(TRADES))
        reference = _client(tmp_path)._process_gz_chunked(path)
        original = pd.read_csv
        monkeypatch.setattr(cr.pd, "read_csv", lambda *a, **kw: original(*a, **{**kw, "chunksize": 2}))
        pd.testing.assert_frame_equal(_client(tmp_path)._process_gz_chunked(path), reference)

    def test_a_file_with_unexpected_columns_gives_an_empty_frame_and_an_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "bad.csv.gz"
        path.write_bytes(gzip.compress(b"timestamp,side,size\n1704067201,Buy,1.0\n"))
        with caplog.at_level(logging.ERROR, logger=cr.logger.name):
            assert _client(tmp_path)._process_gz_chunked(path).empty
        assert "Unexpected Bybit CSV columns" in caplog.text

    def test_a_header_only_file_gives_an_empty_frame(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv.gz"
        path.write_bytes(gzip.compress(HEADER.encode()))
        assert _client(tmp_path)._process_gz_chunked(path).empty


# --------------------------------------------------------------------------- #
# downloading
# --------------------------------------------------------------------------- #
class _Content:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def read(self, n: int) -> bytes:
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk


class _Resp:
    def __init__(self, status: int, data: bytes = b"") -> None:
        self.status, self.content = status, _Content(data)

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Session:
    def __init__(self, script: list) -> None:
        self.script, self.urls = list(script), []

    def get(self, url: str) -> _Resp:
        self.urls.append(url)
        return self.script.pop(0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fast(_s: float) -> None:
        return None
    monkeypatch.setattr(cr.asyncio, "sleep", fast)


class TestDownload:
    def test_a_missing_day_is_an_empty_frame(self, tmp_path: Path) -> None:
        for status in (404, 403):
            session = _Session([_Resp(status)])
            assert _run(_client(tmp_path)._download_and_parse_to_disk(session, "u", "f.csv.gz")).empty

    def test_a_downloaded_day_is_parsed_and_the_temp_file_removed(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        session = _Session([_Resp(200, _csv(TRADES))])
        bars = _run(client._download_and_parse_to_disk(session, "u", "day.csv.gz"))
        assert len(bars) == 2 and not (client.temp_dir / "day.csv.gz").exists()

    def test_other_statuses_are_retried_and_give_up_after_three_attempts(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        flaky = _Session([_Resp(500), _Resp(502), _Resp(200, _csv(TRADES))])
        assert len(_run(client._download_and_parse_to_disk(flaky, "u", "a.csv.gz"))) == 2
        dead = _Session([_Resp(500), _Resp(503), _Resp(500)])
        assert _run(client._download_and_parse_to_disk(dead, "u", "b.csv.gz")).empty and len(dead.urls) == 3

    def test_leftover_temp_files_are_cleaned_after_a_failed_download(self, tmp_path: Path) -> None:
        client = _client(tmp_path)
        (client.temp_dir / "c.csv.gz").write_bytes(b"stale")
        assert _run(client._download_and_parse_to_disk(_Session([_Resp(500)] * 3), "u", "c.csv.gz")).empty
        assert not (client.temp_dir / "c.csv.gz").exists()

    def test_the_url_is_built_from_symbol_and_date(self, tmp_path: Path) -> None:
        session = _Session([_Resp(404)])
        _run(_client(tmp_path, data={"bybit_symbol": "ETHUSDT"})._process_single_day(session, datetime(2024, 3, 5, 13, tzinfo=UTC)))
        assert session.urls == ["https://public.bybit.com/trading/ETHUSDT/ETHUSDT2024-03-05.csv.gz"]


class TestDaysInRange:
    def _stub_days(self, client: BybitPublicClient, monkeypatch: pytest.MonkeyPatch, frames: dict[str, pd.DataFrame]) -> None:
        async def day(session, day_dt):
            return frames.get(day_dt.strftime("%Y-%m-%d"), pd.DataFrame())
        monkeypatch.setattr(client, "_process_single_day", day)

    def _frame(self, ts: str) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": [pd.Timestamp(ts, tz="UTC")], "close": [1.0]})

    def test_days_are_combined_in_time_order_and_empty_days_dropped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client(tmp_path)
        self._stub_days(client, monkeypatch, {"2024-01-03": self._frame("2024-01-03 00:00:05"),
                                              "2024-01-01": self._frame("2024-01-01 00:00:05")})
        out = _run(client._process_days_in_range(None, datetime(2024, 1, 1, 6, tzinfo=UTC), datetime(2024, 1, 3, 12, tzinfo=UTC)))
        assert out["timestamp"].is_monotonic_increasing and len(out) == 2

    def test_no_days_or_no_data_gives_an_empty_frame(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client(tmp_path)
        self._stub_days(client, monkeypatch, {})
        assert _run(client._process_days_in_range(None, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))).empty
        assert _run(client._process_days_in_range(None, datetime(2024, 1, 5, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))).empty


class TestFetchPeriod:
    def _patch(self, client: BybitPublicClient, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
        class FakeSession:
            def __init__(self, **kw) -> None:
                pass

            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *exc) -> bool:
                return False

        async def days(session, start, end):
            return frame
        monkeypatch.setattr(cr.aiohttp, "ClientSession", FakeSession)
        monkeypatch.setattr(client, "_process_days_in_range", days)

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="5s", tz="UTC"), "close": 1.0})

    def test_a_closed_window_includes_its_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client(tmp_path)
        self._patch(client, monkeypatch, self._frame())
        out = _run(client.fetch_period_processed(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC)))
        assert out["timestamp"].iloc[-1] == pd.Timestamp("2024-01-01 00:00:10", tz="UTC") and len(out) == 3

    def test_a_window_reaching_into_the_present_excludes_its_open_end(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client(tmp_path)
        frame = self._frame()
        self._patch(client, monkeypatch, frame)
        future = datetime.now(UTC) + timedelta(days=1)
        out = _run(client.fetch_period_processed(datetime(2024, 1, 1, tzinfo=UTC), future))
        assert len(out) == len(frame)
        out = _run(client.fetch_period_processed(datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC), datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC)))
        assert out["timestamp"].tolist() == [pd.Timestamp("2024-01-01 00:00:05", tz="UTC"), pd.Timestamp("2024-01-01 00:00:10", tz="UTC")]

    def test_no_data_comes_back_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _client(tmp_path)
        self._patch(client, monkeypatch, pd.DataFrame())
        assert _run(client.fetch_period_processed(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC))).empty


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
def _bars(start: str, n: int = 4, tz: str | None = "UTC", price: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq="12h", tz=tz),
        "open": price, "high": price, "low": price, "close": price,
        "tick_volume": 1.0, "real_volume": 2.0, "taker_buy_volume": 1.0, "taker_sell_volume": 1.0})


class TestStorage:
    def test_chunks_are_partitioned_per_month_and_load_back_sorted(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        _run(store.save_chunk(_bars("2024-01-30", n=6), "BTCUSDT"))
        files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet"))
        assert files == ["BTCUSDT/2024/BTCUSDT_2024-01.parquet", "BTCUSDT/2024/BTCUSDT_2024-02.parquet"]
        df = _run(store.load_data("BTCUSDT"))
        assert df.index.is_monotonic_increasing and len(df) == 6 and df["tick_volume"].dtype == np.float32
        assert not list(tmp_path.rglob("*.tmp"))

    def test_saving_over_an_existing_month_keeps_the_newest_rows_and_stays_sorted(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        _run(store.save_chunk(_bars("2024-01-01", n=4, price=1.0), "BTCUSDT"))
        _run(store.save_chunk(_bars("2024-01-02", n=4, price=9.0), "BTCUSDT"))
        df = _run(store.load_data("BTCUSDT"))
        assert len(df) == 6 and df["close"].tolist() == [1.0, 1.0, 9.0, 9.0, 9.0, 9.0] and df.index.is_unique

    def test_naive_and_foreign_timezones_are_normalised_to_utc(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        _run(store.save_chunk(_bars("2024-01-01", tz=None), "A"))
        _run(store.save_chunk(_bars("2024-01-01", tz="Europe/Amsterdam"), "B"))
        a, b = _run(store.load_data("A")), _run(store.load_data("B"))
        assert str(a.index.tz) == "UTC" and b.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")

    def test_string_timestamps_are_parsed_and_an_empty_chunk_is_ignored(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        frame = _bars("2024-01-01")
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        _run(store.save_chunk(frame, "A"))
        _run(store.save_chunk(pd.DataFrame(), "EMPTY"))
        assert len(_run(store.load_data("A"))) == 4 and not (tmp_path / "EMPTY").exists()

    def test_the_last_timestamp_is_read_from_the_newest_file(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        assert _run(store.get_last_timestamp("A")) is None
        (tmp_path / "A").mkdir()
        assert _run(store.get_last_timestamp("A")) is None                     # map zonder bestanden
        _run(store.save_chunk(_bars("2024-01-30", n=6), "A"))
        assert _run(store.get_last_timestamp("A")) == datetime(2024, 2, 1, 12, tzinfo=UTC)

    def test_a_naive_stored_timestamp_is_treated_as_utc(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        folder = tmp_path / "A" / "2024"
        folder.mkdir(parents=True)
        _bars("2024-01-01", tz=None).to_parquet(folder / "A_2024-01.parquet", index=False)
        assert _run(store.get_last_timestamp("A")) == datetime(2024, 1, 2, 12, tzinfo=UTC)

    def test_loading_an_unknown_symbol_gives_an_empty_frame(self, tmp_path: Path) -> None:
        assert _run(ParquetStorage(str(tmp_path)).load_data("NOPE")).empty


class TestPrune:
    def _seed(self, store: ParquetStorage, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        for year in (now.year - 8, now.year - 6, now.year):
            folder = tmp_path / "A" / str(year)
            folder.mkdir(parents=True)
            for month in (1, 6):
                _bars(f"{year}-{month:02d}-01", n=1).to_parquet(folder / f"A_{year}-{month:02d}.parquet", index=False)
        (tmp_path / "A" / "notes.txt").write_text("not a dir")
        (tmp_path / "A" / "backup").mkdir()                                     # niet-numerieke jaarmap
        (tmp_path / "A" / str(now.year) / "A_garbage.parquet").write_bytes(b"x")  # naam zonder datum

    def test_old_years_and_months_beyond_retention_are_removed_and_the_rest_survives(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = ParquetStorage(str(tmp_path))
        self._seed(store, tmp_path)
        now = datetime.now(UTC)
        with caplog.at_level(logging.INFO, logger=cr.logger.name):
            store.prune_old_data("A", retention_years=5)
        assert not (tmp_path / "A" / str(now.year - 8)).exists()
        assert (tmp_path / "A" / str(now.year) / f"A_{now.year}-01.parquet").exists()
        assert (tmp_path / "A" / "backup").is_dir() and (tmp_path / "A" / "notes.txt").exists()
        assert (tmp_path / "A" / str(now.year) / "A_garbage.parquet").exists() and "Cleanup complete" in caplog.text

    def test_a_year_inside_the_window_loses_only_its_stale_months(self, tmp_path: Path) -> None:
        store = ParquetStorage(str(tmp_path))
        now = datetime.now(UTC)
        year = now.year - 5
        folder = tmp_path / "A" / str(year)
        folder.mkdir(parents=True)
        for month in (1, 12):
            _bars(f"{year}-{month:02d}-01", n=1).to_parquet(folder / f"A_{year}-{month:02d}.parquet", index=False)
        store.prune_old_data("A", retention_years=5)
        remaining = {p.name for p in folder.glob("*.parquet")}
        cutoff = (now - timedelta(days=365 * 5)).replace(day=1)
        for month in (1, 12):
            stale = datetime(year, month, 1, tzinfo=UTC) < cutoff.replace(tzinfo=UTC)
            assert (f"A_{year}-{month:02d}.parquet" in remaining) is (not stale)

    def test_pruning_an_unknown_symbol_is_a_no_op(self, tmp_path: Path) -> None:
        ParquetStorage(str(tmp_path)).prune_old_data("NOPE")


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class _FakeClient:
    instances: list[_FakeClient] = []  # noqa: RUF012

    def __init__(self, cfg, symbol_override=None) -> None:
        self.symbol, self.windows = symbol_override, []
        _FakeClient.instances.append(self)

    async def fetch_period_processed(self, start: datetime, end: datetime) -> pd.DataFrame:
        self.windows.append((start, end))
        if start.month == 2:
            return pd.DataFrame()
        return _bars(start.strftime("%Y-%m-%d"), n=2)


class TestEngine:
    def _engine(self, tmp_path: Path, **cfg) -> CryptoIngestionEngine:
        return CryptoIngestionEngine({"machine": {"data_dir": str(tmp_path)}, **cfg})

    def test_the_universe_is_filtered_to_crypto_pairs_with_fallbacks(self, tmp_path: Path) -> None:
        e = self._engine(tmp_path, training={"training_universe": ["BTCUSDT", "AAPL", "ETHusdc", 7]},
                         data={"bybit_symbol": "SOLUSDT", "history_fetch_years": 2, "inter_symbol_sleep": 0.0})
        assert e.symbols == ["BTCUSDT", "ETHusdc"] and e.target_symbol == "SOLUSDT" and e.retention_years == 2
        assert e.inter_symbol_sleep == 0.0
        assert self._engine(tmp_path, training={"training_universe": ["AAPL"]}, data={"bybit_symbol": "SOLUSDT"}).symbols == ["SOLUSDT"]
        default = self._engine(tmp_path)
        assert default.symbols == ["BTCUSDT"] and default.retention_years == 5 and default.inter_symbol_sleep == 0.5

    def test_sync_runs_one_symbol_for_an_override_the_universe_for_many_and_the_target_for_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(tmp_path, training={"training_universe": ["BTCUSDT", "ETHUSDT"]})
        synced: list[str] = []

        async def fake(symbol: str) -> None:
            synced.append(symbol)
        monkeypatch.setattr(engine, "_sync_one", fake)
        _run(engine.sync_raw_data(symbol_override="XRPUSDT"))
        _run(engine.sync_raw_data())
        single = self._engine(tmp_path, data={"bybit_symbol": "SOLUSDT"})
        monkeypatch.setattr(single, "_sync_one", fake)
        _run(single.sync_raw_data())
        assert synced == ["XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_an_up_to_date_symbol_is_not_fetched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                caplog: pytest.LogCaptureFixture) -> None:
        engine = self._engine(tmp_path)
        _FakeClient.instances.clear()
        monkeypatch.setattr(cr, "BybitPublicClient", _FakeClient)

        async def future(symbol: str):
            return datetime.now(UTC) + timedelta(days=1)
        monkeypatch.setattr(engine.storage, "get_last_timestamp", future)
        with caplog.at_level(logging.INFO, logger=cr.logger.name):
            _run(engine._sync_one("BTCUSDT"))
        assert "already up to date" in caplog.text and _FakeClient.instances == []

    def test_a_first_sync_fetches_the_whole_retention_window_month_by_month(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._engine(tmp_path, data={"history_fetch_years": 1})
        _FakeClient.instances.clear()
        monkeypatch.setattr(cr, "BybitPublicClient", _FakeClient)
        _run(engine._sync_one("BTCUSDT"))
        windows = _FakeClient.instances[0].windows
        assert 12 <= len(windows) <= 14 and windows[0][0] < windows[1][0]
        assert all(windows[k][1] < windows[k + 1][0] for k in range(len(windows) - 1))

    def test_a_resumed_sync_starts_one_second_after_the_last_stored_bar_and_crosses_year_ends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(tmp_path)
        _FakeClient.instances.clear()
        monkeypatch.setattr(cr, "BybitPublicClient", _FakeClient)
        start, end = datetime(2023, 12, 20, tzinfo=UTC), datetime(2024, 2, 10, tzinfo=UTC)
        _run(engine._sync_bybit_chunked("BTCUSDT", "BTCUSDT", start, end))
        windows = _FakeClient.instances[0].windows
        assert [w[0].strftime("%Y-%m") for w in windows] == ["2023-12", "2024-01", "2024-02"]
        assert windows[0][0] == start and windows[-1][1] == end
        stored = _run(engine.storage.load_data("BTCUSDT"))
        assert len(stored) == 4                                                   # december + januari; februari was leeg
