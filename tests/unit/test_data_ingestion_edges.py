# tests/unit/test_data_ingestion_edges.py
"""`data/ingestion/`: de Bybit-client (retry en pagination), de drie bronnen, het legacy-pad en het contract.

Geen netwerk: de client krijgt een nep-`requests.Session` met een script van
antwoorden; `time.sleep` is vervangen zodat retries direct lopen.
"""
from __future__ import annotations

import asyncio

import pandas as pd
import pytest
import requests
from omegaconf import OmegaConf

from tradebot.data.ingestion import bybit as bb
from tradebot.data.ingestion import legacy
from tradebot.data.ingestion.bybit import MS_TO_NS, BybitV5Client
from tradebot.data.ingestion.contract import IngestionResult, IngestionSpec, _partition_date
from tradebot.data.ingestion.crypto_sources import FundingSource, OhlcvSource, OpenInterestSource
from tradebot.data.validation.schema import OHLCV_SPEC
from tradebot.utils.failfast import DataContractError

DAY_MS = 86_400_000


class _Resp:
    def __init__(self, status: int = 200, payload: dict | None = None, text: str = "") -> None:
        self.status_code, self._payload, self.text = status, payload or {}, text

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, script: list) -> None:
        self.script, self.calls = list(script), []

    def get(self, url: str, params=None, timeout=None):
        self.calls.append((url, dict(params), timeout))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _ok(rows: list) -> _Resp:
    return _Resp(200, {"retCode": 0, "result": {"list": rows}})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []
    monkeypatch.setattr(bb.time, "sleep", sleeps.append)

    def make(script: list) -> BybitV5Client:
        c = BybitV5Client(base_url="https://example.test/", category="linear")
        c._session = _Session(script)                                     # type: ignore[assignment]
        c.sleeps = sleeps                                                  # type: ignore[attr-defined]
        return c
    return make


# --------------------------------------------------------------------------- #
# _get
# --------------------------------------------------------------------------- #
class TestGet:
    def test_a_healthy_response_is_returned_and_the_url_is_normalised(self, client) -> None:
        c = client([_ok([1])])
        assert c._get("/v5/x", {"a": 1})["result"]["list"] == [1]
        url, params, timeout = c._session.calls[0]
        assert url == "https://example.test/v5/x" and params == {"a": 1} and timeout == bb._HTTP_TIMEOUT_S

    def test_transport_errors_and_throttling_are_retried_with_backoff(self, client) -> None:
        c = client([requests.Timeout("slow"), requests.ConnectionError("reset"), _Resp(429), _Resp(503), _ok([7])])
        assert c._get("/v5/x", {})["result"]["list"] == [7]
        assert c.sleeps == [bb._BACKOFF_BASE_S ** k for k in (1, 2, 3, 4)]

    def test_after_the_last_attempt_it_crashes_instead_of_returning_a_partial_series(self, client) -> None:
        c = client([requests.Timeout("slow")] * bb._MAX_ATTEMPTS)
        with pytest.raises(DataContractError, match="onbereikbaar na 5 pogingen"):
            c._get("/v5/x", {})

    def test_a_client_error_is_not_retried(self, client) -> None:
        c = client([_Resp(404, text="nope")])
        with pytest.raises(DataContractError, match="niet-herstelbare HTTP-status"):
            c._get("/v5/x", {})
        assert len(c._session.calls) == 1

    def test_an_application_error_with_http_200_is_an_error_too(self, client) -> None:
        c = client([_Resp(200, {"retCode": 10001, "retMsg": "params error"})])
        with pytest.raises(DataContractError, match="retCode != 0"):
            c._get("/v5/x", {})


# --------------------------------------------------------------------------- #
# pagination
# --------------------------------------------------------------------------- #
def _kline_rows(*starts: int) -> list[list[str]]:
    return [[str(s), "1", "2", "0.5", "1.5", "10", "15"] for s in starts]


class TestPagination:
    def test_pages_are_walked_backwards_until_the_window_start_is_reached(self, client) -> None:
        c = client([_ok(_kline_rows(5 * DAY_MS, 4 * DAY_MS)), _ok(_kline_rows(3 * DAY_MS, 2 * DAY_MS))])
        df = c.kline("BTCUSDT", "D", start_ms=2 * DAY_MS, end_ms=5 * DAY_MS)
        assert df["start_ms"].tolist() == [5 * DAY_MS, 4 * DAY_MS, 3 * DAY_MS, 2 * DAY_MS]
        assert str(df["close"].dtype) == "float64" and str(df["start_ms"].dtype) == "int64"
        first, second = (call[1] for call in c._session.calls)
        assert first["end"] == 5 * DAY_MS and second["end"] == 4 * DAY_MS - 1 and first["limit"] == 1000
        assert first["interval"] == "D" and first["category"] == "linear" and c.sleeps == [bb._THROTTLE_S]

    def test_an_empty_page_ends_the_walk_and_nothing_at_all_is_an_error(self, client) -> None:
        c = client([_ok(_kline_rows(5 * DAY_MS)), _ok([])])
        assert len(c.kline("X", "D", 2 * DAY_MS, 5 * DAY_MS)) == 1
        with pytest.raises(DataContractError, match="nul rijen"):
            client([_ok([])]).kline("X", "D", 2 * DAY_MS, 5 * DAY_MS)

    def test_a_cursor_that_does_not_advance_is_an_error(self, client) -> None:
        c = client([_ok(_kline_rows(9 * DAY_MS))])                          # nieuwer dan de cursor (end = 5 dagen)
        with pytest.raises(DataContractError, match="cursor schuift niet op"):
            c.kline("X", "D", 1 * DAY_MS, 5 * DAY_MS)

    def test_an_exhausted_page_budget_is_an_error_and_never_a_partial_series(
        self, client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(bb, "_MAX_PAGES", 3)
        c = client([_ok(_kline_rows(1_000 - k)) for k in range(3)])
        with pytest.raises(DataContractError, match=r"paginabudget \(3\) uitgeput"):
            c.kline("X", "D", start_ms=1, end_ms=2_000)


class TestEndpoints:
    def test_funding_history_returns_settlement_times_and_rates(self, client) -> None:
        rows = [{"fundingRateTimestamp": str(2 * DAY_MS), "fundingRate": "0.0001"},
                {"fundingRateTimestamp": str(DAY_MS), "fundingRate": "-0.0002"}]
        df = client([_ok(rows)]).funding_history("BTCUSDT", DAY_MS, 3 * DAY_MS)
        assert df["settle_ms"].tolist() == [2 * DAY_MS, DAY_MS] and df["funding_rate"].tolist() == [0.0001, -0.0002]

    def test_open_interest_returns_timestamps_and_levels(self, client) -> None:
        rows = [{"timestamp": str(DAY_MS), "openInterest": "123.5"}]
        c = client([_ok(rows)])
        df = c.open_interest("BTCUSDT", "1h", DAY_MS, 2 * DAY_MS)
        assert df.to_dict("records") == [{"ts_ms": DAY_MS, "open_interest": 123.5}]
        assert c._session.calls[0][1]["intervalTime"] == "1h" and c._session.calls[0][1]["limit"] == 200

    def test_the_launch_time_of_an_instrument_and_an_unknown_symbol(self, client) -> None:
        assert client([_ok([{"launchTime": "1700000000000"}])]).instrument_launch_ms("BTCUSDT") == 1_700_000_000_000
        with pytest.raises(DataContractError, match="kent dit symbool niet"):
            client([_ok([])]).instrument_launch_ms("NOPEUSDT")


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def kline(self, symbol, interval, start_ms, end_ms):
        self.calls.append(("kline", symbol, interval, start_ms, end_ms))
        return pd.DataFrame({"start_ms": [3 * DAY_MS, DAY_MS, 3 * DAY_MS, 9 * DAY_MS],
                             "open": [1.0, 1.0, 2.0, 5.0], "high": 2.0, "low": 0.5, "close": [1.5, 1.5, 2.5, 5.5],
                             "volume": 10.0, "turnover": 15.0})

    def funding_history(self, symbol, start_ms, end_ms):
        self.calls.append(("funding", symbol, start_ms, end_ms))
        return pd.DataFrame({"settle_ms": [2 * DAY_MS, DAY_MS, 2 * DAY_MS], "funding_rate": [0.001, 0.002, 0.003]})

    def open_interest(self, symbol, interval, start_ms, end_ms):
        self.calls.append(("oi", symbol, interval, start_ms, end_ms))
        return pd.DataFrame({"ts_ms": [2 * DAY_MS, DAY_MS, 2 * DAY_MS], "open_interest": [5.0, 4.0, 6.0]})


def _spec(granularity: str = "1d", dataset: str = "ohlcv") -> IngestionSpec:
    return IngestionSpec(asset_class="crypto", dataset=dataset, symbol="BTCUSDT", granularity=granularity,
                         series_spec=OHLCV_SPEC, gap_policy="register", max_abs_log_return=0.5,
                         allow_price_jumps=False, source_name="bybit")


class TestOhlcvSource:
    def test_it_fetches_through_the_interval_mapping(self) -> None:
        client = _FakeClient()
        OhlcvSource(client, DAY_MS, 9 * DAY_MS).fetch(_spec("1h"))                # type: ignore[arg-type]
        assert client.calls == [("kline", "BTCUSDT", "60", DAY_MS, 9 * DAY_MS)]

    def test_an_empty_window_and_an_unknown_granularity_are_refused(self) -> None:
        with pytest.raises(DataContractError, match="leeg of omgekeerd"):
            OhlcvSource(_FakeClient(), 5, 5).fetch(_spec())                       # type: ignore[arg-type]
        with pytest.raises(DataContractError, match="Granulariteit niet ondersteund"):
            OhlcvSource(_FakeClient(), 1, 5).fetch(_spec("7d"))                   # type: ignore[arg-type]

    def test_bars_become_known_at_their_close_and_duplicates_and_out_of_window_rows_disappear(self) -> None:
        source = OhlcvSource(_FakeClient(), DAY_MS, 5 * DAY_MS)                  # type: ignore[arg-type]
        out = source.normalise(source.fetch(_spec()), _spec())
        assert out["event_ts_ns"].tolist() == [DAY_MS * MS_TO_NS, 3 * DAY_MS * MS_TO_NS]
        assert (out["asof_ts_ns"] - out["event_ts_ns"]).tolist() == [DAY_MS * MS_TO_NS] * 2     # sluitingstijd
        assert out["close"].tolist() == [1.5, 2.5] and set(out["symbol"]) == {"BTCUSDT"}   # laatste duplicaat wint


class TestFundingSource:
    def test_funding_is_known_exactly_at_settlement(self) -> None:
        client = _FakeClient()
        source = FundingSource(client, DAY_MS, 3 * DAY_MS, funding_interval_hours=4)       # type: ignore[arg-type]
        spec = _spec("8h", "funding")
        out = source.normalise(source.fetch(spec), spec)
        assert client.calls == [("funding", "BTCUSDT", DAY_MS, 3 * DAY_MS)]
        assert out["event_ts_ns"].equals(out["asof_ts_ns"]) and out["funding_rate"].tolist() == [0.002, 0.003]
        assert set(out["funding_interval_hours"]) == {4}

    def test_an_inverted_window_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="leeg of omgekeerd"):
            FundingSource(_FakeClient(), 9, 3, 8).fetch(_spec("8h", "funding"))      # type: ignore[arg-type]


class TestOpenInterestSource:
    def test_it_maps_the_interval_and_keeps_the_last_duplicate(self) -> None:
        client = _FakeClient()
        source = OpenInterestSource(client, DAY_MS, 3 * DAY_MS)                  # type: ignore[arg-type]
        spec = _spec("15m", "open_interest")
        out = source.normalise(source.fetch(spec), spec)
        assert client.calls == [("oi", "BTCUSDT", "15min", DAY_MS, 3 * DAY_MS)]
        assert out["open_interest"].tolist() == [4.0, 6.0] and out["event_ts_ns"].equals(out["asof_ts_ns"])

    def test_unsupported_granularity_and_empty_window_are_refused(self) -> None:
        with pytest.raises(DataContractError, match="open-interest endpoint"):
            OpenInterestSource(_FakeClient(), 1, 5).fetch(_spec("1m", "open_interest"))   # type: ignore[arg-type]
        with pytest.raises(DataContractError, match="leeg of omgekeerd"):
            OpenInterestSource(_FakeClient(), 5, 1).fetch(_spec("1h", "open_interest"))   # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# contract helpers
# --------------------------------------------------------------------------- #
class TestContractHelpers:
    def test_daily_and_slower_bars_are_partitioned_per_year_and_intraday_per_day(self) -> None:
        ts = int(pd.Timestamp("2024-03-05 14:30", tz="UTC").value)
        assert _partition_date(ts, "1d") == "2024-01-01" and _partition_date(ts, "4h") == "2024-01-01"
        assert _partition_date(ts, "5m") == "2024-03-05"

    def test_the_result_reports_its_span_as_iso_timestamps(self) -> None:
        start, end = int(pd.Timestamp("2024-01-01", tz="UTC").value), int(pd.Timestamp("2024-01-31", tz="UTC").value)
        result = IngestionResult(_spec(), 31, 1, "h", start, end, 0, 0)
        assert result.first_iso.startswith("2024-01-01") and result.last_iso.startswith("2024-01-31")


# --------------------------------------------------------------------------- #
# legacy path
# --------------------------------------------------------------------------- #
class _Storage:
    def __init__(self, df) -> None:
        self.df = df

    async def load_data(self, sym: str):
        return self.df


class _Engine:
    def __init__(self, cfg: dict, df) -> None:
        self.cfg, self.synced, self.storage = cfg, [], _Storage(df)

    async def sync_raw_data(self, symbol_override: str) -> None:
        self.synced.append(symbol_override)


class TestLegacyIngest:
    def _patch(self, monkeypatch: pytest.MonkeyPatch, df) -> list:
        built: list = []

        def factory(cfg: dict) -> _Engine:
            built.append(_Engine(cfg, df))
            return built[-1]
        monkeypatch.setattr(legacy, "_BybitEngine", factory)
        return built

    def test_a_usdt_symbol_is_routed_to_the_bybit_engine_and_its_data_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = pd.DataFrame({"close": [1.0, 2.0]})
        built = self._patch(monkeypatch, frame)
        out = asyncio.run(legacy.ingest_raw(OmegaConf.create({"data": {"bybit_symbol": "BTCUSDT"}}), "ETHUSDT"))
        assert out is frame and built[0].synced == ["ETHUSDT"]
        assert built[0].cfg["model"]["symbols"]["target"] == "ETHUSDT"

    def test_the_configured_bybit_symbol_is_routed_even_without_usdt_in_its_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built = self._patch(monkeypatch, pd.DataFrame({"close": [1.0]}))
        asyncio.run(legacy.ingest_raw(OmegaConf.create({"data": {"bybit_symbol": "BTCPERP"}}), "BTCPERP"))
        assert len(built) == 1

    def test_a_symbol_no_engine_serves_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, None)
        with pytest.raises(RuntimeError, match="No suitable data engine"):
            asyncio.run(legacy.ingest_raw(OmegaConf.create({"data": {}}), "AAPL"))

    @pytest.mark.parametrize("empty", [None, pd.DataFrame()])
    def test_empty_data_after_the_sync_is_an_error(self, monkeypatch: pytest.MonkeyPatch, empty) -> None:
        self._patch(monkeypatch, empty)
        with pytest.raises(RuntimeError, match="No data loaded after sync"):
            asyncio.run(legacy.ingest_raw(OmegaConf.create({}), "BTCUSDT"))
