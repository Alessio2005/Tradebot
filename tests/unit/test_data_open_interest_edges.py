# tests/unit/test_data_open_interest_edges.py
"""Restgaten van `data/open_interest.py`: paginering, foutpaden, `run_pipeline` en de laad-varianten van de per-bar loader.

Geen netwerk: `requests.get` en `time.sleep` zijn vervangen. De happy path en de causaliteit van de loader staan in
`test_open_interest.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from tradebot.data import open_interest as oi
from tradebot.data.open_interest import OpenInterestFetcher, load_per_bar_open_interest
from tradebot.utils.failfast import DataContractError


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _page(rows: list[tuple[int, str]], cursor: str = "") -> dict:
    return {"retCode": 0, "result": {"list": [{"timestamp": str(ts), "openInterest": v} for ts, v in rows],
                                     "nextPageCursor": cursor}}


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Registreert de query-parameters van elke request en slaat het wachten tussen pagina's over."""
    seen: list[dict] = []
    monkeypatch.setattr(oi.time, "sleep", lambda s: None)
    return seen


def _serve(monkeypatch: pytest.MonkeyPatch, calls: list[dict], script: list) -> None:
    def fake_get(url, params=None, timeout=None):
        calls.append(dict(params))
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Resp(item)
    monkeypatch.setattr(oi.requests, "get", fake_get)


def _fetcher(tmp_path: Path, **kw) -> OpenInterestFetcher:
    return OpenInterestFetcher(data_dir=str(tmp_path), symbol="BTCUSDT", **kw)


class TestPagination:
    def test_pages_are_followed_through_the_cursor_until_it_runs_out(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([(3_000, "30"), (2_000, "20")], cursor="c1"), _page([(1_000, "10")])])
        f = _fetcher(tmp_path)
        f.start_ts = 0
        df = f.fetch_open_interest()
        assert df["openInterest"].tolist() == [10.0, 20.0, 30.0]
        assert "cursor" not in calls[0] and calls[1]["cursor"] == "c1"
        assert calls[0]["intervalTime"] == "1h" and calls[0]["category"] == "linear" and calls[0]["limit"] == 200

    def test_paging_stops_once_the_oldest_row_reaches_the_requested_start(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([(3_000, "30"), (2_000, "20")], cursor="more")])
        f = _fetcher(tmp_path)
        f.start_ts = 2_500                                   # oudste rij (2.000) ligt al voor de start
        assert len(f._paginate_backward()) == 2 and len(calls) == 1

    def test_an_empty_page_ends_the_history(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [{"retCode": 0, "result": None}])
        assert _fetcher(tmp_path)._paginate_backward() == []

    def test_a_non_zero_return_code_ends_the_history_without_raising(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [{"retCode": 10006, "retMsg": "rate limited"}])
        assert _fetcher(tmp_path)._paginate_backward() == []

    def test_a_failure_halfway_is_an_error_naming_the_rows_already_read(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([(3_000, "30")], cursor="c1"), requests.ConnectionError("down")])
        f = _fetcher(tmp_path)
        f.start_ts = 0
        with pytest.raises(DataContractError, match="na 1 rijen"):
            f.fetch_open_interest()


class TestFetchAndPersist:
    def test_duplicate_timestamps_keep_the_last_row(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([(2_000, "21"), (2_000, "20"), (1_000, "10")])])
        df = _fetcher(tmp_path).fetch_open_interest()
        assert df["openInterest"].tolist() == [10.0, 20.0]

    def test_an_empty_history_gives_an_empty_frame(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([])])
        assert _fetcher(tmp_path).fetch_open_interest().empty

    def test_an_empty_history_is_not_persisted(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([])])
        assert _fetcher(tmp_path).run_pipeline().empty
        assert not list(tmp_path.glob("*.parquet"))

    def test_the_pipeline_writes_one_parquet_per_symbol(self, monkeypatch, tmp_path, calls) -> None:
        _serve(monkeypatch, calls, [_page([(2_000, "20"), (1_000, "10")])])
        out = _fetcher(tmp_path).run_pipeline()
        stored = pd.read_parquet(tmp_path / "oi_crypto_BTCUSDT.parquet")
        assert out["openInterest"].tolist() == [10.0, 20.0] and stored["openInterest"].tolist() == [10.0, 20.0]
        assert "timestamp" in stored.columns


class TestLoaderVariants:
    BARS = pd.date_range("2024-01-01 00:30", periods=3, freq="1h", tz="UTC")
    OBS = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"])

    def test_a_file_without_the_level_column_gives_zeros(self, tmp_path: Path) -> None:
        pd.DataFrame({"timestamp": self.OBS.tz_localize("UTC"), "other": [1.0, 2.0, 3.0]}).to_parquet(
            tmp_path / "oi_crypto_BTCUSDT.parquet", index=False)
        assert load_per_bar_open_interest("BTCUSDT", self.BARS, tmp_path).tolist() == [0.0, 0.0, 0.0]

    def test_a_datetime_index_instead_of_a_timestamp_column_is_accepted(self, tmp_path: Path) -> None:
        pd.DataFrame({"openInterest": [100.0, 200.0, 300.0]}, index=self.OBS.tz_localize("UTC")).to_parquet(
            tmp_path / "oi_crypto_BTCUSDT.parquet")
        assert load_per_bar_open_interest("BTCUSDT", self.BARS, tmp_path).tolist() == [100.0, 200.0, 300.0]

    def test_a_string_index_is_parsed_as_utc(self, tmp_path: Path) -> None:
        idx = pd.Index([str(t) for t in self.OBS])
        pd.DataFrame({"openInterest": [100.0, 200.0, 300.0]}, index=idx).to_parquet(
            tmp_path / "oi_crypto_BTCUSDT.parquet")
        assert load_per_bar_open_interest("BTCUSDT", self.BARS, tmp_path).tolist() == [100.0, 200.0, 300.0]

    def test_naive_observation_and_bar_timestamps_are_treated_as_utc(self, tmp_path: Path) -> None:
        pd.DataFrame({"openInterest": [100.0, 200.0, 300.0]}, index=self.OBS).to_parquet(
            tmp_path / "oi_crypto_BTCUSDT.parquet")
        naive_bars = pd.DatetimeIndex(self.BARS.tz_localize(None))
        out = load_per_bar_open_interest("BTCUSDT", naive_bars, tmp_path)
        assert out.tolist() == [100.0, 200.0, 300.0] and out.dtype == np.float64
