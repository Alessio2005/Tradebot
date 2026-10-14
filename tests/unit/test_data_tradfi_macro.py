# tests/unit/test_data_tradfi_macro.py
"""`data/tradfi_macro.py`: Yahoo-tickers met retry, Deribit-DVOL, Fear & Greed, afgeleide features en de publicatielag.

Geen netwerk: `yfinance.download`, `requests.get` en `time.sleep` zijn vervangen.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from tradebot.data import tradfi_macro as tm
from tradebot.data.tradfi_macro import MacroDataFetcher
from tradebot.utils.failfast import DataContractError

DAY_MS = 86_400_000
TODAY_MS = int(pd.Timestamp.now(tz="UTC").normalize().timestamp() * 1000)


class _Resp:
    def __init__(self, payload: dict, error: Exception | None = None) -> None:
        self.payload, self.error = payload, error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error

    def json(self) -> dict:
        return self.payload


@pytest.fixture
def fetcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MacroDataFetcher:
    monkeypatch.setattr(tm.time, "sleep", lambda s: None)
    return MacroDataFetcher(data_dir=str(tmp_path / "macro"))


# --------------------------------------------------------------------------- #
# DVOL
# --------------------------------------------------------------------------- #
def _dvol_rows(n: int = 10) -> list[list]:
    return [[TODAY_MS - (n - 1 - k) * DAY_MS, 50.0 + k, 52.0 + k, 49.0 + k, 51.0 + k] for k in range(n)]


class TestDvol:
    def test_the_data_dir_is_created(self, fetcher: MacroDataFetcher, tmp_path: Path) -> None:
        assert (tmp_path / "macro").is_dir() and "10Y_Yield" in fetcher.tickers

    def test_daily_closes_become_a_sorted_deduplicated_series_named_after_the_currency(
        self, fetcher: MacroDataFetcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = _dvol_rows()
        rows.append([rows[-1][0], 1, 1, 1, 99.0])                              # duplicaat: laatste wint
        calls = []

        def get(url, params=None, timeout=None):
            calls.append(params)
            return _Resp({"result": {"data": rows}})
        monkeypatch.setattr(requests, "get", get)
        df = fetcher.fetch_dvol("eth")
        assert list(df.columns) == ["dvol_eth"] and len(df) == 10 and df["dvol_eth"].iloc[-1] == 99.0
        assert df.index.is_monotonic_increasing and df.index.tz is not None
        assert calls[0]["currency"] == "ETH" and calls[0]["resolution"] == 86400 and len(calls) == 1

    def test_pagination_continues_after_the_last_row_until_today(self, fetcher: MacroDataFetcher,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        pages = [[[TODAY_MS - 5 * DAY_MS, 1, 1, 1, 10.0]],
                 [[TODAY_MS - 4 * DAY_MS, 1, 1, 1, 11.0], [TODAY_MS, 1, 1, 1, 12.0]]]
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({"result": {"data": pages.pop(0)}}))
        df = fetcher.fetch_dvol()
        assert df["dvol_btc"].tolist() == [10.0, 11.0, 12.0]

    def test_empty_pages_and_pages_that_do_not_advance_end_the_walk(self, fetcher: MacroDataFetcher,
                                                                   monkeypatch: pytest.MonkeyPatch,
                                                                   caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({"result": {"data": []}}))
        with caplog.at_level(logging.WARNING, logger=tm.logger.name):
            assert fetcher.fetch_dvol().empty
        assert "no data received" in caplog.text
        stale = [[0, 1, 1, 1, 7.0]]                                              # laatste ts <= huidige start
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({"result": {"data": stale}}))
        assert fetcher.fetch_dvol()["dvol_btc"].tolist() == [7.0]

    def test_a_failing_request_is_a_contract_error(self, fetcher: MacroDataFetcher, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({}, requests.HTTPError("500")))
        with pytest.raises(DataContractError, match=r"DVOL-pagination \(BTC\) afgebroken"):
            fetcher.fetch_dvol()


# --------------------------------------------------------------------------- #
# Fear & Greed
# --------------------------------------------------------------------------- #
class TestFearGreed:
    def test_the_index_is_parsed_into_a_daily_series(self, fetcher: MacroDataFetcher, monkeypatch: pytest.MonkeyPatch) -> None:
        data = [{"value": "30", "timestamp": "1704153600"}, {"value": "70", "timestamp": "1704067200"},
                {"value": "31", "timestamp": "1704153600"}]
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({"data": data}))
        df = fetcher.fetch_fear_greed()
        assert df["fear_greed"].tolist() == [70.0, 31.0] and df.index.is_monotonic_increasing

    def test_an_empty_payload_gives_an_empty_frame_and_an_http_error_propagates(
        self, fetcher: MacroDataFetcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({"data": []}))
        assert fetcher.fetch_fear_greed().empty
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp({}, requests.HTTPError("503")))
        with pytest.raises(requests.HTTPError):
            fetcher.fetch_fear_greed()


# --------------------------------------------------------------------------- #
# Yahoo
# --------------------------------------------------------------------------- #
def _prices(n: int = 5, name: str = "Close") -> pd.DataFrame:
    return pd.DataFrame({name: np.arange(1.0, n + 1)}, index=pd.date_range("2024-01-01", periods=n, tz="UTC"))


class TestExtractClose:
    def test_nothing_gives_an_empty_frame(self, fetcher: MacroDataFetcher) -> None:
        assert fetcher._extract_close_price(None, "X").empty and fetcher._extract_close_price(pd.DataFrame(), "X").empty

    def test_flat_columns_with_and_without_a_close_column(self, fetcher: MacroDataFetcher) -> None:
        assert fetcher._extract_close_price(_prices(), "HYG")["HYG"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert fetcher._extract_close_price(_prices(name="Adj"), "HYG").columns.tolist() == ["HYG"]

    def test_multiindex_columns_take_the_first_close_series(self, fetcher: MacroDataFetcher) -> None:
        wide = pd.DataFrame({("Close", "A"): [1.0, 2.0], ("Close", "B"): [3.0, 4.0], ("Volume", "A"): [9.0, 9.0]})
        assert fetcher._extract_close_price(wide, "X")["X"].tolist() == [1.0, 2.0]
        blank = pd.DataFrame({("Close", ""): [5.0, 6.0], ("Volume", ""): [1.0, 1.0]})
        assert fetcher._extract_close_price(blank, "X")["X"].tolist() == [5.0, 6.0]

    def test_multiindex_columns_without_close_fall_back_to_the_first_column(self, fetcher: MacroDataFetcher) -> None:
        frame = pd.DataFrame({("Open", "A"): [7.0, 8.0], ("High", "A"): [9.0, 9.0]})
        assert fetcher._extract_close_price(frame, "X")["X"].tolist() == [7.0, 8.0]


class TestDownloadWithRetry:
    def test_the_first_usable_download_wins(self, fetcher: MacroDataFetcher, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [None, pd.DataFrame(), _prices()]
        calls = []
        monkeypatch.setattr(tm.yf, "download", lambda ticker, **kw: calls.append((ticker, kw)) or responses.pop(0))
        out = fetcher._download_ticker_with_retry("HYG", "HYG", "2020-01-01")
        assert len(out) == 5 and len(calls) == 3 and calls[0][1]["start"] == "2020-01-01" and calls[0][1]["timeout"] == 30

    def test_after_the_last_attempt_an_empty_frame_is_returned_and_logged(self, fetcher: MacroDataFetcher,
                                                                          monkeypatch: pytest.MonkeyPatch,
                                                                          caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setattr(tm.yf, "download", lambda *a, **kw: None)
        with caplog.at_level(logging.ERROR, logger=tm.logger.name):
            assert fetcher._download_ticker_with_retry("HYG", "HYG", "2020-01-01").empty
        assert "definitively failed after 3 attempts" in caplog.text


class TestFetchDailyData:
    def _stub(self, fetcher: MacroDataFetcher, monkeypatch: pytest.MonkeyPatch, ok: set[str], dvol=True, fg=True) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        monkeypatch.setattr(fetcher, "_download_ticker_with_retry",
                            lambda name, ticker, start: pd.DataFrame({name: np.linspace(1, 2, 30)}, index=idx) if name in ok else pd.DataFrame())
        monkeypatch.setattr(fetcher, "fetch_dvol", lambda currency="BTC": pd.DataFrame(
            {"dvol_btc": np.linspace(40, 50, 30)}, index=idx) if dvol else pd.DataFrame())
        monkeypatch.setattr(fetcher, "fetch_fear_greed", lambda: pd.DataFrame(
            {"fear_greed": np.linspace(10, 90, 30)}, index=idx) if fg else pd.DataFrame())

    def test_with_no_ticker_at_all_nothing_is_returned(self, fetcher, monkeypatch, caplog) -> None:
        self._stub(fetcher, monkeypatch, ok=set())
        with caplog.at_level(logging.ERROR, logger=tm.logger.name):
            assert fetcher.fetch_daily_data().empty
        assert "No single TradFi macro ticker" in caplog.text

    def test_all_sources_are_joined_and_forward_filled(self, fetcher, monkeypatch) -> None:
        self._stub(fetcher, monkeypatch, ok={"10Y_Yield", "HYG", "DXY"})
        df = fetcher.fetch_daily_data()
        assert {"10Y_Yield", "HYG", "DXY", "dvol_btc", "fear_greed"} <= set(df.columns) and len(df) == 30

    def test_missing_dvol_and_fear_greed_are_skipped_with_a_warning(self, fetcher, monkeypatch, caplog) -> None:
        self._stub(fetcher, monkeypatch, ok={"HYG"}, dvol=False, fg=False)
        with caplog.at_level(logging.WARNING, logger=tm.logger.name):
            df = fetcher.fetch_daily_data()
        assert list(df.columns) == ["HYG"] and "DVOL not available" in caplog.text and "Fear & Greed not available" in caplog.text


# --------------------------------------------------------------------------- #
# features and pipeline
# --------------------------------------------------------------------------- #
def _raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "10Y_Yield": 4 + rng.normal(0, 0.05, n).cumsum(), "2Y_Yield": 3 + rng.normal(0, 0.05, n).cumsum(),
        "HYG": 75 + rng.normal(0, 0.3, n).cumsum(), "TLT": 95 + rng.normal(0, 0.3, n).cumsum(),
        "DXY": 28 + rng.normal(0, 0.1, n).cumsum(), "dvol_btc": 55 + rng.normal(0, 1.0, n).cumsum(),
        "fear_greed": np.clip(50 + rng.normal(0, 8, n).cumsum(), 0, 100)}, index=idx)


class TestFeatures:
    def test_nothing_in_nothing_out(self, fetcher: MacroDataFetcher) -> None:
        assert fetcher.calculate_macro_features(None).empty and fetcher.calculate_macro_features(pd.DataFrame()).empty    # type: ignore[arg-type]

    def test_every_feature_group_is_built_and_only_features_are_returned(self, fetcher: MacroDataFetcher) -> None:
        out = fetcher.calculate_macro_features(_raw())
        assert set(out.columns) == {
            "feat_macro_yc_spread", "feat_macro_yc_steepness", "feat_macro_credit_spread", "feat_macro_credit_momentum",
            "feat_macro_dxy_momentum", "feat_macro_dxy_zscore", "feat_macro_dvol_zscore", "feat_macro_dvol_regime",
            "feat_macro_dvol_pct", "feat_macro_dvol_momentum", "feat_macro_fg_norm", "feat_macro_fg_extreme_fear",
            "feat_macro_fg_extreme_greed", "feat_macro_fg_momentum"}
        assert np.isfinite(out.to_numpy()).all() and out["feat_macro_dvol_regime"].abs().max() <= 3.0
        raw = _raw()
        np.testing.assert_allclose(out["feat_macro_yc_spread"], raw["10Y_Yield"] - raw["2Y_Yield"])
        np.testing.assert_allclose(out["feat_macro_fg_norm"], (raw["fear_greed"] - 50) / 50)

    def test_groups_whose_inputs_are_missing_are_left_out(self, fetcher: MacroDataFetcher) -> None:
        out = fetcher.calculate_macro_features(_raw()[["DXY", "fear_greed"]])
        assert set(out.columns) == {"feat_macro_dxy_momentum", "feat_macro_dxy_zscore", "feat_macro_fg_norm",
                                    "feat_macro_fg_extreme_fear", "feat_macro_fg_extreme_greed", "feat_macro_fg_momentum"}

    def test_a_zero_denominator_gives_finite_features(self, fetcher: MacroDataFetcher) -> None:
        raw = _raw(60)
        raw["TLT"] = 0.0
        raw["DXY"] = 28.0
        assert np.isfinite(fetcher.calculate_macro_features(raw).to_numpy()).all()


class TestPipeline:
    def test_without_raw_data_nothing_is_written(self, fetcher, monkeypatch, caplog, tmp_path: Path) -> None:
        monkeypatch.setattr(fetcher, "fetch_daily_data", pd.DataFrame)
        with caplog.at_level(logging.ERROR, logger=tm.logger.name):
            fetcher.run_pipeline()
        assert "No data received" in caplog.text and not list((tmp_path / "macro").glob("*.parquet"))

    def test_every_feature_is_shifted_by_its_publication_lag_and_the_index_becomes_a_timestamp_column(
        self, fetcher, monkeypatch, tmp_path: Path
    ) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        features = pd.DataFrame({"feat_macro_fomc_rate": [1.0, 2.0, 3.0], "feat_macro_cpi": [4.0, 5.0, 6.0],
                                 "feat_macro_other": [7.0, 8.0, 9.0]}, index=idx)
        monkeypatch.setattr(fetcher, "fetch_daily_data", lambda: _raw(5))
        monkeypatch.setattr(fetcher, "calculate_macro_features", lambda df: features)
        fetcher.run_pipeline()
        out = pd.read_parquet(tmp_path / "macro" / "macro_tradfi_daily.parquet").set_index("timestamp")
        assert out["feat_macro_fomc_rate"].dropna().index[0] == idx[0] + pd.Timedelta(hours=18)
        assert out["feat_macro_cpi"].dropna().index[0] == idx[0] + pd.Timedelta(hours=12)
        assert out["feat_macro_other"].dropna().index[0] == idx[0] + pd.Timedelta(hours=12)

    @pytest.mark.parametrize("name", ["Date", "date", None])
    def test_a_date_like_index_name_is_normalised_to_timestamp(self, fetcher, monkeypatch, tmp_path: Path, name) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC", name=name)
        features = pd.DataFrame({"feat_macro_x": [1.0, 2.0, 3.0]}, index=idx)
        monkeypatch.setattr(fetcher, "fetch_daily_data", lambda: _raw(5))
        monkeypatch.setattr(fetcher, "calculate_macro_features", lambda df: features)
        fetcher.run_pipeline()
        assert "timestamp" in pd.read_parquet(tmp_path / "macro" / "macro_tradfi_daily.parquet").columns

    def test_a_real_end_to_end_run_on_synthetic_inputs_writes_the_feature_file(self, fetcher, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.setattr(fetcher, "fetch_daily_data", _raw)
        fetcher.run_pipeline()
        out = pd.read_parquet(tmp_path / "macro" / "macro_tradfi_daily.parquet")
        assert len(out) > 250 and all(c == "timestamp" or c.startswith("feat_macro_") for c in out.columns)
