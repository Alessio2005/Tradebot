# tests/unit/test_data_crypto_macro.py
"""`data/crypto_macro.py`: funding-historie van Bybit, ECDF-features per asset en de marktbrede aggregaten.

Geen netwerk: `requests.get` is een nep-exchange die uit een vaste historie put,
net als de echte endpoint achterwaarts pagineert (`endTime`, aflopend, `limit`).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from tradebot.data import crypto_macro as cm
from tradebot.data.crypto_macro import CryptoMacroFetcher, MultiCryptoMacroFetcher
from tradebot.utils.failfast import DataContractError

NOW = pd.Timestamp.now(tz="UTC").floor("8h")


def _history(n: int = 120, seed: int = 0) -> list[tuple[int, float]]:
    rng = np.random.default_rng(seed)
    stamps = [int((NOW - pd.Timedelta(hours=8 * k)).timestamp() * 1000) for k in range(n)]
    return list(zip(stamps, rng.normal(0.0001, 0.0002, n), strict=True))


class _Exchange:
    """Nep-Bybit: levert per call hoogstens `limit` rijen met ts <= endTime, aflopend."""

    def __init__(self, data: dict[str, list[tuple[int, float]]], ret_code: int = 0, fail: Exception | None = None) -> None:
        self.data, self.ret_code, self.fail, self.calls = data, ret_code, fail, []

    def get(self, url: str, params=None, timeout=None):
        self.calls.append(dict(params))
        if self.fail is not None:
            raise self.fail
        rows = [r for r in self.data.get(params["symbol"], []) if r[0] <= params["endTime"]]
        rows.sort(reverse=True)
        payload = {"retCode": self.ret_code, "retMsg": "boom",
                   "result": {"list": [{"fundingRateTimestamp": str(t), "fundingRate": str(v)}
                                       for t, v in rows[: params["limit"]]]}}
        return _Resp(payload)


class _Resp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


@pytest.fixture
def exchange(monkeypatch: pytest.MonkeyPatch):
    def install(data=None, **kw) -> _Exchange:
        ex = _Exchange(data or {}, **kw)
        monkeypatch.setattr(requests, "get", ex.get)
        monkeypatch.setattr(cm.time, "sleep", lambda s: None)
        return ex
    return install


def _fetcher(tmp_path: Path, symbol: str = "BTCUSDT", days: int = 60) -> CryptoMacroFetcher:
    f = CryptoMacroFetcher(data_dir=str(tmp_path), symbol=symbol, fetch_years=1)
    f.start_ts = int((NOW - pd.Timedelta(days=days)).timestamp() * 1000)
    return f


class TestFundingFetch:
    def test_the_data_dir_is_created_and_the_window_starts_fetch_years_back(self, tmp_path: Path) -> None:
        f = CryptoMacroFetcher(data_dir=str(tmp_path / "a" / "b"), symbol="ETHUSDT", fetch_years=2)
        assert (tmp_path / "a" / "b").is_dir() and f.category == "linear"
        two_years_ms = 2 * 365 * 86_400_000
        assert abs((pd.Timestamp.now(tz="UTC").timestamp() * 1000 - f.start_ts) - two_years_ms) < 3 * 86_400_000

    def test_pages_are_walked_backwards_until_the_history_runs_out(self, tmp_path: Path, exchange) -> None:
        ex = exchange({"BTCUSDT": _history(450)})
        rows = _fetcher(tmp_path, days=400)._paginate_backward(limit=200)
        assert len(rows) == 450 and [c["limit"] for c in ex.calls] == [200, 200, 200, 200]
        assert ex.calls[1]["endTime"] == int(rows[199]["fundingRateTimestamp"]) - 1

    def test_the_walk_stops_at_the_requested_start(self, tmp_path: Path, exchange) -> None:
        ex = exchange({"BTCUSDT": _history(450)})
        rows = _fetcher(tmp_path, days=20)._paginate_backward(limit=200)
        assert len(ex.calls) == 1 and len(rows) == 200

    def test_an_application_error_stops_the_walk_and_is_logged(self, tmp_path: Path, exchange,
                                                               caplog: pytest.LogCaptureFixture) -> None:
        exchange({"BTCUSDT": _history(10)}, ret_code=10001)
        with caplog.at_level(logging.ERROR, logger=cm.logger.name):
            assert _fetcher(tmp_path)._paginate_backward() == []
        assert "retCode=10001" in caplog.text

    def test_a_transport_error_is_a_contract_error_and_never_a_partial_history(self, tmp_path: Path, exchange) -> None:
        exchange(fail=requests.ConnectionError("reset"))
        with pytest.raises(DataContractError, match="GEEN gedeeltelijke historie"):
            _fetcher(tmp_path)._paginate_backward()

    def test_the_frame_is_sorted_deduplicated_and_typed(self, tmp_path: Path, exchange) -> None:
        data = _history(30)
        data.append((data[0][0], 0.5))                                        # duplicaat: laatste wint
        exchange({"BTCUSDT": data})
        df = _fetcher(tmp_path).fetch_funding_rate()
        assert df.index.is_monotonic_increasing and df.index.is_unique and df.index.tz is not None
        assert list(df.columns) == ["fundingRate"] and df["fundingRate"].dtype == float
        assert len(df) == 30

    def test_no_data_gives_an_empty_frame(self, tmp_path: Path, exchange) -> None:
        exchange({})
        assert _fetcher(tmp_path).fetch_funding_rate().empty


class TestFeatures:
    def _funding(self, n: int = 90) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq="8h", tz="UTC")
        return pd.DataFrame({"fundingRate": np.random.default_rng(0).normal(0, 1e-4, n)}, index=idx)

    def test_an_empty_frame_passes_through(self, tmp_path: Path) -> None:
        empty = pd.DataFrame()
        assert _fetcher(tmp_path).calculate_macro_features(empty) is empty

    def test_the_ecdf_feature_lives_in_minus_one_to_one_and_the_rate_is_kept(self, tmp_path: Path) -> None:
        out = _fetcher(tmp_path).calculate_macro_features(self._funding())
        assert list(out.columns) == ["feat_macro_funding_zscore", "fundingRate"]
        assert out["feat_macro_funding_zscore"].between(-1.0, 1.0).all() and np.isfinite(out.to_numpy()).all()
        assert (out["feat_macro_funding_zscore"].iloc[:9] == 0.0).all()      # min_periods = 10: warm-up wordt 0

    def test_the_rank_fallback_gives_the_same_feature(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        funding = self._funding()
        expected = _fetcher(tmp_path).calculate_macro_features(funding.copy())

        def broken(self, *a, **kw):
            raise TypeError("rank unsupported")
        monkeypatch.setattr(pd.core.window.rolling.Rolling, "rank", broken)
        got = _fetcher(tmp_path).calculate_macro_features(funding.copy())
        pd.testing.assert_frame_equal(got, expected)

    def test_infinite_rates_are_neutralised(self, tmp_path: Path) -> None:
        funding = self._funding()
        funding.iloc[40, 0] = np.inf
        out = _fetcher(tmp_path).calculate_macro_features(funding)
        assert np.isfinite(out.to_numpy()).all()


class TestSinglePipeline:
    def test_btc_writes_its_asset_file_and_the_legacy_alias(self, tmp_path: Path, exchange) -> None:
        exchange({"BTCUSDT": _history(60)})
        out = _fetcher(tmp_path).run_pipeline()
        assert not out.empty
        for name in ("macro_crypto_BTCUSDT.parquet", "macro_crypto_daily.parquet"):
            frame = pd.read_parquet(tmp_path / name)
            assert "timestamp" in frame.columns or frame.index.name == "timestamp"
            assert {"feat_macro_funding_zscore", "fundingRate"} <= set(frame.columns)

    def test_other_symbols_get_no_legacy_alias(self, tmp_path: Path, exchange) -> None:
        exchange({"ETHUSDT": _history(60)})
        _fetcher(tmp_path, "ETHUSDT").run_pipeline()
        assert (tmp_path / "macro_crypto_ETHUSDT.parquet").exists() and not (tmp_path / "macro_crypto_daily.parquet").exists()

    def test_an_unnamed_index_still_ends_up_as_a_timestamp_column(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        idx = pd.date_range("2024-01-01", periods=40, freq="8h", tz="UTC")
        frame = pd.DataFrame({"fundingRate": np.linspace(0, 1e-3, 40)}, index=idx)
        monkeypatch.setattr(CryptoMacroFetcher, "fetch_funding_rate", lambda self: frame)
        _fetcher(tmp_path, "ETHUSDT").run_pipeline()
        assert "timestamp" in pd.read_parquet(tmp_path / "macro_crypto_ETHUSDT.parquet").columns

    def test_a_failed_stream_aborts_with_an_empty_frame(self, tmp_path: Path, exchange, caplog: pytest.LogCaptureFixture) -> None:
        exchange({})
        with caplog.at_level(logging.ERROR, logger=cm.logger.name):
            assert _fetcher(tmp_path).run_pipeline().empty
        assert "Funding rate stream failed" in caplog.text and not list(tmp_path.glob("*.parquet"))


SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class TestMultiAsset:

    def test_the_defaults_cover_six_majors(self, tmp_path: Path) -> None:
        assert MultiCryptoMacroFetcher(data_dir=str(tmp_path)).symbols == MultiCryptoMacroFetcher.DEFAULT_SYMBOLS
        assert len(MultiCryptoMacroFetcher.DEFAULT_SYMBOLS) == 6

    def test_every_asset_and_the_market_file_are_written(self, tmp_path: Path, exchange, monkeypatch: pytest.MonkeyPatch) -> None:
        exchange({s: _history(120, seed=k) for k, s in enumerate(SYMS)})
        multi = MultiCryptoMacroFetcher(data_dir=str(tmp_path), symbols=[*SYMS, "DOWNUSDT"], fetch_years=1)
        monkeypatch.setattr(cm.CryptoMacroFetcher, "__init__", _patched_init(cm.CryptoMacroFetcher.__init__))
        per_asset = multi.run_pipeline()
        assert set(per_asset) == set(SYMS)                               # DOWNUSDT levert niets op
        market = pd.read_parquet(tmp_path / "macro_crypto_market.parquet")
        assert {"feat_macro_market_funding", "feat_macro_btc_dominance_proxy", "feat_macro_funding_dispersion"} <= set(market.columns)

    def test_an_unnamed_market_index_still_ends_up_as_a_timestamp_column(self, tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="8h", tz="UTC")
        monkeypatch.setattr(cm.CryptoMacroFetcher, "run_pipeline",
                            lambda self: pd.DataFrame({"feat_macro_funding_zscore": np.zeros(5)}, index=idx))
        monkeypatch.setattr(MultiCryptoMacroFetcher, "_build_market_features",
                            lambda self, per_asset: pd.DataFrame({"m": np.zeros(5)}, index=idx))
        MultiCryptoMacroFetcher(data_dir=str(tmp_path), symbols=["BTCUSDT"]).run_pipeline()
        assert "timestamp" in pd.read_parquet(tmp_path / "macro_crypto_market.parquet").columns

    def test_when_every_stream_fails_nothing_is_written(self, tmp_path: Path, exchange, caplog: pytest.LogCaptureFixture) -> None:
        exchange({})
        with caplog.at_level(logging.ERROR, logger=cm.logger.name):
            assert MultiCryptoMacroFetcher(data_dir=str(tmp_path), symbols=["BTCUSDT"]).run_pipeline() == {}
        assert "All macro streams failed" in caplog.text and not (tmp_path / "macro_crypto_market.parquet").exists()


def _patched_init(original):
    def init(self, data_dir="x", symbol="BTCUSDT", fetch_years=5):
        original(self, data_dir=data_dir, symbol=symbol, fetch_years=fetch_years)
        self.start_ts = int((NOW - pd.Timedelta(days=30)).timestamp() * 1000)
    return init


class TestMarketFeatures:
    def _asset(self, seed: int, n: int = 40, step: str = "8h") -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=n, freq=step, tz="UTC")
        return pd.DataFrame({"feat_macro_funding_zscore": np.random.default_rng(seed).uniform(-1, 1, n)}, index=idx)

    def test_dominance_is_btc_minus_the_mean_of_the_alts_and_dispersion_their_spread(self, tmp_path: Path) -> None:
        multi = MultiCryptoMacroFetcher(data_dir=str(tmp_path))
        per_asset = {"BTCUSDT": self._asset(0), "ETHUSDT": self._asset(1), "SOLUSDT": self._asset(2)}
        out = multi._build_market_features(per_asset)
        z = pd.DataFrame({s: d["feat_macro_funding_zscore"] for s, d in per_asset.items()})
        np.testing.assert_allclose(out["feat_macro_market_funding"], z.mean(axis=1))
        np.testing.assert_allclose(out["feat_macro_btc_dominance_proxy"], z["BTCUSDT"] - z[["ETHUSDT", "SOLUSDT"]].mean(axis=1))
        np.testing.assert_allclose(out["feat_macro_funding_dispersion"], z.std(axis=1, ddof=0))

    def test_without_btc_or_with_one_asset_there_is_no_dominance_signal(self, tmp_path: Path) -> None:
        multi = MultiCryptoMacroFetcher(data_dir=str(tmp_path))
        only_alts = multi._build_market_features({"ETHUSDT": self._asset(1), "SOLUSDT": self._asset(2)})
        single = multi._build_market_features({"BTCUSDT": self._asset(0)})
        assert (only_alts["feat_macro_btc_dominance_proxy"] == 0.0).all()
        assert (single["feat_macro_btc_dominance_proxy"] == 0.0).all() and (single["feat_macro_funding_dispersion"] == 0.0).all()

    def test_assets_without_a_zscore_are_skipped_and_nothing_usable_is_an_empty_frame(self, tmp_path: Path) -> None:
        multi = MultiCryptoMacroFetcher(data_dir=str(tmp_path))
        assert multi._build_market_features({"BTCUSDT": pd.DataFrame({"x": [1.0]}), "ETHUSDT": pd.DataFrame()}).empty

    def test_a_mostly_forward_filled_asset_is_flagged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        multi = MultiCryptoMacroFetcher(data_dir=str(tmp_path))
        sparse = self._asset(3, n=10, step="24h")                                # elke 3e 8h-bucket echt, de rest ffill
        with caplog.at_level(logging.WARNING, logger=cm.logger.name):
            multi._build_market_features({"BTCUSDT": sparse})
        assert "ffilled (>30%)" in caplog.text
