# tests/unit/test_data_macro.py
"""`data/macro.py`: het verversen van de macro-bestanden en het lag-bewuste samenvoegen op de bars."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tradebot.data import crypto_macro, macro, tradfi_macro
from tradebot.data.macro import _load_macro_df, _merge_asof_lagged, merge_macro, update_macro


def _cfg(tmp_path: Path, universe=None):
    return OmegaConf.create({"machine": {"data_dir": str(tmp_path)}, "training": {"training_universe": universe or []}})


# --------------------------------------------------------------------------- #
# update_macro
# --------------------------------------------------------------------------- #
class TestUpdateMacro:
    def _record(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
        calls: list[tuple] = []

        def stub(name):
            class Fetcher:
                def __init__(self, **kw) -> None:
                    self.kw = kw

                def run_pipeline(self) -> None:
                    calls.append((name, self.kw))
            return Fetcher
        monkeypatch.setattr(tradfi_macro, "MacroDataFetcher", stub("tradfi"))
        monkeypatch.setattr(crypto_macro, "CryptoMacroFetcher", stub("crypto"))
        monkeypatch.setattr(crypto_macro, "MultiCryptoMacroFetcher", stub("multi"))
        return calls

    def test_tradfi_and_the_multi_asset_crypto_fetcher_run_for_a_crypto_universe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record(monkeypatch)
        update_macro(_cfg(tmp_path, ["BTCUSDT", "ETHUSDC", "AAPL", 5]))
        assert calls[0] == ("tradfi", {"data_dir": f"{tmp_path}/macro"})
        assert calls[1] == ("multi", {"data_dir": f"{tmp_path}/macro", "symbols": ["BTCUSDT", "ETHUSDC"]})

    def test_a_universe_without_crypto_falls_back_to_the_single_symbol_fetcher(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record(monkeypatch)
        update_macro(_cfg(tmp_path, ["AAPL"]))
        assert [c[0] for c in calls] == ["tradfi", "crypto"]

    def test_the_default_data_dir_is_used_when_the_machine_config_has_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record(monkeypatch)
        update_macro(OmegaConf.create({"machine": {}, "training": {"training_universe": []}}))
        assert calls[0][1]["data_dir"] == "market_data_parquet/macro"


# --------------------------------------------------------------------------- #
# loading and lagged merging
# --------------------------------------------------------------------------- #
class TestLoad:
    def test_a_missing_file_is_none_with_a_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=macro.logger.name):
            assert _load_macro_df(tmp_path / "nope.parquet", "BTC") is None
        assert "Macro file not found" in caplog.text

    def test_a_timestamp_column_becomes_a_sorted_utc_index(self, tmp_path: Path) -> None:
        path = tmp_path / "a.parquet"
        pd.DataFrame({"timestamp": ["2024-01-03", "2024-01-01"], "x": [3.0, 1.0]}).to_parquet(path)
        df = _load_macro_df(path, "BTC")
        assert df.index.is_monotonic_increasing and str(df.index.tz) == "UTC" and df["x"].tolist() == [1.0, 3.0]

    def test_a_plain_index_is_converted_to_datetimes(self, tmp_path: Path) -> None:
        path = tmp_path / "b.parquet"
        pd.DataFrame({"x": [1.0, 2.0]}, index=["2024-01-02", "2024-01-01"]).to_parquet(path)
        df = _load_macro_df(path, "BTC")
        assert isinstance(df.index, pd.DatetimeIndex) and df["x"].tolist() == [2.0, 1.0]

    def test_a_datetime_index_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "c.parquet"
        idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
        pd.DataFrame({"x": [1.0, 2.0]}, index=idx).to_parquet(path)
        assert _load_macro_df(path, "BTC").index.equals(idx)


class TestLaggedMerge:
    def _bars(self) -> pd.DataFrame:
        return pd.DataFrame({"close": np.arange(48.0)}, index=pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"))

    def test_lagged_and_realtime_columns_are_merged_with_their_own_tolerance(self) -> None:
        macro_idx = pd.DatetimeIndex(["2024-01-01 06:00"], tz="UTC")
        df_macro = pd.DataFrame({"feat_macro_cpi": [1.5], "yield_10y": [4.5]}, index=macro_idx)
        out = _merge_asof_lagged(self._bars(), df_macro, "T")
        assert pd.isna(out.loc["2024-01-01 17:00", "feat_macro_cpi"]) and out.loc["2024-01-01 18:00", "feat_macro_cpi"] == 1.5
        assert out.loc["2024-01-01 06:00", "yield_10y"] == 4.5 and pd.isna(out.loc["2024-01-02 07:00", "yield_10y"])

    def test_a_zero_lag_makes_the_value_visible_immediately(self) -> None:
        df_macro = pd.DataFrame({"feat_macro_x": [2.0]}, index=pd.DatetimeIndex(["2024-01-01 06:00"], tz="UTC"))
        out = _merge_asof_lagged(self._bars(), df_macro, "T", lag=pd.Timedelta(0))
        assert out.loc["2024-01-01 06:00", "feat_macro_x"] == 2.0

    def test_only_realtime_or_only_lagged_inputs_work(self) -> None:
        idx = pd.DatetimeIndex(["2024-01-01 00:00"], tz="UTC")
        only_rt = _merge_asof_lagged(self._bars(), pd.DataFrame({"live_px": [1.0]}, index=idx), "T")
        only_lag = _merge_asof_lagged(self._bars(), pd.DataFrame({"feat_macro_z": [1.0]}, index=idx), "T")
        assert "live_px" in only_rt.columns and "feat_macro_z" in only_lag.columns


class TestMergeMacro:
    def _bars(self, n: int = 72) -> pd.DataFrame:
        return pd.DataFrame({"close": np.arange(float(n))}, index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))

    def _write(self, root: Path, name: str, **cols) -> None:
        (root / "macro").mkdir(parents=True, exist_ok=True)
        idx = pd.date_range("2023-12-30", periods=6, freq="D", tz="UTC", name="timestamp")
        pd.DataFrame(cols, index=idx).reset_index().to_parquet(root / "macro" / name, index=False)

    def test_a_crypto_symbol_gets_its_asset_file_and_the_market_file(self, tmp_path: Path) -> None:
        self._write(tmp_path, "macro_crypto_BTCUSDT.parquet", feat_macro_funding_zscore=np.linspace(-1, 1, 6))
        self._write(tmp_path, "macro_crypto_market.parquet", feat_macro_market_funding=np.linspace(1, -1, 6))
        out = merge_macro(self._bars(), "BTCUSDT", _cfg(tmp_path))
        assert {"feat_macro_funding_zscore", "feat_macro_market_funding", "feat_macro_available"} <= set(out.columns)
        assert out["feat_macro_available"].iloc[-1] == 1 and out["feat_macro_available"].dtype == np.int8
        assert not out.isna().any().any()

    def test_without_an_asset_file_the_btc_daily_alias_is_used(self, tmp_path: Path) -> None:
        self._write(tmp_path, "macro_crypto_daily.parquet", feat_macro_funding_zscore=np.zeros(6))
        out = merge_macro(self._bars(), "ETHUSDT", _cfg(tmp_path))
        assert "feat_macro_funding_zscore" in out.columns

    def test_a_non_crypto_symbol_uses_the_tradfi_file(self, tmp_path: Path) -> None:
        self._write(tmp_path, "macro_tradfi_daily.parquet", feat_macro_yc_spread=np.ones(6))
        out = merge_macro(self._bars(), "SPY", _cfg(tmp_path))
        assert "feat_macro_yc_spread" in out.columns

    def test_missing_files_give_zero_availability_and_a_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=macro.logger.name):
            out = merge_macro(self._bars(), "BTCUSDT", _cfg(tmp_path))
        assert (out["feat_macro_available"] == 0).all() and "Continuing without per-asset macro" in caplog.text

    def test_the_default_data_dir_is_market_data_parquet(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        out = merge_macro(self._bars(3), "SPY", OmegaConf.create({"machine": {}}))
        assert len(out) == 3

    def test_infinite_values_and_duplicate_timestamps_are_cleaned(self, tmp_path: Path) -> None:
        self._write(tmp_path, "macro_tradfi_daily.parquet", feat_macro_x=[1.0, np.inf, 2.0, 3.0, 4.0, 5.0])
        bars = self._bars(10)
        dup = pd.concat([bars, bars.iloc[[2, 2, 5]]]).sort_index()
        out = merge_macro(dup, "SPY", _cfg(tmp_path))
        assert out.index.is_unique and np.isfinite(out.to_numpy()).all() and len(out) == len(dup)

    def test_exact_duplicates_that_survive_the_nanosecond_offset_are_dropped(self, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
        bars = self._bars(4)
        original = pd.to_timedelta
        monkeypatch.setattr(macro.pd, "to_timedelta", lambda values, unit=None: original(np.zeros(len(values)), unit="ns"))
        dup = pd.concat([bars, bars.iloc[[1]]]).sort_index()
        out = merge_macro(dup, "SPY", _cfg(tmp_path))
        assert out.index.is_unique and len(out) == 4
