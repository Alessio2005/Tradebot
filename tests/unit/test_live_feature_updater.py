# tests/unit/test_live_feature_updater.py
"""`live/feature_updater.py`: rollend venster, verversingsritme, force-refresh en pariteitscontrole.

`build_features` en de `FeatureStore` zijn gestubd; de echte pipeline wordt in
`test_features_regime_edges.py` op minuutdata getest.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.live import feature_updater as fu
from tradebot.live.feature_updater import FeatureUpdater, FeatureUpdaterConfig

SYM = "ETHUSDT"


def _history(n: int = 50, freq: str = "min", **extra) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.arange(n, dtype=float)
    df = pd.DataFrame({"open": close - 0.5, "high": close + 1.0, "low": close - 1.0, "close": close,
                       "volume": np.full(n, 10.0)}, index=idx)
    for k, v in extra.items():
        df[k] = v
    return df


def _bar(ts: str = "2024-01-01 01:00", close: float = 200.0, **kw) -> pd.Series:
    data = {"open": close - 1, "high": close + 2, "low": close - 2, "close": close, "volume": 5.0, **kw}
    return pd.Series(data, name=pd.Timestamp(ts, tz="UTC"))


def _features(n: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"feat_vol_gk": np.full(n, 0.01), "feat_x": np.arange(float(n)),
                         "close": np.full(n, 1.0), "open": 1.0, "high": 1.0, "low": 1.0, "volume": 1.0}, index=idx)


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch):
    """Vervangt `build_features` door een stub die de aanroepen bijhoudt."""
    calls: list[dict] = []
    state = {"result": _features(), "error": None}

    def fake(cfg, symbol, df, orthogonalizer_save_dir=None):
        calls.append({"symbol": symbol, "rows": len(df), "save_dir": orthogonalizer_save_dir})
        if state["error"] is not None:
            raise state["error"]
        return state["result"], {}
    monkeypatch.setattr("tradebot.features.pipeline.build_features", fake)
    return SimpleNamespace(calls=calls, state=state)


def _updater(**cfg) -> FeatureUpdater:
    return FeatureUpdater(FeatureUpdaterConfig(**cfg))


# --------------------------------------------------------------------------- #
# construction and pre-population
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_defaults_and_the_refresh_interval_floor(self) -> None:
        cfg = FeatureUpdaterConfig(refresh_every_n_bars=0)
        assert cfg.refresh_every_n_bars == 1 and cfg.force_refresh_window_overrides == {} and cfg.window_bars == 600

    def test_a_feature_store_is_opened_only_when_asked_for_with_a_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        opened = []

        class FakeStore:
            def __init__(self, root) -> None:
                opened.append(root)
        monkeypatch.setattr("tradebot.featurestore.store.FeatureStore", FakeStore)
        assert _updater(use_feature_store=True)._store is None and opened == []
        assert isinstance(_updater(use_feature_store=True, feature_store_root="/x")._store, FakeStore)
        assert opened == ["/x"]

    def test_a_drift_monitor_can_be_attached(self) -> None:
        up = _updater()
        monitor = object()
        up.attach_drift_monitor(monitor)
        assert up._drift_monitor is monitor


class TestPrePopulate:
    def test_ohlcv_history_fills_the_buffer_trimmed_to_the_window(self) -> None:
        up = _updater(window_bars=30)
        up.pre_populate(SYM, _history(50))
        buf = up.get_buffer(SYM)
        assert len(buf) == 30 and buf.index[0] == _history(50).index[20]
        assert set(buf.columns) == {"open", "high", "low", "close", "volume"}

    def test_missing_ohlc_columns_leave_the_buffer_empty_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        up = _updater()
        with caplog.at_level(logging.WARNING, logger=fu.logger.name):
            up.pre_populate(SYM, _history().drop(columns=["high"]))
        assert up.get_buffer(SYM).empty and "missing OHLCV columns" in caplog.text

    def test_real_volume_is_renamed_and_a_missing_volume_becomes_zero_and_nans_are_filled(self) -> None:
        up = _updater()
        hist = _history(5).drop(columns=["volume"])
        hist["real_volume"] = [1.0, np.nan, 3.0, 4.0, 5.0]
        up.pre_populate(SYM, hist)
        assert up.get_buffer(SYM)["volume"].tolist() == [1.0, 0.0, 3.0, 4.0, 5.0]
        up2 = _updater()
        up2.pre_populate(SYM, _history(5).drop(columns=["volume"]))
        assert (up2.get_buffer(SYM)["volume"] == 0.0).all()

    def test_taker_columns_are_kept_and_repopulating_replaces_the_old_buffer(self) -> None:
        up = _updater()
        up.pre_populate(SYM, _history(10, taker_buy_volume=1.0, taker_sell_volume=2.0, junk=9.0))
        buf = up.get_buffer(SYM)
        assert {"taker_buy_volume", "taker_sell_volume"} <= set(buf.columns) and "junk" not in buf.columns
        up.pre_populate(SYM, _history(4))
        assert len(up.get_buffer(SYM)) == 4

    def test_a_single_bar_history_uses_the_default_bar_length(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=fu.logger.name):
            _updater().pre_populate(SYM, _history(1))
        assert "median bar = 60s" in caplog.text

    def test_the_median_bar_length_is_measured_from_the_last_hundred_bars(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=fu.logger.name):
            _updater().pre_populate(SYM, _history(20, freq="5s"))
        assert "median bar = 5s" in caplog.text


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #
class TestUpdateWithoutPipeline:
    def test_the_first_bar_gives_nothing_and_later_bars_pass_the_last_row_through(self) -> None:
        up = _updater()
        assert up.update(SYM, _bar("2024-01-01 00:00")) is None
        out = up.update(SYM, _bar("2024-01-01 00:01", close=300.0))
        assert len(out) == 1 and out["close"].iloc[0] == 300.0

    def test_feature_columns_select_what_is_returned(self) -> None:
        up = _updater(feature_columns=["close", "nope"])
        up.update(SYM, _bar("2024-01-01 00:00"))
        assert list(up.update(SYM, _bar("2024-01-01 00:01")).columns) == ["close"]
        everything = _updater(feature_columns=["nope"])
        everything.update(SYM, _bar("2024-01-01 00:00"))
        assert "close" in everything.update(SYM, _bar("2024-01-01 00:01")).columns

    def test_the_window_drops_the_oldest_bar(self) -> None:
        up = _updater(window_bars=3)
        for k in range(6):
            up.update(SYM, _bar(f"2024-01-01 00:0{k}", close=100.0 + k))
        buf = up.get_buffer(SYM)
        assert len(buf) == 3 and buf["close"].tolist() == [103.0, 104.0, 105.0]


class TestUpdateWithPipeline:
    def test_the_pipeline_runs_on_the_refresh_schedule_and_the_cache_serves_the_bars_in_between(
        self, stub_pipeline
    ) -> None:
        up = _updater(refresh_every_n_bars=3, feature_pipeline_cfg=object())
        for k in range(3):                                                         # de eerste bar telt niet mee (buffer < 2)
            assert up.update(SYM, _bar(f"2024-01-01 00:0{k}")) is None
        assert stub_pipeline.calls == []
        out = up.update(SYM, _bar("2024-01-01 00:03", close=250.0))                # derde getelde bar: refresh
        assert len(stub_pipeline.calls) == 1 and out["close"].iloc[0] == 250.0
        up.update(SYM, _bar("2024-01-01 00:04"))
        assert len(stub_pipeline.calls) == 1 and stub_pipeline.calls[0]["save_dir"] is None

    def test_the_cached_row_gets_the_current_bar_and_a_volatility_proxy_patched_in(self, stub_pipeline) -> None:
        up = _updater(feature_pipeline_cfg=object())
        up.update(SYM, _bar("2024-01-01 00:00"))
        out = up.update(SYM, _bar("2024-01-01 00:01", close=200.0))
        assert (out["open"].iloc[0], out["high"].iloc[0], out["low"].iloc[0]) == (199.0, 202.0, 198.0)
        assert out["volume"].iloc[0] == 5.0 and out["feat_vol_gk"].iloc[0] == pytest.approx(4.0 / 200.0)
        assert out["feat_x"].iloc[0] == 2.0                                        # features uit de pipeline blijven

    def test_a_flat_bar_keeps_the_pipeline_volatility_and_a_missing_column_is_not_invented(self, stub_pipeline) -> None:
        stub_pipeline.state["result"] = _features().drop(columns=["volume", "open"])
        up = _updater(feature_pipeline_cfg=object())
        up.update(SYM, _bar("2024-01-01 00:00"))
        flat = pd.Series({"open": 7.0, "high": 7.0, "low": 7.0, "close": 7.0}, name=pd.Timestamp("2024-01-01 00:01", tz="UTC"))
        out = up.update(SYM, flat)
        assert out["feat_vol_gk"].iloc[0] == 0.01 and "volume" not in out.columns and "open" not in out.columns

    def test_an_empty_or_failed_refresh_leaves_the_cache_alone(self, stub_pipeline) -> None:
        stub_pipeline.state["result"] = pd.DataFrame()
        up = _updater(feature_pipeline_cfg=object())
        up.update(SYM, _bar("2024-01-01 00:00"))
        assert up.update(SYM, _bar("2024-01-01 00:01")) is None

    def test_every_refresh_is_appended_to_the_feature_store(self, stub_pipeline) -> None:
        stored = []
        up = _updater(feature_pipeline_cfg=object())
        up._store = SimpleNamespace(append=lambda feats, sym: stored.append((len(feats), sym)))
        up.update(SYM, _bar("2024-01-01 00:00"))
        up.update(SYM, _bar("2024-01-01 00:01"))
        assert stored == [(3, SYM)]

    def test_the_refresh_counter_reports_the_distance_to_the_last_refresh(self, stub_pipeline) -> None:
        up = _updater(refresh_every_n_bars=4, feature_pipeline_cfg=object())
        for k in range(1, 7):
            up.update(SYM, _bar(f"2024-01-01 00:0{k}"))
        assert up.bars_since_last_refresh(SYM) == 5 % 4 and _updater().bars_since_last_refresh(SYM) == 0


# --------------------------------------------------------------------------- #
# force refresh
# --------------------------------------------------------------------------- #
class TestForceRefresh:
    def test_a_short_buffer_gives_nothing(self, stub_pipeline) -> None:
        up = _updater(feature_pipeline_cfg=object())
        up.pre_populate(SYM, _history(1))
        assert up.force_refresh(SYM) is None and stub_pipeline.calls == []

    def test_it_runs_the_pipeline_now_and_remembers_the_usable_bars(self, stub_pipeline) -> None:
        up = _updater(feature_pipeline_cfg=object())
        up.pre_populate(SYM, _history(30))
        assert up.get_last_usable_bars(SYM) == 0
        out = up.force_refresh(SYM)
        assert len(out) == 1 and up.get_last_usable_bars(SYM) == 3 and up._feature_cache[SYM] is not None
        assert stub_pipeline.calls[0]["rows"] == 30

    def test_the_default_window_caps_the_input(self, stub_pipeline, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FeatureUpdater, "_FORCE_REFRESH_WINDOW", 10)
        up = _updater(window_bars=100, feature_pipeline_cfg=object())
        up.pre_populate(SYM, _history(50))
        up.force_refresh(SYM)
        assert stub_pipeline.calls[0]["rows"] == 10
        short = _updater(window_bars=100, feature_pipeline_cfg=object())
        short.pre_populate(SYM, _history(8))
        short.force_refresh(SYM)
        assert stub_pipeline.calls[1]["rows"] == 8

    def test_per_symbol_overrides_cap_or_lift_the_window(self, stub_pipeline) -> None:
        cfg = {"feature_pipeline_cfg": object(), "window_bars": 100}
        capped = _updater(force_refresh_window_overrides={SYM: 12}, **cfg)
        capped.pre_populate(SYM, _history(50))
        capped.force_refresh(SYM)
        full = _updater(force_refresh_window_overrides={SYM: None}, **cfg)
        full.pre_populate(SYM, _history(50))
        full.force_refresh(SYM)
        roomy = _updater(force_refresh_window_overrides={SYM: 500}, **cfg)
        roomy.pre_populate(SYM, _history(50))
        roomy.force_refresh(SYM)
        assert [c["rows"] for c in stub_pipeline.calls] == [12, 50, 50]

    def test_an_empty_pipeline_result_is_a_warning_and_nothing(self, stub_pipeline, caplog: pytest.LogCaptureFixture) -> None:
        stub_pipeline.state["result"] = pd.DataFrame()
        up = _updater(feature_pipeline_cfg=object())
        up.pre_populate(SYM, _history(30))
        with caplog.at_level(logging.WARNING, logger=fu.logger.name):
            assert up.force_refresh(SYM) is None
        assert "pipeline returned empty" in caplog.text

    def test_the_async_wrapper_runs_it_in_an_executor(self, stub_pipeline) -> None:
        up = _updater(feature_pipeline_cfg=object())
        up.pre_populate(SYM, _history(30))
        out = asyncio.run(up.force_refresh_async(SYM))
        assert len(out) == 1


# --------------------------------------------------------------------------- #
# _compute_features and helpers
# --------------------------------------------------------------------------- #
class TestComputeFeatures:
    def test_a_first_pipeline_failure_is_a_burn_in_not_a_crash(self, stub_pipeline, caplog: pytest.LogCaptureFixture) -> None:
        stub_pipeline.state["error"] = ValueError("too few bars")
        up = _updater(feature_pipeline_cfg=object())
        with caplog.at_level(logging.WARNING, logger=fu.logger.name):
            assert up._compute_features(_history(5), symbol=SYM) is None
        assert "failed on first refresh" in caplog.text

    def test_a_failure_after_a_good_refresh_halts_instead_of_serving_stale_features(
        self, stub_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        up = _updater(feature_pipeline_cfg=object())
        up._feature_cache[SYM] = _features(1)
        stub_pipeline.state["error"] = ValueError("kapot")
        with caplog.at_level(logging.CRITICAL, logger=fu.logger.name), pytest.raises(RuntimeError, match="failure after warm-up"):
            up._compute_features(_history(5), symbol=SYM)
        assert "Refusing to return stale cache" in caplog.text

    def test_unhealthy_features_are_reported(self, stub_pipeline, monkeypatch: pytest.MonkeyPatch,
                                              caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setattr("tradebot.monitoring.feature_health.check_feature_health",
                            lambda df, nan_threshold: SimpleNamespace(has_critical=True, n_issues=2, issues=["a", "b"]))
        up = _updater(feature_pipeline_cfg=object())
        with caplog.at_level(logging.WARNING, logger=fu.logger.name):
            up._compute_features(_history(5), symbol=SYM)
        assert "2 feature health issues" in caplog.text

    def test_healthy_features_and_frames_without_feature_columns_pass_silently(
        self, stub_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        up = _updater(feature_pipeline_cfg=object())
        with caplog.at_level(logging.WARNING, logger=fu.logger.name):
            assert up._compute_features(_history(5), symbol=SYM) is stub_pipeline.state["result"]
            stub_pipeline.state["result"] = _features()[["close"]]
            up._compute_features(_history(5), symbol=SYM)
        assert "health issues" not in caplog.text

    def test_the_feature_names_are_compared_with_training_once_per_symbol(
        self, stub_pipeline, caplog: pytest.LogCaptureFixture
    ) -> None:
        expected = list(_features().columns)
        up = _updater(feature_pipeline_cfg=object(), expected_feature_names=expected)
        with caplog.at_level(logging.INFO, logger=fu.logger.name):
            up._compute_features(_history(5), symbol=SYM)
        assert "parity check PASS" in caplog.text
        caplog.clear()
        drifting = _updater(feature_pipeline_cfg=object(), expected_feature_names=[*expected, "feat_gone"])
        with caplog.at_level(logging.WARNING, logger=fu.logger.name):
            drifting._compute_features(_history(5), symbol=SYM)
            drifting._compute_features(_history(5), symbol=SYM)
        assert caplog.text.count("FEATURE DRIFT") == 1 and "feat_gone" in caplog.text

    def test_an_attached_drift_monitor_sees_the_last_feature_row(self, stub_pipeline) -> None:
        seen = []
        up = _updater(feature_pipeline_cfg=object())
        up.attach_drift_monitor(SimpleNamespace(observe=lambda sym, row: seen.append((sym, len(row)))))
        up._compute_features(_history(5), symbol=SYM)
        assert seen == [(SYM, 1)]


class TestBufferHelpers:
    def test_an_empty_buffer_is_an_empty_frame(self) -> None:
        assert FeatureUpdater._buf_to_df([]).empty and _updater().get_buffer("NOPE").empty

    def test_the_buffer_becomes_a_utc_frame(self) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = FeatureUpdater._buf_to_df([(ts, {"close": 1.0}), (ts + pd.Timedelta(minutes=1), {"close": 2.0})])
        assert df["close"].tolist() == [1.0, 2.0] and str(df.index.tz) == "UTC"
