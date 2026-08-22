"""tests/unit/test_wave6.py — Wave 6 smoke tests.

Covers: featurestore, alpha library, orderbook, experiment tracker.
All tests use synthetic in-memory data — no I/O dependencies.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 30_000.0 * np.cumprod(1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({
        "open": close,
        "high": close * 1.002,
        "low":  close * 0.998,
        "close": close,
        "volume": rng.exponential(1_000, n),
        "taker_buy_volume":  rng.exponential(500, n),
        "taker_sell_volume": rng.exponential(500, n),
        "funding_rate": rng.normal(0.0001, 0.00005, n),
    }, index=idx)


# ============================================================================
# FeatureStore
# ============================================================================

class TestFeatureStore:

    def test_append_and_read(self, ohlcv_df: pd.DataFrame, tmp_path: Path) -> None:
        from tradebot.featurestore import FeatureStore

        store = FeatureStore(tmp_path / "fs")
        store.append(ohlcv_df[["close", "volume"]], symbol="BTCUSDT")

        start = ohlcv_df.index[0]
        end   = ohlcv_df.index[-1]
        result = store.read("BTCUSDT", start, end)

        assert not result.empty
        assert "close" in result.columns
        assert len(result) == len(ohlcv_df)

    def test_upsert_deduplication(self, ohlcv_df: pd.DataFrame, tmp_path: Path) -> None:
        from tradebot.featurestore import FeatureStore

        store = FeatureStore(tmp_path / "fs")
        store.append(ohlcv_df[["close"]], symbol="BTCUSDT")
        store.append(ohlcv_df[["close"]], symbol="BTCUSDT")  # duplicate

        result = store.read("BTCUSDT", ohlcv_df.index[0], ohlcv_df.index[-1])
        assert len(result) == len(ohlcv_df), "Upsert must deduplicate"

    def test_latest_timestamp(self, ohlcv_df: pd.DataFrame, tmp_path: Path) -> None:
        from tradebot.featurestore import FeatureStore

        store = FeatureStore(tmp_path / "fs")
        store.append(ohlcv_df[["close"]], symbol="BTCUSDT")
        latest = store.latest_timestamp("BTCUSDT")
        assert latest == ohlcv_df.index[-1]

    def test_empty_read_on_missing_symbol(self, tmp_path: Path) -> None:
        from tradebot.featurestore import FeatureStore

        store = FeatureStore(tmp_path / "fs")
        result = store.read("ETHUSDT", pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC"))
        assert result.empty


# ============================================================================
# Alpha — SignalResult + AlphaSignal protocol
# ============================================================================

class TestAlphaBase:

    def test_signal_result_frozen(self) -> None:
        from tradebot.alpha import SignalResult

        sr = SignalResult(
            symbol="BTCUSDT",
            timestamp=pd.Timestamp("2022-06-01", tz="UTC"),
            signal=0.5,
            confidence=0.75,
            horizon_bars=21,
            signal_id="BTCUSDT_test_abc12345",
        )
        assert sr.signal == 0.5
        with pytest.raises((AttributeError, TypeError)):
            sr.signal = 0.0  # type: ignore[misc]

    def test_alpha_signal_protocol(self) -> None:
        from tradebot.alpha import AlphaSignal, TSMomentum

        assert isinstance(TSMomentum("BTCUSDT"), AlphaSignal)


# ============================================================================
# Alpha — TSMomentum
# ============================================================================

class TestTSMomentum:

    def test_predict_returns_signal_result(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import TSMomentum

        sig = TSMomentum("BTCUSDT", lookback_bars=63, skip_bars=5)
        sig.fit(ohlcv_df)
        result = sig.predict(ohlcv_df)

        assert result.symbol == "BTCUSDT"
        assert -1.0 <= result.signal <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_insufficient_data_returns_neutral(self) -> None:
        from tradebot.alpha import TSMomentum

        idx = pd.date_range("2022-01-01", periods=10, freq="1h", tz="UTC")
        tiny = pd.DataFrame({"close": np.ones(10) * 30_000}, index=idx)
        sig = TSMomentum("BTCUSDT", lookback_bars=252)
        sig.fit(tiny)
        result = sig.predict(tiny)
        assert result.signal == 0.0
        assert result.confidence == 0.5


# ============================================================================
# Alpha — OUMeanReversion
# ============================================================================

class TestOUMeanReversion:

    def test_fit_and_predict(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import OUMeanReversion

        sig = OUMeanReversion("BTCUSDT", fit_window=100)
        sig.fit(ohlcv_df)
        result = sig.predict(ohlcv_df)

        assert -1.0 <= result.signal <= 1.0
        assert result.symbol == "BTCUSDT"


# ============================================================================
# Alpha — FundingCarry
# ============================================================================

class TestFundingCarry:

    def test_with_funding_col(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import FundingCarry

        sig = FundingCarry("BTCUSDT", funding_col="funding_rate")
        sig.fit(ohlcv_df)
        result = sig.predict(ohlcv_df)

        assert -1.0 <= result.signal <= 1.0

    def test_missing_col_returns_neutral(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import FundingCarry

        df = ohlcv_df.drop(columns=["funding_rate"])
        sig = FundingCarry("BTCUSDT", funding_col="funding_rate")
        sig.fit(df)
        result = sig.predict(df)
        assert result.signal == 0.0


# ============================================================================
# Alpha — OFISignal
# ============================================================================

class TestOFISignal:

    def test_predict(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import OFISignal

        sig = OFISignal("BTCUSDT", window=10)
        sig.fit(ohlcv_df)
        result = sig.predict(ohlcv_df)
        assert -1.0 <= result.signal <= 1.0


# ============================================================================
# Alpha — ICWeightedCombiner
# ============================================================================

class TestICWeightedCombiner:

    def test_fit_predict(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import ICWeightedCombiner, TSMomentum, OUMeanReversion

        signals_df = pd.DataFrame({
            "ts_mom": np.random.default_rng(0).normal(0, 0.3, len(ohlcv_df)),
            "ou_rev": np.random.default_rng(1).normal(0, 0.3, len(ohlcv_df)),
        }, index=ohlcv_df.index)

        fwd_rets = ohlcv_df["close"].pct_change().shift(-1).fillna(0.0)

        combiner = ICWeightedCombiner(["ts_mom", "ou_rev"], lookback=100)
        combiner.fit(signals_df, fwd_rets)
        combined = combiner.predict(signals_df)

        assert len(combined) == len(ohlcv_df)
        assert combined.between(-1.0, 1.0).all()

    def test_ic_table(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import ICWeightedCombiner

        signals_df = pd.DataFrame({
            "s1": np.zeros(len(ohlcv_df)),
        }, index=ohlcv_df.index)
        fwd = pd.Series(np.zeros(len(ohlcv_df)), index=ohlcv_df.index)
        c = ICWeightedCombiner(["s1"])
        c.fit(signals_df, fwd)
        tbl = c.ic_table()
        assert "ic_raw" in tbl.columns
        assert "weight" in tbl.columns


# ============================================================================
# Alpha — MacroRegimeOverlay
# ============================================================================

class TestMacroRegimeOverlay:

    def test_regime_state(self, ohlcv_df: pd.DataFrame) -> None:
        from tradebot.alpha import MacroRegimeOverlay
        from tradebot.alpha.macro_regime import RegimeLabel

        overlay = MacroRegimeOverlay("BTCUSDT")
        state = overlay.regime_state(ohlcv_df)
        assert state.label in list(RegimeLabel)
        assert 0.0 <= state.multiplier <= 1.0


# ============================================================================
# OrderBook
# ============================================================================

class TestOrderBook:

    def _make_snapshot(self) -> "OrderBookSnapshot":  # type: ignore[name-defined]
        from tradebot.data.orderbook import OrderBookSnapshot

        bids = np.array([[30_000.0, 1.5], [29_990.0, 2.0], [29_980.0, 3.0]])
        asks = np.array([[30_010.0, 1.0], [30_020.0, 2.5], [30_030.0, 1.8]])
        return OrderBookSnapshot(
            timestamp=pd.Timestamp("2022-01-01 12:00:00", tz="UTC"),
            bids=bids,
            asks=asks,
        )

    def test_best_bid_ask(self) -> None:
        snap = self._make_snapshot()
        assert snap.best_bid() == 30_000.0
        assert snap.best_ask() == 30_010.0

    def test_spread(self) -> None:
        snap = self._make_snapshot()
        assert snap.spread() == pytest.approx(10.0)

    def test_book_imbalance_range(self) -> None:
        from tradebot.data.orderbook import compute_book_imbalance

        snap = self._make_snapshot()
        obi = compute_book_imbalance(snap, depth_levels=3)
        assert -1.0 <= obi <= 1.0

    def test_rolling_book_features(self) -> None:
        from tradebot.data.orderbook import OrderBookSnapshot, rolling_book_features

        snapshots = []
        for i in range(30):
            bids = np.array([[30_000.0 - i * 0.1, 1.0]])
            asks = np.array([[30_010.0 + i * 0.1, 1.0]])
            snapshots.append(OrderBookSnapshot(
                timestamp=pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(hours=i),
                bids=bids,
                asks=asks,
            ))

        df = rolling_book_features(snapshots, window=5)
        assert "book_imbalance" in df.columns
        assert len(df) == 30


# ============================================================================
# ExperimentTracker
# ============================================================================

class TestExperimentTracker:

    def test_log_and_load(self, tmp_path: Path) -> None:
        from tradebot.registry.experiment import ExperimentTracker

        tracker = ExperimentTracker("test_exp", tmp_path, use_mlflow=False)
        run_id = tracker.start_run(0, {"lr": 0.01, "depth": 6})
        tracker.log_metrics({"sharpe": 1.23, "logloss": 0.45})
        tracker.end_run("FINISHED")

        df = tracker.load_runs()
        assert len(df) == 1
        assert df.iloc[0]["experiment_name"] == "test_exp"

    def test_best_run(self, tmp_path: Path) -> None:
        from tradebot.registry.experiment import ExperimentTracker

        tracker = ExperimentTracker("test_exp", tmp_path, use_mlflow=False)
        for i, sharpe in enumerate([0.5, 1.8, 1.2]):
            tracker.start_run(i, {"trial": i})
            tracker.log_metrics({"sharpe": sharpe})
            tracker.end_run("FINISHED")

        best = tracker.best_run("sharpe")
        assert best is not None
        assert best.metrics["sharpe"] == pytest.approx(1.8)
