"""tests/unit/test_wave8.py — Wave 8 unit tests.

Covers:
  - utils/hashing.py
  - utils/time.py
  - schemas/orders.py (OrderAuditSchema)
  - oms/order.py (Order, Fill, OrderState)
  - oms/audit_log.py (AuditLog)
  - oms/position_tracker.py (PositionTracker)
  - oms/reconciler.py (Reconciler)
  - oms/paper_oms.py (PaperOMS)
  - live/state.py (SystemState)
  - live/circuit_breaker.py (CircuitBreaker — trips at DD > threshold)
  - live/feature_updater.py (FeatureUpdater)
  - live/signal_runner.py (SignalRunner — no signals → None)
"""
from __future__ import annotations

import pandas as pd
import pytest

# ─── utils ────────────────────────────────────────────────────────────────────
from tradebot.utils.hashing import (
    hash_config,
    hash_content,
    hash_feature_names,
    hash_file,
)
from tradebot.utils.time import (
    align_to_bar,
    bar_cadence_seconds,
    now_utc,
    seconds_since,
    to_utc,
)


class TestHashConfig:
    def test_deterministic(self):
        cfg = {"a": 1, "b": [1, 2, 3]}
        assert hash_config(cfg) == hash_config(cfg)

    def test_key_order_invariant(self):
        assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})

    def test_different_configs_differ(self):
        assert hash_config({"a": 1}) != hash_config({"a": 2})

    def test_length(self):
        assert len(hash_config({})) == 16
        assert len(hash_config({}, length=8)) == 8

    def test_feature_names_hash_deterministic(self):
        names = ["close", "volume", "rsi_14"]
        h1 = hash_feature_names(names)
        h2 = hash_feature_names(names)
        assert h1 == h2

    def test_feature_names_order_sensitive(self):
        assert hash_feature_names(["a", "b"]) != hash_feature_names(["b", "a"])

    def test_hash_file(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"hello world")
        h = hash_file(p)
        assert len(h) == 16
        assert hash_file(p) == h

    def test_hash_content(self):
        h = hash_content(b"test data")
        assert len(h) == 16
        assert hash_content(b"test data") == h


class TestTimeUtils:
    def test_now_utc_tz_aware(self):
        ts = now_utc()
        assert ts.tzinfo is not None

    def test_to_utc_naive_string(self):
        ts = to_utc("2024-01-01 12:00:00")
        assert ts.tzinfo is not None
        assert ts.hour == 12

    def test_to_utc_aware_string(self):
        ts = to_utc("2024-01-01T12:00:00+00:00")
        assert ts.tz is not None

    def test_to_utc_datetime(self):
        from datetime import datetime
        dt = datetime(2024, 6, 1, 12, 0, 0)
        ts = to_utc(dt)
        assert ts.tzinfo is not None

    def test_bar_cadence_seconds(self):
        assert bar_cadence_seconds("1h") == 3600
        assert bar_cadence_seconds("4h") == 14400
        assert bar_cadence_seconds("1d") == 86400

    def test_bar_cadence_unknown(self):
        with pytest.raises(KeyError):
            bar_cadence_seconds("99m")

    def test_align_to_bar(self):
        ts = pd.Timestamp("2024-01-01 12:37:00", tz="UTC")
        aligned = align_to_bar(ts, "1h")
        assert aligned == pd.Timestamp("2024-01-01 12:00:00", tz="UTC")

    def test_seconds_since(self):
        past = now_utc() - pd.Timedelta(seconds=5)
        elapsed = seconds_since(past)
        assert 4.0 < elapsed < 10.0


# ─── OMS core ─────────────────────────────────────────────────────────────────

from tradebot.oms.order import Fill, Order, OrderSide, OrderStatus, OrderType


def _make_order(symbol="BTCUSDT", side=OrderSide.BUY, qty=0.1) -> Order:
    return Order(
        order_id=f"ord_20240101_{symbol}_{side.value}_0001",
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        qty_base=qty,
        signal_prob=0.65,
        kelly_fraction=0.05,
        model_version="v1.0.0",
        git_sha="abc1234",
        feature_hash="deadbeef12345678",
        portfolio_weight=0.20,
    )


def _make_fill(order: Order, price=40000.0) -> Fill:
    return Fill(
        order_id=order.order_id,
        fill_price=price,
        fill_qty=order.qty_base,
        fill_ts=pd.Timestamp.now(tz="UTC"),
        notional_usdt=price * order.qty_base,
    )


class TestOrderDataclasses:
    def test_order_defaults(self):
        o = _make_order()
        assert o.status == OrderStatus.PENDING
        assert o.limit_price is None

    def test_fill_notional(self):
        o = _make_order(qty=0.1)
        f = _make_fill(o, price=50_000.0)
        assert abs(f.notional_usdt - 5_000.0) < 0.01


# ─── AuditLog ─────────────────────────────────────────────────────────────────

from tradebot.oms.audit_log import _REQUIRED_FIELDS, AuditLog


class TestAuditLog:
    def test_record_and_read(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        order = _make_order()
        fill = _make_fill(order)
        log.record(order, fill)
        df = log.read_as_dataframe()
        assert len(df) == 1
        assert log.records_written == 1

    def test_all_14_fields_present(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        order = _make_order()
        fill = _make_fill(order)
        log.record(order, fill)
        df = log.read_as_dataframe()
        for field in _REQUIRED_FIELDS:
            assert field in df.columns, f"Missing required field: {field}"

    def test_integrity_hash(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        order = _make_order()
        fill = _make_fill(order)
        log.record(order, fill)
        digest = log.write_integrity_hash()
        assert len(digest) == 64  # full SHA-256

    def test_append_only(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        for i in range(3):
            log = AuditLog(log_path)
            o = _make_order(qty=float(i + 1) * 0.01)
            f = _make_fill(o, price=40_000.0 + i * 1000)
            log.record(o, f)
        df = AuditLog(log_path).read_as_dataframe()
        assert len(df) == 3


# ─── PositionTracker ──────────────────────────────────────────────────────────

from tradebot.oms.position_tracker import PositionTracker


class TestPositionTracker:
    def test_open_long(self):
        tracker = PositionTracker(100_000.0)
        o = _make_order(qty=1.0)
        f = _make_fill(o, price=40_000.0)
        tracker.apply_fill("BTCUSDT", f, OrderSide.BUY)
        pos = tracker.get_position("BTCUSDT")
        assert pos is not None
        assert abs(pos.qty - 1.0) < 1e-9
        assert abs(pos.avg_entry_price - 40_000.0) < 1e-3

    def test_close_long_realises_pnl(self):
        tracker = PositionTracker(100_000.0)
        o_buy = _make_order(qty=1.0)
        f_buy = _make_fill(o_buy, price=40_000.0)
        tracker.apply_fill("BTCUSDT", f_buy, OrderSide.BUY)

        o_sell = _make_order(side=OrderSide.SELL, qty=1.0)
        f_sell = _make_fill(o_sell, price=41_000.0)
        tracker.apply_fill("BTCUSDT", f_sell, OrderSide.SELL)

        pos = tracker.get_position("BTCUSDT")
        assert abs(pos.qty) < 1e-9
        assert abs(pos.realised_pnl - 1_000.0) < 0.01

    def test_mark_updates_unrealised(self):
        tracker = PositionTracker(100_000.0)
        o = _make_order(qty=1.0)
        f = _make_fill(o, price=40_000.0)
        tracker.apply_fill("BTCUSDT", f, OrderSide.BUY)
        tracker.mark("BTCUSDT", 42_000.0)
        pos = tracker.get_position("BTCUSDT")
        assert abs(pos.unrealised_pnl - 2_000.0) < 0.01

    def test_equity_includes_unrealised(self):
        tracker = PositionTracker(100_000.0)
        o = _make_order(qty=1.0)
        f = _make_fill(o, price=40_000.0)
        tracker.apply_fill("BTCUSDT", f, OrderSide.BUY)
        tracker.mark("BTCUSDT", 41_000.0)
        assert abs(tracker.equity - 101_000.0) < 0.01


# ─── Reconciler ───────────────────────────────────────────────────────────────

from tradebot.oms.reconciler import Reconciler, ReconcilerConfig


class TestReconciler:
    def test_paper_mode_no_discrepancies(self):
        tracker = PositionTracker(100_000.0)
        rec = Reconciler(tracker, ReconcilerConfig(paper_mode=True))
        results = rec.reconcile({"BTCUSDT": 1.0})
        assert results == []

    def test_live_mode_detects_discrepancy(self):
        tracker = PositionTracker(100_000.0)
        o = _make_order(qty=1.0)
        f = _make_fill(o, price=40_000.0)
        tracker.apply_fill("BTCUSDT", f, OrderSide.BUY)

        rec = Reconciler(tracker, ReconcilerConfig(paper_mode=False, qty_tolerance=0.001))
        results = rec.reconcile({"BTCUSDT": 0.5})  # exchange shows 0.5, we have 1.0
        assert len(results) == 1
        assert results[0].is_discrepant

    def test_live_mode_ok_when_matching(self):
        tracker = PositionTracker(100_000.0)
        o = _make_order(qty=1.0)
        f = _make_fill(o, price=40_000.0)
        tracker.apply_fill("BTCUSDT", f, OrderSide.BUY)

        rec = Reconciler(tracker, ReconcilerConfig(paper_mode=False))
        results = rec.reconcile({"BTCUSDT": 1.0})
        assert not results[0].is_discrepant


# ─── PaperOMS ─────────────────────────────────────────────────────────────────

from tradebot.oms.paper_oms import PaperOMS


class TestPaperOMS:
    def test_fill_price_within_05pct(self):
        oms = PaperOMS(initial_equity=100_000.0)
        oms.set_bar_prices({"BTCUSDT": 40_000.0})
        o = _make_order()
        fill = oms.place_order(o)
        pct_diff = abs(fill.fill_price - 40_000.0) / 40_000.0
        assert pct_diff <= 0.005, f"Fill slippage too large: {pct_diff:.4%}"

    def test_equity_decreases_on_buy(self):
        oms = PaperOMS(initial_equity=100_000.0)
        oms.set_bar_prices({"BTCUSDT": 40_000.0})
        o = _make_order(qty=1.0)
        oms.place_order(o)
        oms.tracker.mark("BTCUSDT", 38_000.0)
        assert oms.tracker.equity < 100_000.0

    def test_equity_curve_grows(self):
        oms = PaperOMS(initial_equity=100_000.0)
        for i in range(5):
            oms.set_bar_prices({"BTCUSDT": 40_000.0 + i * 100})
        curve = oms.equity_curve()
        assert len(curve) >= 1

    def test_get_positions(self):
        oms = PaperOMS(initial_equity=100_000.0)
        oms.set_bar_prices({"BTCUSDT": 40_000.0})
        o = _make_order(qty=0.5)
        oms.place_order(o)
        positions = oms.get_positions()
        assert "BTCUSDT" in positions
        assert abs(positions["BTCUSDT"] - 0.5) < 1e-6


# ─── SystemState ──────────────────────────────────────────────────────────────

from tradebot.live.state import SystemState


class TestSystemState:
    def test_drawdown_zero_at_start(self):
        s = SystemState(equity=100_000.0, equity_peak=100_000.0)
        assert s.current_drawdown == 0.0

    def test_drawdown_after_loss(self):
        s = SystemState(equity=90_000.0, equity_peak=100_000.0)
        assert abs(s.current_drawdown - 0.10) < 1e-9

    def test_update_equity_updates_peak(self):
        s = SystemState(equity=100_000.0, equity_peak=100_000.0)
        s.update_equity(110_000.0)
        assert s.equity_peak == 110_000.0

    def test_daily_pnl_pct(self):
        s = SystemState(equity=97_000.0, equity_peak=100_000.0, daily_pnl_open=100_000.0)
        s.daily_pnl = -3_000.0
        assert abs(s.daily_pnl_pct - (-0.03)) < 1e-9


# ─── CircuitBreaker ───────────────────────────────────────────────────────────

from tradebot.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, HaltReason


class TestCircuitBreaker:
    def test_no_trip_normal_conditions(self):
        state = SystemState(equity=100_000.0, equity_peak=100_000.0)
        cb = CircuitBreaker(CircuitBreakerConfig(), state)
        now = pd.Timestamp.now(tz="UTC")
        state.last_feed_ts = now
        reason = cb.check(now)
        assert reason is None
        assert not cb.is_active

    def test_trips_on_max_drawdown(self):
        state = SystemState(equity=88_000.0, equity_peak=100_000.0)
        state.daily_pnl_open = 100_000.0
        cfg = CircuitBreakerConfig(max_drawdown_pct=0.08)
        cb = CircuitBreaker(cfg, state)
        now = pd.Timestamp.now(tz="UTC")
        state.last_feed_ts = now
        reason = cb.check(now)
        assert reason == HaltReason.MAX_DRAWDOWN
        assert cb.is_active

    def test_trips_on_daily_loss(self):
        state = SystemState(equity=95_000.0, equity_peak=100_000.0)
        state.daily_pnl_open = 100_000.0
        state.daily_pnl = -4_000.0  # 4% loss > 3% threshold
        cfg = CircuitBreakerConfig(max_daily_loss_pct=0.03)
        cb = CircuitBreaker(cfg, state)
        now = pd.Timestamp.now(tz="UTC")
        state.last_feed_ts = now
        reason = cb.check(now)
        assert reason == HaltReason.DAILY_LOSS
        assert cb.is_active

    def test_trips_on_feed_timeout(self):
        state = SystemState(equity=100_000.0, equity_peak=100_000.0)
        state.daily_pnl_open = 100_000.0
        old_ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=60)
        state.last_feed_ts = old_ts
        cfg = CircuitBreakerConfig(feed_timeout_sec=30)
        cb = CircuitBreaker(cfg, state)
        now = pd.Timestamp.now(tz="UTC")
        reason = cb.check(now)
        assert reason == HaltReason.FEED_TIMEOUT

    def test_no_auto_resume_after_trip(self):
        state = SystemState(equity=88_000.0, equity_peak=100_000.0)
        state.daily_pnl_open = 100_000.0
        cb = CircuitBreaker(CircuitBreakerConfig(max_drawdown_pct=0.08), state)
        now = pd.Timestamp.now(tz="UTC")
        state.last_feed_ts = now
        cb.check(now)
        assert cb.is_active
        # Even after equity recovery, still tripped
        state.equity = 100_000.0
        reason2 = cb.check(now)
        assert reason2 == HaltReason.MAX_DRAWDOWN  # original reason returned

    def test_manual_reset(self):
        state = SystemState(equity=88_000.0, equity_peak=100_000.0)
        state.daily_pnl_open = 100_000.0
        cb = CircuitBreaker(CircuitBreakerConfig(max_drawdown_pct=0.08), state)
        now = pd.Timestamp.now(tz="UTC")
        state.last_feed_ts = now
        cb.check(now)
        assert cb.is_active
        # STAGE D, C2: een reset vereist een operator en een motivering; zij
        # gaat door `HaltStore.release` en wordt gejournaliseerd.
        cb.reset(operator="test-operator", justification="unit test")
        assert not cb.is_active

    def test_trips_on_hash_mismatch(self):
        state = SystemState(equity=100_000.0, equity_peak=100_000.0)
        state.daily_pnl_open = 100_000.0
        cfg = CircuitBreakerConfig(model_hash_mismatch=True)
        cb = CircuitBreaker(cfg, state)
        now = pd.Timestamp.now(tz="UTC")
        state.last_feed_ts = now
        reason = cb.check(now, feature_hash="aabbcc", expected_hash="112233")
        assert reason == HaltReason.HASH_MISMATCH


# ─── FeatureUpdater ───────────────────────────────────────────────────────────

from tradebot.live.feature_updater import FeatureUpdater, FeatureUpdaterConfig


class TestFeatureUpdater:
    def _make_bar(self, close: float, ts: pd.Timestamp) -> pd.Series:
        return pd.Series(
            {"open": close * 0.99, "high": close * 1.01,
             "low": close * 0.98, "close": close, "volume": 1000.0},
            name=ts,
        )

    def test_returns_none_on_single_bar(self):
        fu = FeatureUpdater(FeatureUpdaterConfig())
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        bar = self._make_bar(40000.0, ts)
        result = fu.update("BTCUSDT", bar)
        assert result is None

    def test_returns_dataframe_after_two_bars(self):
        fu = FeatureUpdater(FeatureUpdaterConfig())
        for i in range(2):
            ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)
            bar = self._make_bar(40000.0 + i * 100, ts)
            result = fu.update("BTCUSDT", bar)
        assert result is not None
        assert len(result) == 1

    def test_window_capped(self):
        fu = FeatureUpdater(FeatureUpdaterConfig(window_bars=5))
        for i in range(10):
            ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)
            bar = self._make_bar(40000.0 + i, ts)
            fu.update("BTCUSDT", bar)
        buf = fu.get_buffer("BTCUSDT")
        assert len(buf) <= 5


# ─── SignalRunner ─────────────────────────────────────────────────────────────

from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig


class TestSignalRunner:
    def test_no_signals_returns_none(self):
        sr = SignalRunner(signals=[], config=SignalRunnerConfig())
        features = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
        )
        result = sr.predict("BTCUSDT", features)
        assert result is None

    def test_schema_orders_import(self):
        from tradebot.schemas.orders import OrderAuditSchema
        assert OrderAuditSchema is not None
