# tests/unit/test_execution_drift.py
"""Executiedrift per order. Stage D-3, en de teller van D-6.

De scherpste eigenschap die hier wordt afgedwongen is de RICHTING van de
prijsdrift. Wie het absolute verschil meet, ziet een boek dat systematisch te
duur koopt en te goedkoop verkoopt als "gemiddeld nul drift". Deze tests eisen
dat een meevaller nooit een breach is en een tegenvaller altijd, aan beide
zijden van de markt.

Ref: fase-opdracht Stage D-3 en D-6.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.monitoring.execution_drift import (
    FILL_RATIO,
    LATENCY,
    PRICE,
    ExecutionDriftMonitor,
    adverse_drift_bps,
)
from tradebot.oms.order import Fill, Order, OrderSide, OrderType
from tradebot.schemas.config import monitoring_config
from tradebot.utils.failfast import DataContractError

CFG = monitoring_config()
T0 = pd.Timestamp("2026-09-01T12:00:00Z")


def _order(side: OrderSide = OrderSide.BUY, qty: float = 1.0) -> Order:
    return Order(
        order_id="ord_1", symbol="BTCUSDT", side=side,
        order_type=OrderType.MARKET, qty_base=qty, signal_prob=0.5,
        kelly_fraction=0.0, model_version="v1", git_sha="deadbeef",
        feature_hash="", portfolio_weight=0.1, created_at=T0)


def _fill(price: float, qty: float = 1.0, seconds: float = 1.0,
          order_id: str = "ord_1") -> Fill:
    return Fill(
        order_id=order_id, fill_price=price, fill_qty=qty,
        fill_ts=T0 + pd.Timedelta(seconds=seconds),
        notional_usdt=price * qty, post_trade_cost_bps=0.0,
        exchange_order_id="x1", circuit_breaker_state="OPEN")


class TestTheDirectionOfTheDrift:
    def test_paying_more_on_a_buy_is_adverse(self) -> None:
        assert adverse_drift_bps(OrderSide.BUY, 100.0, 101.0) == pytest.approx(100.0)

    def test_paying_less_on_a_buy_is_a_windfall(self) -> None:
        assert adverse_drift_bps(OrderSide.BUY, 100.0, 99.0) == pytest.approx(-100.0)

    def test_receiving_less_on_a_sell_is_adverse(self) -> None:
        assert adverse_drift_bps(OrderSide.SELL, 100.0, 99.0) == pytest.approx(100.0)

    def test_receiving_more_on_a_sell_is_a_windfall(self) -> None:
        assert adverse_drift_bps(OrderSide.SELL, 100.0, 101.0) == pytest.approx(-100.0)

    def test_a_windfall_is_never_a_breach(self) -> None:
        """De asymmetrie die een absolute meting zou missen."""
        monitor = ExecutionDriftMonitor()
        huge = 100.0 * (1.0 - 10 * CFG.max_adverse_price_drift_bps / 10_000.0)
        drift = monitor.record(_order(OrderSide.BUY), _fill(huge),
                               expected_price=100.0)
        assert drift.adverse_bps < 0
        assert not drift.is_breach

    def test_a_zero_price_crashes_instead_of_reporting_drift(self) -> None:
        with pytest.raises(DataContractError):
            adverse_drift_bps(OrderSide.BUY, 0.0, 100.0)


class TestTheThreeGates:
    def test_an_adverse_price_beyond_the_threshold_breaches(self) -> None:
        monitor = ExecutionDriftMonitor()
        bad = 100.0 * (1.0 + 2 * CFG.max_adverse_price_drift_bps / 10_000.0)
        drift = monitor.record(_order(), _fill(bad), expected_price=100.0)
        assert PRICE in drift.breached

    def test_a_slow_fill_breaches(self) -> None:
        monitor = ExecutionDriftMonitor()
        late = CFG.max_fill_latency_seconds + 5.0
        drift = monitor.record(_order(), _fill(100.0, seconds=late),
                               expected_price=100.0)
        assert LATENCY in drift.breached

    def test_a_partial_fill_breaches(self) -> None:
        monitor = ExecutionDriftMonitor()
        drift = monitor.record(_order(qty=1.0), _fill(100.0, qty=0.5),
                               expected_price=100.0)
        assert FILL_RATIO in drift.breached
        assert drift.fill_ratio == pytest.approx(0.5)

    def test_a_clean_fill_breaches_nothing(self) -> None:
        """Negatieve controle: de poort moet ook groen kunnen zijn."""
        monitor = ExecutionDriftMonitor()
        drift = monitor.record(_order(), _fill(100.0), expected_price=100.0)
        assert not drift.is_breach
        assert not drift.breaks_the_clock

    def test_a_fill_for_another_order_crashes(self) -> None:
        """Een drift over twee verschillende orders is een verwisseling."""
        monitor = ExecutionDriftMonitor()
        with pytest.raises(DataContractError, match="hoort niet bij"):
            monitor.record(_order(), _fill(100.0, order_id="ord_999"),
                           expected_price=100.0)


class TestTheClockCounter:
    """D-6: een verklaard verschil breekt de reeks niet, een onverklaard wel."""

    def test_an_unexplained_breach_breaks_the_clock(self) -> None:
        monitor = ExecutionDriftMonitor()
        bad = 100.0 * (1.0 + 2 * CFG.max_adverse_price_drift_bps / 10_000.0)
        monitor.record(_order(), _fill(bad), expected_price=100.0)
        assert len(monitor.unexplained()) == 1

    def test_an_explained_breach_does_not(self) -> None:
        monitor = ExecutionDriftMonitor()
        bad = 100.0 * (1.0 + 2 * CFG.max_adverse_price_drift_bps / 10_000.0)
        drift = monitor.record(
            _order(), _fill(bad), expected_price=100.0,
            explanation="Bybit maintenance window 2026-09-01 12:00-12:05 UTC")
        assert drift.is_breach
        assert not drift.breaks_the_clock
        assert monitor.unexplained() == ()
        assert len(monitor.breaches()) == 1

    def test_the_summary_reports_the_worst_and_not_the_mean(self) -> None:
        """Een gemiddelde middelt de ene ernstige misfill weg."""
        monitor = ExecutionDriftMonitor()
        bad = 100.0 * (1.0 + 2 * CFG.max_adverse_price_drift_bps / 10_000.0)
        for _ in range(20):
            monitor.record(_order(), _fill(100.0), expected_price=100.0)
        monitor.record(_order(), _fill(bad), expected_price=100.0)
        summary = monitor.summary()
        assert summary["n_orders"] == 21
        assert summary["n_breaches"] == 1
        assert summary["worst_adverse_bps"] > CFG.max_adverse_price_drift_bps

    def test_an_empty_monitor_reports_nothing_rather_than_zero(self) -> None:
        summary = ExecutionDriftMonitor().summary()
        assert summary["n_orders"] == 0
        assert summary["worst_adverse_bps"] != summary["worst_adverse_bps"]  # NaN


class TestTheThresholdsComeFromConf:
    def test_the_monitor_uses_the_frozen_config(self) -> None:
        monitor = ExecutionDriftMonitor()
        assert monitor.cfg.max_adverse_price_drift_bps == (
            CFG.max_adverse_price_drift_bps)
        assert monitor.cfg.max_fill_latency_seconds == (
            CFG.max_fill_latency_seconds)
        assert monitor.cfg.min_fill_ratio == CFG.min_fill_ratio
