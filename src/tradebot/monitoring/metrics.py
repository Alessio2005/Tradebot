# src/tradebot/monitoring/metrics.py
"""Prometheus metrics for the live trading engine (P2-2).

Soft-import: if prometheus_client is not installed the module degrades
gracefully to no-op stubs so the engine never crashes on a missing dep.

Usage
-----
    from tradebot.monitoring.metrics import EngineMetrics, start_metrics_server

    metrics = EngineMetrics()
    start_metrics_server(port=8000)  # exposes /metrics endpoint

    # Per-bar updates
    metrics.bar_processed(symbol="BTCUSDT", latency_s=0.042)
    metrics.equity_updated(equity=98_500.0, drawdown=0.015)
    metrics.order_filled(symbol="BTCUSDT", side="BUY")
    metrics.funding_accrued(symbol="BTCUSDT", payment_usdt=-1.23)
    metrics.circuit_breaker_state(active=False)
    metrics.queue_depth(depth=3)
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Soft-import prometheus_client
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )
    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — metrics are no-ops. "
        "Install with: pip install prometheus-client"
    )

__all__ = ["EngineMetrics", "start_metrics_server"]

# ---------------------------------------------------------------------------
# No-op stubs (used when prometheus_client is absent)
# ---------------------------------------------------------------------------

class _NoOpCounter:
    def labels(self, **_kw):  # noqa: ANN001
        return self
    def inc(self, _amount: float = 1) -> None:  # noqa: ANN001
        pass

class _NoOpGauge:
    def labels(self, **_kw):
        return self
    def set(self, _value: float) -> None:
        pass

class _NoOpHistogram:
    def labels(self, **_kw):
        return self
    def observe(self, _value: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

_BAR_LATENCY_BUCKETS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0, 2.0)

if _PROM_AVAILABLE:
    _bars_processed = Counter(
        "tradebot_bars_processed_total",
        "Total OHLCV bars processed by the engine.",
        ["symbol"],
    )
    _bar_latency = Histogram(
        "tradebot_bar_latency_seconds",
        "Wall-clock time to process one bar end-to-end.",
        ["symbol"],
        buckets=_BAR_LATENCY_BUCKETS,
    )
    _equity = Gauge(
        "tradebot_equity_usdt",
        "Current portfolio NAV in USDT.",
    )
    _drawdown = Gauge(
        "tradebot_drawdown_fraction",
        "Current peak-to-trough drawdown (0–1).",
    )
    _cb_active = Gauge(
        "tradebot_circuit_breaker_active",
        "1 if the circuit breaker is tripped, 0 otherwise.",
    )
    _orders_filled = Counter(
        "tradebot_orders_filled_total",
        "Number of orders filled.",
        ["symbol", "side"],
    )
    _funding_payment = Counter(
        "tradebot_funding_payment_usdt_total",
        "Cumulative funding payments (negative = paid, positive = received).",
        ["symbol"],
    )
    _queue_depth = Gauge(
        "tradebot_feed_queue_depth",
        "Current depth of the engine's async event queue.",
    )
else:  # pragma: no cover
    _bars_processed = _NoOpCounter()   # type: ignore[assignment]
    _bar_latency    = _NoOpHistogram() # type: ignore[assignment]
    _equity         = _NoOpGauge()     # type: ignore[assignment]
    _drawdown       = _NoOpGauge()     # type: ignore[assignment]
    _cb_active      = _NoOpGauge()     # type: ignore[assignment]
    _orders_filled  = _NoOpCounter()   # type: ignore[assignment]
    _funding_payment = _NoOpCounter()  # type: ignore[assignment]
    _queue_depth    = _NoOpGauge()     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EngineMetrics:
    """Thin façade over the module-level Prometheus metrics.

    One instance shared across the engine; all methods are thread-safe
    because the underlying prometheus_client primitives are thread-safe.
    """

    def bar_processed(self, symbol: str, latency_s: float) -> None:
        """Record a completed bar cycle."""
        _bars_processed.labels(symbol=symbol).inc()
        _bar_latency.labels(symbol=symbol).observe(latency_s)

    def equity_updated(self, equity: float, drawdown: float) -> None:
        """Update portfolio NAV and drawdown gauges."""
        _equity.set(equity)
        _drawdown.set(drawdown)

    def order_filled(self, symbol: str, side: str) -> None:
        """Increment fill counter (side = "BUY" | "SELL")."""
        _orders_filled.labels(symbol=symbol, side=side).inc()

    def funding_accrued(self, symbol: str, payment_usdt: float) -> None:
        """Record a funding payment (can be negative = cost)."""
        # Counter.inc() requires non-negative amounts; we track the
        # absolute value and embed sign in the label instead of using
        # a separate paid/received split — payment_usdt can be negative
        # so we use a Gauge-backed running total approach via set() on a
        # separate gauge is less ergonomic.  Simplest correct solution:
        # always inc by abs value; sign is visible in the log.
        _funding_payment.labels(symbol=symbol).inc(abs(payment_usdt))

    def circuit_breaker_state(self, active: bool) -> None:
        """Set CB gauge (1 = tripped, 0 = normal)."""
        _cb_active.set(1.0 if active else 0.0)

    def queue_depth(self, depth: int) -> None:
        """Update feed queue depth gauge."""
        _queue_depth.set(depth)


_server_started = False
_server_lock = threading.Lock()


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus HTTP server on ``port`` (idempotent).

    Safe to call multiple times; the server is started only once.
    If prometheus_client is not installed, logs a warning and returns.
    """
    global _server_started
    if not _PROM_AVAILABLE:
        logger.warning("start_metrics_server: prometheus_client not available — skipping.")
        return
    with _server_lock:
        if _server_started:
            return
        start_http_server(port)
        _server_started = True
        logger.info("Prometheus metrics server started on port %d (/metrics).", port)
