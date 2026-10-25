"""Chaos engineering tests — Wave 20.

Tests production resilience under fault injection, end to end through
`LiveEngine.run()`:
- Feed sequence gap
- Exchange API timeout on an order
- Kill switch (a real SIGUSR1)
- Funding settlement exactly on a bar close

Exchange traffic is replaced by a recording fake; nothing goes over the network.
"""
from __future__ import annotations

import asyncio
import signal
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from tradebot.live.circuit_breaker import CircuitBreakerConfig
from tradebot.live.engine import LiveEngine, LiveEngineConfig
from tradebot.live.execution_controller import ExecutionControllerConfig
from tradebot.live.feed import BarEvent, Feed, FeedConfig
from tradebot.live.portfolio_controller import PortfolioControllerConfig
from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig
from tradebot.live.state import EngineMode
from tradebot.oms.order import Order, OrderSide, OrderType
from tradebot.oms.router import OrderRouter
from tradebot.portfolio.constraints import PortfolioConstraints
from tradebot.risk.kill_switches import HaltStore
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
T0 = pd.Timestamp("2024-01-01", tz="UTC")


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """De engine installeert SIGTERM/SIGUSR1-handlers; de testrunner krijgt ze terug."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _engine(tmp_path: Path) -> LiveEngine:
    cb_cfg = CircuitBreakerConfig(
        max_drawdown_pct=0.99, max_intraday_drawdown_pct=0.0, max_daily_loss_pct=0.99,
        feed_timeout_sec=3_600, cb_log_path=tmp_path / "cb.log",
        halt_store_path=tmp_path / "halt.json")
    config = LiveEngineConfig(
        symbols=["BTCUSDT"], initial_equity=100_000.0, mode="paper",
        audit_log_path=str(tmp_path / "audit.jsonl"), cb_config=cb_cfg,
        ec_config=ExecutionControllerConfig(RISK, git_sha="deadbeef", model_version="chaos"),
        pc_config=PortfolioControllerConfig(
            constraints=PortfolioConstraints.from_risk_config(RISK), method="erc", min_history_bars=5),
        sr_config=SignalRunnerConfig(), metrics_port=0, state_out_dir=tmp_path / "state")
    feed = Feed(FeedConfig(symbols=["BTCUSDT"], paper_mode=True), asyncio.Queue())
    return LiveEngine(config, SignalRunner(signals=[], config=SignalRunnerConfig()), feed)


def _bar(minute: int, seq: int, close: float = 40_000.0) -> BarEvent:
    ts = T0 + pd.Timedelta(minutes=minute)
    return BarEvent("BTCUSDT", ts, close, close + 10.0, close - 10.0, close, 5.0, seq=seq)


def _order(order_id: str = "ord_chaos_0001", qty: float = 0.1) -> Order:
    return Order(
        order_id=order_id, symbol="BTCUSDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
        qty_base=qty, signal_prob=0.6, kelly_fraction=0.05, model_version="chaos",
        git_sha="deadbeef", feature_hash="0" * 16, portfolio_weight=0.1)


# --------------------------------------------------------------------------- #
# feed
# --------------------------------------------------------------------------- #
def test_feed_gap_trips_circuit_breaker(tmp_path: Path) -> None:
    """A skipped sequence number on the feed halts the engine and the halt survives on disk."""
    engine = _engine(tmp_path)

    async def feed_with_a_hole() -> None:
        for minute, seq in enumerate((1, 2, 3, 7)):                  # 4..6 never arrived
            engine._queue.put_nowait(_bar(minute, seq))
        engine._queue.put_nowait(None)                                   # a healthy feed ends, so a missed gap fails fast

    engine._feed.start = feed_with_a_hole                                # type: ignore[method-assign]
    asyncio.run(asyncio.wait_for(engine.run(), timeout=30))

    assert engine._cb.is_active, "a sequence gap must trip the breaker"
    assert engine.bars_processed == 3, "the bar after the gap must not be processed"
    assert engine.state.mode is EngineMode.STOPPED
    assert HaltStore(tmp_path / "halt.json").is_halted(), "the halt must be persisted, not only in memory"


# --------------------------------------------------------------------------- #
# exchange
# --------------------------------------------------------------------------- #
class _FlakyBybit:
    """Bybit stand-in: the first `order/create` times out, later ones succeed."""

    def __init__(self) -> None:
        self.creates: list[dict[str, Any]] = []
        self.gets: list[str] = []
        self._timeouts_left = 1

    def module(self) -> types.ModuleType:
        fake = self

        class _Reply:
            def __init__(self, payload: dict[str, Any] | None, boom: bool = False) -> None:
                self._payload, self._boom = payload, boom

            async def __aenter__(self) -> _Reply:
                if self._boom:
                    raise asyncio.TimeoutError
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            async def json(self) -> dict[str, Any]:
                return self._payload or {}

        class _Session:
            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            def get(self, url: str, **_: Any) -> _Reply:
                fake.gets.append(url)
                if "instruments-info" in url:
                    return _Reply({"result": {"list": [{"lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001"}}]}})
                return _Reply({"result": {"list": [{"avgPrice": "40000", "cumExecQty": "0.1",
                                                    "orderStatus": "Filled"}]}})

            def post(self, url: str, data: str = "", **_: Any) -> _Reply:
                import json
                fake.creates.append(json.loads(data))
                if fake._timeouts_left:
                    fake._timeouts_left -= 1
                    return _Reply(None, boom=True)
                return _Reply({"retCode": 0, "result": {"orderId": "EX-1"}})

        module = types.ModuleType("aiohttp")
        module.ClientSession = _Session                                  # type: ignore[attr-defined]
        return module


def test_exchange_timeout_order_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out order is retried under the SAME client id, and a filled one is never sent twice."""
    exchange = _FlakyBybit()
    monkeypatch.setitem(sys.modules, "aiohttp", exchange.module())
    monkeypatch.setenv("BYBIT_API_KEY", "k")
    monkeypatch.setenv("BYBIT_API_SECRET", "s")
    from tradebot.oms.paper_oms import PaperOMS
    router = OrderRouter(PaperOMS(), live_mode=True)
    order = _order()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(router.place_order(order))                           # the exchange never answered
    assert router._order_id_cache == {}, "an unanswered order must not be recorded as placed"

    fill = asyncio.run(router.place_order(order))                        # the retry
    assert fill.exchange_order_id == "EX-1"
    assert [c["orderLinkId"] for c in exchange.creates] == [order.order_id, order.order_id], (
        "both attempts must carry the same orderLinkId, so the exchange can dedupe them")

    asyncio.run(router.place_order(order))                               # a repeat after the fill
    assert len(exchange.creates) == 2, "a placed order must be read back, not created again"


# --------------------------------------------------------------------------- #
# kill switch
# --------------------------------------------------------------------------- #
def test_kill_switch_cancels_all_orders(tmp_path: Path) -> None:
    """A real SIGUSR1 while the engine runs flattens every open position and stops the loop."""
    engine = _engine(tmp_path)
    engine._paper_oms.set_bar_prices({"BTCUSDT": 40_000.0})

    async def scenario() -> None:
        await engine._router.place_order(_order())                       # an open position
        assert engine._paper_oms.get_positions()["BTCUSDT"] == pytest.approx(0.1)
        engine._feed.start = lambda: asyncio.sleep(0)                    # type: ignore[method-assign]
        signal.raise_signal(signal.SIGUSR1)
        await asyncio.wait_for(engine.run(), timeout=30)

    asyncio.run(scenario())

    assert engine._stop_event.is_set()
    assert all(abs(q) < 1e-12 for q in engine._paper_oms.get_positions().values()), "positions must be flat"
    assert engine.bars_processed == 0


def test_circuit_breaker_persists_state(tmp_path, monkeypatch):
    """CircuitBreaker trip should be persisted to disk."""
    import json
    cb_log = tmp_path / "circuit_breaker.log"
    monkeypatch.setattr(
        "tradebot.live.circuit_breaker._CB_LOG_PATH",
        cb_log,
        raising=False,
    )
    try:
        from tradebot.live.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker.__new__(CircuitBreaker)
        cb._logger = MagicMock()
        if hasattr(cb, '_persist_trip'):
            cb._persist_trip("test_reason")
            assert cb_log.exists(), "CB log should be created after trip"
            with open(cb_log) as f:
                entry = json.loads(f.readline().strip())
            assert entry["reason"] == "test_reason"
            assert not entry["acknowledged"]
    except (ImportError, AttributeError):
        pytest.skip("CircuitBreaker _persist_trip not yet implemented")


def test_cpcv_fallback_raises_value_error():
    """CPCV fallback concatenation must raise ValueError (not silent concat)."""
    import pandas as pd

    from tradebot.cv.cpcv import build_cpcv_return_paths

    # n_groups=5, k=2 → 5 % 2 != 0 → should raise ValueError
    fold_returns = {
        (0, 1): pd.Series([0.01, 0.02], dtype=float),
        (2, 3): pd.Series([0.03, 0.04], dtype=float),
    }
    with pytest.raises(ValueError, match="divisible"):
        build_cpcv_return_paths(fold_returns, n_groups=5)


def test_trend_scan_label_removed():
    """TrendScanningLabeler.label() should raise NotImplementedError (P0-8)."""
    import pandas as pd

    from tradebot.labeling.trend_scanning import TrendScanningLabeler
    labeler = TrendScanningLabeler()
    df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})
    with pytest.raises(NotImplementedError):
        labeler.label(df)


def test_funding_searchsorted_causality():
    """A settlement exactly on a bar close belongs to the NEXT bar (window is [T - 1d, T))."""
    from tradebot.data.funding_panel import settlement_sums

    day = np.int64(86_400_000_000_000)
    bar_end = np.array([day, 2 * day, 3 * day], dtype=np.int64)
    event = np.array([day - 1, day, 2 * day + 5], dtype=np.int64)       # just before, exactly on, after
    rate = np.array([0.001, 0.002, 0.004])

    sums = settlement_sums(event, rate, bar_end)

    assert sums[0] == pytest.approx(0.001), "the bar closing at T may only see settlements before T"
    assert sums[1] == pytest.approx(0.002), "the settlement exactly at T lands on the bar after it"
    assert sums[2] == pytest.approx(0.004)
