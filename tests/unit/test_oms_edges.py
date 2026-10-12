# tests/unit/test_oms_edges.py
"""De OMS-paden die de hoofdtests niet raken, inclusief de live Bybit-route.

De live-route praat met Bybit via `aiohttp`. Hier staat in `sys.modules` een
vervanger die elke endpoint met een vast antwoord beantwoordt en elke aanroep
vastlegt. Er gaat niets over het netwerk; wat getoetst wordt is wat de router
met de antwoorden DOET: afronden op qtyStep, weigeren onder minOrderQty,
idempotent herhalen, en een retCode die geen 0 is niet als fill lezen.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import types
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tradebot.oms import audit_log as audit_module
from tradebot.oms.audit_log import AuditLog
from tradebot.oms.order import Fill, Order, OrderSide, OrderState, OrderStatus, OrderType
from tradebot.oms.paper_oms import PaperOMS
from tradebot.oms.position_tracker import PositionRecord, PositionTracker
from tradebot.oms.reconciler import Reconciler, ReconcilerConfig
from tradebot.oms.router import OrderRouter


def _order(symbol: str = "BTCUSDT", side: OrderSide = OrderSide.BUY,
           qty: float = 0.1, order_id: str | None = None) -> Order:
    return Order(
        order_id=order_id or f"ord_20240101_{symbol}_{side.value}_0001",
        symbol=symbol, side=side, order_type=OrderType.MARKET, qty_base=qty,
        signal_prob=0.65, kelly_fraction=0.05, model_version="v1",
        git_sha="abc1234", feature_hash="deadbeef12345678", portfolio_weight=0.2)


def _fill(order: Order, price: float = 40_000.0, qty: float | None = None) -> Fill:
    qty = order.qty_base if qty is None else qty
    return Fill(order_id=order.order_id, fill_price=price, fill_qty=qty,
                fill_ts=pd.Timestamp("2024-01-01", tz="UTC"),
                notional_usdt=price * qty)


# --------------------------------------------------------------------------- #
# Een vervanger voor aiohttp
# --------------------------------------------------------------------------- #
class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeBybit:
    """Beantwoordt de vier V5-endpoints die de router gebruikt."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.instruments: dict[str, Any] = {"result": {"list": [{
            "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001"}}]}}
        self.create: dict[str, Any] = {"retCode": 0, "result": {"orderId": "X1"}}
        self.realtime: dict[str, Any] = {"result": {"list": [
            {"avgPrice": "40000", "cumExecQty": "0.1"}]}}
        self.positions: dict[str, Any] = {"result": {"list": [
            {"symbol": "BTCUSDT", "size": "0.2", "side": "Buy"},
            {"symbol": "ETHUSDT", "size": "1.5", "side": "Sell"},
            {"symbol": "SOLUSDT", "size": "0", "side": "Buy"}]}}

    def _answer(self, url: str) -> dict[str, Any]:
        for key, payload in (("instruments-info", self.instruments),
                             ("order/create", self.create),
                             ("order/realtime", self.realtime),
                             ("position/list", self.positions)):
            if key in url:
                return payload
        raise AssertionError(f"onverwachte endpoint: {url}")

    def module(self) -> types.ModuleType:
        fake = self

        class _Session:
            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            def get(self, url: str, **_: Any) -> _Response:
                fake.calls.append(("GET", url))
                return _Response(fake._answer(url))

            def post(self, url: str, **_: Any) -> _Response:
                fake.calls.append(("POST", url))
                return _Response(fake._answer(url))

        module = types.ModuleType("aiohttp")
        module.ClientSession = _Session  # type: ignore[attr-defined]
        return module


@pytest.fixture
def bybit(monkeypatch: pytest.MonkeyPatch) -> _FakeBybit:
    fake = _FakeBybit()
    monkeypatch.setitem(sys.modules, "aiohttp", fake.module())
    return fake


def _live_router() -> OrderRouter:
    return OrderRouter(PaperOMS(), live_mode=True)


class TestRouterPaperMode:
    def test_paper_mode_delegates_and_has_no_exchange_positions(self) -> None:
        paper = PaperOMS()
        paper.set_bar_prices({"BTCUSDT": 40_000.0})
        router = OrderRouter(paper)
        fill = asyncio.run(router.place_order(_order()))
        assert fill.fill_qty == pytest.approx(0.1)
        assert asyncio.run(router.get_exchange_positions()) == {}
        closes = asyncio.run(router.close_all())
        assert len(closes) == 1

    def test_a_non_positive_step_leaves_the_quantity_alone(self) -> None:
        assert OrderRouter._round_to_step(0.12345, "0") == Decimal("0.12345")


class TestRouterLive:
    def test_an_order_is_rounded_placed_and_read_back(self, bybit: _FakeBybit) -> None:
        router = _live_router()
        order = _order(qty=0.1234)
        fill = asyncio.run(router.place_order(order))
        assert fill.exchange_order_id == "X1"
        assert fill.fill_price == 40_000.0 and fill.fill_qty == pytest.approx(0.1)
        assert order.status is OrderStatus.FILLED
        assert [c[1].split("?")[0].rsplit("/", 1)[-1] for c in bybit.calls] == [
            "instruments-info", "create", "realtime"]

    def test_a_retry_reads_the_cached_order_instead_of_placing_again(
        self, bybit: _FakeBybit,
    ) -> None:
        router = _live_router()
        order = _order()
        asyncio.run(router.place_order(order))
        bybit.calls.clear()
        asyncio.run(router.place_order(order))
        assert [c[0] for c in bybit.calls] == ["GET"]  # alleen de status

    def test_the_symbol_filter_is_cached(self, bybit: _FakeBybit) -> None:
        router = _live_router()
        first = asyncio.run(router._get_symbol_filters("BTCUSDT"))
        second = asyncio.run(router._get_symbol_filters("BTCUSDT"))
        assert first == second == ("0.001", "0.001")
        assert sum("instruments-info" in c[1] for c in bybit.calls) == 1

    def test_an_unknown_instrument_keeps_the_conservative_default(
        self, bybit: _FakeBybit,
    ) -> None:
        bybit.instruments = {"result": {"list": []}}
        assert asyncio.run(_live_router()._get_symbol_filters("NEWUSDT")) == (
            "0.00000001", "0")

    def test_a_quantity_below_the_minimum_is_refused(self, bybit: _FakeBybit) -> None:
        with pytest.raises(RuntimeError, match="minOrderQty"):
            asyncio.run(_live_router().place_order(_order(qty=0.0004)))

    def test_a_rejected_order_is_not_a_fill(self, bybit: _FakeBybit) -> None:
        bybit.create = {"retCode": 10001, "retMsg": "params error"}
        with pytest.raises(RuntimeError, match="Bybit order failed"):
            asyncio.run(_live_router().place_order(_order()))

    def test_positions_are_signed_and_close_all_flattens_them(
        self, bybit: _FakeBybit,
    ) -> None:
        router = _live_router()
        assert asyncio.run(router.get_exchange_positions()) == {
            "BTCUSDT": 0.2, "ETHUSDT": -1.5, "SOLUSDT": 0.0}
        fills = asyncio.run(router.close_all())
        assert len(fills) == 2  # SOL staat vlak en wordt overgeslagen
        assert sum(c[0] == "POST" for c in bybit.calls) == 2


# --------------------------------------------------------------------------- #
# Paper-OMS
# --------------------------------------------------------------------------- #
class TestPaperOms:
    def test_a_crisis_multiplier_is_never_below_one(self) -> None:
        oms = PaperOMS()
        oms.set_crisis_multiplier(0.3)
        assert oms._crisis_multiplier == 1.0
        oms.set_crisis_multiplier(3.0)
        assert oms._crisis_multiplier == 3.0

    def test_an_order_without_a_price_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no reference price"):
            PaperOMS().place_order(_order())

    def test_a_sell_fills_below_the_close_with_the_schedule_fee(
        self, tmp_path: Path,
    ) -> None:
        log = AuditLog(tmp_path / "audit.jsonl")
        oms = PaperOMS(audit_log=log, fee_schedule=SimpleNamespace(taker_bps=2.0))  # type: ignore[arg-type]
        oms.set_bar_prices({"BTCUSDT": 40_000.0}, sigma_map={"BTCUSDT": 0.01},
                           volume_map={"BTCUSDT": 500.0},
                           spread_bps_map={"BTCUSDT": 1.0})
        fill = oms.place_order(_order(side=OrderSide.SELL))
        assert fill.fill_price < 40_000.0
        assert log.records_written == 1

    def test_close_all_skips_flat_and_unpriced_positions(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        oms = PaperOMS()
        oms.set_bar_prices({"BTCUSDT": 40_000.0, "ETHUSDT": 2_000.0})
        oms.place_order(_order())
        oms.place_order(_order("ETHUSDT", OrderSide.SELL, qty=1.0))
        oms.tracker._positions["SOLUSDT"] = PositionRecord(symbol="SOLUSDT", qty=0.0)
        oms.tracker._positions["XRPUSDT"] = PositionRecord(symbol="XRPUSDT", qty=5.0)
        with caplog.at_level(logging.WARNING):
            fills = oms.close_all()
        assert len(fills) == 2
        assert "no price for XRPUSDT" in caplog.text

    def test_funding_is_paid_long_and_received_short(self) -> None:
        oms = PaperOMS()
        oms.set_bar_prices({"BTCUSDT": 40_000.0, "ETHUSDT": 2_000.0})
        ts = pd.Timestamp("2024-01-01 08:00", tz="UTC")
        assert oms.accrue_funding("BTCUSDT", 0.0001, ts) == 0.0  # geen positie
        oms.place_order(_order(qty=0.1))
        oms.place_order(_order("ETHUSDT", OrderSide.SELL, qty=1.0))
        oms.set_bar_prices({"BTCUSDT": 40_000.0, "ETHUSDT": 2_000.0})  # mark de posities
        long_pay = oms.accrue_funding("BTCUSDT", 0.0001, ts)
        short_pay = oms.accrue_funding("ETHUSDT", 0.0001, ts)
        assert long_pay < 0.0 < short_pay


# --------------------------------------------------------------------------- #
# Positieboek, reconciliatie, auditlog
# --------------------------------------------------------------------------- #
class TestPositionTracker:
    def test_adding_reducing_and_reversing(self) -> None:
        tracker = PositionTracker(initial_equity=10_000.0)
        buy = _order(qty=1.0)
        tracker.apply_fill("BTC", _fill(buy, 100.0, 1.0), OrderSide.BUY)
        tracker.apply_fill("BTC", _fill(buy, 110.0, 1.0), OrderSide.BUY)
        pos = tracker.get_position("BTC")
        assert pos.qty == 2.0 and pos.avg_entry_price == pytest.approx(105.0)
        tracker.apply_fill("BTC", _fill(buy, 120.0, 0.5), OrderSide.SELL)
        assert pos.qty == 1.5 and pos.avg_entry_price == pytest.approx(105.0)
        tracker.apply_fill("BTC", _fill(buy, 130.0, 2.5), OrderSide.SELL)
        assert pos.qty == pytest.approx(-1.0)
        assert pos.avg_entry_price == 130.0

    def test_leverage_marks_and_the_equity_series(self) -> None:
        tracker = PositionTracker(initial_equity=10_000.0)
        assert PositionRecord(symbol="X", qty=1.0, last_mark_price=0.0).notional == 0.0
        tracker.apply_fill("BTC", _fill(_order(qty=1.0), 100.0, 1.0), OrderSide.BUY)
        tracker.mark("BTC", 110.0)
        assert tracker.gross_leverage == pytest.approx(110.0 / tracker.equity)
        assert list(tracker.equity_series()) == [tracker.equity]
        flat = PositionTracker(initial_equity=0.0)
        assert flat.gross_leverage == 0.0
        flat._positions["ETH"] = PositionRecord(symbol="ETH", qty=0.0)
        flat.mark("ETH", 2_000.0)
        assert flat.get_position("ETH").unrealised_pnl == 0.0

    def test_the_legacy_on_fill_hook_only_registers_the_key(self) -> None:
        tracker = PositionTracker()
        order = _order()
        tracker.on_fill(_fill(order), OrderSide.BUY)
        assert "BTCUSDT" in tracker._positions


class TestReconciler:
    def test_grace_and_first_strike_are_pending_not_discrepant(self) -> None:
        tracker = PositionTracker()
        tracker.apply_fill("BTC", _fill(_order(qty=1.0), 100.0, 1.0), OrderSide.BUY)
        rec = Reconciler(tracker, ReconcilerConfig(
            paper_mode=False, require_two_strikes=True, fill_grace_period_sec=5.0))
        rec.note_fill("BTC", ts=1_000.0)
        in_grace = rec.reconcile({"BTC": 0.0}, now=1_002.0)
        assert in_grace[0].is_discrepant is False
        first = rec.reconcile({"BTC": 0.0}, now=1_100.0)
        assert first[0].is_discrepant is False
        second = rec.reconcile({"BTC": 0.0}, now=1_200.0)
        assert second[0].is_discrepant is True
        rec.note_fill("BTC")  # zonder tijdstip: nu
        assert rec._last_fill_ts["BTC"] > 1_200.0


class TestAuditLog:
    def test_an_incomplete_record_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(audit_module, "_build_record",
                            lambda order, fill: {"order_id": order.order_id})
        order = _order()
        with pytest.raises(ValueError, match="missing required fields"):
            AuditLog(tmp_path / "a.jsonl").record(order, _fill(order))

    def test_an_absent_log_reads_empty_and_hashes_empty(self, tmp_path: Path) -> None:
        log = AuditLog(tmp_path / "never.jsonl")
        assert log.read_as_dataframe().empty
        assert log.write_integrity_hash() == ""

    def test_broken_lines_are_skipped_on_read_and_on_chain_resume(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        order = _order()
        log.record(order, _fill(order))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{broken\n\n")
        with caplog.at_level(logging.WARNING):
            assert len(log.read_as_dataframe()) == 1
        assert "malformed" in caplog.text
        resumed = AuditLog(path)
        assert resumed._prev_hash == log._prev_hash

    def test_an_unreadable_log_starts_a_fresh_chain(self, tmp_path: Path) -> None:
        directory = tmp_path / "audit.jsonl"
        directory.mkdir()
        assert AuditLog(directory)._prev_hash == "0" * 64


def test_an_order_state_is_terminal_only_when_done() -> None:
    order = _order()
    state = OrderState(order=order)
    assert state.is_terminal is False
    order.status = OrderStatus.CANCELLED
    assert state.is_terminal is True


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_an_invalid_mark_keeps_the_last_valid_one(bad: float) -> None:
    """Een gat in de feed is geen verlies van 100 %."""
    tracker = PositionTracker(initial_equity=10_000.0)
    tracker.mark("UNKNOWN", 1.0)  # geen positie: niets te doen
    tracker.apply_fill("BTC", _fill(_order(qty=1.0), 100.0, 1.0), OrderSide.BUY)
    tracker.mark("BTC", 110.0)
    tracker.mark("BTC", bad)
    pos = tracker.get_position("BTC")
    assert pos.last_mark_price == 110.0
    assert pos.unrealised_pnl == pytest.approx(10.0)
