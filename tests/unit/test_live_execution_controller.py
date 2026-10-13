# tests/unit/test_live_execution_controller.py
"""`live/execution_controller.py`: inertiefilter, orderopbouw en de fat-finger-controle.

De soevereine limieten (`_verify_target_book`) hebben hun eigen bestand,
`test_live_limits_are_sovereign.py`; hier gaat het om wat daarna gebeurt.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from tradebot.live import execution_controller as ec
from tradebot.live.execution_controller import ExecutionController, ExecutionControllerConfig
from tradebot.oms.order import OrderSide, OrderType
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
PRICES = {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0}
EQUITY = 100_000.0


def _controller(**cfg) -> ExecutionController:
    base = {"git_sha": "abc", "model_version": "mv", "min_notional_per_trade": 0.0,
            "min_weight_change": 0.0, "max_weight_change": 1.0}
    return ExecutionController(ExecutionControllerConfig(RISK, **{**base, **cfg}))


def _size(ctrl: ExecutionController, target: dict[str, float], current: dict[str, float] | None = None,
          prices: dict[str, float] | None = None, equity: float = EQUITY, **kw):
    return ctrl.size_orders(pd.Series(target), current or {}, prices or PRICES, equity, **kw)


class TestInertiaFilter:
    def test_a_small_relative_change_is_not_traded(self) -> None:
        ctrl = _controller(min_weight_change=0.02)
        assert _size(ctrl, {"BTCUSDT": 0.211}, {"BTCUSDT": 0.20}) == []

    def test_a_small_absolute_notional_is_not_traded(self) -> None:
        ctrl = _controller(min_notional_per_trade=1_000.0)
        assert _size(ctrl, {"BTCUSDT": 0.005}, equity=100_000.0) == []             # $500

    def test_a_trade_clearing_both_gates_goes_through(self) -> None:
        ctrl = _controller(min_weight_change=0.02, min_notional_per_trade=1_000.0)
        assert len(_size(ctrl, {"BTCUSDT": 0.10})) == 1

    def test_the_legacy_dust_floor_still_holds_when_the_gates_are_switched_off(self) -> None:
        assert _size(_controller(), {"BTCUSDT": 0.0005}) == []

    def test_a_single_bar_move_is_limited_to_max_weight_change(self) -> None:
        orders = _size(_controller(max_weight_change=0.05), {"BTCUSDT": 0.40})
        assert orders[0].qty_base == pytest.approx(0.05 * EQUITY / PRICES["BTCUSDT"])
        assert orders[0].portfolio_weight == 0.40                                  # het doel blijft het doel
        sell = _size(_controller(max_weight_change=0.05), {"BTCUSDT": 0.0}, {"BTCUSDT": 0.40})
        assert sell[0].side is OrderSide.SELL and sell[0].qty_base == pytest.approx(0.05 * EQUITY / 50_000.0)


class TestOrderConstruction:
    def test_orders_carry_the_audit_trail(self) -> None:
        ctrl = _controller(kelly_divisor=4.0, max_kelly_fraction=0.25)
        orders = _size(ctrl, {"BTCUSDT": 0.20, "ETHUSDT": -0.10}, {"ETHUSDT": 0.05},
                       signal_probs={"BTCUSDT": 0.7}, feature_hashes={"BTCUSDT": "h1"})
        buy, sell = orders
        assert (buy.side, sell.side) == (OrderSide.BUY, OrderSide.SELL) and buy.order_type is OrderType.MARKET
        assert buy.qty_base == pytest.approx(0.20 * EQUITY / 50_000.0) and sell.qty_base == pytest.approx(0.15 * EQUITY / 3_000.0)
        assert (buy.signal_prob, sell.signal_prob) == (0.7, 0.5) and (buy.feature_hash, sell.feature_hash) == ("h1", "")
        assert buy.kelly_fraction == pytest.approx(0.05) and buy.model_version == "mv" and buy.git_sha == "abc"
        assert buy.order_id.endswith("_BTCUSDT_" + buy.side.value + "_0001") and sell.order_id.endswith("_0002")

    def test_the_kelly_fraction_is_capped(self) -> None:
        order = _size(_controller(kelly_divisor=1.0, max_kelly_fraction=0.1), {"BTCUSDT": 0.5})[0]
        assert order.kelly_fraction == 0.1

    def test_symbols_without_a_price_or_with_dust_notional_are_skipped(self) -> None:
        assert _size(_controller(), {"BTCUSDT": 0.2, "ETHUSDT": 0.2}, prices={"BTCUSDT": 0.0, "ETHUSDT": -1.0}) == []
        assert _size(_controller(), {"BTCUSDT": 0.002}, equity=1_000.0) == []       # $2 < $10


class TestFatFinger:
    def test_a_second_order_more_than_twice_the_previous_one_is_rejected_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ctrl = _controller()
        assert len(_size(ctrl, {"BTCUSDT": 0.10})) == 1
        with caplog.at_level(logging.ERROR, logger=ec.logger.name):
            assert _size(ctrl, {"BTCUSDT": 0.30}) == []
        assert "FAT FINGER" in caplog.text and ">2× last qty" in caplog.text
        assert len(_size(ctrl, {"BTCUSDT": 0.15})) == 1                             # binnen 2x gaat wel
        assert ctrl._last_order_qty["BTCUSDT"] == pytest.approx(0.15 * EQUITY / 50_000.0)

    def test_an_order_above_half_a_percent_of_daily_volume_is_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        ctrl = _controller()
        ctrl._daily_volume_notional["BTCUSDT"] = 1_000_000.0                         # 0,5 % = $5.000
        with caplog.at_level(logging.ERROR, logger=ec.logger.name):
            assert _size(ctrl, {"BTCUSDT": 0.10}) == []                             # $10.000
        assert "0.5% daily volume" in caplog.text
        assert len(_size(ctrl, {"BTCUSDT": 0.04})) == 1                             # $4.000

    def test_without_volume_data_there_is_no_volume_limit(self) -> None:
        assert len(_size(_controller(), {"ETHUSDT": 0.5})) == 1
