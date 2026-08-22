# src/tradebot/oms/order.py
"""Core OMS dataclasses: Order, Fill, OrderState.

These are the unit-of-work types that flow through the OMS.  Every field
required by the R-8 audit trail is present so the AuditLog can write a
complete record without any external look-ups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

__all__ = ["OrderSide", "OrderType", "OrderStatus", "Order", "Fill", "OrderState"]


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """A single order request.

    Parameters
    ----------
    order_id :
        Unique identifier: ``f"ord_{ts:%Y%m%d}_{symbol}_{side}_{seq:03d}"``.
    symbol :
        Trading pair (e.g. ``"BTCUSDT"``).
    side :
        BUY or SELL.
    order_type :
        MARKET or LIMIT.
    qty_base :
        Order size in base-asset units (e.g. BTC).
    limit_price :
        Required for LIMIT orders; None for MARKET.
    signal_prob :
        Calibrated model probability at order creation time.
    kelly_fraction :
        Kelly sizing fraction applied.
    model_version :
        Model version string from catalog.
    git_sha :
        Short git commit hash of the running code.
    feature_hash :
        SHA-256[:16] of the feature vector used for this signal.
    portfolio_weight :
        Target portfolio weight this order is intended to achieve.
    pre_trade_cost_bps :
        Pre-trade Almgren-Chriss cost estimate in basis points.
    created_at :
        UTC timestamp of order creation.
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty_base: float
    signal_prob: float
    kelly_fraction: float
    model_version: str
    git_sha: str
    feature_hash: str
    portfolio_weight: float
    pre_trade_cost_bps: float = 0.0
    limit_price: float | None = None
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now(tz="UTC"))
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Fill:
    """A confirmed fill event for one order.

    Parameters
    ----------
    order_id :
        Matches ``Order.order_id``.
    fill_price :
        Execution price in quote currency.
    fill_qty :
        Filled quantity in base-asset units.
    fill_ts :
        UTC timestamp of fill confirmation.
    notional_usdt :
        ``fill_price × fill_qty``.
    post_trade_cost_bps :
        Realised IS cost in basis points (computed post-fill).
    exchange_order_id :
        Exchange-assigned order ID for reconciliation.
    circuit_breaker_state :
        CB state at fill time ("OPEN" | "TRIPPED").
    """

    order_id: str
    fill_price: float
    fill_qty: float
    fill_ts: pd.Timestamp
    notional_usdt: float
    post_trade_cost_bps: float = 0.0
    exchange_order_id: str = ""
    circuit_breaker_state: str = "OPEN"


@dataclass
class OrderState:
    """Live lifecycle state of an order."""

    order: Order
    fill: Fill | None = None

    @property
    def is_terminal(self) -> bool:
        return self.order.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        )
