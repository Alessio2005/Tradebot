# src/tradebot/oms/position_tracker.py
"""Real-time position and mark-to-market PnL tracker.

Maintains a per-symbol position book.  Updated on every fill and
mid-price tick.  Does NOT touch the exchange — it is a pure in-memory
accounting layer.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .order import Fill, OrderSide

logger = logging.getLogger(__name__)

__all__ = ["PositionRecord", "PositionTracker"]


@dataclass
class PositionRecord:
    """Accounting record for a single symbol.

    Attributes
    ----------
    symbol :
        Trading pair.
    qty :
        Net position in base-asset units.  Positive = long, negative = short.
    avg_entry_price :
        Volume-weighted average entry price.
    realised_pnl :
        Cumulative realised PnL in USDT.
    unrealised_pnl :
        Current mark-to-market PnL vs. ``avg_entry_price``.
    last_mark_price :
        Most recent mid-price used for mark-to-market.
    """

    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    last_mark_price: float = 0.0

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.last_mark_price if self.last_mark_price else 0.0


class PositionTracker:
    """Maintains a portfolio of per-symbol positions updated on fills and ticks.

    Parameters
    ----------
    initial_equity :
        Starting equity in USDT.  Used to compute gross leverage.
    """

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        self._equity = initial_equity
        self._positions: Dict[str, PositionRecord] = {}

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def on_fill(self, fill: Fill, side: OrderSide) -> None:
        """Update position accounting on a confirmed fill.

        For a BUY fill, qty increases; for SELL, qty decreases.
        Realised PnL is computed on any quantity reduction (closing a leg).
        """
        pos = self._positions.setdefault(
            fill.order_id.split("_")[2] if "_" in fill.order_id else fill.order_id,
            PositionRecord(symbol=fill.order_id),
        )
        # PositionRecord symbol should be the actual symbol — caller passes via symbol arg
        # This method is called with the symbol already set; see apply_fill() below.

    def apply_fill(self, symbol: str, fill: Fill, side: OrderSide) -> None:
        """Update position for ``symbol`` on a confirmed fill."""
        pos = self._positions.setdefault(symbol, PositionRecord(symbol=symbol))
        signed_qty = fill.fill_qty if side == OrderSide.BUY else -fill.fill_qty
        price = fill.fill_price

        if pos.qty == 0.0:
            pos.avg_entry_price = price
            pos.qty = signed_qty
        elif (pos.qty > 0) == (signed_qty > 0):
            # Adding to existing position — update VWAP
            total = pos.qty + signed_qty
            pos.avg_entry_price = (
                pos.avg_entry_price * abs(pos.qty) + price * abs(signed_qty)
            ) / abs(total)
            pos.qty = total
        else:
            # Reducing or reversing position — realise PnL on closed portion
            close_qty = min(abs(pos.qty), abs(signed_qty))
            pnl = close_qty * (price - pos.avg_entry_price) * (1 if pos.qty > 0 else -1)
            pos.realised_pnl += pnl
            self._equity += pnl
            remainder = pos.qty + signed_qty
            if abs(remainder) < 1e-12:
                pos.qty = 0.0
                pos.avg_entry_price = 0.0
            else:
                pos.qty = remainder
                if (remainder > 0) != (pos.qty - signed_qty > 0):
                    # Full reversal — new entry at fill price
                    pos.avg_entry_price = price

        logger.debug(
            "apply_fill: %s %s %.6f @ %.2f → qty=%.6f avg=%.2f rpnl=%.2f",
            symbol, side.value, fill.fill_qty, price,
            pos.qty, pos.avg_entry_price, pos.realised_pnl,
        )

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark(self, symbol: str, mid_price: float) -> None:
        """Update unrealised PnL for ``symbol`` given the latest mid-price.

        L-3 FIX (CHIEF AUDIT 2026-05-28): reject missing/invalid prices.
        A ``mid_price`` of 0 / NaN / negative (e.g. a feed gap on one symbol
        during a multi-asset rebalance) used to book ``qty * (0 - avg_entry)``
        — a phantom 100% loss on that position.  With one of five equal-weight
        legs unpriced this drops marked equity ~15-20% for a single bar, and
        because the engine's ``peak_drawdown`` only ratchets upward that phantom
        dip permanently corrupts the live Max-Drawdown KPI and can trip the
        circuit breaker.  We keep the last valid mark instead.
        """
        if symbol not in self._positions:
            return
        if not math.isfinite(mid_price) or mid_price <= 0.0:
            logger.warning(
                "PositionTracker.mark[%s]: ignoring invalid mark price %r — "
                "retaining last valid mark %.6f (phantom-loss guard).",
                symbol, mid_price, self._positions[symbol].last_mark_price,
            )
            return
        pos = self._positions[symbol]
        pos.last_mark_price = mid_price
        if pos.qty == 0.0 or pos.avg_entry_price == 0.0:
            pos.unrealised_pnl = 0.0
        else:
            pos.unrealised_pnl = pos.qty * (mid_price - pos.avg_entry_price)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[PositionRecord]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, PositionRecord]:
        return dict(self._positions)

    @property
    def equity(self) -> float:
        """Total equity = cash + unrealised PnL across all positions."""
        unrealised = sum(p.unrealised_pnl for p in self._positions.values())
        return self._equity + unrealised

    @property
    def gross_leverage(self) -> float:
        """Sum of absolute notional exposures divided by equity."""
        eq = self.equity
        if eq <= 1e-12:
            return 0.0
        total_notional = sum(p.notional for p in self._positions.values())
        return total_notional / eq

    def equity_series(self) -> pd.Series:
        """Return a single-element equity Series (for CircuitBreaker use)."""
        return pd.Series([self.equity])
