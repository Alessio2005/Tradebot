# src/tradebot/oms/paper_oms.py
"""Paper-trade OMS — identical interface to the live OMS router.

Simulates Bybit fills on historical or live bars without placing real orders.
Fill price = bar.close * (1 + slippage).
Funding costs are accrued every 8h (Bybit perpetuals convention).
"""
from __future__ import annotations

import logging
import uuid

import pandas as pd

from ..execution.fees import FeeSchedule
from ..execution.slippage import SlippageModel, compute_slippage
from .audit_log import AuditLog
from .order import Fill, Order, OrderSide, OrderStatus
from .position_tracker import PositionTracker

logger = logging.getLogger(__name__)

__all__ = ["PaperOMS"]

_FUNDING_INTERVAL_H = 8


class PaperOMS:
    """Simulated OMS for paper-trading.

    Parameters
    ----------
    initial_equity :
        Starting NAV in USDT.
    slippage_model :
        Slippage model applied to every fill.
    audit_log :
        Optional audit log.  If None, fills are not persisted.
    taker_fee_bps :
        Taker fee in basis points (default 4 bps = 0.04 %).
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        slippage_model: SlippageModel | None = None,
        audit_log: AuditLog | None = None,
        taker_fee_bps: float = 4.0,
        fee_schedule: FeeSchedule | None = None,
    ) -> None:
        self._tracker = PositionTracker(initial_equity=initial_equity)
        self._slippage = slippage_model or SlippageModel(mode="fixed", fixed_bps=5.0)
        self._audit = audit_log
        self._taker_fee_bps = taker_fee_bps
        # Rec 4 (Sim-to-Reality #19): VIP-tier fee schedule overrides taker_fee_bps
        self._fee_schedule: FeeSchedule | None = fee_schedule
        # Rec 2 (Sim-to-Reality #6): per-bar market metadata for slippage wiring
        self._bar_sigma: dict[str, float] = {}
        self._bar_volume: dict[str, float] = {}
        self._bar_spread_bps: dict[str, float] = {}
        # Rec 3 (Sim-to-Reality #5): crisis regime multiplier on market impact
        self._crisis_multiplier: float = 1.0
        self._fills: list[Fill] = []
        self._equity_history: list[float] = [initial_equity]
        self._last_funding_ts: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    # Core interface (mirrors live OMS)
    # ------------------------------------------------------------------

    def set_crisis_multiplier(self, multiplier: float) -> None:
        """Update crisis regime multiplier on market impact (Rec 3 / Sim-to-Reality #5).

        Called by the engine after each portfolio optimisation step.
        Multiplier is clamped to [1.0, ∞) — calm regime = 1.0, crisis ≈ 3.0.
        """
        self._crisis_multiplier = max(1.0, float(multiplier))

    def place_order(self, order: Order) -> Fill:
        """Simulate a market fill at ``order`` time.

        Fill price is the current bar close adjusted for slippage and fees.
        Slippage is always adverse (increases cost for both buy and sell).
        Uses per-bar sigma, volume and observed spread (Rec 2) and the
        crisis multiplier (Rec 3) when they have been set via set_bar_prices()
        / set_crisis_multiplier().
        """
        # Rec 2: pull per-bar metadata for realistic slippage computation
        sigma_bar = self._bar_sigma.get(order.symbol, 0.0)
        bar_volume = self._bar_volume.get(order.symbol, 1.0)
        spread_bps = self._bar_spread_bps.get(order.symbol, 0.0)

        slip = compute_slippage(
            self._slippage,
            price=1.0,  # relative mode: returns fraction
            sigma_bar=sigma_bar,
            bar_volume=bar_volume,
            observed_spread_bps=spread_bps,
            crisis_multiplier=self._crisis_multiplier,
        )
        # We don't have bar.close here — caller sets a reference price via
        # last_close attribute before calling place_order.
        ref_price = getattr(self, "_last_close", {}).get(order.symbol, 0.0)
        if ref_price <= 0.0:
            raise ValueError(
                f"PaperOMS: no reference price for {order.symbol}. "
                "Call set_bar_prices() before place_order()."
            )

        # Rec 4: use VIP-tier fee schedule when configured
        if self._fee_schedule is not None:
            fee_fraction = self._fee_schedule.taker_bps * 1e-4
        else:
            fee_fraction = self._taker_fee_bps * 1e-4
        adverse_fraction = slip + fee_fraction  # both costs on same side

        if order.side == OrderSide.BUY:
            fill_price = ref_price * (1.0 + adverse_fraction)
        else:
            fill_price = ref_price * (1.0 - adverse_fraction)

        notional = fill_price * order.qty_base
        post_bps = abs(fill_price - ref_price) / ref_price * 1e4

        fill = Fill(
            order_id=order.order_id,
            fill_price=fill_price,
            fill_qty=order.qty_base,
            fill_ts=pd.Timestamp.now(tz="UTC"),
            notional_usdt=notional,
            post_trade_cost_bps=round(post_bps, 4),
            exchange_order_id=f"paper_{uuid.uuid4().hex[:8]}",
            circuit_breaker_state="OPEN",
        )

        order.status = OrderStatus.FILLED
        self._tracker.apply_fill(order.symbol, fill, order.side)
        self._fills.append(fill)
        self._equity_history.append(self._tracker.equity)

        if self._audit is not None:
            self._audit.record(order, fill)

        logger.debug(
            "PaperOMS.fill: %s %s %.6f @ %.4f (slip+fee=%.2fbps)",
            order.symbol, order.side.value, order.qty_base,
            fill_price, post_bps,
        )
        return fill

    def close_all(self) -> list[Fill]:
        """Flatten all open positions at current mark prices."""
        fills: list[Fill] = []
        for symbol, pos in self._tracker.get_all_positions().items():
            if abs(pos.qty) < 1e-12:
                continue
            ref_price = getattr(self, "_last_close", {}).get(symbol, 0.0)
            if ref_price <= 0.0:
                logger.warning("PaperOMS.close_all: no price for %s, skipping.", symbol)
                continue
            side = OrderSide.SELL if pos.qty > 0 else OrderSide.BUY
            close_order = Order(
                order_id=f"ord_close_{symbol}_{uuid.uuid4().hex[:6]}",
                symbol=symbol,
                side=side,
                order_type=__import__("tradebot.oms.order", fromlist=["OrderType"]).OrderType.MARKET,
                qty_base=abs(pos.qty),
                signal_prob=0.5,
                kelly_fraction=0.0,
                model_version="close",
                git_sha="",
                feature_hash="",
                portfolio_weight=0.0,
            )
            fills.append(self.place_order(close_order))
        return fills

    def get_positions(self) -> dict[str, float]:
        """Return symbol → net_qty mapping."""
        return {
            sym: pos.qty
            for sym, pos in self._tracker.get_all_positions().items()
        }

    def equity_curve(self) -> pd.Series:
        """Return equity curve as a Series (index = fill number)."""
        return pd.Series(self._equity_history, name="equity")

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def set_bar_prices(
        self,
        prices: dict[str, float],
        sigma_map: dict[str, float] | None = None,
        volume_map: dict[str, float] | None = None,
        spread_bps_map: dict[str, float] | None = None,
    ) -> None:
        """Update reference close prices and per-bar market metadata.

        Must be called once per bar before ``place_order``.

        Parameters
        ----------
        prices : symbol → close price for mark-to-market.
        sigma_map : symbol → bar volatility fraction (Rec 2 / Sim-to-Reality #6).
        volume_map : symbol → bar volume in base-asset units (Rec 2).
        spread_bps_map : symbol → half-spread in bps from live order book or
            intrabar range proxy (Rec 2).  Used as a floor on modelled slippage.
        """
        if not hasattr(self, "_last_close"):
            self._last_close: dict[str, float] = {}
        self._last_close.update(prices)
        for sym, price in prices.items():
            self._tracker.mark(sym, price)
        if sigma_map:
            self._bar_sigma.update(sigma_map)
        if volume_map:
            self._bar_volume.update(volume_map)
        if spread_bps_map:
            self._bar_spread_bps.update(spread_bps_map)

    # ------------------------------------------------------------------
    # Funding simulation
    # ------------------------------------------------------------------

    def accrue_funding(
        self,
        symbol: str,
        funding_rate: float,
        ts: pd.Timestamp,
    ) -> float:
        """Accrue 8h funding cost for a perpetual position.

        Parameters
        ----------
        funding_rate :
            Bybit 8h funding rate (e.g. 0.0001 = 0.01 %).
        ts :
            Current bar timestamp.

        Returns
        -------
        float : funding payment (negative = paid, positive = received).
        """
        pos = self._tracker.get_position(symbol)
        if pos is None or abs(pos.qty) < 1e-12:
            return 0.0
        notional = pos.notional  # altijd positief (abs qty × prijs)

        # P0-I FIX: funding is SIDE-AWARE.
        # Bij positief funding_rate: longs BETALEN, shorts ONTVANGEN.
        # Oude code: -notional * rate → altijd negatief → shorts werden fout belast.
        # Fix: gebruik sign(qty) zodat shorts een positieve payment ontvangen.
        # Backtest-conventie (portfolio.py:302): signed_now * fr — idem.
        if pos.qty > 0:
            # Long: betaal funding (negatief PnL)
            payment = -notional * funding_rate
        else:
            # Short: ontvang funding bij positief rate (positief PnL)
            payment = notional * funding_rate
        self._tracker._equity += payment
        logger.debug(
            "Funding: %s rate=%.6f notional=%.2f payment=%.4f",
            symbol, funding_rate, notional, payment,
        )
        return payment

    @property
    def tracker(self) -> PositionTracker:
        return self._tracker
