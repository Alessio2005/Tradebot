# src/tradebot/live/execution_controller.py
"""Convert target portfolio weights to orders with impact-aware sizing.

Compares current positions against target weights, computes the required
delta, applies Kelly sizing and position limits, and emits Order objects
for the OMS router.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..oms.order import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)

__all__ = ["ExecutionControllerConfig", "ExecutionController"]

# Legacy dust-trade floor (USDT).  Kept only as an absolute minimum below which
# nothing trades regardless of inertia config — config.min_notional_per_trade
# is the operational gate (default $1 000).
_MIN_ORDER_NOTIONAL = 10.0  # USDT


class ExecutionControllerConfig:
    """Configuration for the execution controller.

    Parameters
    ----------
    kelly_divisor :
        Fractional Kelly divisor for conservative sizing (default 4).
    max_kelly_fraction :
        Absolute cap on Kelly fraction per signal.
    min_confidence :
        Minimum signal confidence to place a trade.
    max_weight_change :
        Maximum allowed single-bar weight change (turnover limiter).
    git_sha :
        Current git commit SHA for audit trail.
    model_version :
        Current model version string.
    min_notional_per_trade :
        CHIEF-1 (2026-05-28) — INERTIA FILTER, absolute USDT floor.  Any
        rebalance order below this notional is dropped so the portfolio
        does not churn $200-$300 trades on $40 k positions.  With
        post-trade cost ≈ 10 bps round-trip + maker-taker imbalance, a
        $300 trade costs ≈ $0.30 → over 8 760 bars/year that compounds
        away alpha.  Default $1 000 (≈ 0.5 % of starting NAV).
    min_weight_change :
        CHIEF-1 — INERTIA FILTER, relative gate.  Drop rebalances where
        |target_w - current_w| < this threshold (default 2 %).  This is
        the "no trade band" used by every serious portfolio manager.
        Stacked with ``min_notional_per_trade`` (AND-gate): both must
        be exceeded for an order to fire.
    """

    def __init__(
        self,
        kelly_divisor: float = 4.0,
        max_kelly_fraction: float = 0.25,
        min_confidence: float = 0.55,
        max_weight_change: float = 0.10,
        git_sha: str = "",
        model_version: str = "v1.0.0",
        min_notional_per_trade: float = 1_000.0,
        min_weight_change: float = 0.02,
    ) -> None:
        self.kelly_divisor = kelly_divisor
        self.max_kelly_fraction = max_kelly_fraction
        self.min_confidence = min_confidence
        self.max_weight_change = max_weight_change
        self.git_sha = git_sha
        self.model_version = model_version
        # CHIEF-1 inertia parameters
        self.min_notional_per_trade = float(min_notional_per_trade)
        self.min_weight_change = float(min_weight_change)


class ExecutionController:
    """Converts target weights to Order objects.

    Parameters
    ----------
    config :
        Controller configuration.
    """

    def __init__(self, config: ExecutionControllerConfig) -> None:
        self._cfg = config
        self._seq: int = 0

        # Wave 15 P0-5.2 — position limit parameters (set via attributes after init)
        self._max_notional_per_symbol: float = float("inf")
        self._max_gross_notional: float = float("inf")

        # Wave 15 P0-5.3 — fat-finger state
        self._last_order_qty: dict[str, float] = {}
        self._daily_volume_notional: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Per-bar order generation
    # ------------------------------------------------------------------

    def size_orders(
        self,
        target_weights: pd.Series,
        current_weights: dict[str, float],
        prices: dict[str, float],
        equity: float,
        signal_probs: dict[str, float] | None = None,
        feature_hashes: dict[str, str] | None = None,
    ) -> list[Order]:
        """Generate orders to move from ``current_weights`` to ``target_weights``.

        Parameters
        ----------
        target_weights :
            Target portfolio weights (indexed by symbol, sum ≤ 1).
        current_weights :
            Current portfolio weights (symbol → fraction of equity).
        prices :
            Current close prices per symbol.
        equity :
            Current portfolio equity in USDT.
        signal_probs :
            symbol → calibrated signal probability (for audit trail).
        feature_hashes :
            symbol → feature hash (for audit trail).

        Returns
        -------
        List of Order objects.  Empty list if no rebalancing is needed.
        """
        orders: list[Order] = []
        now = pd.Timestamp.now(tz="UTC")
        date_str = now.strftime("%Y%m%d")

        for symbol in target_weights.index:
            target_w = float(target_weights[symbol])
            current_w = float(current_weights.get(symbol, 0.0))
            delta_w = target_w - current_w

            # ── CHIEF-1 (2026-05-28) — INERTIA FILTER (anti-churn) ─────────
            # Empirical evidence from Run 4b:
            #   16:02 SELL DOTUSDT $304   (delta_w ≈ 0.0015 on $40 k pos)
            #   16:57 BUY  DOTUSDT $234   (delta_w ≈ 0.0012 on $40 k pos)
            # These were costlier than informative — pure drift correction
            # spending half-spread + maker-taker delta + slippage every time.
            #
            # New gate: drop the order unless BOTH
            #   |delta_w|        ≥ min_weight_change       (relative)
            #   |delta_w|·equity ≥ min_notional_per_trade  (absolute)
            # are exceeded.  An old, looser tier (>1 bp & >$10) still applies
            # as a hard dust floor for sanity.
            inertia_w   = abs(delta_w) < self._cfg.min_weight_change
            inertia_usd = abs(delta_w) * equity < self._cfg.min_notional_per_trade
            if inertia_w or inertia_usd:
                logger.debug(
                    "ExecutionController: INERTIA skip [%s] delta_w=%.4f "
                    "notional=$%.0f (gates: w>=%.4f, $>=%.0f)",
                    symbol, delta_w, abs(delta_w) * equity,
                    self._cfg.min_weight_change, self._cfg.min_notional_per_trade,
                )
                continue
            # Hard dust floor (legacy).  Should never trigger after the
            # inertia gates above unless someone explicitly drops them to 0.
            if abs(delta_w) < 0.001:
                continue
            if abs(delta_w) > self._cfg.max_weight_change:
                delta_w = self._cfg.max_weight_change * (1 if delta_w > 0 else -1)

            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue

            notional = abs(delta_w) * equity
            if notional < _MIN_ORDER_NOTIONAL:
                continue

            qty_base = notional / price
            side = OrderSide.BUY if delta_w > 0 else OrderSide.SELL

            # Kelly fraction (proportional to |delta_w|, capped)
            kelly = min(abs(delta_w) / self._cfg.kelly_divisor, self._cfg.max_kelly_fraction)

            self._seq += 1
            order_id = f"ord_{date_str}_{symbol}_{side.value}_{self._seq:04d}"

            signal_prob = (signal_probs or {}).get(symbol, 0.5)
            feature_hash = (feature_hashes or {}).get(symbol, "")

            # Wave 15 P0-5.2 — position limit pre-trade gate
            current_notionals = {
                sym: abs(float(current_weights.get(sym, 0.0)) * equity)
                for sym in target_weights.index
            }
            try:
                self._check_position_limits(symbol, qty_base, notional, current_notionals)
            except ValueError as exc:
                logger.error("ExecutionController: %s", exc)
                continue

            # Wave 15 P0-5.3 — fat-finger check
            price_for_ff = prices.get(symbol, 0.0)
            try:
                self._fat_finger_check(symbol, qty_base, price_for_ff)
            except ValueError as exc:
                logger.error("ExecutionController: %s", exc)
                continue

            orders.append(
                Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    qty_base=round(qty_base, 8),
                    signal_prob=signal_prob,
                    kelly_fraction=kelly,
                    model_version=self._cfg.model_version,
                    git_sha=self._cfg.git_sha,
                    feature_hash=feature_hash,
                    portfolio_weight=target_w,
                    pre_trade_cost_bps=0.0,
                    created_at=now,
                )
            )
            # Wave 15 P0-5.3 — update last order qty after successful order creation
            self._last_order_qty[symbol] = abs(qty_base)

            logger.debug(
                "ExecutionController: %s %s qty=%.6f delta_w=%.4f",
                symbol, side.value, qty_base, delta_w,
            )

        return orders

    # ------------------------------------------------------------------
    # Wave 15 P0-5.2 — Position limit pre-trade check
    # ------------------------------------------------------------------

    def _check_position_limits(
        self,
        symbol: str,
        new_qty: float,
        new_notional: float,
        current_positions: dict,
    ) -> None:
        """Pre-trade position limit gate (Wave 15 P0-5.2).

        Raises ValueError als een limiet overschreden wordt.
        """
        # Per-symbol notional check
        if new_notional > self._max_notional_per_symbol:
            raise ValueError(
                f"POSITION LIMIT BREACH: {symbol} notional {new_notional:.2f} > "
                f"max {self._max_notional_per_symbol:.2f} (Wave 15 P0-5.2)"
            )

        # Gross notional check
        total_notional = sum(abs(p) for p in current_positions.values()) + abs(new_notional)
        if total_notional > self._max_gross_notional:
            raise ValueError(
                f"GROSS NOTIONAL LIMIT BREACH: total {total_notional:.2f} > "
                f"max {self._max_gross_notional:.2f} (Wave 15 P0-5.2)"
            )

    # ------------------------------------------------------------------
    # Wave 15 P0-5.3 — Fat-finger validator
    # ------------------------------------------------------------------

    def _fat_finger_check(
        self,
        symbol: str,
        qty: float,
        price: float,
    ) -> None:
        """Fat-finger validator: reject abnormally large orders (Wave 15 P0-5.3)."""
        notional = abs(qty * price)

        # 2× jump detector: reject als qty > 2× laatste qty voor dit symbol
        last_qty = self._last_order_qty.get(symbol, 0.0)
        if last_qty > 0.0 and abs(qty) > 2.0 * last_qty:
            raise ValueError(
                f"FAT FINGER: {symbol} qty={qty:.4f} is >2× last qty={last_qty:.4f}. "
                f"Reject to prevent runaway order. (Wave 15 P0-5.3)"
            )

        # % daily volume check: reject als notional > 0.5% daily volume
        daily_vol_notional = self._daily_volume_notional.get(symbol, float("inf"))
        if notional > 0.005 * daily_vol_notional:
            raise ValueError(
                f"FAT FINGER: {symbol} notional={notional:.2f} > 0.5% daily volume. "
                f"Reject. (Wave 15 P0-5.3)"
            )
