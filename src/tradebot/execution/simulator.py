# src/tradebot/execution/simulator.py
"""LOB Queue-Position Simulator (Tier 3 — T3.1).

Models realistic limit-order fills based on Level-2 order-book queue position.

Motivation (AFML §10 / Almgren-Chriss):
  A naïve backtest fills a limit order at the posted price on the bar the
  price touches the limit level.  In reality the order sits in a queue and
  only fills *after* enough volume has traded ahead of it.

Fill rule (Lob-Sim v1):
    Order fills on bar b iff cumulative_volume_at_limit_level > Queue_at_entry + fill_fraction × Bar_volume_b

  Where:
    • Queue_at_entry   = best-side queue depth at the time the order was placed.
    • fill_fraction    = share of each bar's volume that counts toward consuming
                         the queue ahead of us (default 0.4, i.e. 40 % of bar vol
                         is estimated to be in queue ahead of our order).
    • cumulative_volume = volume traded at or better than the limit since entry.

  Conceptually: we entered the queue with ``Queue_at_entry`` units ahead of us.
  Each bar, ``fill_fraction × bar_vol`` is consumed from ahead.  We fill when
  the cumulative consumed volume exceeds Queue_at_entry.

LOB snapshot:
  When Level-2 data is unavailable (common for historical crypto backtests),
  we proxy Queue_at_entry using:

      Queue_at_entry ≈ queue_vol_multiplier × bar_volume_at_entry

  A value of 1.0 means we estimate 1 bar-volume worth of orders ahead of us.
  Typical crypto-exchange depth studies suggest 0.5–2.0 depending on asset
  liquidity.  Default 1.0 (conservative mid-point).

Square-root market impact (AFML §13 / Bouchaud-Bonart):
  When ``apply_market_impact=True``, the fill price is adjusted by:

      impact = eta × sigma × sqrt(size_notional / adv_notional)

  where eta = 0.1 (universal constant from empirical studies).

Usage::

    sim = LOBSimulator(fill_fraction=0.4, queue_vol_multiplier=1.0)
    fills = sim.simulate_fills(bars_df, orders_df)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["LOBSimulator", "OrderRecord", "FillRecord"]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class OrderRecord:
    """A pending limit order placed at ``entry_bar_ts``.

    Attributes
    ----------
    order_id : str
        Unique identifier.
    symbol : str
        Trading pair.
    side : str
        "BUY" or "SELL".
    limit_price : float
        Limit price.  Buy fills when market_price <= limit; sell fills
        when market_price >= limit.
    size_base : float
        Order size in base currency.
    entry_bar_ts : pd.Timestamp
        Bar timestamp at which the order was placed.
    queue_at_entry : float
        Estimated queue ahead (in base-currency volume units).  If None,
        the simulator derives it from ``bar_volume × queue_vol_multiplier``.
    """
    order_id: str
    symbol: str
    side: str          # "BUY" | "SELL"
    limit_price: float
    size_base: float
    entry_bar_ts: pd.Timestamp
    queue_at_entry: Optional[float] = None


@dataclass
class FillRecord:
    """Result of a simulated fill.

    Attributes
    ----------
    order_id : str
    fill_ts : pd.Timestamp
        Bar timestamp when the fill occurred.
    fill_price : float
        Execution price (limit ± market impact).
    fill_size : float
        Amount filled (= ``order.size_base`` for full fills).
    bars_in_queue : int
        Number of bars the order waited before filling.
    market_impact_bps : float
        One-way market impact in basis points.
    """
    order_id: str
    fill_ts: pd.Timestamp
    fill_price: float
    fill_size: float
    bars_in_queue: int
    market_impact_bps: float = 0.0


# =============================================================================
# Simulator
# =============================================================================

class LOBSimulator:
    """Limit-Order-Book queue-position simulator.

    Parameters
    ----------
    fill_fraction : float
        Fraction of each bar's volume that is assumed to trade ahead of our
        order in the queue.  Default 0.4 (40%).
    queue_vol_multiplier : float
        When ``order.queue_at_entry`` is None, proxy it as:
        ``bar_volume_at_entry × queue_vol_multiplier``.  Default 1.0.
    apply_market_impact : bool
        If True, apply the Bouchaud square-root market impact model to the
        fill price.  Default True.
    impact_eta : float
        Universal impact constant η in the square-root model.  Default 0.1.
    adv_window_bars : int
        Rolling window for Average Daily Volume (ADV) used in market impact.
    max_wait_bars : int
        Maximum bars an order can wait before it is cancelled (expired).
        Default 24 (= 1 day of hourly bars).
    """

    def __init__(
        self,
        fill_fraction: float = 0.4,
        queue_vol_multiplier: float = 1.0,
        apply_market_impact: bool = True,
        impact_eta: float = 0.1,
        adv_window_bars: int = 24,
        max_wait_bars: int = 24,
        bars_per_year: float = 365.0 * 24,  # CHIEF AUDIT 2026-05-23: for ann→bar vol
    ) -> None:
        self.fill_fraction = float(np.clip(fill_fraction, 0.0, 1.0))
        self.queue_vol_multiplier = max(0.0, float(queue_vol_multiplier))
        self.apply_market_impact = apply_market_impact
        self.impact_eta = float(impact_eta)
        self.adv_window_bars = int(adv_window_bars)
        self.max_wait_bars = int(max_wait_bars)
        self.bars_per_year = max(float(bars_per_year), 1.0)

    # ------------------------------------------------------------------
    def _compute_market_impact_bps(
        self,
        side: str,
        size_base: float,
        fill_price: float,
        bar_vol: float,
        adv: float,
        ann_vol: float,
        bars_per_year: float = 365.0 * 24,  # hourly bars default
    ) -> float:
        """Square-root market impact in bps (one-way).

        I(Q) = η × σ_bar × √(Q_notional / ADV_notional)

        Returns impact in basis points.
        """
        if adv <= 0 or fill_price <= 0:
            return 0.0
        size_notional = size_base * fill_price
        adv_notional  = adv * fill_price
        # CHIEF AUDIT 2026-05-23 (KRITISCH — LOBSimulator): ann_vol was used
        # directly as σ in the Almgren-Chriss formula, but the formula requires
        # BAR-LEVEL σ (σ_bar), not annualised vol.  Using ann_vol=0.80 directly
        # overstates impact by factor sqrt(bars_per_year) ≈ 93.6× for hourly
        # bars — making simulated fills unrealistically expensive.
        # Fix: convert ann_vol → bar_vol before applying the impact model.
        bar_vol_eff = ann_vol / float(np.sqrt(max(bars_per_year, 1.0)))
        impact_frac = self.impact_eta * bar_vol_eff * float(np.sqrt(size_notional / (adv_notional + 1e-9)))
        return float(np.clip(impact_frac * 1e4, 0.0, 200.0))  # cap at 200 bps

    # ------------------------------------------------------------------
    def simulate_order(
        self,
        order: OrderRecord,
        bars_after_entry: pd.DataFrame,
        ann_vol_proxy: float = 0.80,
        treat_as_maker: bool = True,
    ) -> Optional[FillRecord]:
        """Simulate a single limit order against a bar series.

        Parameters
        ----------
        order :
            The pending limit order.
        bars_after_entry :
            DataFrame of OHLCV bars from (entry bar + 1) onwards, index =
            DatetimeIndex.  Must contain: open, high, low, close, volume.
        ann_vol_proxy :
            Annualised volatility proxy (fraction) used in market impact.
            Default 0.80 (80% — conservative for crypto).

        Returns
        -------
        FillRecord if filled, None if expired.
        """
        if bars_after_entry.empty:
            return None

        # Proxy queue size from entry bar volume if not provided
        queue = (
            order.queue_at_entry
            if order.queue_at_entry is not None
            else self.queue_vol_multiplier * float(
                bars_after_entry["volume"].iloc[0] if len(bars_after_entry) > 0 else 0.0
            )
        )

        cumvol = 0.0
        is_buy = order.side.upper() == "BUY"

        # Rolling ADV for market impact
        adv = float(bars_after_entry["volume"].iloc[: self.adv_window_bars].mean())

        for bars_waited, (ts, row) in enumerate(bars_after_entry.iterrows(), start=1):
            # Expire after max_wait_bars
            if bars_waited > self.max_wait_bars:
                logger.debug(
                    "[%s] Order %s expired after %d bars.",
                    order.symbol, order.order_id, self.max_wait_bars,
                )
                return None

            bar_vol = float(row.get("volume", 0.0))

            # Does the price touch our limit on this bar?
            bar_low  = float(row.get("low",  row.get("close", order.limit_price)))
            bar_high = float(row.get("high", row.get("close", order.limit_price)))

            price_touched = (
                (is_buy  and bar_low  <= order.limit_price) or
                (not is_buy and bar_high >= order.limit_price)
            )

            if not price_touched:
                # Price not at our level — no queue consumed
                continue

            # Accumulate volume traded at our level
            cumvol += self.fill_fraction * bar_vol

            # Fill when cumulative consumed volume > queue ahead
            if cumvol >= queue:
                fill_price = order.limit_price

                # CHIEF AUDIT 2026-05-23 (KRITISCH — dubbele maker-impact):
                # Een LIMIT-fill krijgt fill_price = order.limit_price (de
                # passieve quote-prijs). Een Almgren-Chriss impact-bump bovenop
                # die limit-prijs is fundamenteel onjuist: maker-orders LIEFEREN
                # liquiditeit, ze nemen geen liquiditeit weg. De "adverse
                # selection"-kost van een maker-fill zit IMPLICIET in de queue-
                # wait (we filllen pas nadat genoeg tegenvolume is gepasseerd,
                # wat correleert met onfavoriete prijsbewegingen). Apply impact
                # ALLEEN voor taker-style fills (treat_as_maker=False).
                impact_bps = 0.0
                if self.apply_market_impact and not treat_as_maker:
                    impact_bps = self._compute_market_impact_bps(
                        side=order.side,
                        size_base=order.size_base,
                        fill_price=fill_price,
                        bar_vol=bar_vol,
                        adv=adv,
                        ann_vol=ann_vol_proxy,
                        bars_per_year=self.bars_per_year,
                    )
                    impact_frac = impact_bps / 1e4
                    if is_buy:
                        fill_price *= (1.0 + impact_frac)
                    else:
                        fill_price *= (1.0 - impact_frac)

                logger.debug(
                    "[%s] Order %s filled @ %.6f (impact=%.1f bps) after %d bars.",
                    order.symbol, order.order_id, fill_price, impact_bps, bars_waited,
                )
                return FillRecord(
                    order_id=order.order_id,
                    fill_ts=ts,
                    fill_price=fill_price,
                    fill_size=order.size_base,
                    bars_in_queue=bars_waited,
                    market_impact_bps=impact_bps,
                )

        return None  # Not filled within the available bar window

    # ------------------------------------------------------------------
    def simulate_fills(
        self,
        bars_df: pd.DataFrame,
        orders: List[OrderRecord],
        ann_vol_proxy: float = 0.80,
        treat_as_maker: bool = True,
    ) -> List[FillRecord]:
        """Batch-simulate fills for a list of orders against a bar series.

        Parameters
        ----------
        bars_df :
            Full OHLCV bar DataFrame (DatetimeIndex, UTC-aware).
        orders :
            List of pending orders to simulate.
        ann_vol_proxy :
            Annualised vol proxy for market impact.

        Returns
        -------
        List[FillRecord] — one entry per *filled* order.  Unfilled/expired
        orders are silently dropped.
        """
        fills: List[FillRecord] = []

        for order in orders:
            # Slice bars from one bar after entry
            try:
                entry_loc = bars_df.index.get_loc(order.entry_bar_ts)
            except KeyError:
                # entry bar not in index — find nearest
                entry_loc = bars_df.index.searchsorted(order.entry_bar_ts)

            if entry_loc >= len(bars_df) - 1:
                continue  # No future bars to check

            future_bars = bars_df.iloc[entry_loc + 1 :]
            fill = self.simulate_order(
                order,
                future_bars,
                ann_vol_proxy=ann_vol_proxy,
                treat_as_maker=treat_as_maker,
            )
            if fill is not None:
                fills.append(fill)

        logger.info(
            "LOBSimulator: %d orders → %d fills (%.1f%% fill rate).",
            len(orders),
            len(fills),
            100.0 * len(fills) / max(1, len(orders)),
        )
        return fills

    # ------------------------------------------------------------------
    @staticmethod
    def estimate_fill_probability(
        queue_at_entry: float,
        fill_fraction: float,
        bar_volume: float,
        n_bars: int,
    ) -> float:
        """Estimate the probability of filling within n_bars.

        Uses a simple geometric model: each bar, fill_fraction × bar_vol
        of queue is consumed.  Fill prob is the fraction of expected
        consumed vol / queue that clears within n_bars.

        Returns probability in [0, 1].
        """
        if queue_at_entry <= 0:
            return 1.0
        vol_per_bar = fill_fraction * max(bar_volume, 1e-9)
        bars_to_fill = queue_at_entry / vol_per_bar
        # Geometric CDF: P(fill in n_bars) = min(1, expected_fills / 1)
        prob = float(np.clip(n_bars / (bars_to_fill + 1e-9), 0.0, 1.0))
        return prob

    # ------------------------------------------------------------------
    @staticmethod
    def adjust_cost_for_queue_wait(
        fill_bps: float,
        bars_in_queue: float,
        adverse_selection_bps_per_bar: float = 0.1,
    ) -> float:
        """Augment quoted spread cost with adverse-selection cost from waiting.

        Every bar we wait in the queue, adverse selection erodes our edge
        (the price moves against us).  Default 0.1 bps / bar.

        Returns total_cost_bps = fill_bps + adverse_selection_bps.
        """
        return float(fill_bps) + float(bars_in_queue) * float(adverse_selection_bps_per_bar)
