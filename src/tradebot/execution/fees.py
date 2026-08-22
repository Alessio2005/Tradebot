# src/tradebot/execution/fees.py
"""Exchange fee models for PnL computation.

Bybit USDT-Perpetual (linear) fee schedule (V5 API, 2024-2025):
    Non-VIP Maker: 0.0200 %   (post-only / limit orders resting on the book)
    Non-VIP Taker: 0.0550 %   (market orders / limit orders crossing the spread)

NOTE (venue migration 2026-06-14): the platform was migrated from Binance
USDⓈ-M to **Bybit linear (USDT) perpetuals**.  Bybit's standard taker fee
(5.5 bps) is HIGHER than Binance VIP0 (5.0 bps); this widens the per-trade
cost floor by ~10 % and must be reflected in both backtest and live.  Bybit
has no BNB-style token discount by default — the historical BIT discount is
deprecated — so the discount lever is a generic ``fee_discount_pct``.

``FeeSchedule.compute`` returns the one-way absolute fee for a given
order at the current mid-price.  Both maker and taker are one-way:
a round-trip costs 2× the stated rate.

CHIEF AUDIT-FIX (Sim-to-Reality #19):
  Mixing a VIP0 backtest with a VIP5 live account misstates per-trade cost
  by 50-200 %.  Use :func:`bybit_perp_schedule` with a tier argument (or
  supply your own ``FeeSchedule``) so backtest and live share assumptions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = [
    "BYBIT_PERP_DEFAULT",
    "BYBIT_PERP_TIERS",
    "FeeSchedule",
    "bybit_perp_schedule",
]


# Public Bybit derivatives VIP schedule (USDT linear perpetuals).
# Maker / Taker in bps.  Source: https://www.bybit.com/en/help-center/article/
# Trading-Fee-Structure  (verify current schedule before live deployment).
BYBIT_PERP_TIERS: dict[str, tuple[float, float]] = {
    "NONVIP": (2.0, 5.5),
    "VIP1":   (1.8, 4.0),
    "VIP2":   (1.6, 3.75),
    "VIP3":   (1.4, 3.5),
    "VIP4":   (1.2, 3.25),
    "VIP5":   (1.0, 3.0),
    "PRO1":   (0.9, 2.5),
    "PRO2":   (0.7, 2.3),
    "PRO3":   (0.5, 2.0),
}


@dataclass(frozen=True)
class FeeSchedule:
    """Fee schedule for a single exchange/account tier.

    Attributes
    ----------
    maker_bps : maker rebate/fee in basis points (positive = cost).
    taker_bps : taker fee in basis points.
    funding_bps_per_8h : expected funding rate in bps per 8-hour period.
        Bybit linear funding posts every 8h (00:00/08:00/16:00 UTC) for the
        majors; some alts use 4h/1h cycles (see ``nextFundingTime`` in the
        tickers endpoint).  Use 0 for inverse contracts.
    """

    maker_bps: float = 2.0    # 0.0200 % Bybit non-VIP maker
    taker_bps: float = 5.5    # 0.0550 % Bybit non-VIP taker
    funding_bps_per_8h: float = 1.0  # rough long-term average positive funding

    def compute(
        self,
        notional: float,
        order_type: Literal["maker", "taker"] = "taker",
    ) -> float:
        """Absolute one-way fee for a given notional.

        Parameters
        ----------
        notional : order notional value (price × quantity).
        order_type : "maker" or "taker".

        Returns
        -------
        float : non-negative absolute fee (same units as notional).
        """
        bps = self.taker_bps if order_type == "taker" else self.maker_bps
        return max(float(notional), 0.0) * bps * 1e-4

    def funding_cost(self, notional: float, holding_bars: int, bars_per_8h: int = 8) -> float:
        """Expected funding cost over a holding period.

        Parameters
        ----------
        notional : position notional.
        holding_bars : number of bars the position is held.
        bars_per_8h : number of bars per 8-hour funding period.

        Returns
        -------
        float : expected funding cost (positive = cost for long).
        """
        n_periods = holding_bars / max(bars_per_8h, 1)
        return max(float(notional), 0.0) * self.funding_bps_per_8h * 1e-4 * n_periods


# Convenience singleton — Bybit linear non-VIP default.
BYBIT_PERP_DEFAULT: FeeSchedule = FeeSchedule(
    maker_bps=2.0,
    taker_bps=5.5,
    funding_bps_per_8h=1.0,
)


def bybit_perp_schedule(
    tier: str = "NONVIP",
    fee_discount_pct: float = 0.0,
    referrer_rebate_pct: float = 0.0,
    funding_bps_per_8h: float = 1.0,
) -> FeeSchedule:
    """Build a Bybit linear-perpetual FeeSchedule for a specific account tier.

    Parameters
    ----------
    tier : Bybit tier string (``"NONVIP"``, ``"VIP1"`` … ``"VIP5"``,
        ``"PRO1"`` … ``"PRO3"``).  Case-insensitive.  ``"VIP0"`` is accepted
        as an alias for ``"NONVIP"`` (backward-compat with the Binance config).
    fee_discount_pct : generic fee discount (0.10 = 10 %) for promotions /
        fee-coupon programmes.  Applies multiplicatively.
    referrer_rebate_pct : additional rebate (0.10 = 10 %) for referred
        accounts.  Applies multiplicatively after the fee discount.
    funding_bps_per_8h : carry funding assumption (unchanged from
        ``FeeSchedule``).

    Returns
    -------
    FeeSchedule with the account-specific effective rates.
    """
    key = tier.upper()
    if key in ("VIP0", "VIP_0", "DEFAULT"):
        key = "NONVIP"
    if key not in BYBIT_PERP_TIERS:
        logger.warning(
            "bybit_perp_schedule: unknown tier %r — falling back to NONVIP.", tier,
        )
        key = "NONVIP"
    maker_bps, taker_bps = BYBIT_PERP_TIERS[key]
    discount = 1.0
    discount *= max(1.0 - float(fee_discount_pct), 0.0)
    discount *= max(1.0 - float(referrer_rebate_pct), 0.0)
    return FeeSchedule(
        maker_bps=maker_bps * discount,
        taker_bps=taker_bps * discount,
        funding_bps_per_8h=float(funding_bps_per_8h),
    )
