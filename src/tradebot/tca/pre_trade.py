# src/tradebot/tca/pre_trade.py
"""Pre-trade Transaction Cost Analysis.

Estimates expected execution cost BEFORE order placement.

Cost model (Almgren-Chriss square-root):
  spread_cost  = 0.5 * bid_ask_spread * Q
  impact_cost  = η * σ * sqrt(Q / ADV)
  delay_cost   = 0  (market orders: no delay component)
  total_cost   = spread_cost + impact_cost

Reference:
  Almgren & Chriss (2000) "Optimal execution of portfolio transactions".
  Almgren et al. (2005) "Direct estimation of equity market impact".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PreTradeCostEstimate", "estimate_pre_trade_cost"]

# Default Almgren-Chriss η calibrated for crypto (higher than equity)
_DEFAULT_ETA = 0.142


@dataclass(frozen=True)
class PreTradeCostEstimate:
    """Pre-trade cost breakdown for a single order.

    All monetary values are in fractional (per-unit) terms unless noted.
    """

    notional: float       # order size in base currency
    spread_cost: float    # 0.5 * spread * Q (fraction of notional)
    impact_cost: float    # η * σ * sqrt(Q/ADV) (fraction of notional)
    total_cost: float     # spread_cost + impact_cost
    total_bps: float      # total_cost in basis points


def estimate_pre_trade_cost(
    notional: float,
    volatility: float,
    adv: float,
    bid_ask_spread: float = 0.0002,
    eta: float = _DEFAULT_ETA,
) -> PreTradeCostEstimate:
    """Estimate expected execution cost for an order.

    Parameters
    ----------
    notional :
        Order size in base currency (e.g. USD notional).
    volatility :
        Daily realised volatility (annualised if adv is daily).
    adv :
        Average daily volume in the same currency as notional.
    bid_ask_spread :
        Fractional bid-ask spread (e.g. 0.0002 = 2bps).
    eta :
        Almgren-Chriss market impact coefficient.  Default 0.142
        calibrated from crypto lit (higher than equity ~0.1).

    Returns
    -------
    PreTradeCostEstimate with decomposed cost estimate.
    """
    notional = max(notional, 0.0)
    adv      = max(adv,      1.0)

    participation = notional / adv
    spread_cost   = 0.5 * bid_ask_spread * notional
    impact_cost   = eta * volatility * np.sqrt(participation) * notional
    total_cost    = spread_cost + impact_cost
    total_bps     = (total_cost / max(notional, 1.0)) * 10_000.0

    return PreTradeCostEstimate(
        notional=notional,
        spread_cost=spread_cost,
        impact_cost=impact_cost,
        total_cost=total_cost,
        total_bps=total_bps,
    )
