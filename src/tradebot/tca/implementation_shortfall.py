# src/tradebot/tca/implementation_shortfall.py
"""Implementation Shortfall (IS) decomposition.

IS = (arrival_price - execution_price) * signed_qty

Decomposition:
  delay_component   = (decision_price - arrival_price) * qty
  market_impact     = (execution_price - arrival_price) * qty
  timing_component  = (VWAP - arrival_price) * qty  (limit orders only)

Reference: Perold (1988) "The Implementation Shortfall: Paper versus Reality".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = ["ISDecomposition", "VwapSource", "decompose_implementation_shortfall"]


# CHIEF AUDIT-FIX (Sim-to-Reality #8):
#   ``vwap_price`` was previously accepted as a bare float with no provenance
#   marker.  Crypto has NO consolidated tape, so the upstream feed often
#   supplies a *mid-price* VWAP (sum(mid * vol)/sum(vol)) instead of the
#   actual trade-print VWAP from the exchange.  That introduces a
#   systematic 10-50 bps bias in the `timing` component and silently
#   corrupts every TCA report.  We now require the caller to label the
#   VWAP source.  A "mid" source triggers a loud warning so post-trade
#   review can flag affected reports.
VwapSource = Literal["trade_print", "mid_vwap", "unknown"]


@dataclass(frozen=True)
class ISDecomposition:
    """Full IS decomposition for a single order."""

    symbol: str
    total_is: float            # total implementation shortfall
    delay_component: float     # cost of decision lag (alpha decay)
    market_impact: float       # cost of price movement during execution
    timing_component: float    # residual (limit order timing, opportunity cost)
    total_bps: float           # total_is in basis points
    vwap_source: VwapSource = "unknown"
    timing_is_biased: bool = False  # True iff vwap_source != "trade_print"


def decompose_implementation_shortfall(
    symbol: str,
    decision_price: float,
    arrival_price: float,
    execution_price: float,
    signed_qty: float,
    vwap_price: float | None = None,
    vwap_source: VwapSource = "unknown",
) -> ISDecomposition:
    """Decompose implementation shortfall into delay, impact, and timing.

    Parameters
    ----------
    symbol :
        Ticker.
    decision_price :
        Price at the time of the investment decision (e.g. prior-day close).
    arrival_price :
        Mid-price at order submission time.
    execution_price :
        Volume-weighted average execution price.
    signed_qty :
        Signed quantity in base currency (positive = buy).
    vwap_price :
        VWAP over the execution window.  If None, timing component = 0.
    vwap_source :
        Provenance of ``vwap_price``.  Must be one of:
          * ``"trade_print"`` — actual exchange trade-print VWAP (unbiased).
          * ``"mid_vwap"``    — Σ(mid × vol)/Σvol fallback (~10-50 bps bias).
          * ``"unknown"``     — legacy callers; behaviour unchanged but a
            warning is emitted because the timing component cannot be trusted.

        Crypto has no consolidated tape; many feeds emit mid-VWAP by default.
        ``timing_is_biased=True`` is propagated to downstream consumers so
        biased reports can be filtered or annotated.

    Returns
    -------
    ISDecomposition.
    """
    # Sign convention (Perold 1988): costs are NEGATIVE (they reduce P&L).
    # For a buy: executing above arrival means cost → negative P&L impact.

    # Delay: alpha decay between decision and arrival (negative if price moved against)
    delay = (decision_price - arrival_price) * signed_qty

    # Market impact: P&L cost of executing vs arrival (negative when buy exec > arrival)
    impact = (arrival_price - execution_price) * signed_qty

    # Timing / opportunity cost (vs VWAP) — residual
    biased = False
    if vwap_price is not None and abs(signed_qty) > 0:
        timing = (vwap_price - arrival_price) * signed_qty - impact
        if vwap_source == "mid_vwap":
            biased = True
            logger.warning(
                "IS[%s]: vwap_source='mid_vwap' — timing component carries "
                "10-50 bps systematic bias vs. exchange trade-print VWAP. "
                "Crypto has no consolidated tape; supply trade_print VWAP "
                "from the actual exchange aggregate feed for unbiased TCA.",
                symbol,
            )
        elif vwap_source == "unknown":
            biased = True
            logger.warning(
                "IS[%s]: vwap_source='unknown' — provenance not declared. "
                "Pass vwap_source='trade_print' or 'mid_vwap' to make the "
                "bias state explicit.",
                symbol,
            )
    else:
        timing = 0.0

    total_is = delay + impact + timing
    basis = abs(signed_qty) * max(arrival_price, 1e-9)
    total_bps = total_is / basis * 10_000.0

    return ISDecomposition(
        symbol=symbol,
        total_is=total_is,
        delay_component=delay,
        market_impact=impact,
        timing_component=timing,
        total_bps=total_bps,
        vwap_source=vwap_source,
        timing_is_biased=biased,
    )
