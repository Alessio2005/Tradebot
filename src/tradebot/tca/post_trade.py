# src/tradebot/tca/post_trade.py
"""Post-trade TCA — realised vs expected cost comparison."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .implementation_shortfall import ISDecomposition, decompose_implementation_shortfall
from .pre_trade import PreTradeCostEstimate, estimate_pre_trade_cost

logger = logging.getLogger(__name__)

__all__ = ["PostTradeRecord", "analyse_execution"]


@dataclass(frozen=True)
class PostTradeRecord:
    """Post-trade analysis of a single executed order."""

    symbol: str
    execution_time: pd.Timestamp
    signed_qty: float
    execution_price: float
    expected_cost: PreTradeCostEstimate
    is_decomp: ISDecomposition
    cost_vs_expected_bps: float   # actual - expected (positive = worse than expected)
    alpha_capture: float          # fraction of pre-trade signal captured post-cost


def analyse_execution(
    symbol: str,
    execution_time: pd.Timestamp,
    signed_qty: float,
    execution_price: float,
    decision_price: float,
    arrival_price: float,
    volatility: float,
    adv: float,
    bid_ask_spread: float = 0.0002,
    vwap_price: float | None = None,
    pre_trade_signal: float = 0.0,
) -> PostTradeRecord:
    """Perform post-trade analysis for a single executed order.

    Parameters
    ----------
    symbol, execution_time, signed_qty, execution_price :
        Order identification and execution details.
    decision_price :
        Price at time of investment decision.
    arrival_price :
        Mid-price at order submission.
    volatility :
        Daily realised volatility (same units as pre-trade estimate).
    adv :
        Average daily volume.
    bid_ask_spread :
        Estimated bid-ask spread (fraction).
    vwap_price :
        VWAP over execution window (optional).
    pre_trade_signal :
        Alpha signal strength at decision time (used for alpha-capture calc).

    Returns
    -------
    PostTradeRecord.
    """
    notional = abs(signed_qty) * abs(execution_price)
    expected = estimate_pre_trade_cost(
        notional=notional,
        volatility=volatility,
        adv=adv,
        bid_ask_spread=bid_ask_spread,
    )

    is_decomp = decompose_implementation_shortfall(
        symbol=symbol,
        decision_price=decision_price,
        arrival_price=arrival_price,
        execution_price=execution_price,
        signed_qty=signed_qty,
        vwap_price=vwap_price,
    )

    actual_bps = is_decomp.total_bps
    cost_vs_expected = actual_bps - expected.total_bps

    # Alpha capture: fraction of signal retained after costs
    if abs(pre_trade_signal) > 1e-9 and notional > 0:
        cost_as_signal = is_decomp.total_is / (notional + 1e-9)
        alpha_capture = float(np.clip(1.0 - cost_as_signal / (abs(pre_trade_signal) + 1e-9), 0.0, 2.0))
    else:
        alpha_capture = 1.0

    return PostTradeRecord(
        symbol=symbol,
        execution_time=execution_time,
        signed_qty=signed_qty,
        execution_price=execution_price,
        expected_cost=expected,
        is_decomp=is_decomp,
        cost_vs_expected_bps=cost_vs_expected,
        alpha_capture=alpha_capture,
    )
