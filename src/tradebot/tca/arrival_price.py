# src/tradebot/tca/arrival_price.py
"""Arrival-price benchmark computation.

The arrival price is the mid-price at the moment the trading decision is made
(signal fires).  It is the most common benchmark for market orders.

Reference: Kissell & Glantz (2003) "Optimal Trading Strategies".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["ArrivalPriceBenchmark", "compute_arrival_price_benchmark"]


@dataclass(frozen=True)
class ArrivalPriceBenchmark:
    """Arrival-price benchmark metrics for a single order."""

    symbol: str
    decision_time: pd.Timestamp
    arrival_price: float   # mid-price at decision time
    execution_price: float
    signed_qty: float      # positive = buy, negative = sell
    implementation_shortfall: float  # (arrival - exec) * signed_qty
    slippage_bps: float              # IS in basis points


def compute_arrival_price_benchmark(
    symbol: str,
    decision_time: pd.Timestamp,
    execution_price: float,
    signed_qty: float,
    price_series: pd.Series,
    arrival_offset: int = 0,
) -> ArrivalPriceBenchmark:
    """Compute arrival-price IS for a single execution.

    Parameters
    ----------
    symbol :
        Ticker.
    decision_time :
        Timestamp of the trading signal / decision.
    execution_price :
        Average execution price of the order.
    signed_qty :
        Order quantity (positive = buy).
    price_series :
        Time series of mid-prices with UTC DatetimeIndex.
    arrival_offset :
        Number of bars to look back from ``decision_time`` to find the
        arrival price (0 = use the decision bar itself).

    Returns
    -------
    ArrivalPriceBenchmark.
    """
    if decision_time in price_series.index:
        arrival_price = float(price_series.loc[decision_time])
    else:
        before = price_series[price_series.index <= decision_time]
        if before.empty:
            arrival_price = execution_price
        else:
            arrival_price = float(before.iloc[-1])

    if arrival_offset > 0:
        idx = price_series.index.get_indexer([decision_time], method="ffill")[0]
        idx = max(0, idx - arrival_offset)
        arrival_price = float(price_series.iloc[idx])

    if arrival_price <= 0:
        arrival_price = execution_price

    is_val = (arrival_price - execution_price) * signed_qty
    slippage_bps = (is_val / (abs(signed_qty) * arrival_price + 1e-9)) * 10_000.0

    return ArrivalPriceBenchmark(
        symbol=symbol,
        decision_time=decision_time,
        arrival_price=arrival_price,
        execution_price=execution_price,
        signed_qty=signed_qty,
        implementation_shortfall=is_val,
        slippage_bps=slippage_bps,
    )
