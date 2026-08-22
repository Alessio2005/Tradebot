# src/tradebot/tca/__init__.py
"""Transaction Cost Analysis — pre-trade, post-trade, IS decomposition."""
from __future__ import annotations

from .arrival_price import ArrivalPriceBenchmark, compute_arrival_price_benchmark
from .implementation_shortfall import ISDecomposition, decompose_implementation_shortfall
from .post_trade import PostTradeRecord, analyse_execution
from .pre_trade import PreTradeCostEstimate, estimate_pre_trade_cost
from .report import TCAReport, build_tca_report

__all__ = [
    # pre-trade
    "PreTradeCostEstimate",
    "estimate_pre_trade_cost",
    # arrival price
    "ArrivalPriceBenchmark",
    "compute_arrival_price_benchmark",
    # IS
    "ISDecomposition",
    "decompose_implementation_shortfall",
    # post-trade
    "PostTradeRecord",
    "analyse_execution",
    # report
    "TCAReport",
    "build_tca_report",
]
