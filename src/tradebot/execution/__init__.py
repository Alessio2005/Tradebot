# src/tradebot/execution/__init__.py
"""Execution sub-package — market impact, slippage, fees, LOB simulator."""
from __future__ import annotations

from .fees import (
    BYBIT_PERP_DEFAULT,
    FeeSchedule,
    bybit_perp_schedule,
)
from .market_impact import (
    ImpactResult,
    RankQuantileScaler,
    cov_to_corr,
    ehlers_super_smoother,
    kalman_smoother_1d,
    ledoit_wolf_shrunk_corr,
    ledoit_wolf_shrunk_cov,
    max_size_for_alpha,
    negative_skew_crisis_multiplier,
    square_root_impact,
    trade_passes_impact_gate,
)
from .simulator import FillRecord, LOBSimulator, OrderRecord
from .slippage import SlippageModel, compute_slippage, fixed_bps_slippage

__all__ = [
    "BYBIT_PERP_DEFAULT",
    "FeeSchedule",
    "bybit_perp_schedule",
    "FillRecord",
    "ImpactResult",
    "LOBSimulator",
    "OrderRecord",
    "RankQuantileScaler",
    "SlippageModel",
    "compute_slippage",
    "cov_to_corr",
    "ehlers_super_smoother",
    "fixed_bps_slippage",
    "kalman_smoother_1d",
    "ledoit_wolf_shrunk_corr",
    "ledoit_wolf_shrunk_cov",
    "max_size_for_alpha",
    "negative_skew_crisis_multiplier",
    "square_root_impact",
    "trade_passes_impact_gate",
]
