# src/tradebot/risk/liquidity_risk.py
"""Liquidity risk — bid-ask spread × position × volatility proxy.

Estimates the cost and risk of being unable to exit a position under
adverse market conditions.

Metrics:
  - Liquidation cost: spread × notional (minimum exit cost)
  - Liquidation horizon: days to exit at 20% ADV participation
  - Liquidity-adjusted VaR (LVaR): VaR + liquidation cost over horizon
  - Market-impact risk: stress scenario where spread widens 5×
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["LiquidityRiskAssessment", "assess_liquidity_risk", "liquidity_adjusted_var"]


@dataclass(frozen=True)
class LiquidityRiskAssessment:
    """Liquidity risk metrics for a single position or portfolio."""

    symbol: str
    notional: float
    bid_ask_spread_bps: float
    liquidation_cost: float          # notional * spread
    liquidation_days: float          # notional / (participation * ADV) / trading_days
    normal_lvar_bps: float           # VaR + liquidation cost (bps)
    stressed_lvar_bps: float         # VaR + 5× liquidation cost (bps)
    liquidity_score: float           # [0, 1] — 1 = highly liquid


def assess_liquidity_risk(
    symbol: str,
    notional: float,
    adv: float,
    bid_ask_spread_bps: float = 5.0,
    var_99_bps: float = 200.0,
    participation_rate: float = 0.20,
    stress_spread_multiplier: float = 5.0,
) -> LiquidityRiskAssessment:
    """Compute liquidity risk metrics for a position.

    Parameters
    ----------
    symbol : ticker
    notional : position size in USD
    adv : average daily volume in USD
    bid_ask_spread_bps : current bid-ask spread in bps
    var_99_bps : 1-day 99% VaR in bps (from market risk model)
    participation_rate : fraction of ADV to trade per day (default 20%)
    stress_spread_multiplier : spread multiplier in stress scenario

    Returns
    -------
    LiquidityRiskAssessment.
    """
    adv = max(adv, 1.0)
    spread_frac = bid_ask_spread_bps / 10_000.0
    liquidation_cost = notional * spread_frac

    # Days to exit at participation_rate × ADV
    daily_capacity = participation_rate * adv
    liquidation_days = notional / max(daily_capacity, 1.0)

    # LVaR = VaR + daily amortised liquidation cost over horizon
    daily_liq_cost_bps = bid_ask_spread_bps / max(liquidation_days, 1.0)
    normal_lvar = var_99_bps + daily_liq_cost_bps
    stressed_lvar = var_99_bps + daily_liq_cost_bps * stress_spread_multiplier

    # Liquidity score: 1 = exit in < 0.5 days; 0 = exit takes > 30 days
    raw_score = 1.0 - float(np.clip(liquidation_days / 30.0, 0.0, 1.0))
    # Penalise for wide spread
    spread_penalty = float(np.clip(bid_ask_spread_bps / 100.0, 0.0, 0.5))
    liquidity_score = float(np.clip(raw_score - spread_penalty, 0.0, 1.0))

    return LiquidityRiskAssessment(
        symbol=symbol,
        notional=notional,
        bid_ask_spread_bps=bid_ask_spread_bps,
        liquidation_cost=liquidation_cost,
        liquidation_days=liquidation_days,
        normal_lvar_bps=normal_lvar,
        stressed_lvar_bps=stressed_lvar,
        liquidity_score=liquidity_score,
    )


def liquidity_adjusted_var(
    weights: pd.Series,
    positions_usd: pd.Series,
    adv_map: pd.Series,
    spread_map: pd.Series,
    var_map: pd.Series,
    confidence: float = 0.99,
    participation_rate: float = 0.20,
) -> pd.DataFrame:
    """Compute Liquidity-adjusted VaR for a portfolio.

    Parameters
    ----------
    weights : portfolio weights indexed by symbol
    positions_usd : notional position sizes in USD
    adv_map : average daily volume per symbol
    spread_map : bid-ask spread in bps per symbol
    var_map : individual 1-day VaR in bps per symbol
    confidence : VaR confidence level
    participation_rate : max daily trading as fraction of ADV

    Returns
    -------
    DataFrame with per-asset liquidity risk metrics and portfolio total.
    """
    symbols = weights.index.tolist()
    records = []

    for sym in symbols:
        notional = float(positions_usd.get(sym, 0.0))
        if notional <= 0:
            continue
        assessment = assess_liquidity_risk(
            symbol=sym,
            notional=notional,
            adv=float(adv_map.get(sym, notional)),
            bid_ask_spread_bps=float(spread_map.get(sym, 5.0)),
            var_99_bps=float(var_map.get(sym, 200.0)),
            participation_rate=participation_rate,
        )
        records.append({
            "symbol":            sym,
            "weight":            float(weights.get(sym, 0.0)),
            "notional_usd":      assessment.notional,
            "spread_bps":        assessment.bid_ask_spread_bps,
            "liq_cost_usd":      assessment.liquidation_cost,
            "liq_days":          assessment.liquidation_days,
            "lvar_bps":          assessment.normal_lvar_bps,
            "stressed_lvar_bps": assessment.stressed_lvar_bps,
            "liq_score":         assessment.liquidity_score,
        })

    return pd.DataFrame(records).set_index("symbol") if records else pd.DataFrame()
