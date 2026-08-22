"""AssetTrackSchema — per-asset backtest result.  Stage 4 boundary.

One row per bar in the backtest window.  Produced by:
  backtest/per_side.py     → internal_backtest()
  backtest/bidirectional.py → bidirectional_backtest()

Then consumed by:
  backtest/portfolio.py    → multi-asset PortfolioBacktest
  risk/portfolio.py        → Kelly sizing + drawdown controls

Side encoding: +1 LONG, -1 SHORT, 0 FLAT (single source of truth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series


class AssetTrackSchema(pa.DataFrameModel):
    """Per-asset equity-curve track.

    Index: DatetimeIndex (bar timestamps), monotonic-increasing.
    One row per bar in the backtest period.
    """

    timestamp: Index[pa.DateTime] = pa.Field(unique=True, check_name=True)

    # ── Position & PnL ────────────────────────────────────────────────────────
    position:      Series[pa.Int8]  = pa.Field(isin=[-1, 0, 1], nullable=False)
    gross_ret:     Series[float]    = pa.Field(nullable=False)   # before costs
    net_ret:       Series[float]    = pa.Field(nullable=False)   # after spread+funding
    equity:        Series[float]    = pa.Field(gt=0, nullable=False)  # cumulative

    # ── Sizing ────────────────────────────────────────────────────────────────
    leverage:      Series[float]    = pa.Field(ge=0, nullable=False)
    notional_usd:  Series[float]    = pa.Field(ge=0, nullable=False)

    # ── Cost components ───────────────────────────────────────────────────────
    spread_cost:   Series[float]    = pa.Field(ge=0, nullable=False)
    funding_cost:  Series[float]    = pa.Field(nullable=False)   # can be negative (received)

    # ── Signal ────────────────────────────────────────────────────────────────
    prob_long:     Series[float]    = pa.Field(ge=0.0, le=1.0, nullable=True)
    prob_short:    Series[float]    = pa.Field(ge=0.0, le=1.0, nullable=True)
    meta_prob:     Series[float]    = pa.Field(ge=0.0, le=1.0, nullable=True)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    drawdown:      Series[float]    = pa.Field(le=0, nullable=False)  # always ≤ 0

    class Config:
        strict = "filter"   # allow extra diagnostic columns from backtest runs
        coerce = False
        ordered = True

    @pa.dataframe_check
    def index_monotonic_increasing(cls, df: DataFrame) -> bool:
        return bool(df.index.is_monotonic_increasing)

    @pa.dataframe_check
    def index_is_utc(cls, df: DataFrame) -> bool:
        """P0-23: All DatetimeIndex must be UTC-aware."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return True  # let pandera's Index[pa.DateTime] catch this
        return df.index.tz is not None and str(df.index.tz) in {"UTC", "tzutc()", "UTC+00:00"}

    @pa.dataframe_check
    def equity_positive(cls, df: DataFrame) -> bool:
        return bool((df["equity"] > 0).all())

    @pa.dataframe_check
    def net_ret_finite(cls, df: DataFrame) -> bool:
        return bool(np.isfinite(df["net_ret"].to_numpy()).all())

    @pa.dataframe_check
    def drawdown_le_zero(cls, df: DataFrame) -> bool:
        return bool((df["drawdown"] <= 1e-10).all())   # 1e-10 float tolerance


# Backward-compat alias (schemas/__init__.py public API)
TrackSchema = AssetTrackSchema
