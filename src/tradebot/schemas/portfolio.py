"""Pandera schema for Stage 4 (backtest_portfolio) output."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Index, Series


class PortfolioResultSchema(pa.DataFrameModel):
    """Equity-curve output of the portfolio backtester."""
    ts: Index[pa.typing.DateTime] = pa.Field(check_name=True)
    equity: Series[float] = pa.Field(gt=0.0)
    drawdown: Series[float] = pa.Field(ge=0.0, le=1.0)
    gross_leverage: Series[float] = pa.Field(ge=0.0)

    class Config:
        strict = False  # allow extra per-asset contribution columns
        coerce = False
        ordered = False
