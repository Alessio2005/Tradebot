"""Bar schemas — Stage 1 output boundary.

BarSchema          : raw OHLCV bars (tick-aggregated, before feature engineering).
ImbalanceBarSchema : volume-imbalance bars from bars.py (adds imbalance_ratio).

Design rules (blueprint §3.3):
  • strict = "filter"  → extra columns stripped; required cols enforced.
  • coerce = False     → no silent dtype casts; wrong dtype = hard crash here.
  • Index dtype validated via dataframe_check (handles both tz-naive and
    tz-aware DatetimeIndex without forcing one specific dtype string).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class BarSchema(pa.DataFrameModel):
    """OHLCV bar — canonical dtype contract for Bybit perps.

    Float columns: float64 (high precision needed for PnL arithmetic).
    Volume: float64 (BTC nominal volumes overflow float32).
    Index: DatetimeIndex (tz-aware OR tz-naive), monotonic-increasing, unique.
    """

    open:   Series[float] = pa.Field(gt=0, nullable=False)
    high:   Series[float] = pa.Field(gt=0, nullable=False)
    low:    Series[float] = pa.Field(gt=0, nullable=False)
    close:  Series[float] = pa.Field(gt=0, nullable=False)
    volume: Series[float] = pa.Field(ge=0, nullable=False)

    class Config:
        strict  = "filter"
        coerce  = False
        ordered = False

    # ── Index integrity ───────────────────────────────────────────────────────

    @pa.dataframe_check
    @classmethod
    def index_is_datetime(cls, df: DataFrame[Any]) -> bool:
        return isinstance(df.index, pd.DatetimeIndex)

    @pa.dataframe_check
    @classmethod
    def index_is_utc(cls, df: DataFrame[Any]) -> bool:
        """P0-23: All DatetimeIndex must be UTC-aware."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return True  # let index_is_datetime catch this
        return df.index.tz is not None and str(df.index.tz) in {"UTC", "tzutc()", "UTC+00:00"}

    @pa.dataframe_check
    @classmethod
    def index_monotonic_increasing(cls, df: DataFrame[Any]) -> bool:
        return bool(df.index.is_monotonic_increasing)

    @pa.dataframe_check
    @classmethod
    def index_unique(cls, df: DataFrame[Any]) -> bool:
        return bool(df.index.is_unique)

    # ── OHLC consistency (the only place these checks live) ──────────────────

    @pa.dataframe_check
    @classmethod
    def high_ge_low(cls, df: DataFrame[Any]) -> bool:
        return bool((df["high"] >= df["low"]).all())

    @pa.dataframe_check
    @classmethod
    def high_ge_open_close(cls, df: DataFrame[Any]) -> bool:
        return bool(
            (df["high"] >= df["open"]).all()
            and (df["high"] >= df["close"]).all()
        )

    @pa.dataframe_check
    @classmethod
    def low_le_open_close(cls, df: DataFrame[Any]) -> bool:
        return bool(
            (df["low"] <= df["open"]).all()
            and (df["low"] <= df["close"]).all()
        )


class ImbalanceBarSchema(BarSchema):
    """Volume-imbalance bar — adds derived fields from bars.py.

    Inherits all BarSchema fields and OHLC checks.
    """

    bar_count:       Series[float] = pa.Field(ge=1, nullable=False)
    dollar_volume:   Series[float] = pa.Field(ge=0, nullable=False)
    vwap:            Series[float] = pa.Field(gt=0, nullable=False)
    imbalance_ratio: Series[float] = pa.Field(ge=-1.0, le=1.0, nullable=False)

    class Config:
        strict  = "filter"
        coerce  = False
        ordered = False


# Backward-compat alias (schemas/__init__.py public API)
BarsSchema = BarSchema
