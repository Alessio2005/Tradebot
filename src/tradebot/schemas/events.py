"""EventSchema — CUSUM-event-driven dataset.  Stage 1 → Stage 2 boundary.

Templated from blueprint Appendix A.  One row per event-bar.
  • feat_* columns   → model inputs
  • aux columns      → labeling / backtest only (close, t1, funding, etc.)

Pragmatic schema:
  • strict = "filter"  → extra feat_* columns kept; required cols enforced.
  • bid/ask cols are NOT required (mid-price-only data is the common case).
  • Index dtype validated via dataframe_check (tz-aware AND tz-naive both OK).
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class EventSchema(pa.DataFrameModel):
    """CUSUM-event-driven dataset.  Output of Stage 1 build_features."""

    # ── Required: OHLCV (labeling + backtest) ─────────────────────────────────
    close:       Series[float] = pa.Field(gt=0, nullable=False)
    high:        Series[float] = pa.Field(gt=0, nullable=False)
    low:         Series[float] = pa.Field(gt=0, nullable=False)
    open:        Series[float] = pa.Field(gt=0, nullable=False)
    volume:      Series[float] = pa.Field(ge=0, nullable=False)

    # Garman-Klass volatility proxy (required by labelers and backtests).
    feat_vol_gk: Series[float] = pa.Field(gt=0, nullable=False)

    class Config:
        strict  = "filter"
        coerce  = False
        ordered = False

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

    @pa.dataframe_check
    @classmethod
    def feat_columns_present(cls, df: DataFrame[Any]) -> bool:
        """At least one feat_* column must be present."""
        return any(c.startswith("feat_") for c in df.columns)

    @pa.dataframe_check
    @classmethod
    def no_all_nan_feat_columns(cls, df: DataFrame[Any]) -> bool:
        """No feat_* column may be entirely NaN — indicates upstream failure."""
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        for col in feat_cols:
            if df[col].isna().all():
                return False
        return True


# Backward-compat alias (schemas/__init__.py public API)
EventsSchema = EventSchema
