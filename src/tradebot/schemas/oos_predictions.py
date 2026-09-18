"""Pandera schema for Stage 3 (train_cpcv) OOS probability output."""
from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series


class OOSPredictionSchema(pa.DataFrameModel):
    """One row per OOS bar × fold."""
    ts: Index[pa.DateTime] = pa.Field(check_name=True)
    prob: Series[float] = pa.Field(ge=0.0, le=1.0)
    sigma: Series[float] = pa.Field(gt=0.0)
    fold_id: Series[int] = pa.Field(ge=0)
    side: Series[str] = pa.Field(isin=["LONG", "SHORT"])

    class Config:
        strict = True
        coerce = False
        ordered = False

    @pa.dataframe_check
    @classmethod
    def index_is_utc(cls, df: DataFrame[Any]) -> bool:
        """P0-23: All DatetimeIndex must be UTC-aware."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return True  # let pandera's Index[pa.typing.DateTime] catch this
        return df.index.tz is not None and str(df.index.tz) in {"UTC", "tzutc()", "UTC+00:00"}
