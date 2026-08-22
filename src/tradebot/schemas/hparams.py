"""Pandera schema for Stage 2 (tune_hparams) output."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

from ._common import StrictConfig


class HParamsSchema(pa.DataFrameModel):
    """Validated hyperparameter record per (symbol, side) run."""
    symbol: Series[str] = pa.Field(isin=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    side: Series[str] = pa.Field(isin=["LONG", "SHORT"])
    learning_rate: Series[float] = pa.Field(gt=0.0, lt=1.0)
    depth: Series[int] = pa.Field(ge=4, le=10)
    l2_leaf_reg: Series[float] = pa.Field(gt=0.0)
    iterations: Series[int] = pa.Field(ge=50)
    oos_logloss: Series[float] = pa.Field(ge=0.0)

    class Config(StrictConfig):
        ordered = False
