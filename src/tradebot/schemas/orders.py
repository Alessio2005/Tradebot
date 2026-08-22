# src/tradebot/schemas/orders.py
"""Pandera schema for OMS order/fill audit-log records (R-8).

Validates the 14-field JSONL audit trail as a DataFrame for downstream
TCA analysis and compliance reporting.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class OrderAuditSchema(pa.DataFrameModel):
    """Schema for a flattened audit-log DataFrame (one row per fill).

    # NOTE (Wave 14 P0): OrderAuditSchema.side uses "BUY"/"SELL" per Bybit API.
    # Internal signals use "LONG"/"SHORT" (OOSPredictionSchema) and Int8 ±1 (labels).
    # Conversion happens at the execution boundary in execution_controller.py.
    # Do NOT change this schema — it mirrors the exchange audit trail format.
    """

    event_ts: Series[str] = pa.Field(str_matches=r"\d{4}-\d{2}-\d{2}T")
    order_id: Series[str] = pa.Field(str_matches=r"ord_")
    symbol: Series[str] = pa.Field(str_length={"min_value": 3})
    side: Series[str] = pa.Field(isin=["BUY", "SELL"])
    qty_base: Series[float] = pa.Field(gt=0.0)
    notional_usdt: Series[float] = pa.Field(gt=0.0)
    order_type: Series[str] = pa.Field(isin=["MARKET", "LIMIT"])
    signal_prob: Series[float] = pa.Field(ge=0.0, le=1.0)
    kelly_fraction: Series[float] = pa.Field(ge=0.0, le=1.0)
    model_version: Series[str] = pa.Field()
    git_sha: Series[str] = pa.Field()
    feature_hash: Series[str] = pa.Field()
    portfolio_weight: Series[float] = pa.Field(ge=0.0, le=1.0)
    fill_price: Series[float] = pa.Field(gt=0.0)

    class Config:
        strict = False  # allow optional extra columns
        coerce = False
