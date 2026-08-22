# src/tradebot/featurestore/schema.py
"""Pandera contract for feature-store output."""
from __future__ import annotations

import pandera.pandas as ppa

__all__ = ["FeatureStoreSchema"]


class FeatureStoreSchema(ppa.DataFrameModel):
    """Minimal schema for rows written to / read from the FeatureStore.

    Columns beyond those declared here are allowed (strict=False) because
    each symbol may have a different feature set.  The required columns are
    the causal-fence anchors that every store write must carry.
    """

    symbol: ppa.typing.Series[str] = ppa.Field(nullable=False)

    class Config:
        strict = False
        coerce = False
        ordered = False
