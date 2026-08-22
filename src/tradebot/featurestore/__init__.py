# src/tradebot/featurestore/__init__.py
"""FeatureStore — append-only Parquet-partitioned feature cache."""
from __future__ import annotations

from .backfill import backfill_from_dataframe, backfill_from_directory
from .schema import FeatureStoreSchema
from .store import FeatureStore

__all__ = [
    "FeatureStore",
    "FeatureStoreSchema",
    "backfill_from_dataframe",
    "backfill_from_directory",
]
