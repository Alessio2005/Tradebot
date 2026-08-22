# src/tradebot/featurestore/backfill.py
"""One-shot initialisation of the FeatureStore from historical feature Parquet files.

Usage:
    python -c "
    from tradebot.featurestore.backfill import backfill_from_directory
    backfill_from_directory('artefacts/features/', 'artefacts/feature_store/')
    "
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .store import FeatureStore

logger = logging.getLogger(__name__)

__all__ = ["backfill_from_directory", "backfill_from_dataframe"]


def backfill_from_dataframe(
    df: pd.DataFrame,
    symbol: str,
    store: FeatureStore,
) -> int:
    """Append all rows in ``df`` to ``store`` under ``symbol``.

    Returns the number of rows written.
    """
    store.append(df, symbol)
    logger.info("backfill: symbol=%s rows=%d", symbol, len(df))
    return len(df)


def backfill_from_directory(
    features_dir: str | Path,
    store_root: str | Path,
    glob: str = "*.parquet",
    symbol_from_stem: bool = True,
) -> dict[str, int]:
    """Populate a FeatureStore from a directory of per-symbol Parquet files.

    Parameters
    ----------
    features_dir :
        Directory containing ``<symbol>.parquet`` files (one per symbol).
    store_root :
        Root directory for the FeatureStore.
    glob :
        Filename pattern to match.
    symbol_from_stem :
        If True, derive the symbol name from the file stem (e.g.
        ``BTCUSDT.parquet`` → ``"BTCUSDT"``).

    Returns
    -------
    dict mapping symbol → number of rows written.
    """
    store = FeatureStore(store_root)
    features_dir = Path(features_dir)
    results: dict[str, int] = {}

    for path in sorted(features_dir.glob(glob)):
        symbol = path.stem if symbol_from_stem else path.name
        df = pd.read_parquet(path)
        n = backfill_from_dataframe(df, symbol, store)
        results[symbol] = n

    logger.info("backfill complete: %d symbols", len(results))
    return results
