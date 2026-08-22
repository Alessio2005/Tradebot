# src/tradebot/featurestore/store.py
"""Append-only Parquet-partitioned feature cache.

Partition scheme: <root>/<symbol>/year=YYYY/month=MM/*.parquet

Guarantees
----------
- No duplicate timestamps per symbol (upsert semantics via pandas dedup).
- Schema validated via FeatureStoreSchema on every write.
- Causal fence: callers must never pass feature rows where the timestamp
  reflects information from bars t' > t (enforced by contract, not code).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

__all__ = ["FeatureStore"]

_PARTITION_COLS = ["symbol", "year", "month"]


class FeatureStore:
    """Append-only, symbol-partitioned feature cache backed by Parquet.

    Parameters
    ----------
    root : Path
        Root directory for the store.  Created on first write.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(self, df: pd.DataFrame, symbol: str) -> None:
        """Upsert ``df`` rows for ``symbol`` into the store.

        Rows are keyed by their DatetimeIndex.  Existing rows with the
        same timestamp are overwritten (upsert).  ``df`` must have a
        timezone-aware UTC DatetimeIndex.
        """
        if df.empty:
            return

        df = df.copy()
        df["symbol"] = symbol
        df["year"]   = df.index.year.astype("int32")
        df["month"]  = df.index.month.astype("int32")

        for (yr, mo), chunk in df.groupby(["year", "month"]):
            part_dir = self.root / symbol / f"year={yr}" / f"month={mo}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_file = part_dir / "data.parquet"

            if part_file.exists():
                existing = pd.read_parquet(part_file)
                merged = pd.concat([existing, chunk])
                # CHIEF AUDIT 2026-05-23 (FIX 6 / silent feature overwrite):
                # ``keep="last"`` upsert means a re-run of the same period
                # over-writes the previously cached feature values without
                # any audit trail.  If a feature definition or its upstream
                # changes between runs the model trained on the cached data
                # is silently no longer reproducible.  We still upsert
                # (backwards-compatible, no breaking change) but emit a
                # warning so the operator can investigate.
                n_existing = len(existing)
                n_chunk    = len(chunk)
                merged = merged[~merged.index.duplicated(keep="last")]
                merged.sort_index(inplace=True)
                n_merged = len(merged)
                # Number of rows that existed in BOTH frames (i.e. silently
                # overwritten) = (rows before dedup) - (rows after dedup).
                overlap = (n_existing + n_chunk) - n_merged
                if overlap > 0:
                    overlap_index = existing.index.intersection(chunk.index)
                    if len(overlap_index) > 0:
                        first_ts = overlap_index.min()
                        last_ts  = overlap_index.max()
                    else:
                        first_ts = last_ts = None
                    logger.warning(
                        "FeatureStore.append: silent upsert overwrote %d "
                        "existing bar(s) for symbol=%s partition=%04d-%02d "
                        "(first=%s, last=%s, new=%d, kept-from-new=last). "
                        "Re-runs of the same period are now non-deterministic — "
                        "verify the new feature values are intentional.",
                        overlap, symbol, yr, mo, first_ts, last_ts, n_chunk,
                    )
            else:
                merged = chunk.sort_index()

            merged.to_parquet(part_file, engine="pyarrow", index=True)

        logger.debug("FeatureStore.append: symbol=%s rows=%d", symbol, len(df))

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def read(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Read features for ``symbol`` in [start, end].

        Returns empty DataFrame if no data exists for the requested window.
        """
        sym_root = self.root / symbol
        if not sym_root.exists():
            return pd.DataFrame()

        years  = range(start.year, end.year + 1)
        frames: list[pd.DataFrame] = []

        for yr in years:
            yr_dir = sym_root / f"year={yr}"
            if not yr_dir.exists():
                continue
            for mo_dir in sorted(yr_dir.iterdir()):
                part_file = mo_dir / "data.parquet"
                if not part_file.exists():
                    continue
                df_part = pd.read_parquet(part_file)
                frames.append(df_part)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        mask = (combined.index >= start) & (combined.index <= end)
        return combined.loc[mask]

    def latest_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        """Return the most recent timestamp stored for ``symbol``, or None."""
        sym_root = self.root / symbol
        if not sym_root.exists():
            return None

        latest: Optional[pd.Timestamp] = None
        for part_file in sym_root.rglob("data.parquet"):
            # Read only the index by reading one column and discarding it.
            df_part = pd.read_parquet(part_file)
            if df_part.empty:
                continue
            ts = df_part.index.max()
            if latest is None or ts > latest:
                latest = ts
        return latest
