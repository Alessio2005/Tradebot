"""CPCVFoldSchema — CPCV fold metadata.  Stage 2 / Stage 3 boundary.

One row per (fold_id, split_id) combination.  Captures the train/test
bar-index ranges so that each DAG stage can independently reconstruct
the exact split without re-running the combinatorial generator.

Bar-space (not calendar-day) embargo and purge offsets are stored here
so downstream code never has to re-derive them — eliminating the
purge_days / embargo_days inconsistency diagnosed in blueprint §2.4.
"""
from __future__ import annotations

from typing import Any

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class CPCVFoldSchema(pa.DataFrameModel):
    """CPCV fold descriptor table.

    Index: RangeIndex (integer).
    One row per fold-split pair (N_groups × C(N,k) combinations).
    """

    # Fold identification
    fold_id:  Series[pa.Int32] = pa.Field(ge=0, nullable=False)
    split_id: Series[pa.Int32] = pa.Field(ge=0, nullable=False)

    # Side (+1 LONG / -1 SHORT) — folds are computed separately per side
    side: Series[pa.Int8] = pa.Field(isin=[-1, 1], nullable=False)

    # Train set boundaries (bar-index, inclusive)
    train_start: Series[pa.Int32] = pa.Field(ge=0, nullable=False)
    train_end:   Series[pa.Int32] = pa.Field(ge=0, nullable=False)

    # Test set boundaries (bar-index, inclusive)
    test_start: Series[pa.Int32] = pa.Field(ge=0, nullable=False)
    test_end:   Series[pa.Int32] = pa.Field(ge=0, nullable=False)

    # Effective purge/embargo applied (bar-space), stored for audit trail
    effective_embargo_bars: Series[pa.Int32] = pa.Field(ge=0, nullable=False)
    effective_purge_bars:   Series[pa.Int32] = pa.Field(ge=0, nullable=False)

    # Number of train/test events (after purge+embargo)
    n_train: Series[pa.Int32] = pa.Field(ge=1, nullable=False)
    n_test:  Series[pa.Int32] = pa.Field(ge=1, nullable=False)

    class Config:
        strict = True
        coerce = False
        ordered = True

    @pa.dataframe_check
    @classmethod
    def train_before_test_or_no_overlap(cls, df: DataFrame[Any]) -> bool:
        """train_end < test_start (purge gap in between).

        Note: CPCV can produce non-contiguous train sets; this check
        only enforces that the reported ranges don't invert.
        """
        return bool((df["train_end"] < df["test_start"]).all())

    @pa.dataframe_check
    @classmethod
    def embargo_covers_t_max(cls, df: DataFrame[Any]) -> bool:
        """embargo_bars must be > 0 — never zero (blueprint §4.2 step 3)."""
        return bool((df["effective_embargo_bars"] > 0).all())


# Backward-compat alias (schemas/__init__.py public API)
FoldsSchema = CPCVFoldSchema
