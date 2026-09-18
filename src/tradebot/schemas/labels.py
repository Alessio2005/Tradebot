"""LabelSchema — Triple Barrier + Meta-Label output.  Stage 1 → Stage 2.

Columns produced by:
  labeling/triple_barrier.py → barrier_label, t1_idx, realized_ret
  labeling/trend_scanning.py → tstat, best_span
  labeling/meta.py           → meta_label, meta_prob (after Stage 3)

Side encoding (SINGLE source of truth, blueprint §2.4):
  +1 → LONG
  -1 → SHORT
   0 → FLAT / no-trade

This eliminates the internal 2/0/1 encoding found in train_regime.py
_calc_non_overlapping_stats (noted as SK-candidate in session summary).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame, Index, Series


class LabelSchema(pa.DataFrameModel):
    """Triple Barrier label set, one row per CUSUM event.

    Produced at the end of Stage 1 and consumed by Stage 2 (HPO objective)
    and Stage 3 (CPCV training).
    """

    timestamp: Index[pa.DateTime] = pa.Field(unique=True, check_name=True)

    # ── Primary label ─────────────────────────────────────────────────────────
    # 1 = profit-taking barrier hit (positive label for meta-labeling)
    # 0 = stop-loss or time-out barrier
    barrier_label: Series[pa.Int8] = pa.Field(isin=[0, 1], nullable=False)

    # Side that triggered the event (-1 SHORT, +1 LONG)
    side: Series[pa.Int8] = pa.Field(isin=[-1, 1], nullable=False)

    # Bar-index of trade exit (t1).  int32 sufficient (max bars << 2^31).
    t1_idx: Series[pa.Int32] = pa.Field(ge=0, nullable=False)

    # Realised net return (after spread), float64 for PnL precision.
    realized_ret: Series[float] = pa.Field(nullable=False)

    # ── Trend scanner auxiliary ───────────────────────────────────────────────
    tstat:     Series[float] = pa.Field(nullable=True)   # may be NaN for filtered events
    best_span: Series[pa.Int32] = pa.Field(ge=1, nullable=True)

    # ── Sample weight (AFML ch.4 uniqueness-based) ───────────────────────────
    sample_weight: Series[float] = pa.Field(gt=0, nullable=False)

    # ── Optional meta-label (added by Stage 3, absent in Stage 1 output) ─────
    meta_label: Series[pa.Int8] = pa.Field(isin=[0, 1], nullable=True)
    meta_prob:  Series[float]   = pa.Field(ge=0.0, le=1.0, nullable=True)

    class Config:
        strict = "filter"   # meta columns added in Stage 3 → forward-compat
        coerce = False
        ordered = True

    @pa.dataframe_check
    @classmethod
    def index_monotonic_increasing(cls, df: DataFrame[Any]) -> bool:
        return bool(df.index.is_monotonic_increasing)

    @pa.dataframe_check
    @classmethod
    def index_is_utc(cls, df: DataFrame[Any]) -> bool:
        """P0-23: All DatetimeIndex must be UTC-aware."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return True  # let pandera's Index[pa.DateTime] catch this
        return df.index.tz is not None and str(df.index.tz) in {"UTC", "tzutc()", "UTC+00:00"}

    @pa.dataframe_check
    @classmethod
    def t1_ge_t0(cls, df: DataFrame[Any]) -> bool:
        """t1_idx must be ≥ event bar index (no backward barriers)."""
        t0_pos = df.index.get_indexer(df.index)   # 0,1,2,...
        return bool((df["t1_idx"].to_numpy() >= t0_pos).all())

    @pa.dataframe_check
    @classmethod
    def realized_ret_finite(cls, df: DataFrame[Any]) -> bool:
        return bool(np.isfinite(df["realized_ret"].to_numpy()).all())


# Backward-compat alias (schemas/__init__.py public API)
LabelsSchema = LabelSchema
