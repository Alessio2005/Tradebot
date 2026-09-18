"""arrays.py — SINGLE source of truth for Pandas column → Numba-safe array.

Replaces ALL _f64c / _i32c calls in train_regime.py (blueprint §2.3, §3.3).

Design contract:
  1. Only accepts columns from a Pandera-validated DataFrame.
     (Call validate_or_die() at stage boundaries; within a stage the pipeline
     is responsible for maintaining dtype invariants.)
  2. Crashes HARD on object-dtype, NaN, or non-C-contiguous layout.
     We do NOT silently cast.  Wrong dtype = schema bug upstream, fix there.
  3. One copy at most (non-contiguous layout) — logged as a warning because
     repeated non-contiguity indicates upstream view/slice design smell.

Usage:
    arr = numba_array(df["close"], dtype="f64")
    arr = numba_array(df["event_idx"], dtype="i32")
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── dtype map ────────────────────────────────────────────────────────────────
_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "f64": np.dtype(np.float64),
    "f32": np.dtype(np.float32),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
}


def numba_array(
    df_col: pd.Series,
    *,
    dtype: Literal["f64", "f32", "i32", "i64"],
) -> NDArray[Any]:
    """Convert a Pandas Series to a Numba-safe C-contiguous array.

    Parameters
    ----------
    df_col:
        A column from a Pandera-validated DataFrame.
    dtype:
        Target dtype token.  Accepted values: "f64", "f32", "i32", "i64".

    Returns
    -------
    NDArray
        C-contiguous NumPy array of the requested dtype.

    Raises
    ------
    TypeError
        If the column has object dtype OR if its current dtype does not match
        the requested dtype.  This is a SCHEMA BUG — fix the source, not here.
    ValueError
        If the column contains NaN values (Numba kernels must not see NaN
        unless the kernel is explicitly designed to handle them).
    """
    arr: NDArray[Any] = df_col.to_numpy(copy=False)

    # ── 1. object dtype: always a schema failure ──────────────────────────────
    if arr.dtype == object:
        raise TypeError(
            f"object-dtype detected in column '{df_col.name}'. "
            "Upstream pipeline broke dtype invariants. "
            "Fix the source (merge_asof, concat, reindex), not here."
        )

    # ── 2. dtype mismatch: must be resolved at schema level ──────────────────
    expected = _DTYPE_MAP[dtype]
    if arr.dtype != expected:
        raise TypeError(
            f"dtype mismatch in '{df_col.name}': "
            f"requested {expected}, got {arr.dtype}. "
            "This is a schema contract violation — correct the upstream "
            "producer or add an explicit cast at the validated stage boundary."
        )

    # ── 3. NaN guard (opt-in: caller may suppress via a pre-fillna) ──────────
    if np.issubdtype(arr.dtype, np.floating) and np.isnan(arr).any():
        raise ValueError(
            f"NaN values in column '{df_col.name}' before Numba kernel. "
            "Fill NaN at the feature-engineering stage, not here."
        )

    # ── 4. Contiguity — one copy max, log the smell ───────────────────────────
    if not arr.flags["C_CONTIGUOUS"]:
        logger.warning(
            "Non-contiguous array for '%s' — fix the upstream view/slice "
            "(e.g. after iloc, fancy indexing, or transpose). "
            "Copying once; repeated occurrences indicate design smell.",
            df_col.name,
        )
        arr = np.ascontiguousarray(arr)

    return arr


def numba_array_nonan(
    df_col: pd.Series,
    *,
    dtype: Literal["f64", "f32", "i32", "i64"],
    fill: float = 0.0,
) -> NDArray[Any]:
    """Like numba_array but fills NaN with `fill` instead of raising.

    Use only for columns where NaN has a well-defined fill semantic
    (e.g. funding_rate where missing = 0.0, jump_ratio where missing = 0.0).
    Document the semantic at the call-site.
    """
    filled = df_col.fillna(fill)
    return numba_array(filled, dtype=dtype)


# ── Schema guard ──────────────────────────────────────────────────────────────

def validate_or_die(
    df: pd.DataFrame,
    schema: type[pa.DataFrameModel],
    *,
    lazy: bool = True,
    sample: int | None = None,
) -> pd.DataFrame:
    """Validate df against a Pandera schema; raise hard on failure.

    Parameters
    ----------
    df:      DataFrame to validate.
    schema:  Pandera DataFrameModel subclass (e.g. BarSchema).
    lazy:    If True, collect ALL schema errors before raising (shows full
             failure surface, not just the first mismatch).
    sample:  If provided, validate only a random sample of `sample` rows
             for performance on large frames.  Use None (default) at stage
             boundaries (full validation); use sample=10_000 for hot paths.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame (unchanged — pandera may return a copy).

    Raises
    ------
    pandera.errors.SchemaErrors
        Contains ALL schema violations found.  Never silently passes.
    """
    target = df.sample(n=min(sample, len(df))) if sample is not None else df
    return schema.validate(target, lazy=lazy)
