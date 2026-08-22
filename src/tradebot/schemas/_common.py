"""Shared Pandera validators and Config base used by all schemas."""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa


class StrictConfig:
    """Strict Pandera config: no extra cols, no silent coercion."""
    strict = True
    coerce = False
    ordered = False


def monotonic_increasing_index(idx: pd.Index) -> bool:
    return bool(idx.is_monotonic_increasing)


def no_all_nan(s: pd.Series) -> pd.Series:
    assert not s.isna().all(), f"Column {s.name!r} is entirely NaN"
    return s


def is_tz_aware_utc(idx: pd.DatetimeIndex) -> bool:
    return idx.tzinfo is not None


check_monotonic = pa.Check(
    lambda s: s.index.is_monotonic_increasing,
    element_wise=False,
    error="Index must be monotonically increasing",
)
