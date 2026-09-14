"""test_schemas.py — Pandera schema validation contract tests.

The DAG architecture relies on schemas as the single source of truth for
dtypes / column-presence at every stage boundary.  These tests pin that
contract: a valid frame passes; a perturbed frame fails *predictably*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors


def _make_bar_frame(n: int = 200, *, with_index: bool = True) -> pd.DataFrame:
    """Build a valid OHLCV frame: high >= max(O,C), low <= min(O,C)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC") if with_index else range(n)
    open_  = rng.uniform(99, 101, n)
    close_ = rng.uniform(99, 101, n)
    high_  = np.maximum(open_, close_) + rng.uniform(0.01, 0.5, n)
    low_   = np.minimum(open_, close_) - rng.uniform(0.01, 0.5, n)
    return pd.DataFrame({
        "open":   open_,
        "high":   high_,
        "low":    low_,
        "close":  close_,
        "volume": rng.uniform(1, 10, n),
    }, index=idx)


def test_bar_schema_accepts_clean_frame() -> None:
    from tradebot.schemas.bars import BarSchema
    df = _make_bar_frame()
    BarSchema.validate(df, lazy=True)


def test_event_schema_accepts_with_required_cols() -> None:
    from tradebot.schemas.events import EventSchema
    df = _make_bar_frame()
    df["feat_vol_gk"] = 0.01
    df["feat_dummy"]  = 0.0
    EventSchema.validate(df, lazy=True)


def test_validate_or_die_reraises_on_bad_dtype() -> None:
    """validate_or_die must escalate, never silently coerce."""
    from tradebot.schemas.bars import BarSchema
    from tradebot.utils.arrays import validate_or_die

    df = _make_bar_frame()
    df["close"] = df["close"].astype(object)  # wreck dtype
    # `pytest.raises(Exception)` stond hier: die slaagt ook op een typefout in
    # de testregel zelf. Gemeten welke uitzondering er valt: pandera's
    # SchemaErrors (meervoud, want lazy=True verzamelt).
    with pytest.raises(SchemaErrors):
        validate_or_die(df, BarSchema, lazy=True)


def test_numba_array_rejects_object_dtype() -> None:
    from tradebot.utils.arrays import numba_array
    s = pd.Series(["a", "b", "c"], name="bad")
    with pytest.raises(TypeError):
        numba_array(s, dtype="f64")


def test_numba_array_rejects_dtype_mismatch() -> None:
    from tradebot.utils.arrays import numba_array
    s = pd.Series(np.array([1, 2, 3], dtype=np.int64), name="x")
    with pytest.raises(TypeError):
        numba_array(s, dtype="f64")


def test_numba_array_passes_clean_input() -> None:
    from tradebot.utils.arrays import numba_array
    s = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float64), name="x")
    arr = numba_array(s, dtype="f64")
    assert arr.dtype == np.float64
    assert arr.flags["C_CONTIGUOUS"]
