# src/tradebot/utils/time.py
"""UTC-only time utilities for the live trading stack.

All timestamps in tradebot are UTC-aware pd.Timestamp.  This module
provides conversion helpers that enforce that invariant at system boundaries
(user input, exchange API responses, config files).
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

__all__ = [
    "now_utc",
    "to_utc",
    "bar_cadence_seconds",
    "align_to_bar",
    "seconds_since",
    "asof_join",
]

_BAR_CADENCES: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def now_utc() -> pd.Timestamp:
    """Return the current UTC time as a timezone-aware pd.Timestamp."""
    return pd.Timestamp.now(tz="UTC")


def to_utc(ts: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert any timestamp representation to a UTC-aware pd.Timestamp.

    Raises ValueError if the input has no timezone and cannot be assumed UTC.
    Naive datetime inputs are assumed UTC (exchange convention).
    """
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return pd.Timestamp(ts, tz="UTC")
        return pd.Timestamp(ts).tz_convert("UTC")

    # string — let pandas parse, then enforce UTC
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def bar_cadence_seconds(interval: str) -> int:
    """Return the number of seconds in a bar interval string (e.g. ``"1h"``).

    Raises KeyError for unknown interval strings.
    """
    if interval not in _BAR_CADENCES:
        raise KeyError(
            f"Unknown interval {interval!r}. Known: {sorted(_BAR_CADENCES)}"
        )
    return _BAR_CADENCES[interval]


def align_to_bar(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Floor a timestamp to the nearest bar boundary for the given interval."""
    cadence = bar_cadence_seconds(interval)
    epoch_s = int(ts.timestamp())
    floored_s = (epoch_s // cadence) * cadence
    return pd.Timestamp(floored_s, unit="s", tz="UTC")


def seconds_since(ts: pd.Timestamp) -> float:
    """Return the number of seconds elapsed since ``ts`` (UTC)."""
    return (now_utc() - ts).total_seconds()


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    asof_col: str = "asof_ts",
    by: str | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Join ``right`` onto ``left`` using only information AVAILABLE at t.

    For every row of ``left`` (UTC DatetimeIndex = decision time t), take the
    most recent row of ``right`` whose ``asof_col`` (availability moment,
    Mandate v3 §7) is STRICTLY <= t. Rows of ``right`` published after t can
    never leak in — this is the single sanctioned way to merge PIT sources
    (R-1; tested in tests/lookahead/).

    Parameters
    ----------
    left  : decision-time frame, tz-aware UTC DatetimeIndex.
    right : PIT frame with ``asof_col`` tz-aware UTC column.
    by    : optional key column (e.g. ``symbol``) present in both frames for
            grouped as-of joins; ``left`` must then carry it as a column.
    suffix: appended to right-hand column names on collision.
    """
    if not isinstance(left.index, pd.DatetimeIndex) or left.index.tz is None:
        raise ValueError("left must have a tz-aware UTC DatetimeIndex")
    s = right[asof_col]
    if not pd.api.types.is_datetime64_any_dtype(s) or s.dt.tz is None:
        raise ValueError(f"right[{asof_col!r}] must be tz-aware UTC datetimes")

    lf = left.reset_index().rename(columns={left.index.name or "index": "_t"})
    rf = right.sort_values(asof_col, kind="stable")
    overlap = (set(lf.columns) & set(rf.columns)) - {asof_col, by}
    if overlap:
        rf = rf.rename(columns={c: f"{c}{suffix or '_r'}" for c in overlap})
    merged = pd.merge_asof(
        lf.sort_values("_t", kind="stable"),
        rf,
        left_on="_t",
        right_on=asof_col,
        by=by,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.set_index("_t").rename_axis(left.index.name)
