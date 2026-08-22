# tests/lookahead/test_pit_sources.py
"""Lookahead guards for the PIT data layer (Wave 20, step 0.2; G6).

All synthetic — no network. Per-source *fetch* smoke tests live in the
integration suite; THESE tests guard the contract that makes lookahead
impossible by construction: asof_ts stamping + asof_join.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.data.sources.base import ASOF_COL, EVENT_COL, stamp_asof, validate_pit
from tradebot.data.sources.wiki_constituents import build_membership_calendar
from tradebot.utils.time import asof_join

UTC = "UTC"


def _pit_frame(event_dates, lag_days=1, **cols) -> pd.DataFrame:
    df = pd.DataFrame({EVENT_COL: pd.to_datetime(event_dates, utc=True), **cols})
    return stamp_asof(df, pd.Timedelta(days=lag_days))


# ── contract guards ─────────────────────────────────────────────────────────

def test_validate_pit_rejects_asof_before_event() -> None:
    df = _pit_frame(["2024-01-02"], x=[1.0])
    df.loc[0, ASOF_COL] = df.loc[0, EVENT_COL] - pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="lookahead"):
        validate_pit(df)


def test_validate_pit_rejects_naive_timestamps() -> None:
    df = _pit_frame(["2024-01-02"], x=[1.0])
    df[EVENT_COL] = df[EVENT_COL].dt.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        validate_pit(df)


def test_stamp_asof_rejects_negative_lag() -> None:
    df = pd.DataFrame({EVENT_COL: pd.to_datetime(["2024-01-02"], utc=True)})
    with pytest.raises(ValueError, match=">= 0"):
        stamp_asof(df, pd.Timedelta(days=-1))


# ── asof_join: the single sanctioned merge ──────────────────────────────────

def test_asof_join_never_uses_future_rows() -> None:
    t = pd.date_range("2024-01-01", periods=5, freq="D", tz=UTC)
    left = pd.DataFrame({"sig": np.arange(5.0)}, index=t)
    right = _pit_frame(["2024-01-02"], lag_days=1, value=[10.0])
    out = asof_join(left, right)
    # asof = 2024-01-03 -> rows before that must be NaN
    assert out.loc["2024-01-01", "value"] != out.loc["2024-01-01", "value"]  # NaN
    assert np.isnan(out.loc["2024-01-02", "value"])
    assert out.loc["2024-01-03", "value"] == 10.0
    assert out.loc["2024-01-04", "value"] == 10.0


def test_asof_join_lag_shifts_availability() -> None:
    t = pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC)
    left = pd.DataFrame({"sig": np.zeros(10)}, index=t)
    for lag in (0, 1, 3):
        right = _pit_frame(["2024-01-05"], lag_days=lag, value=[1.0])
        out = asof_join(left, right)
        first = out["value"].first_valid_index()
        assert first == pd.Timestamp("2024-01-05", tz=UTC) + pd.Timedelta(days=lag)


def test_asof_join_grouped_by_symbol() -> None:
    t = pd.date_range("2024-01-01", periods=4, freq="D", tz=UTC)
    left = pd.DataFrame(
        {"symbol": ["A", "B"] * 4},
        index=t.repeat(2),
    )
    right = pd.concat(
        [
            _pit_frame(["2024-01-01"], lag_days=1, symbol=["A"], value=[1.0]),
            _pit_frame(["2024-01-02"], lag_days=1, symbol=["B"], value=[2.0]),
        ],
        ignore_index=True,
    )
    out = asof_join(left, right, by="symbol")
    a = out[out["symbol"] == "A"]["value"]
    b = out[out["symbol"] == "B"]["value"]
    assert np.isnan(a.iloc[0]) and a.iloc[1] == 1.0  # A known from Jan 2
    assert np.isnan(b.iloc[1]) and b.iloc[2] == 2.0  # B known from Jan 3


def test_asof_join_deterministic() -> None:
    t = pd.date_range("2024-01-01", periods=50, freq="D", tz=UTC)
    rng = np.random.default_rng(7)
    left = pd.DataFrame({"sig": rng.normal(size=50)}, index=t)
    right = _pit_frame(
        pd.date_range("2024-01-01", periods=20, freq="2D"),
        lag_days=2,
        value=rng.normal(size=20),
    )
    out1 = asof_join(left, right)
    out2 = asof_join(left, right)
    pd.testing.assert_frame_equal(out1, out2)  # R-5


# ── universe calendar: no membership before knowable ────────────────────────

def test_membership_calendar_no_lookahead() -> None:
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "action": ["add", "remove", "add"],
            EVENT_COL: pd.to_datetime(["2024-01-05", "2024-03-01", "2024-02-01"], utc=True),
            "date_unknown": [False, False, False],
        }
    )
    events[ASOF_COL] = events[EVENT_COL] + pd.Timedelta(days=1)
    dates = pd.date_range("2024-01-01", "2024-04-01", freq="D", tz=UTC)
    cal = build_membership_calendar(events, dates)

    # AAA: knowable from Jan 6, removed knowable Mar 2
    assert not cal.loc["2024-01-05", "AAA"]
    assert cal.loc["2024-01-06", "AAA"]
    assert cal.loc["2024-03-01", "AAA"]
    assert not cal.loc["2024-03-02", "AAA"]
    # BBB: knowable from Feb 2 onward
    assert not cal.loc["2024-02-01", "BBB"]
    assert cal.loc["2024-02-02", "BBB"]
    assert cal.loc["2024-04-01", "BBB"]
