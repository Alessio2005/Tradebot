# tests/lookahead/test_event_units_causality.py
"""G6 guards for event_to_panel + PEAD/quality event construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import eq_pead, eq_quality
from tradebot.alpha.xs_unit import event_to_panel

UTC = "UTC"


def _idx(n: int = 60) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B", tz=UTC)


def test_event_to_panel_strictly_asof_and_expires() -> None:
    idx = _idx()
    ev = pd.DataFrame(
        {"symbol": ["A"], "asof_ts": [idx[10] + pd.Timedelta(hours=18)],
         "value": [2.5]}
    )
    out = event_to_panel(ev, idx, pd.Index(["A", "B"]), expiry_bdays=5)
    assert out["A"].iloc[:11].isna().all()        # 18:00 after close day10 -> day11
    assert (out["A"].iloc[11:16] == 2.5).all()    # 5 bdays valid
    assert out["A"].iloc[16:].isna().all()        # expired
    assert out["B"].isna().all()


def test_event_to_panel_later_event_overrides() -> None:
    idx = _idx()
    ev = pd.DataFrame(
        {"symbol": ["A", "A"], "asof_ts": [idx[5], idx[8]], "value": [1.0, 9.0]}
    )
    out = event_to_panel(ev, idx, pd.Index(["A"]), expiry_bdays=20)
    assert out["A"].iloc[6] == 1.0
    assert out["A"].iloc[8] == 9.0


def test_pead_ar3_window_is_realised_before_asof() -> None:
    idx = _idx()
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (60, 12)), axis=0)),
        index=idx, columns=[f"S{i}" for i in range(12)],
    )
    acc = idx[20] + pd.Timedelta(hours=22)  # after close day 20
    filings = pd.DataFrame({"ticker": ["S0"], "form": ["10-Q"], "asof_ts": [acc]})
    ev = eq_pead.announcement_events(prices, filings)
    assert len(ev) == 1
    # d = day21; window days 21-23; asof = close day 23
    assert ev["asof_ts"].iloc[0] == idx[23]
    rets = prices.pct_change(fill_method=None)
    ab = (rets["S0"] - rets.mean(axis=1)).iloc[21:24].sum()
    assert ev["value"].iloc[0] == pytest.approx(float(ab))
    # perturbing prices AFTER day 23 must not change the event value
    bumped = prices.copy()
    bumped.iloc[30:] *= 1.5
    ev2 = eq_pead.announcement_events(bumped, filings)
    assert ev2["value"].iloc[0] == pytest.approx(ev["value"].iloc[0])


def test_quality_first_filing_wins_and_pit() -> None:
    pe = pd.Timestamp("2023-12-31", tz=UTC)
    fund = pd.DataFrame({
        "ticker": ["A"] * 4,
        "concept": ["GrossProfit", "Assets", "GrossProfit", "Assets"],
        "form": ["10-K", "10-K", "10-K", "10-K"],
        "event_ts": [pe] * 4,
        "asof_ts": [pd.Timestamp("2024-02-15", tz=UTC)] * 2
        + [pd.Timestamp("2025-02-15", tz=UTC)] * 2,   # re-disclosure a year later
        "value": [40.0, 100.0, 999.0, 100.0],
    })
    ev = eq_quality.gpa_events(fund)
    assert len(ev) == 1
    assert ev["value"].iloc[0] == pytest.approx(0.40)  # first filing, not 9.99
    assert ev["asof_ts"].iloc[0] == pd.Timestamp("2024-02-15", tz=UTC)
