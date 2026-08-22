# src/tradebot/data/sources/cboe.py
"""CBOE index histories — VIX / VIX3M term structure (free CSV).

Use ONLY as conditioner/de-grosser on equity sleeves (Mandate §5.4; F5
binds: market-wide vol data is falsified as a standalone directional timer,
breadth ≈ 1).

PIT: index close on day D -> conservative availability next calendar day
00:00 UTC (same convention as Stooq EOD).
"""
from __future__ import annotations

import pandas as pd

from .base import EVENT_COL, SourceMeta, http_get_text, read_csv_text, stamp_asof, validate_pit

__all__ = ["META", "fetch_index_history", "fetch_vix_term_structure"]

_BASE = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
META = SourceMeta(
    name="cboe",
    url=f"{_BASE}/<INDEX>_History.csv",
    license="free (CBOE published index histories)",
    publication_lag="EOD close -> next calendar day 00:00 UTC (conservative)",
    coverage="VIX 1990->, VIX3M 2002->; daily",
    survivorship="n/a (index)",
    quality_notes="F5: conditioner/de-grosser only — never a standalone timer",
)

_DEFAULT_LAG = pd.Timedelta(days=1)


def fetch_index_history(index: str = "VIX", lag: pd.Timedelta = _DEFAULT_LAG) -> pd.DataFrame:
    """One CBOE index history as PIT frame [index, event_ts, open..close, asof_ts]."""
    text = http_get_text(f"{_BASE}/{index.upper()}_History.csv")
    df = read_csv_text(text)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Unexpected CBOE columns: {list(df.columns)}")
    df[EVENT_COL] = pd.to_datetime(df["date"], utc=True)
    df["index"] = index.upper()
    keep = ["index", EVENT_COL] + [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = df[keep].dropna(subset=["close"])
    df = stamp_asof(df, lag)
    return validate_pit(df, required=("index", "close"))


def fetch_vix_term_structure(lag: pd.Timedelta = _DEFAULT_LAG) -> pd.DataFrame:
    """VIX3M/VIX ratio — the term-structure slope conditioner.

    slope > 1: contango (calm); slope < 1: backwardation (stress) — the
    de-grossing input for equity sleeves (Mandate §5.4).
    """
    vix = fetch_index_history("VIX", lag=lag)[["event_ts", "close", "asof_ts"]]
    v3m = fetch_index_history("VIX3M", lag=lag)[["event_ts", "close"]]
    merged = vix.merge(v3m, on="event_ts", suffixes=("_vix", "_vix3m"), how="inner")
    merged["ts_slope"] = merged["close_vix3m"] / merged["close_vix"]
    merged["index"] = "VIX_TS"
    out = merged[["index", "event_ts", "close_vix", "close_vix3m", "ts_slope", "asof_ts"]]
    out = out.sort_values("asof_ts", kind="stable").reset_index(drop=True)
    return validate_pit(out, required=("ts_slope",))
