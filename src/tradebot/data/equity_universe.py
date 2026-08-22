# src/tradebot/data/equity_universe.py
"""Wave 21 — point-in-time equity universe + survivorship accounting (G8).

Pipeline: wiki_constituents (PIT membership events) -> Stooq bulk OHLCV for
the FULL historical member set (incl. removed names) -> coverage report
quantifying exactly which ex-members have no price history (the residual
survivorship gap that must be documented and stressed, never ignored).

Outputs under market_data_parquet/equities/:
  membership_events.parquet  (PIT add/remove events)
  ohlcv.parquet              (long frame, all fetchable members ever)
  universe_coverage.json     (G8 evidence: fetched / failed / unknown-date)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tradebot.data.sources import stooq, wiki_constituents
from tradebot.data.sources.base import ASOF_COL

__all__ = ["to_stooq_symbol", "build_universe", "load_price_panel"]


def to_stooq_symbol(ticker: str) -> str:
    """Map a Wikipedia/S&P ticker to Stooq's US convention.

    'BRK.B' -> 'brk-b.us', 'AAPL' -> 'aapl.us'.
    """
    return f"{ticker.strip().lower().replace('.', '-')}.us"


def build_universe(
    root: Path | str = "market_data_parquet",
    max_symbols: int | None = None,
    source: str = "auto",
) -> dict:
    """Fetch membership events + all price history; write coverage report.

    Returns the coverage dict (also persisted as universe_coverage.json).
    ``max_symbols`` exists only for smoke runs; a real Wave-21 build fetches
    everything. ``source``: "stooq" | "yfinance" | "auto" (= stooq with
    fail-fast, then yfinance fallback — Stooq blocked wholesale 2026-06-10).
    """
    root = Path(root)
    events = wiki_constituents.fetch_membership_events()
    dest = root / "equities"
    dest.mkdir(parents=True, exist_ok=True)
    events.to_parquet(dest / "membership_events.parquet", index=False)

    tickers = sorted(events["symbol"].unique())
    if max_symbols is not None:
        tickers = tickers[:max_symbols]

    used = source
    if source in ("auto", "stooq"):
        try:
            _, failures = stooq.fetch_universe(
                [to_stooq_symbol(t) for t in tickers],
                market="equities", name="ohlcv", root=root,
            )
            used = "stooq"
        except ConnectionError:
            if source == "stooq":
                raise
            used = "yfinance"
    if used == "yfinance":
        from tradebot.data.sources.yfinance_backup import fetch_universe_yf

        _, failures = fetch_universe_yf(
            tickers, market="equities", name="ohlcv", root=root
        )

    unknown = sorted(
        events.loc[events["date_unknown"], "symbol"].unique().tolist()
    )
    coverage = {
        "price_source": used,
        "n_members_ever": len(tickers),
        "n_fetched": len(tickers) - len(failures),
        "n_failed": len(failures),
        "failed_symbols": failures,
        "n_unknown_add_date": len(unknown),
        "unknown_add_date_symbols": unknown,
        "note": (
            "failed symbols are predominantly delisted/renamed names — this "
            "IS the residual survivorship gap (G8); quantify its effect by "
            "stressing accepted units against a fetched-only vs full-member "
            "universe and record the delta in the wave log."
        ),
    }
    (dest / "universe_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return coverage


def load_price_panel(
    root: Path | str = "market_data_parquet",
    min_history_days: int = 252,
    with_open: bool = False,
) -> tuple[pd.DataFrame, ...]:
    """Load (close-price panel, PIT membership calendar) for the XS units.

    Panel cells exist only where the bar's ``asof_ts`` has passed — daily
    bars enter the panel on their own date and are consumed by the harness
    at t+1 via ``w.shift(1)`` (consistent with the Stooq next-day lag).
    """
    root = Path(root)
    ohlcv = pd.read_parquet(root / "equities" / "ohlcv.parquet")
    events = pd.read_parquet(root / "equities" / "membership_events.parquet")
    events[ASOF_COL] = pd.to_datetime(events[ASOF_COL], utc=True)

    panel = ohlcv.pivot_table(index="event_ts", columns="symbol", values="close")
    panel.index = pd.to_datetime(panel.index, utc=True)
    panel = panel.sort_index()
    panel = panel.loc[:, panel.notna().sum() >= min_history_days]

    # Glitch guard (G8): a one-day move beyond +300%/-75% in an S&P-class
    # name is a data error (bad print / residual split artifact), not a
    # return. Mask the offending prices to NaN and REPORT the count —
    # never silently. (First W22 run: split artifacts produced a +16078%
    # fake year; adjusted prices fix the bulk, this catches the rest.)
    import numpy as np

    logret = np.log(panel / panel.shift(1))
    bad = logret.abs() > np.log(4.0)
    n_bad = int(bad.to_numpy().sum())
    if n_bad:
        print(f"glitch guard: masked {n_bad} price points with |1d move| > 300% "
              f"({n_bad / max(panel.notna().to_numpy().sum(), 1):.5%} of cells)")
        panel = panel.mask(bad)
    # ticker case: events use 'AAPL', stooq uses 'aapl.us'
    rename = {to_stooq_symbol(t): t for t in events["symbol"].unique()}
    panel = panel.rename(columns=rename)

    calendar = wiki_constituents.build_membership_calendar(events, panel.index)
    calendar = calendar.reindex(columns=panel.columns).fillna(False)
    if not with_open:
        return panel, calendar

    open_panel = ohlcv.pivot_table(index="event_ts", columns="symbol", values="open")
    open_panel.index = pd.to_datetime(open_panel.index, utc=True)
    open_panel = open_panel.sort_index().rename(columns=rename)
    open_panel = open_panel.reindex(index=panel.index, columns=panel.columns)
    # same discipline as close: mask glitch opens (gap vs SAME-day close
    # beyond 4x is a bad print, not a price)
    ratio = (open_panel / panel).abs()
    bad_open = (ratio > 4.0) | (ratio < 0.25)
    n_bad = int(bad_open.to_numpy().sum())
    if n_bad:
        print(f"glitch guard (open): masked {n_bad} open prints with open/close beyond 4x")
        open_panel = open_panel.mask(bad_open)
    open_panel = open_panel.mask(panel.isna())
    return panel, calendar, open_panel
