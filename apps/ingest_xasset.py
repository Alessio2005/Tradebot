#!/usr/bin/env python
"""Ingest the Wave-27 cross-asset total-return proxy panel (R-6: stateless CLI).

    python apps/ingest_xasset.py [--start 2004-01-01] [--out market_data_parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tradebot.data.sources.base import write_market_parquet
from tradebot.data.xasset_proxy import XASSET_UNIVERSE, fetch_panel

_MIN_ROWS = 50_000  # staleness/truncation guard (FRED incident, R-8)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2004-01-01")
    p.add_argument("--out", default="market_data_parquet")
    p.add_argument("--max-stale-days", type=int, default=7)
    a = p.parse_args(argv)

    long = fetch_panel(start=a.start)
    last = long["event_ts"].max()
    stale = (pd.Timestamp.now(tz="UTC") - last).days
    n_sym = long["symbol"].nunique()

    if len(long) < _MIN_ROWS:
        print(f"REFUSED: {len(long)} rows < {_MIN_ROWS} — truncated fetch", file=sys.stderr)
        return 2
    if stale > a.max_stale_days:
        print(f"REFUSED: last bar {last.date()} is {stale}d stale", file=sys.stderr)
        return 2
    if n_sym < len(XASSET_UNIVERSE):
        missing = set(XASSET_UNIVERSE) - set(long["symbol"].unique())
        print(f"REFUSED: missing symbols {sorted(missing)}", file=sys.stderr)
        return 2

    path = write_market_parquet(long, market="xasset", name="tr_panel", root=Path(a.out))
    print(f"OK {len(long):,} rows · {n_sym} symbols · "
          f"{long['event_ts'].min().date()} -> {last.date()} · {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
