# apps/ingest_fx.py
"""Stateless CLI — Wave 25 FX data ingest via FRED (R-6, <=80 LOC).

  python apps/ingest_fx.py            # G10 rates + spots -> market_data_parquet/fx/

Network: FRED only (verified source). Run locally; the sandbox egress is
blocked, evaluation runs agent-side afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    from tradebot.data.sources.fred import (
        G10_RATE_SERIES,
        G10_SPOT_SERIES,
        fetch_series,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="market_data_parquet")
    p.add_argument("--only", choices=["rates", "spots", "all"], default="all")
    args = p.parse_args(argv)
    dest = Path(args.root) / "fx"
    dest.mkdir(parents=True, exist_ok=True)

    fails: dict[str, str] = {}

    # Spots only matter from ~1999 (when >=6 rate series exist for carry);
    # pre-1990 spot history is dead weight and just lengthens the slow DEX
    # fetch. Rates keep full history (USD DTB3 back to 1971).
    start_for = {"rates": 1971, "spots": 1990}

    def leg(name: str, series: dict, lag_for) -> None:
        frames = []
        for ccy, spec in series.items():
            sid = spec[0]
            try:
                df = fetch_series(sid, lag_for(spec), start_year=start_for[name])
                frames.append(df)
                print(f"{name} {ccy:4s} {sid:18s} {len(df):6d} rows")
            except Exception as exc:
                fails[sid] = str(exc)
                print(f"{name} {ccy:4s} {sid:18s} FAIL: {exc}")
        if not frames:
            print(f"{name}: 0 series fetched — existing parquet left untouched")
            return
        pd.concat(frames, ignore_index=True).to_parquet(
            dest / f"{name}.parquet", index=False
        )

    if args.only in ("rates", "all"):
        leg("rates", G10_RATE_SERIES, lambda s: pd.Timedelta(days=s[1]))
    if args.only in ("spots", "all"):
        leg("spots", G10_SPOT_SERIES, lambda s: pd.Timedelta(days=1))

    print(f"\ndone; {len(fails)} failures: {sorted(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
