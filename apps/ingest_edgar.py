# apps/ingest_edgar.py
"""Stateless CLI — Wave 23a/b EDGAR bulk ingest (R-6, <=80 LOC).

  python apps/ingest_edgar.py                  # full universe (~15-25 min, 10 req/s cap)
  python apps/ingest_edgar.py --max-tickers 25 # smoke

Network: data.sec.gov + www.sec.gov. Run locally; evaluation runs
agent-side afterwards on the parquets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    from tradebot.data.edgar_universe import bulk_ingest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="market_data_parquet")
    p.add_argument("--max-tickers", type=int, default=None)
    args = p.parse_args(argv)

    events = pd.read_parquet(Path(args.root) / "equities" / "membership_events.parquet")
    tickers = sorted(events["symbol"].unique())
    if args.max_tickers:
        tickers = tickers[: args.max_tickers]
    print(f"ingesting EDGAR for {len(tickers)} tickers...")
    cov = bulk_ingest(tickers, root=args.root)
    print(f"done: no_cik={cov['n_no_cik']} failed={cov['n_failed']} "
          f"-> equities/fundamentals.parquet + filings.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
