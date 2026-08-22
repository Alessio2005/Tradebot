# apps/fetch_factors.py
"""Stateless CLI — Ken French FF5+MOM daily factorset -> parquet (R-6, <=80 LOC).

  python apps/fetch_factors.py
  python apps/fetch_factors.py --out market_data_parquet/equities/factors_daily.parquet

BAB (AQR) is a separate manual download; until it is added the G4-equities
regression runs on FF5+MOM and is labeled as such in the gate row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from tradebot.data.sources.kenfrench import fetch_factors_daily

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", default="market_data_parquet/equities/factors_daily.parquet"
    )
    args = p.parse_args(argv)

    df = fetch_factors_daily()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"{len(df)} rows {df['event_ts'].min():%Y-%m-%d} -> "
        f"{df['event_ts'].max():%Y-%m-%d} | columns: "
        f"{[c for c in df.columns if c not in ('event_ts', 'asof_ts')]} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
