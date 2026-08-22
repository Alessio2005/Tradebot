# apps/ingest_eia.py
"""Stateless CLI — ingest the EIA NYMEX Contract 1-4 term-structure archive (R-6).

  python apps/ingest_eia.py
  python apps/ingest_eia.py --products WTI,NG --out market_data_parquet

Writes market_data_parquet/commodities/eia_term_structure.parquet (long format:
product, tenor, event_ts, settle, asof_ts). Refuses to write if the archive
bound moved — a truncated .xls and a resumed publication both change what a
carry backtest means.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from tradebot.data.sources.eia import (
        ARCHIVE_LAST_BAR,
        PRODUCTS,
        assert_archive_bounds,
        fetch_term_structure,
    )
    from tradebot.data.sources.base import write_market_parquet

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products", default=",".join(PRODUCTS))
    p.add_argument("--out", default="market_data_parquet")
    p.add_argument("--name", default="eia_term_structure")
    p.add_argument("--tolerance-days", type=int, default=5)
    args = p.parse_args(argv)

    products = tuple(s.strip().upper() for s in args.products.split(",") if s.strip())
    print(f"fetching EIA contract 1-4 for {products} (archive ends {ARCHIVE_LAST_BAR.date()})")

    df = fetch_term_structure(products)
    assert_archive_bounds(df, tolerance_days=args.tolerance_days)

    for prod, g in df.groupby("product"):
        span = f"{g['event_ts'].min().date()} -> {g['event_ts'].max().date()}"
        tenors = sorted(g["tenor"].unique())
        print(f"  {prod:5s} n={len(g):>6d}  tenors={tenors}  {span}")

    # A tenor that is missing on a date makes the slope undefined there; report
    # the usable overlap per product rather than discovering it mid-backtest.
    wide = df.pivot_table(index=["product", "event_ts"], columns="tenor", values="settle")
    complete = wide.dropna()
    print(f"\nrows with all 4 tenors present: {len(complete)} / {len(wide)}")
    for prod, g in complete.groupby(level="product"):
        idx = g.index.get_level_values("event_ts")
        print(f"  {prod:5s} usable {len(g):>6d}  {idx.min().date()} -> {idx.max().date()}")

    path = write_market_parquet(df, "commodities", args.name, root=args.out)
    print(f"\nwrote {path}  ({len(df)} rows, {df['product'].nunique()} products)")
    print(f"asof range {pd.to_datetime(df['asof_ts']).min()} -> {pd.to_datetime(df['asof_ts']).max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
