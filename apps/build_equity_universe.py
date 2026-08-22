# apps/build_equity_universe.py
"""Stateless CLI — Wave 21 equity universe build (R-6, <=80 LOC).

Examples:
  python apps/build_equity_universe.py build
  python apps/build_equity_universe.py build --max-symbols 25   # smoke run
  python apps/build_equity_universe.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from tradebot.data.equity_universe import build_universe, load_price_panel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="market_data_parquet")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build", help="fetch membership + OHLCV, write coverage")
    bp.add_argument("--max-symbols", type=int, default=None)

    sub.add_parser("report", help="print coverage + panel shape")

    args = p.parse_args(argv)

    if args.cmd == "build":
        cov = build_universe(root=args.root, max_symbols=args.max_symbols)
        print(json.dumps({k: v for k, v in cov.items() if not isinstance(v, (dict, list))}, indent=2))
        print(f"failed: {cov['n_failed']} (see universe_coverage.json)")
        return 0

    cov_path = Path(args.root) / "equities" / "universe_coverage.json"
    if cov_path.exists():
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        print(f"members ever: {cov['n_members_ever']}  fetched: {cov['n_fetched']}  "
              f"failed: {cov['n_failed']}  unknown add-date: {cov['n_unknown_add_date']}")
    panel, cal = load_price_panel(root=args.root)
    print(f"panel: {panel.shape[0]} days x {panel.shape[1]} symbols  "
          f"({panel.index.min():%Y-%m-%d} -> {panel.index.max():%Y-%m-%d})")
    print(f"avg tradeable names/day: {cal.sum(axis=1).mean():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
