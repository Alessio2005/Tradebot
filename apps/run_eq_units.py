# apps/run_eq_units.py
"""Stateless CLI — Wave 22a/b/c evaluation (R-6, <=80 LOC).

Runs the three equity units on the Wave-21 PIT panel, prints per-unit
summary + G4 row + pairwise correlations, and stages ledger entries.

  python apps/run_eq_units.py --factors market_data_parquet/equities/factors_daily.parquet
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from tradebot.alpha import eq_lowvol, eq_strev, eq_xsmom, factor_residual_alpha
    from tradebot.data.equity_universe import load_price_panel
    from tradebot.registry import HypothesisLedger, LedgerEntry

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="market_data_parquet")
    p.add_argument("--factors", default="", help="parquet with Ken French + BAB daily")
    p.add_argument("--stage", default="", help="ledger staging file (wave close merges)")
    # Verplicht zodra er wordt gestaged, en niet met een lege default:
    # `LedgerEntry` weigert een entry zonder herkomst (audit par. 26).
    # Tot 2026-09-18 gaf deze aanroep deze drie velden NIET door en gooide
    # `--stage` dus altijd `TypeError: missing 3 required positional
    # arguments`. Er is nooit een entry uit dit script in de ledger beland.
    # De waarden hier verzinnen zou precies DI-20 herhalen: een
    # provenance-veld dat niet te herleiden is, is geen provenance. Ze
    # worden daarom gevraagd, zoals `apps/ledger_append.py` dat ook doet.
    p.add_argument("--git-sha", default="")
    p.add_argument("--data-hash", default="")
    p.add_argument("--preregistration-id", default="")
    p.add_argument("--start", default="2000-01-01",
                   help="evaluation window start (membership reliable ~2000+)")
    args = p.parse_args(argv)
    if args.stage and not (args.git_sha and args.data_hash and args.preregistration_id):
        p.error("--stage vereist --git-sha, --data-hash en --preregistration-id; "
                "een ledger-entry zonder herkomst wordt door LedgerEntry geweigerd")

    panel, calendar = load_price_panel(root=args.root)
    print(f"panel {panel.shape}, avg names/day {calendar.sum(axis=1).mean():.0f}, "
          f"eval from {args.start}")

    results = {}
    for mod in (eq_xsmom, eq_strev, eq_lowvol):
        res = mod.run(panel, membership=calendar)
        results[res.unit] = res
        s = res.summary(start=args.start)
        print(f"\n== {res.unit} ({mod.PRIOR}) ==")
        print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2, default=str))

        if args.factors:
            facs = pd.read_parquet(args.factors)
            facs.index = pd.to_datetime(facs["event_ts"], utc=True)
            cols = [c for c in ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "BAB") if c in facs.columns]
            g4 = factor_residual_alpha(
                res.net_returns.loc[args.start:], facs[cols], unit=res.unit,
                market="equities" if "BAB" in cols else "book", periods_per_year=252,
            )
            print(g4.gate_row())

        if args.stage:
            HypothesisLedger.append_to_staging(args.stage, LedgerEntry.from_config(
                wave=22, unit=res.unit, market="equities",
                config={**res.config, "eval_start": args.start, "price_adjust": "yf_auto_adjust"},
                git_sha=args.git_sha,
                data_hash=args.data_hash,
                preregistration_id=args.preregistration_id,
                metrics={"net_sharpe": s["net_sharpe"], "net_cagr": s["net_cagr"]},
                notes=mod.PRIOR,
            ))

    rets = pd.DataFrame({u: r.net_returns.loc[args.start:] for u, r in results.items()})
    print("\npairwise corr:\n", rets.corr().round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
