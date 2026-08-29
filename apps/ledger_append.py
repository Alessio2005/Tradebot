# apps/ledger_append.py
"""Stateless CLI for the append-only hypothesis ledger (R-6, <=80 LOC).

Examples:
  python apps/ledger_append.py total
  python apps/ledger_append.py append --wave 21 --unit eq_xsmom_12_1 \
      --market equities --config '{"lookback": 252, "skip": 21}' \
      --n-trials 1 --result interim --notes "Jegadeesh-Titman 1993"
  python apps/ledger_append.py merge --staging \
      artefacts/governance/hypothesis_ledger_staging_22a.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    from tradebot.registry import DEFAULT_LEDGER_PATH, HypothesisLedger, LedgerEntry

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("total", help="print honest cumulative n_hypotheses")

    ap = sub.add_parser("append", help="append one entry to the main ledger")
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--market", required=True)
    ap.add_argument("--config", required=True, help="JSON dict of the config")
    ap.add_argument("--n-trials", type=int, default=1)
    # Verplicht, en niet met een lege default: `LedgerEntry` weigert een entry
    # zonder herkomst (audit §26), en een CLI die die velden niet doorgeeft,
    # crasht op precies het contract dat hem beschermt.
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--data-hash", required=True)
    ap.add_argument("--preregistration-id", required=True)
    ap.add_argument("--result", default="interim")
    ap.add_argument("--metrics", default="{}", help="JSON dict of metrics")
    ap.add_argument("--notes", default="")
    # Een amendement herziet het OORDEEL over trials die al zijn geteld; het
    # draagt daarom n_trials=0. Zie `LedgerEntry.amends`.
    ap.add_argument("--amends", default="",
                    help="config_hash van de entry die wordt geamendeerd")
    ap.add_argument("--staging", default="", help="write to staging file instead")

    mp = sub.add_parser("merge", help="serially merge a staging file")
    mp.add_argument("--staging", required=True)

    args = p.parse_args(argv)
    ledger = HypothesisLedger(args.ledger)

    if args.cmd == "total":
        print(ledger.total_n_hypotheses())
        return 0

    if args.cmd == "merge":
        total = ledger.merge_staging(args.staging)
        print(f"merged; total_n_hypotheses={total}")
        return 0

    entry = LedgerEntry.from_config(
        wave=args.wave,
        unit=args.unit,
        market=args.market,
        config=json.loads(args.config),
        git_sha=args.git_sha,
        data_hash=args.data_hash,
        preregistration_id=args.preregistration_id,
        n_trials=args.n_trials,
        result=args.result,
        metrics=json.loads(args.metrics),
        notes=args.notes,
        amends=args.amends,
    )
    if args.staging:
        HypothesisLedger.append_to_staging(args.staging, entry)
        print(f"staged -> {args.staging}")
    else:
        total = ledger.append(entry)
        print(f"appended; total_n_hypotheses={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
