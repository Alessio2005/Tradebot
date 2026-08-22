"""Parse live-engine state into a human-readable paper-trade report.

Usage:
    python scripts/paper_trade_report.py \
        --state-dir .tradebot_state \
        --output    paper_trade_report.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone

import pandas as pd


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(state_dir: pathlib.Path) -> str:
    audit = _read_jsonl(state_dir / "audit_log.jsonl")
    circuit = _read_jsonl(state_dir / "circuit_log.jsonl")

    out: list[str] = ["# Paper-trade report", ""]
    out.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    out.append("")

    out.append("## Audit log summary")
    if not audit:
        out.append("- _no audit records found_")
    else:
        df = pd.DataFrame(audit)
        n_orders = df["order_id"].nunique() if "order_id" in df else len(df)
        n_fills = df["fill_ts"].notna().sum() if "fill_ts" in df else 0
        notional = df["notional_usdt"].sum() if "notional_usdt" in df else 0.0
        out += [
            f"- records:           {len(df)}",
            f"- distinct orders:   {n_orders}",
            f"- fills:             {n_fills}",
            f"- gross notional:    {notional:,.0f} USDT",
        ]

    out += ["", "## Circuit breaker"]
    if not circuit:
        out.append("- ✅ no halts during run")
    else:
        for entry in circuit:
            out.append(
                f"- 🔴 {entry.get('event_ts')} — {entry.get('halt_reason')} "
                f"(equity={entry.get('equity'):.2f})"
            )

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--output",    required=True)
    args = parser.parse_args()

    state_dir = pathlib.Path(args.state_dir)
    output    = pathlib.Path(args.output)
    output.write_text(build_report(state_dir), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
