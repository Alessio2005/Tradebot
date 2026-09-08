# apps/freeze_ledger_reset.py
"""Bevries de eenmalige ledger-reset van AD-24. Draai dit EEN keer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.registry.ledger_reset import ResetAlreadyExists, freeze_reset
from tradebot.registry.lineage import get_git_sha

OUT = Path("artefacts/governance/ledger_reset.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-new", type=int, required=True)
    parser.add_argument("--rationale", type=str, required=True)
    args = parser.parse_args()
    try:
        record = freeze_reset(
            m_new=args.m_new, rationale=args.rationale,
            git_sha=get_git_sha(short=True), out=OUT,
        )
    except ResetAlreadyExists as exc:
        print(f"GEWEIGERD: {exc}")
        return 1
    print(f"M_new = {record.m_new} bevroren op {record.frozen_utc}; "
          f"gearchiveerd: {record.archived_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
