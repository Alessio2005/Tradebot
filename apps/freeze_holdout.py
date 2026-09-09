# apps/freeze_holdout.py
"""Bevries het poortsample van AD-24 R7/R8. Draai dit EEN keer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.registry.lineage import get_git_sha
from tradebot.utils.failfast import TradebotContractError
from tradebot.validation.holdout import freeze_holdout

OUT = Path("artefacts/governance/holdout_lock.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-utc", type=str, required=True)
    args = parser.parse_args()
    try:
        lock = freeze_holdout(
            split_utc=args.split_utc, out=OUT, git_sha=get_git_sha(short=True),
        )
    except TradebotContractError as exc:
        print(f"GEWEIGERD: {exc}")
        return 1
    print(f"split_utc = {lock.split_utc} bevroren op {lock.frozen_utc} "
          f"(git_sha={lock.git_sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
