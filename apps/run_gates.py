"""Draai de research-gates lokaal, vóór een PR — Stage B-5.

Dezelfde vier poorten die `.github/workflows/research_gates.yml` blokkerend
draait, via dezelfde functie. De wetenschap staat in
`validation/gate_runner.py`; deze app doet argumenten en artefact (R-6: apps
<= 80 LOC).

    python apps/run_gates.py                 # alle poorten, rapport, exit 1 bij rood
    python apps/run_gates.py --fail-fast     # stop bij de eerste rode poort
    python apps/run_gates.py --out x.json    # schrijf het oordeel weg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.registry.lineage import get_git_sha
from tradebot.validation.gate_runner import GATE_SPECS, run_research_gates


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop bij de eerste rode poort")
    ap.add_argument("--out", default=None,
                    help="schrijf het oordeel als JSON naar dit pad")
    args = ap.parse_args(argv)

    print(f"research gates — {len(GATE_SPECS)} poorten, git_sha={get_git_sha()}")
    print()

    runs = run_research_gates(
        ROOT, stop_on_first_failure=args.fail_fast)

    for run in runs:
        print(run.render())
        if not run.passed and run.stdout_tail:
            print("       ---- uitvoer (staart) ----")
            for line in run.stdout_tail.splitlines()[-15:]:
                print(f"       {line}")
        print()

    failed = [r for r in runs if not r.passed]
    not_run = [s.name for s in GATE_SPECS
               if s.name not in {r.name for r in runs}]

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "git_sha": get_git_sha(),
            "gates": [r.as_dict() for r in runs],
            "not_run": not_run,
            "passed": not failed and not not_run,
        }, indent=2), encoding="utf-8")
        print(f"artefact: {out}")

    if not_run:
        print(f"NIET GEDRAAID (fail-fast): {', '.join(not_run)}")
    if failed:
        print(f"ROOD: {', '.join(r.name for r in failed)}")
        print("Een merge op deze staat zou een promotiebesluit toelaten dat "
              "door niets wordt afgedwongen.")
        return 1

    print("Alle poorten groen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
