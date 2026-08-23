"""Phase 3, deliverable 14 — draai de Level-1 baseline en de statistische gates.

Vereist een bevroren pre-registratie-ID: zonder dat crasht de run (sectie 18.1).
De wetenschap staat in `backtest/baseline_report.py`; deze app doet argumenten,
artefact en ledger.

    python apps/run_baseline.py --preregistration-id <id> --wave 29
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.backtest.baseline_report import (
    PRIMARY_TRACK,
    execute_baseline_wave,
    json_safe,
)
from tradebot.features.registry import current_git_sha
from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.registry.preregistration import (
    freeze_metadata,
    require_preregistration,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preregistration-id", required=True)
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--out", default="artefacts/baseline/phase3_baseline.json")
    args = ap.parse_args(argv)

    gov = ROOT / "artefacts" / "governance"
    prereg = require_preregistration(args.preregistration_id, directory=gov)
    ledger = HypothesisLedger(gov / "hypothesis_ledger.json")

    # M uit de BEVRIEZING, niet live uit de ledger: anders telt de
    # resultaat-entry van deze meting zichzelf mee en levert een tweede run een
    # andere DSR op.
    freeze = freeze_metadata(args.preregistration_id, directory=gov)
    payload, gate = execute_baseline_wave(
        ROOT, prereg, m_trials=freeze["m_trials"], git_sha=current_git_sha()
    )
    payload["ledger_total_at_freeze"] = freeze["ledger_total_at_freeze"]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True),
                   encoding="utf-8")

    unit_name = f"{prereg.wave}_result"
    if not any(e["unit"] == unit_name for e in ledger.entries()):
        ledger.append(LedgerEntry(
            wave=args.wave, unit=unit_name, market="crypto",
            config_hash=prereg.preregistration_id[:16], n_trials=1,
            result="accepted" if gate.promoted else "falsified",
            metrics={k: (None if v is None else v)
                     for k, v in json_safe(payload["measured"]).items()},
            notes=(f"Baseline-resultaat; preregistration_id="
                   f"{prereg.preregistration_id}; git_sha={payload['git_sha']}"),
        ))

    print(json.dumps(json_safe({
        "primary_track": PRIMARY_TRACK,
        **payload["measured"],
        "m_trials": payload["m_trials"],
        "n_binding_stop_criteria": gate.n_binding,
        "promoted_to_candidate": gate.promoted,
        "artefact": str(out.relative_to(ROOT)),
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
