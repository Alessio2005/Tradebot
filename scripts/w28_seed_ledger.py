#!/usr/bin/env python
"""W28 step 0.1 — rebuild the canonical hypothesis ledger from the wave log.

The canonical ``artefacts/governance/hypothesis_ledger.json`` was lost from
both working copies (recorded in ``docs/WAVE_LOG.md`` W27 as an open G8
governance defect). ``HypothesisLedger`` refuses to auto-create it, which is
correct: a seed reconstruction must always be deliberate and documented.

This script is that deliberate reconstruction. It does NOT invent a number.
Every row below is quoted from ``docs/WAVE_LOG.md`` and the arithmetic
reconciles end-to-end against two independent statements in that log:

    seed 2363  (W20 §0.1, itemised: 2000 audit-§10/§11 + W14:204 + W15:21
                + W16:8 + W17:49 + W18:1 + W19:80)
      + 5      W20   line 143   "2363->2368"
      + 3      W22r1 line 202   "= 2371"
      + 3      W22r3 line 227   "= 2374"
      + 4      W23c  line 242   "= 2378"
      + 3      W24   line 276   "= 2381"
      + 321    W26   line 366   "+321 trials -> 2702"      <- reconciles
      ------
      = 2702   (W26 close, stated independently in the W26 gate table)

W27 (+4) is NOT seeded here: its staging file carries the real config hash,
so it is merged afterwards through the sanctioned path::

    python apps/ledger_append.py merge \
        --staging artefacts/governance/hypothesis_ledger_staging_w27.json

giving the 2706 that W27 states. Known inconsistency, resolved and recorded:
the "Wave 22 — GESLOTEN" section states 2377 where the running chain says
2381. That section was written after a merge whose staging unlink failed on
the mount; the 2381 chain is the one that reconciles with W26's independent
+321 -> 2702, so 2381 is used and 2377 is treated as a stale restatement.
Individual config hashes for W20-W26 did not survive the file loss; rows are
therefore reconstructed at WAVE granularity and flagged RECONSTRUCTED in
``notes``. Trial COUNTS are exact; per-config identity is not recoverable.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.registry import DEFAULT_LEDGER_PATH, LedgerEntry

SEED_TOTAL = 2363
SEED_NOTE = (
    "Itemised reconstruction quoted verbatim from docs/WAVE_LOG.md W20 §0.1: "
    "2000 (audit §10/§11) + W14:204 + W15:21 + W16:8 + W17:49 + W18:1 + "
    "W19:80 = 2363. Conservative floor."
)

# (wave, unit, market, n_trials, result, wave-log evidence)
RECONSTRUCTED: list[tuple[int, str, str, int, str, str]] = [
    (20, "w20_rebaseline_plus_4_crypto_units", "book", 5, "accepted",
     "WAVE_LOG line 143: rebaseline entry + 4 accepted crypto units "
     "(ML-XS, LOWVOL, REVERSAL-k10, CARRY); 2363->2368."),
    (22, "w22_run1_void", "equities", 3, "archived",
     "WAVE_LOG line 202: run 1 void (data artifact); +3 archived -> 2371."),
    (22, "w22_run3_eq_units", "equities", 3, "interim",
     "WAVE_LOG line 227: run-3 evals of eq_xsmom/eq_lowvol/eq_strev -> 2374."),
    (23, "w23c_eq_overnight", "equities", 4, "archived",
     "WAVE_LOG line 242: W23c merge -> 2378. Closed as F15."),
    (24, "w23ab_w24_eq_units", "equities", 3, "archived",
     "WAVE_LOG line 276: W23a PEAD / W23b quality / W24 resid-strev -> 2381. "
     "Closed as F16/F17/F18."),
    (26, "w26_crypto_clock_seasonality", "crypto", 321, "falsified",
     "WAVE_LOG line 366: 24 HOD + 7 DOW + 2 funding + 288 window scan = 321 "
     "-> 2702. Closed as F19."),
]

EXPECTED_TOTAL_BEFORE_W27 = 2702


def main() -> int:
    path = Path(DEFAULT_LEDGER_PATH)
    if path.exists():
        print(f"REFUSING: {path} already exists — seeding is a one-shot act.")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        asdict(
            LedgerEntry.from_config(
                wave=wave,
                unit=unit,
                market=market,
                config={"reconstruction": unit, "wave": wave},
                n_trials=n,
                result=result,
                notes=f"RECONSTRUCTED (config hashes lost with the original "
                      f"ledger file; trial count exact). {evidence}",
            )
        )
        for wave, unit, market, n, result, evidence in RECONSTRUCTED
    ]
    total = SEED_TOTAL + sum(int(e["n_trials"]) for e in entries)
    if total != EXPECTED_TOTAL_BEFORE_W27:
        print(f"REFUSING: reconstruction sums to {total}, "
              f"expected {EXPECTED_TOTAL_BEFORE_W27}")
        return 1

    doc = {
        "seed_total": SEED_TOTAL,
        "seed_note": SEED_NOTE,
        "reconstructed_utc": "2026-08-10",
        "reconstruction_note": (
            "Rebuilt in Wave 28 step 0.1 after the canonical file was lost. "
            "Source of truth: docs/WAVE_LOG.md. Arithmetic reconciles to the "
            "independently stated W26 total of 2702; W27 (+4) is merged from "
            "its staging file afterwards -> 2706."
        ),
        "entries": entries,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)
    print(f"seeded {path}: seed_total={SEED_TOTAL}, "
          f"entries={len(entries)}, total_n_hypotheses={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
