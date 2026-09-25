"""Fase 11, breedte en tijdschaal — stages A t/m D: breedte, klok, muur en poort.

Nul trials: alleen tweede momenten, besluitpanelen en synthetische rendementen,
geen enkel gemiddeld rendement. Alle logica staat in
`src/tradebot/validation/phase11_breadth_measurement.py`; dit is een compositie
(R-6).

    python apps/run_breadth_measurement.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps"))

from run_phase5_baseline import load_market
from tradebot.backtest.baseline_report import json_safe, load_baseline_configs
from tradebot.features.base import DataRegister
from tradebot.features.registry import current_git_sha
from tradebot.registry.ledger_reset import active_trial_count
from tradebot.risk.engine import risk_config_hash
from tradebot.schemas.config import RiskConfig, inference_config, load_config
from tradebot.validation.breadth import breadth_config
from tradebot.validation.phase11_breadth_measurement import build_artefact

GOV = ROOT / "artefacts" / "governance"
CONF = ROOT / "conf" / "research" / "breadth.yaml"
OUT = "artefacts/governance/phase11_breadth.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    base = load_baseline_configs(ROOT)
    register = DataRegister(GOV / "data_hashes.json")
    trials = active_trial_count(reset_path=GOV / "ledger_reset.json",
                                ledger_path=GOV / "hypothesis_ledger.json")
    provenance = {
        "config": {"path": str(CONF.relative_to(ROOT)),
                   "sha256": hashlib.sha256(CONF.read_bytes()).hexdigest()},
        "data_hashes": {s: h for s, h in sorted(register.hashes.items()) if "/1d" in s},
    }
    payload = build_artefact(
        ROOT, market=load_market(ROOT, base), base=base, cfg=breadth_config(CONF),
        inference=inference_config(), lock_path=GOV / "holdout_lock.json",
        m_new=trials.total, git_sha=current_git_sha(), provenance=provenance,
        current_policy_hash=risk_config_hash(
            load_config(ROOT / "conf/risk/default.yaml", RiskConfig)))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
