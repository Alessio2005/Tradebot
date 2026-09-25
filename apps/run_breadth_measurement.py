"""Fase 11, breedte en tijdschaal — stages A en B: breedte en signaalklok meten.

Nul trials: alleen tweede momenten en besluitpanelen, geen gemiddeld rendement.
Alle logica staat in `src/tradebot/validation/`; dit is een compositie (R-6).

    python apps/run_breadth_measurement.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "apps"))

from run_phase5_baseline import load_market
from tradebot.backtest.baseline_report import json_safe, load_baseline_configs
from tradebot.features.base import DataRegister
from tradebot.features.registry import current_git_sha
from tradebot.schemas.config import inference_config
from tradebot.validation.breadth import breadth_config
from tradebot.validation.phase11_breadth_measurement import (
    baseline_weight_tracks,
    measure_breadth,
    measure_signal_clock,
    split_windows,
)

CONF = ROOT / "conf" / "research" / "breadth.yaml"
LOCK = ROOT / "artefacts" / "governance" / "holdout_lock.json"
OUT = "artefacts/governance/phase11_breadth.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    cfg, base, inf = breadth_config(CONF), load_baseline_configs(ROOT), inference_config()
    market = load_market(ROOT, base)
    usable = market["sigma"].dropna(how="any").index
    usable = usable[usable.isin(market["adv"].dropna(how="any").index)]
    returns = np.log(market["prices"]).diff().loc[usable]
    windows = split_windows(returns, lock_path=LOCK)
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    payload = {
        "git_sha": current_git_sha(),
        "config": {"path": str(CONF.relative_to(ROOT)),
                   "sha256": hashlib.sha256(CONF.read_bytes()).hexdigest()},
        "inference": {"n_boot": inf.n_boot, "seed": inf.seed, "ci_level": inf.ci_level,
                      "block_length": inf.block_length},
        "data_hashes": {s: h for s, h in sorted(register.hashes.items()) if "/1d" in s},
        "risk_policy_hash": None,
        "risk_policy_note": "geen risicobesluit: alles hier is L0/data (AD-27 n.v.t.)",
        "trials": 0,
        "breadth": measure_breadth(windows, cfg=cfg, n_boot=inf.n_boot, seed=inf.seed,
                                   ci_level=inf.ci_level, block_length=inf.block_length),
        "signal_clock": measure_signal_clock(
            baseline_weight_tracks(ROOT, base, git_sha=current_git_sha()),
            prices=market["prices"], usable=usable,
            development_index=windows["W_DEV"].index, cfg=cfg,
            bars_per_year=base["bt"].bars_per_year),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
