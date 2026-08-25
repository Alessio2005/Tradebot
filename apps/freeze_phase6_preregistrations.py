"""Bevries de drie Phase 6-pre-registraties en registreer ze in de ledger.

Phase 6, stap 1. De parameterruimte EN de power-analyse worden hier uit `conf/`
en uit het adequaatheidsartefact geresolveerd en ingespoten, zodat beide in de
`preregistration_id` worden gehasht en niet achteraf kunnen verschuiven.

    python apps/freeze_phase6_preregistrations.py --wave 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.features.base import DataRegister
from tradebot.features.registry import current_git_sha
from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.registry.phase6_power import power_block
from tradebot.registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
)
from tradebot.schemas.config import (
    AdequacyConfig,
    LabelingConfig,
    ValidationConfig,
    VolatilityConfig,
    load_config,
)

SPECS = {
    "H1": "conf/research/preregistration_h1_garch_vs_ewma.yaml",
    "H2": "conf/research/preregistration_h2_hmm_vs_m0.yaml",
    "H3": "conf/research/preregistration_h3_meta_labeling.yaml",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", type=int, default=30)
    ap.add_argument("--market", default="crypto")
    args = ap.parse_args(argv)

    adequacy = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
    measured = json.loads(
        (ROOT / "artefacts/governance/phase6_data_adequacy.json").read_text("utf-8"))
    register = DataRegister(ROOT / "artefacts/governance/data_hashes.json")
    ledger = HypothesisLedger(
        ROOT / "artefacts/governance/hypothesis_ledger.json")

    shared = {
        "volatility": load_config(
            ROOT / "conf/model/volatility.yaml", VolatilityConfig).model_dump(mode="json"),
        "validation": load_config(
            ROOT / "conf/validation/default.yaml", ValidationConfig).model_dump(mode="json"),
        "labeling": load_config(
            ROOT / "conf/model/labeling.yaml", LabelingConfig).model_dump(mode="json"),
        "adequacy": adequacy.model_dump(mode="json"),
    }
    kwargs = {
        "cfg": adequacy.power,
        "mean_pairwise_correlation":
            measured["verdicts"]["hrp"]["measured"]["mean_pairwise_correlation"],
        "oos_bars_per_symbol":
            measured["details"]["fold_geometry"]["total_oos_bars"],
        "n_symbols": len(measured["universe"]["symbols"]),
        "uniqueness_ratio":
            measured["verdicts"]["meta_labeling"]["measured"]["uniqueness_ratio"],
    }

    git_sha = current_git_sha()
    for name, spec_path in SPECS.items():
        raw = yaml.safe_load(
            (ROOT / spec_path).read_text(encoding="utf-8"))["preregistration"]
        prereg = load_preregistration_spec(
            ROOT / spec_path,
            data_hashes=tuple((s, register.hashes[s]) for s in raw["data_series"]),
            parameters={**shared, "power_analysis": power_block(name, **kwargs)},
        )
        total = ledger.total_n_hypotheses()
        path = freeze_preregistration(
            prereg, git_sha=git_sha, ledger_total_at_freeze=total,
            directory=ROOT / "artefacts" / "governance")
        if not any(e.get("notes", "").endswith(prereg.preregistration_id)
                   for e in ledger.entries()):
            ledger.append(LedgerEntry(
                wave=args.wave, unit=prereg.wave, market=args.market,
                config_hash=prereg.preregistration_id[:16],
                n_trials=prereg.planned_trials, result="interim",
                notes=("Pre-registratie bevroren VOOR de eerste fit; de geplande "
                       "trials tellen mee in M. preregistration_id="
                       f"{prereg.preregistration_id}")))
        print(f"{name}  id={prereg.preregistration_id}  "
              f"trials={prereg.planned_trials}  M {total} -> "
              f"{ledger.total_n_hypotheses()}  {path.relative_to(ROOT)}")
    print(f"\nM na bevriezing van alle drie: {ledger.total_n_hypotheses()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
