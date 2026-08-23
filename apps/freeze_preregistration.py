"""Bevries een pre-registratie en registreer hem in de hypothese-ledger.

Phase 3, stap 1. De parameterruimte wordt hier uit `conf/` geresolveerd en
INGESPOTEN, zodat de registratie precies vastpint waarmee er straks wordt
gedraaid. De data_hashes komen uit het gecertificeerde Phase 1-register.

    python apps/freeze_preregistration.py conf/research/preregistration_baseline_phase3.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.features.base import DataRegister
from tradebot.features.registry import current_git_sha
from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
)
from tradebot.schemas.config import (
    AlphaConfig,
    ExecutionConfig,
    ValidationConfig,
    VolatilityConfig,
    load_config,
)


def resolve_parameters() -> dict[str, object]:
    """De parameterruimte, uit `conf/`. Een bron van waarheid, geen kopie."""
    return {
        "alpha": load_config(ROOT / "conf/model/alpha.yaml", AlphaConfig).model_dump(
            mode="json"),
        "volatility": load_config(
            ROOT / "conf/model/volatility.yaml", VolatilityConfig).model_dump(mode="json"),
        "validation": load_config(
            ROOT / "conf/validation/default.yaml", ValidationConfig).model_dump(mode="json"),
        "execution": load_config(
            ROOT / "conf/execution/fees.yaml", ExecutionConfig).model_dump(mode="json"),
        "allocators": ["equal_weight", "risk_parity"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec", help="pad naar de pre-registratie-YAML")
    ap.add_argument("--wave", type=int, required=True,
                    help="wave-nummer voor de ledger-entry")
    ap.add_argument("--market", default="crypto")
    args = ap.parse_args(argv)

    register = DataRegister(ROOT / "artefacts" / "governance" / "data_hashes.json")
    import yaml

    raw = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))["preregistration"]
    data_hashes = tuple(
        (series, register.hashes[series]) for series in raw["data_series"]
    )

    prereg = load_preregistration_spec(
        args.spec, data_hashes=data_hashes, parameters=resolve_parameters()
    )
    ledger = HypothesisLedger(ROOT / "artefacts" / "governance" / "hypothesis_ledger.json")
    total = ledger.total_n_hypotheses()
    path = freeze_preregistration(
        prereg,
        git_sha=current_git_sha(),
        ledger_total_at_freeze=total,
        directory=ROOT / "artefacts" / "governance",
    )

    already = any(
        e.get("notes", "").endswith(prereg.preregistration_id) for e in ledger.entries()
    )
    if not already:
        ledger.append(LedgerEntry(
            wave=args.wave,
            unit=prereg.wave,
            market=args.market,
            config_hash=prereg.preregistration_id[:16],
            n_trials=prereg.planned_trials,
            result="interim",
            notes=("Pre-registratie bevroren VOOR de eerste run; de geplande "
                   f"trials tellen mee in M. preregistration_id={prereg.preregistration_id}"),
        ))

    print(f"preregistration_id : {prereg.preregistration_id}")
    print(f"artefact           : {path.relative_to(ROOT)}")
    print(f"ledger M at freeze : {total}")
    print(f"planned trials     : {prereg.planned_trials}")
    print(f"ledger M after     : {ledger.total_n_hypotheses()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
