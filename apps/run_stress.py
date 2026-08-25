"""Phase 4, deliverable 13 — draai de stressscenario's en de baseline-overlay.

De wetenschap staat in `risk/stress_test.py` en `risk/stress_report.py`; deze
app doet argumenten en artefact.

    python apps/run_stress.py [--out artefacts/risk/phase4_stress.json]
    python apps/run_stress.py --scenarios-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.backtest.baseline_report import risk_overlay_wave  # noqa: E402
from tradebot.features.registry import current_git_sha  # noqa: E402
from tradebot.registry.risk_registry import RiskConfigRegistry  # noqa: E402
from tradebot.risk.engine import RiskEngine  # noqa: E402
from tradebot.risk.kill_switches import HaltStore  # noqa: E402
from tradebot.risk.stress_test import BaseBook, RiskStressHarness  # noqa: E402
from tradebot.schemas.config import RiskConfig, load_config  # noqa: E402

#: De gemeten uitgangssituatie van dit platform: 72,4% geannualiseerde vol op
#: de Phase 3 1/N-track (reports/BASELINE_BENCHMARK.md sectie 3.1). De
#: stressscenario's schokken hiervandaan, niet vanaf een verzonnen getal.
BASELINE_VOL = 0.724
BASE_ALPHA = 0.6
BASE_ADV_USD = 5e8


def _base_book(cfg: RiskConfig) -> BaseBook:
    symbols = tuple(cfg.clusters)
    return BaseBook(
        symbols=symbols,
        desired_exposure={s: BASE_ALPHA for s in symbols},
        sigma_hat={s: BASELINE_VOL for s in symbols},
        adv_usd={s: BASE_ADV_USD for s in symbols},
        equity=1.0,
        high_water_mark=1.0,
        day_start_equity=1.0,
        asof_ts=pd.Timestamp.utcnow().tz_convert("UTC"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artefacts/risk/phase4_stress.json")
    ap.add_argument("--halt-store", default=None,
                    help="Pad voor de persistente HALTED-state; standaard uit "
                         "(de scenario's mogen de live-state niet aanraken).")
    ap.add_argument("--scenarios-only", action="store_true",
                    help="Sla de baseline-overlay over (S1-S4 alleen).")
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
    store = HaltStore(args.halt_store) if args.halt_store else None
    engine = RiskEngine(cfg, halt_store=store)
    git_sha = current_git_sha()

    # Een risicoconfiguratie zonder hash is niet auditbaar (stap 11). Bewust
    # NIET in de hypothese-ledger: die telt trials en zou de DSR deflateren.
    RiskConfigRegistry(ROOT / "artefacts" / "governance" / "risk_config_registry.json").register(
        config_hash=engine.config_hash, git_sha=git_sha,
        config=cfg.model_dump(mode="json"), audit_header=engine.audit_header(),
        notes="Phase 4 stress run.",
    )

    scenarios = [
        outcome.as_record()
        for outcome in RiskStressHarness(engine).run_all(_base_book(cfg))
    ]
    payload: dict[str, object] = {
        "git_sha": git_sha,
        "risk_config_hash": engine.config_hash,
        "risk_audit_header": engine.audit_header(),
        "base_book": {"sigma_hat": BASELINE_VOL, "a_t": BASE_ALPHA},
        "scenarios": scenarios,
    }
    if not args.scenarios_only:
        payload["baseline_overlay"] = risk_overlay_wave(ROOT, engine, git_sha=git_sha)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    for row in scenarios:
        print(f"{row['scenario']:<24} requested={row['requested_gross']:.3f} "
              f"permitted={row['permitted_gross']:.6f} "
              f"reduction={row['gross_reduction'] * 100:5.1f}% "
              f"bound={row['bound_kinds']}")
    print(f"\nartefact: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
