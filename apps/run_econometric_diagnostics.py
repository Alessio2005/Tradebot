"""Econometrische keten + fractionele differentiëring op elke reeks.

Phase 6, stappen 3 en 4. De wetenschap staat in
`validation/diagnostics_report.py`; deze app doet argumenten en artefacten.

    python apps/run_econometric_diagnostics.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.data.phase6_universe import load_phase6_universe
from tradebot.features.registry import current_git_sha
from tradebot.reporting.phase6_econometrics import render_diagnostics_report
from tradebot.schemas.config import FracDiffConfig, load_config
from tradebot.validation.diagnostics_report import diagnose_universe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out",
                    default="artefacts/governance/phase6_econometrics.json")
    ap.add_argument("--report-out", default="reports/ECONOMETRIC_DIAGNOSTICS.md")
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    frac_cfg = load_config(ROOT / "conf/model/fracdiff.yaml", FracDiffConfig)
    git_sha = current_git_sha()
    universe = load_phase6_universe(
        ROOT, cfg, build_cross_sectional_momentum(cfg["alpha"]), git_sha=git_sha)

    diagnostics = diagnose_universe(universe.prices, frac_cfg)

    payload = {
        "git_sha": git_sha,
        "universe": universe.as_record(),
        "fracdiff_config": frac_cfg.model_dump(mode="json"),
        "symbols": {s: d.as_record() for s, d in diagnostics.items()},
    }
    json_out = ROOT / args.json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")

    report_out = ROOT / args.report_out
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        render_diagnostics_report(payload, diagnostics), encoding="utf-8")

    for symbol, diag in diagnostics.items():
        print(f"{symbol:<10} ARCH {'OPEN ' if diag.arch_gate_open else 'DICHT'}  "
              f"p={diag.log_return.engle_arch.p_value:.3g}  "
              f"d={diag.frac.d:.4f}  geheugen={diag.frac.memory_retained:.4f} "
              f"(d=1: {diag.frac.memory_retained_at_d_one:.4f})")
    print(f"\nartefact: {json_out.relative_to(ROOT)}")
    print(f"rapport:  {report_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
