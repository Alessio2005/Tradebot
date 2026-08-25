"""Data Adequacy Gate — meet alle vijf modelklassen VOOR de eerste fit.

Phase 6, stap 2. De wetenschap staat in `validation/adequacy_report.py` en het
universum in `data/phase6_universe.py`; deze app doet argumenten en artefact
(R-6: apps <= 80 LOC).

    python apps/run_data_adequacy.py
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
from tradebot.schemas.config import (
    AdequacyConfig,
    LabelingConfig,
    ValidationConfig,
    load_config,
)
from tradebot.validation.adequacy_report import measure_all


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artefacts/governance/phase6_data_adequacy.json")
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    adequacy = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
    validation = load_config(ROOT / "conf/validation/default.yaml", ValidationConfig)
    labeling = load_config(ROOT / "conf/model/labeling.yaml", LabelingConfig)

    git_sha = current_git_sha()
    universe = load_phase6_universe(
        ROOT, cfg, build_cross_sectional_momentum(cfg["alpha"]), git_sha=git_sha)

    # `intraday_bars_per_day=None` is geen aanname maar een vaststelling: de
    # gecertificeerde store bevat GEEN 1m/5m-reeks (`docs/DATA_REGISTER.md`,
    # `reports/phase1_data_gap.md` — een expliciet scope-besluit uit Phase 1).
    verdicts, details = measure_all(
        returns=universe.log_returns, ohlc=universe.ohlc,
        sigma=universe.sigma_bar, side=universe.side,
        intraday_bars_per_day=None,
        observed_intraday_granularity="geen (alleen 1d, 8h funding, 1d OI)",
        adequacy=adequacy, validation=validation, labeling=labeling)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "git_sha": git_sha,
        "universe": universe.as_record(),
        "verdicts": {k: v.as_record() for k, v in verdicts.items()},
        "details": details,
    }, indent=2, default=str), encoding="utf-8")

    for name, verdict in verdicts.items():
        print(f"{'PASS' if verdict.adequate else 'FAIL':>4}  {name:<22} "
              f"{verdict.requirement}")
        if not verdict.adequate:
            print(f"        -> {verdict.shortfall}")
    print(f"\nartefact: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
