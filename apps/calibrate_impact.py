"""Phase 5, deliverable 5 — kalibreer het marktimpactmodel, of bewijs dat het niet kan.

De wetenschap staat in `execution/impact_calibration.py`; deze app doet
argumenten, data en artefacten (R-6: apps <= 80 LOC).

    python apps/calibrate_impact.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.data.pit_store import PitStore
from tradebot.execution.impact_calibration import calibrate_impact, render_calibration_report
from tradebot.features.base import ASOF_INDEX_NAME, DataRegister, load_certified_series
from tradebot.features.registry import current_git_sha
from tradebot.schemas.config import DataConfig, VolatilityConfig, load_config
from tradebot.volatility.ewma import ewma_volatility


def load_panels(root: Path) -> tuple[dict, dict, dict, list[str]]:
    """OHLC, causale dagvolatiliteit en `data_hash` per symbool."""
    data = load_config(root / "conf/data/default.yaml", DataConfig)
    vol = load_config(root / "conf/model/volatility.yaml", VolatilityConfig)
    store, register = PitStore(root / data.pit_store_root), DataRegister(
        root / "artefacts/governance/data_hashes.json")
    panels, sigmas, hashes = {}, {}, {}
    for symbol in data.symbols:
        df, _ = load_certified_series(store, register, asset_class="crypto",
                                      dataset="ohlcv", symbol=symbol, granularity="1d")
        idx = pd.DatetimeIndex(pd.to_datetime(df["asof_ts_ns"].to_numpy(), unit="ns",
                                              utc=True), name=ASOF_INDEX_NAME)
        frame = df[["high", "low", "close"]].astype("float64").set_axis(idx)
        # De volatiliteit voor bar t mag bar t niet kennen: L2 levert de
        # geannualiseerde sigma_{t+1|t}, en die wordt hier gedeannualiseerd.
        annual = ewma_volatility(frame["close"], lam=vol.ewma_lambda,
                                 burn_in_bars=vol.burn_in_bars,
                                 annualisation_factor=vol.annualisation_factor)
        panels[symbol] = frame
        sigmas[symbol] = annual / float(np.sqrt(vol.annualisation_factor))
        hashes[f"crypto/ohlcv/{symbol}/1d"] = register.hashes[f"crypto/ohlcv/{symbol}/1d"]
    available = sorted({p.dataset for p in store.partitions(asset_class="crypto")})
    return panels, sigmas, hashes, available


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artefacts/execution/impact_params.json")
    ap.add_argument("--report", default="reports/TCA_CALIBRATION_REPORT.md")
    args = ap.parse_args(argv)

    panels, sigmas, hashes, available = load_panels(ROOT)
    params, evidence = calibrate_impact(
        panels, sigmas, available_datasets=available, data_hashes=hashes)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"params": params.as_record(), "evidence": evidence.as_record(),
         "git_sha": current_git_sha()},
        indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_calibration_report(
        params, evidence, git_sha=current_git_sha()), encoding="utf-8")

    print(f"status={params.status.value} eta={params.eta:.4f} "
          f"kappa_d={params.kappa_d:.4f} n={params.sample_size}")
    print(f"wrote {out}\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
