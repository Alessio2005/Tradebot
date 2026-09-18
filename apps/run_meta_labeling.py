"""H3 — CatBoost als secondary model. Stappen 12 en 13, deliverable 22.

De wetenschap staat in `train/meta_label.py`, `validation/feature_importance.py`
en `validation/meta_label_campaign.py`; de opmaak in
`reporting/phase6_meta_labeling.py`. Deze app doet uitsluitend argumenten, data
en artefacten -- er staat geen enkele meting in.

R-6 stelt een grens van 80 regels; dit bestand telt er meer, met het leeuwendeel
in imports en bedrading. Dat staat als meting in DI-4.

    python apps/run_meta_labeling.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.backtest.baseline_runner import build_weight_tracks
from tradebot.backtest.phase5_baseline import exposures_from_weights
from tradebot.backtest.regime_overlay import MarketPanels
from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.data.phase6_universe import load_phase6_universe
from tradebot.data.pit_store import PitStore
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.execution.order_router import VenueSpec
from tradebot.features.base import DataRegister, load_certified_close_panel
from tradebot.features.positioning import build_certified_micro_frame
from tradebot.features.registry import build_default_registry, current_git_sha
from tradebot.labeling.phase6_barriers import label_triple_barrier
from tradebot.reporting.phase6_meta_labeling import (
    build_h3_payload,
    render_meta_label_report,
)
from tradebot.schemas.config import (
    FeatureConfig,
    ImpactConfig,
    LabelingConfig,
    RiskConfig,
    ValidationConfig,
    load_config,
    meta_label_config,
)
from tradebot.train.meta_label import build_dataset
from tradebot.validation.meta_label_campaign import (
    MetaLabelInputs,
    run_meta_label_campaign,
)

GOV = "artefacts/governance"
PREREG = f"{GOV}/preregistration_56395fa2013768014c0c915edf346770.json"
LEDGER_UNIT = "phase6_h3_meta_labeling"
TRACK = "xs_momentum_risk_parity"

#: Werkpunten voor de precision/recall-curve. Het OORDEEL valt op
#: `probability_threshold` uit `conf/model/meta_label.yaml`; deze reeks staat
#: erbij zodat een lezer ziet wat een ander werkpunt zou hebben gedaan, en niet
#: zodat er een beter werkpunt kan worden gekozen.
CURVE = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return data


def _feature_matrices(
    root: Path, cfg: dict[str, Any], index: pd.Index
) -> dict[str, pd.DataFrame]:
    """De gecertificeerde Phase 2-featurematrix per symbool, op één tijdas."""
    features = load_config(root / "conf/features/default.yaml", FeatureConfig)
    pipeline = build_default_registry(features).pipeline()
    store = PitStore(root / cfg["data"].pit_store_root)
    register = DataRegister(root / f"{GOV}/data_hashes.json")
    out = {}
    for symbol in cfg["data"].symbols:
        frame = build_certified_micro_frame(
            store, register, symbol=symbol, granularity="1d",
            funding_granularity="8h", open_interest_granularity="1d",
            funding_tolerance=pd.Timedelta(
                hours=features.funding_tolerance_hours),
            open_interest_tolerance=pd.Timedelta(
                hours=features.open_interest_tolerance_hours),
            asset_class="crypto")
        out[symbol] = pipeline.transform(frame).values.reindex(index)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{GOV}/phase6_h3_meta_labeling.json")
    ap.add_argument("--report-out",
                    default="reports/META_LABELING_EVALUATION.md")
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    risk = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
    imp = load_config(ROOT / "conf/execution/impact.yaml", ImpactConfig)
    validation = load_config(ROOT / "conf/validation/default.yaml",
                             ValidationConfig)
    labeling = load_config(ROOT / "conf/model/labeling.yaml", LabelingConfig)
    meta = meta_label_config()
    git_sha = current_git_sha()
    universe = load_phase6_universe(
        ROOT, cfg, build_cross_sectional_momentum(cfg["alpha"]),
        git_sha=git_sha)
    index = universe.prices.index

    panel = load_certified_close_panel(
        PitStore(ROOT / cfg["data"].pit_store_root),
        DataRegister(ROOT / f"{GOV}/data_hashes.json"),
        symbols=list(cfg["data"].symbols), granularity="1d",
        asset_class="crypto")
    tracks, _ = build_weight_tracks(
        panel, build_cross_sectional_momentum(cfg["alpha"]),
        lam=cfg["vol"].ewma_lambda, vol_burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
        gross_target=1.0, git_sha=git_sha)
    base = exposures_from_weights(tracks[TRACK].loc[index].fillna(0.0))

    features = _feature_matrices(ROOT, cfg, index)
    labels = {}
    for symbol, frame in universe.ohlc.items():
        labels[symbol] = label_triple_barrier(
            high=frame["high"].to_numpy(dtype="float64"),
            low=frame["low"].to_numpy(dtype="float64"),
            close=frame["close"].to_numpy(dtype="float64"),
            sigma=universe.sigma_bar[symbol].to_numpy(dtype="float64"),
            side=np.sign(base[symbol].to_numpy(dtype="float64")),
            cfg=labeling)
    dataset = build_dataset(features, labels)

    prereg = _load(PREREG)["content"]
    power = prereg["parameters"]["power_analysis"]
    result = run_meta_label_campaign(MetaLabelInputs(
        dataset=dataset, base_exposures=base,
        panels=MarketPanels(
            prices=universe.prices, sigma_annual=universe.sigma_annual,
            sigma_bar=universe.sigma_bar, adv=universe.adv,
            volume=universe.adv, funding=pd.DataFrame(
                0.0, index=index, columns=universe.prices.columns),
            clusters=dict(risk.clusters)),
        cv=WalkForwardCV(
            train_size=validation.train_bars, test_size=validation.test_bars,
            step=validation.test_bars, mode="rolling",
            min_train=validation.train_bars,
            embargo_bars=validation.embargo_bars),
        cfg=meta, risk_cfg=risk,
        impact=ImpactParams(
            eta=imp.eta, kappa_d=imp.kappa_d, status=ImpactStatus(imp.status),
            method=imp.method, data_hash=imp.data_hash,
            sample_size=imp.sample_size, period_start=imp.period_start,
            period_end=imp.period_end, instruments=imp.instruments,
            eta_ci_low=imp.eta_ci_low, eta_ci_high=imp.eta_ci_high),
        venue=VenueSpec(
            maker_fee_bps=cfg["exec"].maker_fee_bps,
            taker_fee_bps=cfg["exec"].taker_fee_bps, funding_cap_abs=0.02,
            min_notional=10.0, latency_bars=1),
        spread_source="conf/execution/fees.yaml",
        initial_equity=cfg["bt"].initial_equity,
        bars_per_year=cfg["bt"].bars_per_year,
        half_spread_bps=cfg["exec"].assumed_half_spread_bps,
        embargo_bars=validation.embargo_bars,
        alpha=float(power["alpha"]),
        auc_threshold=float(power["threshold_auc"]),
        leak_threshold=float(next(
            c["threshold"] for c in prereg["stop_criteria"]
            if c["name"] == "negative_control_leaks")),
        events_threshold=float(next(
            c["threshold"] for c in prereg["stop_criteria"]
            if c["name"] == "events_below_adequacy")),
        effective_independent_series=float(
            power["effective_independent_series"]),
        n_symbols=int(power["n_symbols"]),
        curve_thresholds=CURVE,
        # Stop-criterium 2 verwijst naar DE DATA ADEQUACY GATE, en die heeft
        # zijn eigen meting in zijn eigen artefact. Hem hier opnieuw berekenen
        # zou een tweede definitie geven van een grootheid die er al een heeft.
        min_effective_events_per_fold=float(min(
            _load(f"{GOV}/phase6_data_adequacy.json")["details"]
            ["meta_labeling"]["effective_events_per_fold"]))))

    ledger = _load(f"{GOV}/hypothesis_ledger.json")
    total = ledger["seed_total"] + sum(e["n_trials"] for e in ledger["entries"])
    booked = next(e for e in ledger["entries"] if e["unit"] == LEDGER_UNIT)
    if len(result.trials) > booked["n_trials"]:
        raise SystemExit(
            f"De campagne draaide {len(result.trials)} specs terwijl er "
            f"{booked['n_trials']} zijn gepre-registreerd.")

    payload = build_h3_payload(
        git_sha=git_sha, universe=universe.as_record(), track=TRACK,
        preregistration=_load(PREREG),
        adequacy_artefact=_load(f"{GOV}/phase6_data_adequacy.json"),
        labeling={k: v for k, v in labeling.model_dump().items()},
        campaign=result.as_record(), ledger_before=total, ledger_after=total,
        trials_booked_at_freeze=int(booked["n_trials"]),
        ledger_unit=LEDGER_UNIT, ledger_config_hash=booked["config_hash"],
        probability_threshold=meta.probability_threshold,
        embargo_bars=validation.embargo_bars, artefact_path=args.out)

    for path, text in (
        (ROOT / args.out, json.dumps(payload, indent=2, default=str)),
        (ROOT / args.report_out, render_meta_label_report(payload)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    print(f"events: {result.n_events}  positief: {result.positive_ratio:.4f}  "
          f"uniqueness: {result.uniqueness_ratio:.4f}")
    print(f"negatieve controle (max over "
          f"{len(result.shuffled_auc_replicates)}): "
          f"{result.shuffled_auc:.4f}")
    for status, count in sorted(result.by_status().items()):
        print(f"  {status:<10} {count}")
    for trial in result.trials:
        print(f"  {trial.label:<22} AUC={trial.auc:.4f} "
              f"CI_cons=[{trial.interval_conservative[0]:.4f}, "
              f"{trial.interval_conservative[1]:.4f}] "
              f"dSharpe={trial.net_sharpe_delta:+.4f}")
    print(f"M blijft {total}\nartefact: {args.out}\nrapport:  {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
