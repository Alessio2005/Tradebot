"""H2 — M0 Causal Vol-Buckets tegen de Markov-familie. Stap 11, deliverable 18.

De wetenschap staat in `validation/regime_benchmark.py`, de opmaak in
`reporting/phase6_regime_benchmark.py`; deze app doet uitsluitend argumenten,
data en artefacten -- er staat geen enkele meting in.

R-6 stelt een grens van 80 regels; dit bestand telt er meer, met het leeuwendeel
in imports en bedrading. Dat staat als meting in DI-4, dat de regel repo-breed
volgt, en wordt hier niet weggeschreven als "past wel".

    python apps/run_regime_benchmark.py
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
from tradebot.features.registry import current_git_sha
from tradebot.regime.buckets import classify_vol_buckets
from tradebot.reporting.phase6_regime_benchmark import (
    build_h2_payload,
    render_regime_report,
)
from tradebot.schemas.config import (
    AdequacyConfig,
    ImpactConfig,
    RiskConfig,
    ValidationConfig,
    adequacy_config,
    load_config,
    regime_config,
)
from tradebot.validation.regime_benchmark import CampaignInputs, run_regime_campaign

GOV = "artefacts/governance"
PREREG = f"{GOV}/preregistration_3d3af28730a6c7f9da48d13139522a05.json"
LEDGER_UNIT = "phase6_h2_hmm_vs_m0"

#: De primaire signaaltrack uit de pre-registratie. `long_only_equal_weight`
#: halteert op 2022-05-10 en handelt ~130 van 1.743 bars; elk regime-experiment
#: daarop meet de eerste zes maanden en daarna niets.
TRACK = "xs_momentum_risk_parity"
#: Aangenomen half-spreads voor de sensitiviteitsanalyse (§0.3). De basis is
#: 1,0 bp uit `conf/execution/fees.yaml`; de grens van de pre-registratie is 3.
SPREAD_SWEEP = (1.0, 2.0, 3.0, 5.0, 10.0)


def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{GOV}/phase6_h2_regime_benchmark.json")
    ap.add_argument("--report-out", default="reports/M0_VS_HMM_BENCHMARK.md")
    ap.add_argument("--control-replicates", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    risk = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
    imp = load_config(ROOT / "conf/execution/impact.yaml", ImpactConfig)
    validation = load_config(ROOT / "conf/validation/default.yaml",
                             ValidationConfig)
    adequacy = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
    regime = regime_config()
    git_sha = current_git_sha()
    universe = load_phase6_universe(
        ROOT, cfg, build_cross_sectional_momentum(cfg["alpha"]),
        git_sha=git_sha)

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
    index = universe.prices.index
    base = exposures_from_weights(tracks[TRACK].loc[index].fillna(0.0))

    buckets = pd.DataFrame(
        {symbol: classify_vol_buckets(
            frame, regime.m0, ewma_lambda=cfg["vol"].ewma_lambda,
            ewma_burn_in_bars=cfg["vol"].burn_in_bars, symbol=symbol).buckets
         for symbol, frame in universe.ohlc.items()},
        columns=list(universe.prices.columns))
    returns = universe.log_returns

    prereg = _load(PREREG)["content"]
    power = prereg["parameters"]["power_analysis"]
    result = run_regime_campaign(CampaignInputs(
        returns=returns, buckets=buckets.loc[returns.index],
        base_exposures=base,
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
        adequacy=adequacy, m2_cfg=regime.m2, risk_cfg=risk,
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
        base_half_spread_bps=cfg["exec"].assumed_half_spread_bps,
        spread_sweep_bps=SPREAD_SWEEP,
        alpha=adequacy_config().power.alpha,
        expected_effect=float(power["expected_sharpe_gain"]),
        minimum_detectable_effect=float(
            power["scenarios"]["rho_0.95"]["minimum_detectable_effect"]),
        target_power=adequacy_config().power.target_power,
        n_control_replicates=args.control_replicates,
        block_length=float(validation.spa_block_length), seed=args.seed))

    # M IS AL GEBOEKT. De zes trials staan sinds het bevriezen van de
    # pre-registratie in de ledger; deze run voert ze uit en boekt niets bij.
    ledger = _load(f"{GOV}/hypothesis_ledger.json")
    total = ledger["seed_total"] + sum(e["n_trials"] for e in ledger["entries"])
    booked = next(e for e in ledger["entries"] if e["unit"] == LEDGER_UNIT)
    if result.n_trials > booked["n_trials"]:
        raise SystemExit(
            f"De campagne draaide {result.n_trials} conditioneerders terwijl "
            f"er {booked['n_trials']} zijn gepre-registreerd. Meer zoeken dan "
            f"vooraf geboekt maakt elke DSR erna te gunstig.")

    payload = build_h2_payload(
        git_sha=git_sha, universe=universe.as_record(),
        track=TRACK, preregistration=_load(PREREG),
        adequacy_artefact=_load(f"{GOV}/phase6_data_adequacy.json"),
        campaign=result.as_record(), ledger_before=total, ledger_after=total,
        trials_booked_at_freeze=int(booked["n_trials"]),
        ledger_unit=LEDGER_UNIT, ledger_config_hash=booked["config_hash"],
        spread_sweep_bps=SPREAD_SWEEP, artefact_path=args.out,
        n_oos_bars=int(result.m0.mask.sum()),
        embargo_bars=validation.embargo_bars)

    for path, text in (
        (ROOT / args.out, json.dumps(payload, indent=2, default=str)),
        (ROOT / args.report_out, render_regime_report(payload)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    print(f"trials: {result.n_trials} van {booked['n_trials']} geboekt; "
          f"M blijft {total}")
    for status, count in sorted(result.by_status().items()):
        print(f"  {status:<10} {count}")
    print(f"M0 netto Sharpe OOS: {result.m0.net_sharpe_oos:+.4f} "
          f"(ongeconditioneerd {result.plain.net_sharpe_oos:+.4f})")
    print(f"artefact: {args.out}\nrapport:  {args.report_out}")
    return int(np.isnan(result.m0.net_sharpe_oos))


if __name__ == "__main__":
    raise SystemExit(main())
