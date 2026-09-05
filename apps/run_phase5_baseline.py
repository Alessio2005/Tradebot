"""Phase 5, §20 — herwaardeer de Phase 3-baseline door de volledige keten.

De wetenschap staat in `backtest/phase5_baseline.py`; deze app doet argumenten,
data en artefact (R-6: apps <= 80 LOC).

    python apps/run_phase5_baseline.py
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

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.backtest.baseline_runner import CostModel, build_weight_tracks
from tradebot.backtest.phase5_baseline import run_all_layers, summarise
from tradebot.data.funding_panel import daily_funding_panel
from tradebot.data.pit_store import PitStore
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.execution.order_router import SpreadModel, SpreadStatus, VenueSpec
from tradebot.features.base import (
    ASOF_INDEX_NAME,
    DataRegister,
    load_certified_close_panel,
    load_certified_series,
)
from tradebot.features.registry import current_git_sha
from tradebot.schemas.config import ImpactConfig, RiskConfig, load_config
from tradebot.volatility.ewma import ewma_volatility_panel


def load_market(root: Path, cfg: dict) -> dict:
    """Prijzen, volatiliteit, causale ADV, bar-volume en funding."""
    store = PitStore(root / cfg["data"].pit_store_root)
    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    symbols = list(cfg["data"].symbols)
    prices = load_certified_close_panel(
        store, register, symbols=symbols, granularity="1d",
        asset_class="crypto").values
    sigma = ewma_volatility_panel(
        prices, lam=cfg["vol"].ewma_lambda, burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor)
    turnover = {}
    for symbol in symbols:
        df, _ = load_certified_series(store, register, asset_class="crypto",
                                      dataset="ohlcv", symbol=symbol,
                                      granularity="1d")
        idx = pd.DatetimeIndex(pd.to_datetime(df["asof_ts_ns"].to_numpy(),
                                              unit="ns", utc=True),
                               name=ASOF_INDEX_NAME)
        turnover[symbol] = pd.Series(df["turnover"].to_numpy(dtype="float64"),
                                     index=idx)
    volume = pd.DataFrame(turnover).reindex(prices.index)
    # Causaal: de turnover van bar t is pas op zijn close bekend.
    adv = volume.rolling(30, min_periods=30).mean().shift(1)
    funding = daily_funding_panel(
        store, register, symbols=symbols, asset_class="crypto",
        funding_granularity="8h", bar_index=prices.index)
    return {"prices": prices, "sigma": sigma, "adv": adv, "volume": volume,
            "funding": funding, "annualisation": cfg["vol"].annualisation_factor}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artefacts/baseline/phase5_revaluation.json")
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    risk = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
    imp = load_config(ROOT / "conf/execution/impact.yaml", ImpactConfig)
    market = load_market(ROOT, cfg)
    tracks, _ = build_weight_tracks(
        load_certified_close_panel(
            PitStore(ROOT / cfg["data"].pit_store_root),
            DataRegister(ROOT / "artefacts/governance/data_hashes.json"),
            symbols=list(cfg["data"].symbols), granularity="1d",
            asset_class="crypto"),
        build_cross_sectional_momentum(cfg["alpha"]),
        lam=cfg["vol"].ewma_lambda, vol_burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor,
        gross_target=1.0, git_sha=current_git_sha())

    usable = market["sigma"].dropna(how="any").index
    usable = usable[usable.isin(market["adv"].dropna(how="any").index)]
    cost = CostModel(taker_fee_bps=cfg["exec"].taker_fee_bps,
                     half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                     is_provisional=cfg["exec"].cost_assumption_is_provisional)
    params = ImpactParams(
        eta=imp.eta, kappa_d=imp.kappa_d, status=ImpactStatus(imp.status),
        method=imp.method, data_hash=imp.data_hash, sample_size=imp.sample_size,
        period_start=imp.period_start, period_end=imp.period_end,
        instruments=imp.instruments, eta_ci_low=imp.eta_ci_low,
        eta_ci_high=imp.eta_ci_high)

    rows = []
    for name, weights in sorted(tracks.items()):
        rows.extend(run_all_layers(
            name, weights.loc[usable].fillna(0.0),
            market["prices"].loc[usable],
            market["sigma"].loc[usable],
            market["sigma"].loc[usable] / float(np.sqrt(market["annualisation"])),
            market["adv"].loc[usable], market["volume"].loc[usable],
            market["funding"].loc[usable],
            dict(risk.clusters), risk_cfg=risk, impact=params,
            venue=VenueSpec(maker_fee_bps=cfg["exec"].maker_fee_bps,
                            taker_fee_bps=cfg["exec"].taker_fee_bps,
                            funding_cap_abs=0.02, min_notional=10.0,
                            latency_bars=1),
            spread=SpreadModel(half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                               status=SpreadStatus.SPREAD_ASSUMED,
                               source="conf/execution/fees.yaml"),
            initial_equity=cfg["bt"].initial_equity,
            cost_per_side=cost.per_side,
            bars_per_year=cfg["bt"].bars_per_year))

    frame = summarise(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"git_sha": current_git_sha(), "n_bars": int(len(usable)),
         "period": [str(usable[0].date()), str(usable[-1].date())],
         "rows": frame.to_dict(orient="records")},
        indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
