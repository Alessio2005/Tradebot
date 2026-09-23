# src/tradebot/backtest/ladder_inputs.py
"""De invoer van de vier-lagen-ladder, op één plek. Fase 11, stap 3.

WAAROM DIT BESTAND BESTAAT
==========================
`apps/run_phase5_baseline.py` bouwde de ladderinvoer (gecertificeerde prijzen,
EWMA-volatiliteit, causale ADV, funding, de vier gewichtstracks, de kosten- en
impactparameters) in zijn eigen `main()`. Twee fase-10-apps
(`run_h10_1_decision_frequency.py`, `run_h10_2_unbalanced_panel.py`) importeerden
`load_market` daaruit en kopieerden de rest. Fase 11 stap 3 moet dezelfde
ladder onder TWEE beleidsregels meten; een vierde kopie zou een audit opleveren
van een paneel dat stil kan gaan afwijken van het paneel van de ladder (R-3).

`load_market` staat hier letterlijk zoals het in de app stond, en de app
re-exporteert het, zodat `from run_phase5_baseline import load_market` in de
fase-10-apps blijft werken. Die twee apps zijn gesloten campagnes en worden niet
aangeraakt.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..alpha.momentum import build_cross_sectional_momentum
from ..data.funding_panel import daily_funding_panel
from ..data.pit_store import PitStore
from ..execution.impact_model import ImpactParams, ImpactStatus
from ..execution.order_router import SpreadModel, SpreadStatus, VenueSpec
from ..features.base import (
    ASOF_INDEX_NAME,
    DataRegister,
    load_certified_close_panel,
    load_certified_series,
)
from ..features.registry import current_git_sha
from ..schemas.config import ImpactConfig, load_config
from ..volatility.ewma import ewma_volatility_panel
from .baseline_report import load_baseline_configs
from .baseline_runner import CostModel, build_weight_tracks
from .phase5_baseline import EngineFactory, LayerResult, run_all_layers

__all__ = ["LadderInputs", "load_ladder_inputs", "load_market", "run_ladder_track"]


def load_market(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
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


@dataclass(frozen=True)
class LadderInputs:
    """Alles wat de ladder leest, behalve het risicobeleid."""

    cfg: dict[str, Any]
    market: dict[str, Any]
    tracks: dict[str, pd.DataFrame]
    usable: pd.DatetimeIndex
    cost: CostModel
    impact: ImpactParams


def load_ladder_inputs(root: Path) -> LadderInputs:
    """De invoer zoals `apps/run_phase5_baseline.py` haar tot fase 11 zelf bouwde."""
    cfg = load_baseline_configs(root)
    imp = load_config(root / "conf/execution/impact.yaml", ImpactConfig)
    market = load_market(root, cfg)
    tracks, _ = build_weight_tracks(
        load_certified_close_panel(
            PitStore(root / cfg["data"].pit_store_root),
            DataRegister(root / "artefacts/governance/data_hashes.json"),
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
    return LadderInputs(cfg=cfg, market=market, tracks=dict(tracks),
                        usable=usable, cost=cost, impact=params)


def run_ladder_track(
    inputs: LadderInputs,
    track: str,
    risk_cfg: Any,
    *,
    engine_factory: EngineFactory | None = None,
) -> list[LayerResult]:
    """Eén track door alle vier de lagen, onder het gegeven risicobeleid."""
    cfg, market, usable = inputs.cfg, inputs.market, inputs.usable
    extra = {} if engine_factory is None else {"engine_factory": engine_factory}
    return run_all_layers(
        track, inputs.tracks[track].loc[usable].fillna(0.0),
        market["prices"].loc[usable],
        market["sigma"].loc[usable],
        market["sigma"].loc[usable] / float(np.sqrt(market["annualisation"])),
        market["adv"].loc[usable], market["volume"].loc[usable],
        market["funding"].loc[usable],
        dict(risk_cfg.clusters), risk_cfg=risk_cfg, impact=inputs.impact,
        venue=VenueSpec(maker_fee_bps=cfg["exec"].maker_fee_bps,
                        taker_fee_bps=cfg["exec"].taker_fee_bps,
                        funding_cap_abs=0.02, min_notional=10.0,
                        latency_bars=1),
        spread=SpreadModel(half_spread_bps=cfg["exec"].assumed_half_spread_bps,
                           status=SpreadStatus.SPREAD_ASSUMED,
                           source="conf/execution/fees.yaml"),
        initial_equity=cfg["bt"].initial_equity,
        cost_per_side=inputs.cost.per_side,
        bars_per_year=cfg["bt"].bars_per_year, **extra)
