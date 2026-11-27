"""Van de gevalideerde config (`schemas.options.PricingConfig`) naar de pricer-instellingen."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from ...schemas.options import PricingConfig
from .engines import EngineSettings, Method
from .greeks import Bumps
from .implied_vol import IVSettings

__all__ = ["PricerSettings", "pricer_settings", "resolve_workers"]

#: Bovengrens van `workers: auto`.
_AUTO_MAX_WORKERS = 4


@dataclass(frozen=True, slots=True)
class PricerSettings:
    engines: EngineSettings
    iv: IVSettings
    bumps: Bumps
    risk_free_rate: float
    #: De keten in batches prijzen (`chain.builder`); uit = per contract met QuantLib.
    vectorized: bool = True
    #: Processen voor het QuantLib-deel van de keten (`chain.builder`); 1 = in-proces.
    workers: int = 1
    #: De pool pas bij meer QuantLib-contracten dan dit in één keten.
    pool_min_contracts: int = 16


def resolve_workers(workers: int | Literal["auto"]) -> int:
    """`auto` = min(4, aantal cores); een getal blijft wat het is."""
    if workers == "auto":
        return min(_AUTO_MAX_WORKERS, os.cpu_count() or 1)
    return workers


def pricer_settings(cfg: PricingConfig) -> PricerSettings:
    return PricerSettings(
        engines=EngineSettings(binomial_steps=cfg.binomial_steps, fd_time_steps=cfg.fd_time_steps,
                               fd_grid_points=cfg.fd_grid_points, scan=Method(cfg.scan_engine),
                               decision=Method(cfg.decision_engine)),
        iv=IVSettings(accuracy=cfg.iv_accuracy, max_evaluations=cfg.iv_max_evaluations,
                      min_vol=cfg.iv_min, max_vol=cfg.iv_max),
        bumps=Bumps(vol=cfg.vega_bump_vol, rate=cfg.rho_bump_rate),
        risk_free_rate=cfg.risk_free_rate,
        vectorized=cfg.vectorized,
        workers=resolve_workers(cfg.workers),
        pool_min_contracts=cfg.pool_min_contracts,
    )
