"""Van de gevalideerde config (`schemas.options.PricingConfig`) naar de pricer-instellingen."""
from __future__ import annotations

from dataclasses import dataclass

from ...schemas.options import PricingConfig
from .engines import EngineSettings, Method
from .greeks import Bumps
from .implied_vol import IVSettings

__all__ = ["PricerSettings", "pricer_settings"]


@dataclass(frozen=True, slots=True)
class PricerSettings:
    engines: EngineSettings
    iv: IVSettings
    bumps: Bumps
    risk_free_rate: float


def pricer_settings(cfg: PricingConfig) -> PricerSettings:
    return PricerSettings(
        engines=EngineSettings(binomial_steps=cfg.binomial_steps, fd_time_steps=cfg.fd_time_steps,
                               fd_grid_points=cfg.fd_grid_points, scan=Method(cfg.scan_engine),
                               decision=Method(cfg.decision_engine)),
        iv=IVSettings(accuracy=cfg.iv_accuracy, max_evaluations=cfg.iv_max_evaluations,
                      min_vol=cfg.iv_min, max_vol=cfg.iv_max),
        bumps=Bumps(vol=cfg.vega_bump_vol, rate=cfg.rho_bump_rate),
        risk_free_rate=cfg.risk_free_rate,
    )
