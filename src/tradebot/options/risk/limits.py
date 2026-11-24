"""De maten van het boek en de grenzen erop (detailontwerp 8.2-8.6).

Alle maten zijn lineair in het aantal eenheden van een nieuwe positie (Greeks, max verlies,
buying power en het verlies per scenario bij vaste spot en vol). De poort rekent de maten van het
boek en van één eenheid daarom één keer uit en verkleint een order zonder opnieuw te waarderen.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...schemas.options import RiskConfig
from ..positions import MarketState, PositionStatus, StrategyPosition
from ..pricing.engines import EngineSettings, Method
from .aggregate import aggregate
from .margin import buying_power
from .scenarios import Scenario, position_values

__all__ = ["BookMetrics", "CheckResult", "check", "measure"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    measured: float
    threshold: float

    @property
    def ok(self) -> bool:
        return self.measured <= self.threshold


@dataclass(frozen=True, slots=True)
class BookMetrics:
    beta_delta_usd: float
    vega_usd: float
    gamma_loss_5pct: float
    max_loss_by_underlying: Mapping[str, float]
    buying_power: float
    scenario_losses: tuple[float, ...] | None     # in de volgorde van het raster
    complete: bool

    def plus(self, other: BookMetrics, k: float = 1.0) -> BookMetrics:
        """`self + k × other`."""
        names = set(self.max_loss_by_underlying) | set(other.max_loss_by_underlying)
        loss = {n: self.max_loss_by_underlying.get(n, 0.0)
                + k * other.max_loss_by_underlying.get(n, 0.0) for n in sorted(names)}
        sc = (None if self.scenario_losses is None or other.scenario_losses is None
              else tuple(a + k * b for a, b in zip(self.scenario_losses, other.scenario_losses,
                                                   strict=True)))
        return BookMetrics(self.beta_delta_usd + k * other.beta_delta_usd,
                           self.vega_usd + k * other.vega_usd,
                           self.gamma_loss_5pct + k * other.gamma_loss_5pct, loss,
                           self.buying_power + k * other.buying_power, sc,
                           self.complete and other.complete)


def measure(positions: Sequence[StrategyPosition], shares: Mapping[str, int],
            market: MarketState, beta: Mapping[str, float], scenarios: Sequence[Scenario],
            method: Method, engines: EngineSettings) -> BookMetrics:
    live = tuple(p for p in positions if p.status is not PositionStatus.CLOSED)
    g = aggregate(live, shares, market.chains, beta)
    loss: dict[str, float] = {}
    for p in live:
        if not math.isinf(p.max_loss):
            loss[p.underlying] = loss.get(p.underlying, 0.0) + p.max_loss
    args = (live, shares, market.chains, market.iv, beta, market.now)
    base = position_values(*args, Scenario(0.0, 0.0), method, engines)
    sc: tuple[float, ...] | None = None
    if base is not None:
        b = sum(base.values())
        sc = tuple(b - sum(position_values(*args, s, method, engines).values())  # type: ignore[union-attr]
                   for s in scenarios)
    return BookMetrics(g.beta_delta_usd, g.vega_usd, g.gamma_loss_5pct, loss,
                       buying_power(live, market.chains), sc, g.complete)


def check(m: BookMetrics, cfg: RiskConfig, capital: float) -> tuple[CheckResult, ...]:
    worst = max(m.scenario_losses) if m.scenario_losses else math.inf
    per_und = max(m.max_loss_by_underlying.values(), default=0.0)
    return (
        CheckResult("greeks_complete", 0.0 if m.complete else 1.0, 0.0),
        CheckResult("beta_delta_usd", abs(m.beta_delta_usd),
                    cfg.max_abs_beta_delta_frac * capital),
        CheckResult("vega_usd", abs(m.vega_usd), cfg.max_vega_frac * capital),
        CheckResult("gamma_loss_5pct", m.gamma_loss_5pct, cfg.max_gamma_loss_5pct_frac * capital),
        CheckResult("max_loss_per_underlying", per_und,
                    cfg.max_loss_per_underlying_frac * capital),
        CheckResult("buying_power", m.buying_power, cfg.max_buying_power_frac * capital),
        CheckResult("scenario_loss", worst, cfg.max_scenario_loss_frac * capital),
    )
