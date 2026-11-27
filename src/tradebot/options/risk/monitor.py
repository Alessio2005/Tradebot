"""Dagverlies, drawdown en de-grossing (detailontwerp 8.7).

* Dagverlies, op de mid, tegen de equity van de vorige sluiting: vanaf
  `daily_loss_stop_entries_frac` geen entries tot de volgende handelsdag; vanaf
  `daily_loss_degross_frac` stelt `degross_plan` sluitingen voor, grootste bijdrage aan het
  slechtste scenario eerst, tot alle grenzen weer gehaald worden.
* Drawdown vanaf de high-water mark vanaf `drawdown_halt_frac`: een `HaltRecord` voor de
  bestaande `HaltStore` (`risk/kill_switches.py`). Die overleeft een herstart en gaat alleen met
  de hand weer open. Exits, stops en het sluiten vóór expiratie gaan in HALT door.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ...risk.kill_switches import HaltRecord
from ...schemas.options import RiskConfig
from ..positions import Book, MarketState, PositionStatus, StrategyPosition
from ..pricing.engines import EngineSettings, Method
from .limits import check, measure
from .scenarios import Scenario, position_values

__all__ = ["RiskStatus", "assess", "degross_plan"]


@dataclass(frozen=True, slots=True)
class RiskStatus:
    daily_loss_frac: float
    drawdown_frac: float
    entries_blocked: bool
    degross: bool
    halt: HaltRecord | None


def assess(equity: float, prev_close_equity: float, high_water_mark: float, cfg: RiskConfig,
           now: datetime) -> RiskStatus:
    daily = max(prev_close_equity - equity, 0.0) / prev_close_equity if prev_close_equity else 0.0
    hwm = max(high_water_mark, equity)
    dd = (hwm - equity) / hwm if hwm > 0 else 0.0
    halt = None
    if dd >= cfg.drawdown_halt_frac:
        halt = HaltRecord("options_drawdown", "drawdown vanaf de high-water mark", dd,
                          cfg.drawdown_halt_frac, "risk.drawdown_halt_frac", now.isoformat())
    return RiskStatus(daily, dd, daily >= cfg.daily_loss_stop_entries_frac,
                      daily >= cfg.daily_loss_degross_frac, halt)


def degross_plan(book: Book, market: MarketState, beta: Mapping[str, float],
                 scenarios: Sequence[Scenario], cfg: RiskConfig, method: Method,
                 engines: EngineSettings) -> list[str]:
    """Positie-ids om te sluiten, in volgorde, tot het boek alle grenzen haalt."""
    live = [p for p in book.positions if p.status is not PositionStatus.CLOSED]

    def ok(ps: Sequence[StrategyPosition],
           shares: Mapping[str, int] = book.shares) -> tuple[bool, tuple[float, ...] | None]:
        m = measure(tuple(ps), shares, market, beta, scenarios, method, engines)
        return all(c.ok for c in check(m, cfg, market.capital_usd)), m.scenario_losses

    good, losses = ok(live)
    if good:
        return []
    worst = (scenarios[max(range(len(scenarios)), key=lambda i: losses[i])]
             if losses else Scenario(0.0, 0.0))
    args = (live, book.shares, market.chains, market.iv, beta, market.now)
    base = position_values(*args, Scenario(0.0, 0.0), method, engines) or {}
    hit = position_values(*args, worst, method, engines) or {}
    order = sorted(live, key=lambda p: (-(base.get(p.position_id, 0.0)
                                          - hit.get(p.position_id, 0.0)), p.position_id))
    plan: list[str] = []
    for p in order:
        plan.append(p.position_id)
        rest = [x for x in live if x.position_id not in plan]
        # De aandelen-legs van een te sluiten positie gaan mee weg (review 2026-11-27, N18):
        # anders meet elke deelverzameling met het volledige aandelenbezit.
        shares = dict(book.shares)
        for x in live:
            if x.position_id in plan and x.shares:
                shares[x.underlying] = shares.get(x.underlying, 0) - x.shares
        if ok(rest, shares)[0]:
            break
    return plan
