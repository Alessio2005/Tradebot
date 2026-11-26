"""Scenario's: het hele boek volledig herwaardeerd bij andere spot en vol (detailontwerp 8.4).

Raster: spot × (1 + s × beta) voor `s` in `scenario_spot` (en 0), IV + `v` volpunten voor `v`
in `scenario_vol_points`, voor alle onderliggende tegelijk. Elke leg wordt herwaardeerd met de
gekozen engine op hetzelfde tijdstip (geen tijdsverloop). Verlies = waarde in het basisscenario
(s = 0, v = 0, met hetzelfde model) − waarde in het scenario, zodat een verschil tussen mid en
model geen schijnverlies geeft.

De vol van een leg is zijn eigen IV uit de keten, anders de ATM-IV van de onderliggende. Zonder
beide is het scenario onbekend (`None`): de poort weigert dan nieuwe entries.

Een leg met een discreet dividend in de looptijd gaat altijd met FD (de andere engines kennen
geen discrete dividenden), op een grover rooster (`RISK_FD_*`): het raster vraagt honderden
waarderingen per controle, en voor een verliesgrens is een fout van een paar cent genoeg.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from ..chain.builder import Chain
from ..iv_metrics import IVMetrics
from ..positions import MULTIPLIER, Leg, PositionStatus, StrategyPosition
from ..pricing.engines import EngineSettings, Method
from ..pricing.vanilla import OptionInputs, price

__all__ = ["Scenario", "ScenarioResult", "grid", "position_values", "run_scenarios"]

MIN_VOL = 0.01
RISK_FD_TIME_STEPS = 50
RISK_FD_GRID_POINTS = 100


@dataclass(frozen=True, slots=True)
class Scenario:
    spot_move: float          # fractie, voor beta 1
    vol_points: float         # volpunten (10 = +0,10)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    worst: Scenario
    worst_loss: float                     # USD, + is verlies
    losses: tuple[tuple[Scenario, float], ...]


def grid(spot_moves: Iterable[float], vol_points: Iterable[float]) -> tuple[Scenario, ...]:
    moves = sorted({0.0, *spot_moves})
    return tuple(Scenario(s, v) for s in moves for v in sorted(set(vol_points)))


def _vol(chain: Chain, pos_leg: Leg, iv: IVMetrics | None) -> float | None:
    c = pos_leg.contract
    o = chain.find(c.expiry, c.strike, c.right)
    if o is not None and o.iv is not None:
        return o.iv
    return iv.atm_iv if iv is not None else None


def position_values(positions: Sequence[StrategyPosition], shares: Mapping[str, int],
                    chains: Mapping[str, Chain], iv: Mapping[str, IVMetrics],
                    beta: Mapping[str, float], now: datetime, scenario: Scenario,
                    method: Method, engines: EngineSettings) -> dict[str, float] | None:
    """Waarde per positie-id (en `shares:<naam>` per aandelenbezit) in één scenario."""
    out: dict[str, float] = {}
    coarse = replace(engines, fd_time_steps=min(engines.fd_time_steps, RISK_FD_TIME_STEPS),
                     fd_grid_points=min(engines.fd_grid_points, RISK_FD_GRID_POINTS))
    for p in positions:
        if p.status is PositionStatus.CLOSED:
            continue
        chain = chains.get(p.underlying)
        if chain is None:
            return None
        spot = chain.spot * (1 + scenario.spot_move * beta[p.underlying])
        total = 0.0
        for leg in p.option_legs:
            vol = _vol(chain, leg, iv.get(p.underlying))
            if vol is None:
                return None
            c = leg.contract
            inp = OptionInputs(spot, c.strike, c.right, c.expiry, now, chain.carry)  # type: ignore[union-attr]
            v = max(vol + scenario.vol_points / 100.0, MIN_VOL)
            if inp.expired:
                px = inp.intrinsic
            elif inp.discrete_dividends:
                px = price(inp, v, Method.FD, coarse)
            else:
                px = price(inp, v, method, engines)
            total += leg.qty * px * MULTIPLIER
        out[p.position_id] = total
    for und, n in sorted(shares.items()):
        if not n:
            continue
        chain = chains.get(und)
        if chain is None:
            return None
        out[f"shares:{und}"] = n * chain.spot * (1 + scenario.spot_move * beta[und])
    return out


def run_scenarios(positions: Sequence[StrategyPosition], shares: Mapping[str, int],
                  chains: Mapping[str, Chain], iv: Mapping[str, IVMetrics],
                  beta: Mapping[str, float], now: datetime, scenarios: Sequence[Scenario],
                  method: Method, engines: EngineSettings) -> ScenarioResult | None:
    base = position_values(positions, shares, chains, iv, beta, now, Scenario(0.0, 0.0),
                           method, engines)
    if base is None:
        return None
    b = sum(base.values())
    losses = []
    for s in scenarios:
        vals = position_values(positions, shares, chains, iv, beta, now, s, method, engines)
        losses.append((s, b - sum(vals.values())))  # type: ignore[union-attr]
    worst = max(losses, key=lambda x: (x[1], -abs(x[0].spot_move), -abs(x[0].vol_points)))
    return ScenarioResult(worst[0], worst[1], tuple(losses))
