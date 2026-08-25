# src/tradebot/risk/stress_report.py
"""Geometrische compounding, variantie-drag, en de baseline mét risicolaag.

Phase 4, stap 10 (tweede helft) en deliverable 12. De wetenschap staat hier;
`apps/run_stress.py` doet argumenten en artefact.

WAAROM DIT RAPPORT BESTAAT
--------------------------
`reports/BASELINE_BENCHMARK.md` sectie 3.1 legt de kern bloot: het 1/N-mandje
haalt een POSITIEVE Sharpe van +0,156 en verliest tegelijk 48,9% van het
kapitaal. Dat is geen fout maar rekenkunde. De Sharpe gebruikt het
REKENKUNDIGE gemiddelde; kapitaal groeit MEETKUNDIG, en het verschil is bij
benadering

    g ~= mu - sigma^2 / 2

Bij sigma = 72% is `sigma^2/2` ongeveer 26 procentpunt - precies de gemeten
drag. Een Sharpe-gate is op ongeschaalde crypto-exposure dus aantoonbaar geen
proxy voor kapitaalgroei.

Volatility targeting grijpt rechtstreeks op die term aan: het schaalt sigma
omlaag naar `sigma_target`, en de drag valt kwadratisch mee terug. Deze module
kwantificeert dat op de Phase 3-baselines in plaats van het te beweren.

WAT ER GEMETEN WORDT, EN WAT NIET
---------------------------------
De baseline wordt OPNIEUW GEDRAAID met de echte `RiskEngine` in het pad, bar
voor bar, met een causale `sigma_hat` uit L2 en een risicostaat die meeloopt
(equity, High-Water Mark, dagopening). Er wordt niets nagebootst: als de
kill switches vuren, vuren ze.

Wat dit NIET is: een claim dat de strategie daarmee winstgevend wordt. De
Phase 3-momentumtracks verliezen BRUTO al geld (bruto Sharpe -0,28); geen
risicolaag repareert een signaal zonder edge. Vol targeting verandert de
Sharpe zelfs nauwelijks - het is een schaaltransformatie. Wat het wel
verandert is het MEETKUNDIGE rendement, en dat is wat kapitaal doet.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 13.1, 14.1, 23 (Phase 4);
`reports/BASELINE_BENCHMARK.md` sectie 3.1.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .contract import MarketState, RiskState
from .engine import RiskEngine
from .kill_switches import advance_high_water_mark

__all__ = [
    "CompoundingProfile",
    "OverlayResult",
    "apply_risk_overlay",
    "compounding_profile",
]


@dataclass(frozen=True)
class CompoundingProfile:
    """Rekenkundig versus meetkundig, met de drag ertussen expliciet."""

    n_bars: int
    arithmetic_ann: float
    volatility_ann: float
    geometric_ann: float
    variance_drag_pp: float
    total_return: float
    max_drawdown: float
    sharpe: float

    def as_record(self) -> dict[str, float | int]:
        return {
            "n_bars": int(self.n_bars),
            "arithmetic_ann": float(self.arithmetic_ann),
            "volatility_ann": float(self.volatility_ann),
            "geometric_ann": float(self.geometric_ann),
            "variance_drag_pp": float(self.variance_drag_pp),
            "total_return": float(self.total_return),
            "max_drawdown": float(self.max_drawdown),
            "sharpe": float(self.sharpe),
        }


def compounding_profile(
    returns: pd.Series, *, periods_per_year: float
) -> CompoundingProfile:
    """Rekenkundig rendement, volatiliteit, CAGR en de drag daartussen.

    De drag wordt GEMETEN (`rekenkundig - meetkundig`), niet benaderd met
    `sigma^2/2`. Die benadering geldt alleen voor kleine returns en breekt
    precies waar het interessant wordt - bij een 30% gap is het verschil tussen
    `ln(1+r)` en `r` niet verwaarloosbaar.
    """
    r = returns.to_numpy(dtype="float64")
    r = r[np.isfinite(r)]
    require(
        r.size > 1,
        "Te weinig eindige rendementen voor een compounding-profiel.",
        DataContractError,
        n=int(r.size),
    )
    require(
        bool((r > -1.0).all()),
        "Een rendement van -100% of erger: het kapitaal is weg en de "
        "meetkundige reeks is niet gedefinieerd.",
        DataContractError,
        worst=float(r.min()),
    )
    ppy = float(periods_per_year)
    arithmetic = float(r.mean()) * ppy
    vol = float(r.std(ddof=1)) * math.sqrt(ppy)

    equity = np.cumprod(1.0 + r)
    total = float(equity[-1] - 1.0)
    n_years = r.size / ppy
    geometric = float(equity[-1] ** (1.0 / n_years) - 1.0)

    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(1.0 - equity / peak))
    sharpe = float(arithmetic / vol) if vol > 0.0 else 0.0

    return CompoundingProfile(
        n_bars=int(r.size),
        arithmetic_ann=arithmetic,
        volatility_ann=vol,
        geometric_ann=geometric,
        # In procentPUNTEN, zoals BASELINE_BENCHMARK.md sectie 3.1 hem noteert.
        variance_drag_pp=(arithmetic - geometric) * 100.0,
        total_return=total,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )


@dataclass(frozen=True)
class OverlayResult:
    """Eén baseline-track, met en zonder risicolaag."""

    track: str
    baseline: CompoundingProfile
    with_risk: CompoundingProfile
    n_halted_bars: int
    halt_reason: str
    binding_counts: Mapping[str, int]
    avg_gross_before: float
    avg_gross_after: float
    turnover_before: float
    turnover_after: float

    def as_record(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "baseline": self.baseline.as_record(),
            "with_risk": self.with_risk.as_record(),
            "n_halted_bars": int(self.n_halted_bars),
            "halt_reason": self.halt_reason,
            "binding_counts": dict(self.binding_counts),
            "avg_gross_before": float(self.avg_gross_before),
            "avg_gross_after": float(self.avg_gross_after),
            "turnover_before": float(self.turnover_before),
            "turnover_after": float(self.turnover_after),
            "drag_reduction_pp": float(
                self.baseline.variance_drag_pp - self.with_risk.variance_drag_pp
            ),
            "geometric_improvement_pp": float(
                (self.with_risk.geometric_ann - self.baseline.geometric_ann) * 100.0
            ),
        }


def apply_risk_overlay(
    track: str,
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    engine: RiskEngine,
    *,
    adv_usd: Mapping[str, float],
    cost_per_side: float,
    periods_per_year: float,
    baseline_returns: pd.Series,
) -> OverlayResult:
    """Draai één baseline-track opnieuw, bar voor bar, mét de risicolaag.

    Causaliteit, expliciet: het besluit op `t` gebruikt `sigma_hat[t]` en een
    risicostaat die uitsluitend equity tot en met `t` kent. De toegestane
    weging verdient vervolgens het rendement van `t+1`. Dat is exact de
    conventie van `backtest/baseline_runner.py` (`held = w.shift(1)`), zodat de
    twee reeksen vergelijkbaar zijn.

    Parameters
    ----------
    weights : de door alpha gevraagde exposure per bar (`a_t`).
    asset_returns : per-asset rendementen op dezelfde tijdas.
    sigma_hat : ex-ante geannualiseerde vol per asset, causaal (L2 EWMA).
    baseline_returns : de netto reeks ZONDER risicolaag, ter vergelijking.
    """
    index = weights.index
    require(
        bool(asset_returns.index.equals(index) and sigma_hat.index.equals(index)),
        "Gewichten, rendementen en vol-schattingen staan niet op dezelfde tijdas.",
        DataContractError,
        track=track,
    )
    symbols = [str(c) for c in weights.columns]

    state = RiskState(equity=1.0, high_water_mark=1.0, day_start_equity=1.0)
    held: dict[str, float] = dict.fromkeys(symbols, 0.0)
    prev_held: dict[str, float] = dict(held)

    net_returns: list[float] = []
    stamps: list[pd.Timestamp] = []
    binding_counts: dict[str, int] = {}
    n_halted = 0
    halt_reason = ""
    gross_after: list[float] = []
    turnover_after: list[float] = []

    for ts in index:
        # 1. Het rendement van deze bar op de weging die aan het EINDE van de
        #    vorige bar is toegestaan.
        bar_return = float(
            sum(held[s] * float(asset_returns.at[ts, s]) for s in symbols
                if np.isfinite(asset_returns.at[ts, s]))
        )
        turnover = float(sum(abs(held[s] - prev_held[s]) for s in symbols))
        net = bar_return - turnover * float(cost_per_side)
        net_returns.append(net)
        stamps.append(ts)
        turnover_after.append(turnover)

        # 2. Equity bijwerken, HWM voorwaarts, dagopening = deze bar.
        equity = max(state.equity * (1.0 + net), 1e-12)
        state = advance_high_water_mark(state, equity)
        state = RiskState(
            equity=state.equity,
            high_water_mark=state.high_water_mark,
            day_start_equity=state.equity / (1.0 + net) if net != -1.0 else state.equity,
            halted=state.halted,
            halt_reason=state.halt_reason,
            halted_at=state.halted_at,
        )

        # 3. Het risicobesluit voor de VOLGENDE bar, op informatie tot en met t.
        sigma_row = {s: float(sigma_hat.at[ts, s]) for s in symbols}
        if not all(np.isfinite(v) and v > 0.0 for v in sigma_row.values()):
            # Burn-in van de vol-estimator: er bestaat geen schatting, dus er
            # wordt niet gehandeld. Geen aanname, geen laatste bekende waarde.
            prev_held, held = dict(held), dict.fromkeys(symbols, 0.0)
            gross_after.append(0.0)
            continue

        desired = {s: float(weights.at[ts, s]) for s in symbols}
        decision = engine.decide(
            desired,
            MarketState(asof_ts=ts, sigma_hat=sigma_row, adv_usd=dict(adv_usd)),
            state,
        )
        state = decision.risk_state_out
        if state.halted:
            n_halted += 1
            halt_reason = halt_reason or state.halt_reason
        for constraint in decision.binding_constraints:
            key = constraint.kind.value
            binding_counts[key] = binding_counts.get(key, 0) + 1

        prev_held = dict(held)
        held = {s: float(decision.permitted_exposure[s]) for s in symbols}
        gross_after.append(decision.gross())

    overlaid = pd.Series(net_returns, index=pd.Index(stamps, name=index.name))
    gross_before = float(weights.abs().sum(axis=1).mean())
    turnover_before = float(
        weights.diff().abs().sum(axis=1).fillna(0.0).mean()
    )

    return OverlayResult(
        track=track,
        baseline=compounding_profile(baseline_returns, periods_per_year=periods_per_year),
        with_risk=compounding_profile(overlaid, periods_per_year=periods_per_year),
        n_halted_bars=n_halted,
        halt_reason=halt_reason,
        binding_counts=dict(sorted(binding_counts.items())),
        avg_gross_before=gross_before,
        avg_gross_after=float(np.mean(gross_after)) if gross_after else 0.0,
        turnover_before=turnover_before,
        turnover_after=float(np.mean(turnover_after)) if turnover_after else 0.0,
    )
