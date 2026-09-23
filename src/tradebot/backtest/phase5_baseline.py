# src/tradebot/backtest/phase5_baseline.py
"""De Phase 3-baseline, opnieuw gedraaid door de volledige Phase 5-keten.

Fase-opdracht §20, exit-criterium 18. De wetenschap achter
`apps/run_phase5_baseline.py`.

WAT HIER OPNIEUW WORDT GEMETEN, EN WAAROM DAT NODIG IS
-------------------------------------------------------
`reports/BASELINE_BENCHMARK.md` en `reports/phase3_exit_report.md` zijn
geproduceerd door een pad dat de soevereine risicolaag NOOIT aanraakte
(`reports/phase5_sovereign_wiring_audit.md` §2.1). Dat is geen verwijt aan die
rapporten - zij zeggen het zelf - maar het betekent dat dit de EERSTE meting is
van het systeem zoals het nu bestaat.

De vergelijking loopt over vier lagen, elk apart aan te zetten, zodat het
verschil toewijsbaar is in plaats van één samengevat getal:

    L0  vectorized, Phase 3-conventie          <- de bestaande baseline
    L1  + soevereine risicolaag                <- wat L7 met de gewichten doet
    L2  + executie-latency (1 bar)             <- de shift(1)/shift(2)-breuk
    L3  + spread, impact, fees, funding        <- volledige execution realism

WAT `a_t` IS EN WAT HET NIET IS
--------------------------------
De engine krijgt `a_t in [-1, +1]` van de alpha-unit, precies zoals het
L4-contract voorschrijft, en de soevereine laag bepaalt de positiegrootte. De
Phase 3-tracks daarentegen werden geSIZED door een allocator (`risk_parity` /
`equal_weight`) en daarna genormaliseerd op `gross_target`.

Dat is een ECHT verschil en het wordt hier niet weggepoetst. De vergelijking
draait daarom per track op de gewichten die die track produceerde, hernormaliseerd
naar `a_t` door te delen door de grootste absolute waarde per bar. Dat behoudt de
RELATIEVE verdeling van de allocator - waar de track over gaat - en laat de
absolute schaal over aan L7, waar hij hoort.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 15, 16.1, 19, 21, 26.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..execution.impact_model import ImpactParams
from ..execution.order_router import (
    OrderRouter,
    SpreadModel,
    VenueSpec,
)
from ..risk.contract import MarketState, RiskState
from ..risk.engine import RiskEngine, risk_config_hash
from ..utils.failfast import DataContractError, require
from .engine import EventDrivenEngine, build_slices, exposures_from_frame
from .metrics import calmar_ratio, max_drawdown, sharpe_ratio
from .vectorized import run_vectorized

__all__ = [
    "LAYERS",
    "LayerResult",
    "build_engine",
    "exposures_from_weights",
    "run_all_layers",
    "summarise",
]

#: De vier lagen, in volgorde van toenemend realisme.
LAYERS: tuple[str, ...] = ("L0_vectorized", "L1_sovereign", "L2_latency",
                           "L3_execution")

#: Verwaarloosbare exposure. Onder deze waarde bestaat de positie niet.
_TOL = 1e-12

#: Bouwt de `RiskEngine` voor één laag (`"L1_sovereign"` of `"L3_execution"`).
#: De standaard is een gewone `RiskEngine`; fase 11 stap 3 geeft hier een
#: `TracingRiskEngine` in, zodat de bindingsaudit DEZE lussen meet in plaats van
#: een nabouw ervan (R-3).
EngineFactory = Callable[[str, Any], RiskEngine]


def _plain_engine(_layer: str, risk_cfg: Any) -> RiskEngine:
    return RiskEngine(risk_cfg)


@dataclass(frozen=True)
class LayerResult:
    """Wat één laag opleverde, met alles wat §20 uitgevraagd wil hebben."""

    layer: str
    track: str
    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    metrics: dict[str, float]
    costs: dict[str, float]
    audit: dict[str, Any]

    def as_record(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "track": self.track,
            **{k: float(v) for k, v in self.metrics.items()},
            **{f"cost_{k}": float(v) for k, v in self.costs.items()},
            **self.audit,
        }


def exposures_from_weights(weights: pd.DataFrame) -> pd.DataFrame:
    """Hernormaliseer allocator-gewichten naar `a_t in [-1, +1]`.

    Deelt per bar door de grootste absolute waarde. De RELATIEVE verdeling van
    de allocator blijft daarmee exact behouden - dat is waar een track over
    gaat - en de absolute schaal komt van L7.

    Een bar waarop alles nul is, blijft nul: dat is een vlak boek, geen
    deling door nul.
    """
    peak = weights.abs().max(axis=1)
    scaled = weights.div(peak.where(peak > _TOL), axis=0)
    return scaled.fillna(0.0)


def build_engine(
    risk_cfg: Any,
    impact: ImpactParams,
    venue: VenueSpec,
    spread: SpreadModel,
    *,
    initial_equity: float,
    risk_engine: RiskEngine | None = None,
) -> EventDrivenEngine:
    """De authoritative engine, met alle vier de lagen aangesloten.

    `risk_engine` is er voor één aanroeper: de bindingsaudit van fase 11 stap 3,
    die een `TracingRiskEngine` inzet om deze engine te meten zonder haar na te
    bouwen. Hij moet `risk_cfg` dragen; een engine met een ander beleid dan de
    router meldt, zou een besluit onder de verkeerde hash laten reizen.
    """
    risk = risk_engine if risk_engine is not None else RiskEngine(risk_cfg)
    require(
        risk.config_hash == risk_config_hash(risk_cfg),
        "De meegegeven RiskEngine draagt een ander beleid dan risk_cfg. De "
        "router zou dan een config_hash melden die niet heeft beslist (AD-27).",
        DataContractError,
        engine_hash=risk.config_hash, config_hash=risk_config_hash(risk_cfg),
    )
    return EventDrivenEngine(
        risk_engine=risk,
        router=OrderRouter(venue=venue, spread=spread, impact=impact,
                           risk_config_hash=risk.config_hash),
        initial_equity=initial_equity,
    )


def _metrics(
    equity: pd.Series, net: pd.Series, gross: pd.Series, *, bars_per_year: float
) -> dict[str, float]:
    values = equity.to_numpy(dtype="float64")
    mdd, _, _ = max_drawdown(values)
    net_arr = net.to_numpy(dtype="float64")
    return {
        "gross_return": float(np.prod(1.0 + gross.to_numpy()) - 1.0),
        "net_return": float(values[-1] / values[0] - 1.0) if values.size else 0.0,
        "volatility": float(np.std(net_arr, ddof=1) * np.sqrt(bars_per_year))
        if net_arr.size > 1 else 0.0,
        "gross_sharpe": float(sharpe_ratio(gross.to_numpy(),
                                           bars_per_year=int(bars_per_year))),
        "net_sharpe": float(sharpe_ratio(net_arr, bars_per_year=int(bars_per_year))),
        "max_drawdown": float(mdd),
        "calmar": float(calmar_ratio(values, bars_per_year=int(bars_per_year))),
    }


def _sovereign_weights(
    engine: RiskEngine,
    exposures: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    adv: pd.DataFrame,
    clusters: Mapping[str, str],
    *,
    equity: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Laat de soevereine laag over elke bar beslissen, zonder executie.

    Dit isoleert L1: wat DOET de risicolaag met deze gewichten, los van hoe ze
    worden uitgevoerd.
    """
    symbols = list(exposures.columns)
    rows: list[list[float]] = []
    bound: dict[str, int] = {}
    for ts in exposures.index:
        market = MarketState(
            asof_ts=ts,
            sigma_hat={s: float(sigma_hat.at[ts, s]) for s in symbols},
            adv_usd={s: float(adv.at[ts, s]) for s in symbols},
            cluster=dict(clusters),
        )
        state = RiskState(equity=equity, high_water_mark=equity,
                          day_start_equity=equity)
        decision = engine.decide(
            {s: float(exposures.at[ts, s]) for s in symbols}, market, state)
        rows.append([decision.permitted_exposure[s] for s in symbols])
        for c in decision.binding_constraints:
            bound[c.kind.value] = bound.get(c.kind.value, 0) + 1
    return pd.DataFrame(rows, index=exposures.index, columns=symbols), bound


def run_all_layers(
    track: str,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    sigma_daily: pd.DataFrame,
    adv: pd.DataFrame,
    bar_volume: pd.DataFrame,
    funding: pd.DataFrame,
    clusters: Mapping[str, str],
    *,
    risk_cfg: Any,
    impact: ImpactParams,
    venue: VenueSpec,
    spread: SpreadModel,
    initial_equity: float,
    cost_per_side: float,
    bars_per_year: float,
    engine_factory: EngineFactory = _plain_engine,
) -> list[LayerResult]:
    """Draai alle vier de lagen op dezelfde track en dezelfde bars.

    `engine_factory(laag, risk_cfg)` levert de `RiskEngine` voor L1 en voor L3.
    Zonder argument is dat een gewone `RiskEngine`, en dan is deze functie
    bit-identiek aan wat zij vóór fase 11 was.
    """
    require(
        bool(weights.index.equals(prices.index)),
        "Gewichten en prijzen staan niet op dezelfde tijdas.",
        DataContractError, track=track,
    )
    out: list[LayerResult] = []
    exposures = exposures_from_weights(weights)

    # ---------------------------------------------------------------- L0
    vec = run_vectorized(weights, prices, cost_per_side=cost_per_side,
                         initial_equity=initial_equity)
    out.append(LayerResult(
        layer="L0_vectorized", track=track,
        equity_curve=vec.equity_curve, returns=vec.returns,
        gross_returns=vec.gross_returns,
        metrics=_metrics(vec.equity_curve, vec.returns, vec.gross_returns,
                         bars_per_year=bars_per_year),
        costs={"turnover_cost": float(
            (vec.turnover * cost_per_side).sum() * initial_equity)},
        audit={"evidence_class": vec.evidence_class,
               "mean_turnover": float(vec.turnover.mean())},
    ))

    # ---------------------------------------------------------------- L1
    permitted, bound = _sovereign_weights(
        engine_factory("L1_sovereign", risk_cfg), exposures, sigma_hat, adv,
        clusters, equity=initial_equity)
    l1 = run_vectorized(permitted, prices, cost_per_side=cost_per_side,
                        initial_equity=initial_equity)
    out.append(LayerResult(
        layer="L1_sovereign", track=track,
        equity_curve=l1.equity_curve, returns=l1.returns,
        gross_returns=l1.gross_returns,
        metrics=_metrics(l1.equity_curve, l1.returns, l1.gross_returns,
                         bars_per_year=bars_per_year),
        costs={"turnover_cost": float(
            (l1.turnover * cost_per_side).sum() * initial_equity)},
        audit={"binding_constraints": bound,
               "mean_gross": float(permitted.abs().sum(axis=1).mean()),
               "max_symbol_share": float(
                   (permitted.abs().div(
                       permitted.abs().sum(axis=1).where(
                           permitted.abs().sum(axis=1) > _TOL), axis=0)
                    ).max().max())},
    ))

    # ---------------------------------------------------------------- L2
    # Latency zonder kosten: dezelfde soevereine gewichten, één bar later.
    delayed = permitted.shift(1).fillna(0.0)
    l2 = run_vectorized(delayed, prices, cost_per_side=cost_per_side,
                        initial_equity=initial_equity)
    out.append(LayerResult(
        layer="L2_latency", track=track,
        equity_curve=l2.equity_curve, returns=l2.returns,
        gross_returns=l2.gross_returns,
        metrics=_metrics(l2.equity_curve, l2.returns, l2.gross_returns,
                         bars_per_year=bars_per_year),
        costs={"turnover_cost": float(
            (l2.turnover * cost_per_side).sum() * initial_equity)},
        audit={"latency_bars": 1},
    ))

    # ---------------------------------------------------------------- L3
    slices = build_slices(prices, sigma_hat, sigma_daily, adv, bar_volume,
                          dict(clusters), funding)
    engine = build_engine(risk_cfg, impact, venue, spread,
                          initial_equity=initial_equity,
                          risk_engine=engine_factory("L3_execution", risk_cfg))
    result = engine.run(slices, exposures_from_frame(exposures))
    equity = result.equity_curve
    net = result.returns
    out.append(LayerResult(
        layer="L3_execution", track=track,
        equity_curve=equity, returns=net, gross_returns=net,
        metrics=_metrics(equity, net, net, bars_per_year=bars_per_year),
        costs={
            "fees": result.total_fees,
            "funding": result.total_funding,
            "spread": result.total_spread_cost,
            "impact": result.total_impact_cost,
        },
        audit={
            "risk_policy_hash": result.risk_policy_hash,
            "impact_status": result.impact_status,
            "spread_status": result.spread_status,
            "n_orders": result.n_orders,
            "n_fills": result.n_fills,
            "n_rejected": result.n_rejected,
            "n_partial": result.n_partial,
            "n_sovereign_clipped": result.n_sovereign_clipped,
            "n_sovereign_halted": result.n_sovereign_halted,
            "fill_ratio": result.fill_ratio,
        },
    ))
    return out


def summarise(results: list[LayerResult]) -> pd.DataFrame:
    """Alle lagen naast elkaar, één rij per (track, laag)."""
    return pd.DataFrame([r.as_record() for r in results])
