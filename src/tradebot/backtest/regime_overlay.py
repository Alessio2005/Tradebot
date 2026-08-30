# src/tradebot/backtest/regime_overlay.py
"""Eén geconditioneerde arm door de authoritative engine. H2, stap 11.

WAAROM DIT NAAST `phase5_baseline.py` STAAT
============================================
`run_all_layers` draait vier lagen op één track en levert per laag één
samenvatting. H2 heeft iets anders nodig: ALLEEN de L3-laag -- de enige die als
promotiebewijs telt -- maar dan met PER-BAR attributie, want het oordeel gaat
over de bars waarop de fit out-of-sample is en niet over het hele venster.

Wat er per bar uit moet komen:

    return          uit de equity-curve van de engine
    fees            uit het verschil van de cumulatieve `fees_paid`
    turnover        uit de fills op die bar, tegen hun arrival-prijs
    exposure        of het boek op die bar uberhaupt een positie hield

Dat laatste is geen boekhoudkundig detail. `long_only_equal_weight` halteert op
2022-05-10 en handelt daarna nooit meer: van 1.743 bars zijn er ~130 echt
actief. Een Sharpe over "1.743 bars" is daar een Sharpe over 130 bars met 1.613
nullen erachter. De fase-opdracht eist daarom expliciet het aantal EFFECTIEF
HANDELENDE bars naast elk resultaat, en dit module telt ze.

DE ARM VERSCHILT IN PRECIES EEN DING
=====================================
`exposures` is de enige ingang die tussen de armen verschilt: de basisexposure
maal de regimefactor. Alles daarna -- risicolaag, router, venue, spread, impact,
funding, boekhouding -- is hetzelfde object met dezelfde configuratie en
dezelfde `config_hash`. Een verschil in uitkomst kan dus nergens anders vandaan
komen dan uit de conditionering, en dat is wat de vergelijking meet.

De spread is de ENE uitzondering, en met opzet: `half_spread_bps` is een
parameter omdat de pre-registratie een sensitiviteitsanalyse eist. Bij welke
aangenomen half-spread verdwijnt een gemeten verbetering? Verdwijnt zij al bij
3 bp, dan is de promotie een spread-aanname en geen modelresultaat (§0.3).

Ref: fase-opdracht stap 11, §0.2, §0.3; ARCHITECTUUR_AUDIT_2026-08-22.md 15, 19.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..execution.impact_model import ImpactParams
from ..execution.order_router import SpreadModel, SpreadStatus, VenueSpec
from ..utils.failfast import DataContractError, require
from .engine import build_slices, exposures_from_frame
from .metrics import max_drawdown, sharpe_ratio
from .phase5_baseline import build_engine

__all__ = ["ArmResult", "MarketPanels", "run_overlay_arm"]


@dataclass(frozen=True)
class MarketPanels:
    """De marktsneden waarop elke arm draait. Identiek tussen de armen."""

    prices: pd.DataFrame
    sigma_annual: pd.DataFrame
    sigma_bar: pd.DataFrame
    adv: pd.DataFrame
    volume: pd.DataFrame
    funding: pd.DataFrame
    clusters: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in ("sigma_annual", "sigma_bar", "adv", "volume", "funding"):
            frame = getattr(self, name)
            require(
                bool(frame.index.equals(self.prices.index)),
                f"{name} staat niet op de tijdas van de markprijzen. Een "
                "stilzwijgende reindex is de onopvallendste manier om een "
                "backtest een bar te laten zien die er niet was.",
                DataContractError, frame=name,
            )


@dataclass(frozen=True)
class ArmResult:
    """Wat één arm opleverde, gescoord op de out-of-sample bars."""

    label: str
    half_spread_bps: float
    #: Per-bar netto returns over het VOLLEDIGE venster; het masker staat apart
    #: zodat een lezer kan narekenen waarop is gescoord.
    returns: pd.Series
    mask: np.ndarray
    net_sharpe_oos: float
    net_return_oos: float
    max_drawdown_oos: float
    fees_oos: float
    spread_cost_oos: float
    impact_cost_oos: float
    funding_oos: float
    turnover_notional_oos: float
    n_orders_oos: int
    n_fills_oos: int
    #: Bars binnen het masker waarop daadwerkelijk is gevuld.
    n_trading_bars_oos: int
    #: Bars binnen het masker waarop het boek een positie hield. Dit is het
    #: getal dat het haltprobleem uit §0.5 zichtbaar maakt.
    n_exposed_bars_oos: int
    mean_gross_notional_oos: float
    risk_policy_hash: str
    impact_status: str
    spread_status: str
    n_sovereign_halted: int
    n_sovereign_clipped: int

    @property
    def oos_returns(self) -> np.ndarray:
        return self.returns.to_numpy(dtype="float64")[self.mask]

    def as_record(self) -> dict[str, Any]:
        return {
            "label": self.label, "half_spread_bps": self.half_spread_bps,
            "n_scored_bars": int(self.mask.sum()),
            "net_sharpe_oos": self.net_sharpe_oos,
            "net_return_oos": self.net_return_oos,
            "max_drawdown_oos": self.max_drawdown_oos,
            "fees_oos": self.fees_oos,
            "spread_cost_oos": self.spread_cost_oos,
            "impact_cost_oos": self.impact_cost_oos,
            "funding_oos": self.funding_oos,
            "turnover_notional_oos": self.turnover_notional_oos,
            "n_orders_oos": self.n_orders_oos, "n_fills_oos": self.n_fills_oos,
            "n_trading_bars_oos": self.n_trading_bars_oos,
            "n_exposed_bars_oos": self.n_exposed_bars_oos,
            "mean_gross_notional_oos": self.mean_gross_notional_oos,
            "risk_policy_hash": self.risk_policy_hash,
            "impact_status": self.impact_status,
            "spread_status": self.spread_status,
            "n_sovereign_halted": self.n_sovereign_halted,
            "n_sovereign_clipped": self.n_sovereign_clipped,
        }


def run_overlay_arm(
    label: str,
    exposures: pd.DataFrame,
    panels: MarketPanels,
    *,
    mask: np.ndarray,
    risk_cfg: Any,
    impact: ImpactParams,
    venue: VenueSpec,
    half_spread_bps: float,
    spread_source: str,
    initial_equity: float,
    bars_per_year: float,
) -> ArmResult:
    """Draai één arm door `EventDrivenEngine` en attribueer per bar."""
    require(
        bool(exposures.index.equals(panels.prices.index)),
        "De exposures staan niet op de tijdas van de markprijzen.",
        DataContractError, label=label,
        n_exposures=len(exposures), n_prices=len(panels.prices),
    )
    require(
        len(mask) == len(panels.prices),
        "Het OOS-masker staat niet op de tijdas van de markprijzen.",
        DataContractError, label=label, n_mask=int(len(mask)),
    )
    spread = SpreadModel(
        half_spread_bps=half_spread_bps, status=SpreadStatus.SPREAD_ASSUMED,
        source=spread_source)
    engine = build_engine(risk_cfg, impact, venue, spread,
                          initial_equity=initial_equity)
    slices = build_slices(
        panels.prices, panels.sigma_annual, panels.sigma_bar, panels.adv,
        panels.volume, dict(panels.clusters), panels.funding)
    result = engine.run(slices, exposures_from_frame(exposures))

    index = pd.DatetimeIndex([s.ts for s in result.snapshots])
    require(
        bool(index.equals(panels.prices.index)),
        "De engine leverde niet één boekstaat per bar; per-bar attributie is "
        "dan niet gedefinieerd.",
        DataContractError, label=label, n_snapshots=len(index),
    )
    scored = pd.Series(mask, index=index)

    fees = pd.Series([s.fees_paid for s in result.snapshots],
                     index=index).diff().fillna(0.0)
    funding = pd.Series([s.funding_paid for s in result.snapshots],
                        index=index).diff().fillna(0.0)
    gross = pd.Series([s.gross_notional for s in result.snapshots], index=index)

    turnover = pd.Series(0.0, index=index)
    spread_cost = pd.Series(0.0, index=index)
    impact_cost = pd.Series(0.0, index=index)
    orders = pd.Series(0, index=index)
    fills = pd.Series(0, index=index)
    for report in result.reports:
        ts = report.order.ts_earliest_fill
        if ts not in turnover.index:
            continue
        orders[ts] += 1
        spread_cost[ts] += report.spread_cost
        impact_cost[ts] += report.impact_cost
        if report.fill is not None:
            fills[ts] += 1
            turnover[ts] += abs(report.filled_qty) * report.arrival_price

    returns = result.returns.reindex(index).fillna(0.0)
    oos = returns.to_numpy(dtype="float64")[mask]
    equity = np.cumprod(1.0 + oos)
    drawdown, _, _ = max_drawdown(equity)
    return ArmResult(
        label=label, half_spread_bps=float(half_spread_bps),
        returns=returns, mask=np.asarray(mask, dtype=bool),
        net_sharpe_oos=float(
            sharpe_ratio(oos, bars_per_year=int(bars_per_year))),
        net_return_oos=float(equity[-1] - 1.0) if equity.size else 0.0,
        max_drawdown_oos=float(drawdown),
        fees_oos=float(fees[scored].sum()),
        spread_cost_oos=float(spread_cost[scored].sum()),
        impact_cost_oos=float(impact_cost[scored].sum()),
        funding_oos=float(funding[scored].sum()),
        turnover_notional_oos=float(turnover[scored].sum()),
        n_orders_oos=int(orders[scored].sum()),
        n_fills_oos=int(fills[scored].sum()),
        n_trading_bars_oos=int((fills[scored] > 0).sum()),
        n_exposed_bars_oos=int((gross[scored] > 0.0).sum()),
        mean_gross_notional_oos=float(gross[scored].mean()),
        risk_policy_hash=result.risk_policy_hash,
        impact_status=result.impact_status,
        spread_status=result.spread_status,
        n_sovereign_halted=int(result.n_sovereign_halted),
        n_sovereign_clipped=int(result.n_sovereign_clipped),
    )
