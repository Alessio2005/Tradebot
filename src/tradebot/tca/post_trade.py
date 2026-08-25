# src/tradebot/tca/post_trade.py
"""Post-trade TCA — realised vs expected cost comparison."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .implementation_shortfall import ISDecomposition, decompose_implementation_shortfall
from .pre_trade import PreTradeCostEstimate, estimate_pre_trade_cost

logger = logging.getLogger(__name__)

__all__ = ["PostTradeRecord", "analyse_execution"]


@dataclass(frozen=True)
class PostTradeRecord:
    """Post-trade analysis of a single executed order."""

    symbol: str
    execution_time: pd.Timestamp
    signed_qty: float
    execution_price: float
    expected_cost: PreTradeCostEstimate
    is_decomp: ISDecomposition
    cost_vs_expected_bps: float   # actual - expected (positive = worse than expected)
    alpha_capture: float          # fraction of pre-trade signal captured post-cost


def analyse_execution(
    symbol: str,
    execution_time: pd.Timestamp,
    signed_qty: float,
    execution_price: float,
    decision_price: float,
    arrival_price: float,
    volatility: float,
    adv: float,
    bid_ask_spread: float = 0.0002,
    vwap_price: float | None = None,
    pre_trade_signal: float = 0.0,
) -> PostTradeRecord:
    """Perform post-trade analysis for a single executed order.

    Parameters
    ----------
    symbol, execution_time, signed_qty, execution_price :
        Order identification and execution details.
    decision_price :
        Price at time of investment decision.
    arrival_price :
        Mid-price at order submission.
    volatility :
        Daily realised volatility (same units as pre-trade estimate).
    adv :
        Average daily volume.
    bid_ask_spread :
        Estimated bid-ask spread (fraction).
    vwap_price :
        VWAP over execution window (optional).
    pre_trade_signal :
        Alpha signal strength at decision time (used for alpha-capture calc).

    Returns
    -------
    PostTradeRecord.
    """
    notional = abs(signed_qty) * abs(execution_price)
    expected = estimate_pre_trade_cost(
        notional=notional,
        volatility=volatility,
        adv=adv,
        bid_ask_spread=bid_ask_spread,
    )

    is_decomp = decompose_implementation_shortfall(
        symbol=symbol,
        decision_price=decision_price,
        arrival_price=arrival_price,
        execution_price=execution_price,
        signed_qty=signed_qty,
        vwap_price=vwap_price,
    )

    actual_bps = is_decomp.total_bps
    cost_vs_expected = actual_bps - expected.total_bps

    # Alpha capture: fraction of signal retained after costs
    if abs(pre_trade_signal) > 1e-9 and notional > 0:
        cost_as_signal = is_decomp.total_is / (notional + 1e-9)
        alpha_capture = float(np.clip(1.0 - cost_as_signal / (abs(pre_trade_signal) + 1e-9), 0.0, 2.0))
    else:
        alpha_capture = 1.0

    return PostTradeRecord(
        symbol=symbol,
        execution_time=execution_time,
        signed_qty=signed_qty,
        execution_price=execution_price,
        expected_cost=expected,
        is_decomp=is_decomp,
        cost_vs_expected_bps=cost_vs_expected,
        alpha_capture=alpha_capture,
    )


# =============================================================================
# PHASE 5 — TCA ROUNDTRIP CLOSURE
# =============================================================================
# `analyse_execution` hierboven blijft bestaan voor losse orders, maar hij kan
# niet BEWIJZEN dat er geen kosten zijn zoekgeraakt: hij vergelijkt één order
# met een schatting, en niets koppelt de som terug aan de boekhouding.
#
# Fase-opdracht §13 eist dat de roundtrip SLUIT:
#
#     arrival-price difference = realized execution costs + residu
#
# De constructie hieronder maakt dat afdwingbaar met een SCHADUWBOEKHOUDING.
# Dezelfde fills worden opnieuw geboekt tegen hun ARRIVAL price en met fee 0,
# met dezelfde funding en dezelfde markprijzen. Het verschil in eindequity
# tussen het echte en het schaduwboek MOET exact gelijk zijn aan de som van de
# gedecomponeerde kosten. Is dat niet zo, dan bestaat er een kostenpost die in
# de decompositie of in de boekhouding ontbreekt - en dat is precies de fout
# die deze test moet vangen.
from dataclasses import replace as _replace
from typing import TYPE_CHECKING

from ..backtest.accounting import Ledger as _Ledger
from ..utils.failfast import DataContractError as _DataContractError
from ..utils.failfast import require as _require

if TYPE_CHECKING:
    from ..backtest.engine import BacktestResult
    from ..execution.order_router import ExecutionReport

__all__ += ["CostDecomposition", "TcaClosure", "close_tca_roundtrip"]


@dataclass(frozen=True)
class CostDecomposition:
    """De opsplitsing die §13 voorschrijft, in quote-valuta.

    `timing` staat er apart bij en telt NIET mee in `execution_total`. Het is de
    opportuniteitskost van wat niet gevuld raakte, en dat is een contrafeitelijk
    getal: het heeft de kas nooit verlaten. Het meetellen zou de roundtrip laten
    sluiten op een bedrag dat nergens is betaald.
    """

    spread: float
    impact: float
    fees: float
    funding: float
    timing: float

    @property
    def execution_total(self) -> float:
        """Wat het uitvoeren daadwerkelijk heeft gekost."""
        return float(self.spread + self.impact + self.fees)

    @property
    def total_with_funding(self) -> float:
        return float(self.execution_total + self.funding)

    def as_record(self) -> dict[str, float]:
        return {
            "spread": float(self.spread),
            "impact": float(self.impact),
            "fees": float(self.fees),
            "funding": float(self.funding),
            "timing_opportunity_cost": float(self.timing),
            "execution_total": self.execution_total,
            "total_with_funding": self.total_with_funding,
        }


@dataclass(frozen=True)
class TcaClosure:
    """Het bewijs dat er geen kosten zijn verdwenen."""

    arrival_equity: float
    realised_equity: float
    gap: float
    decomposition: CostDecomposition
    residual: float
    tolerance: float
    closes: bool

    def as_record(self) -> dict[str, float | bool]:
        return {
            "arrival_equity": float(self.arrival_equity),
            "realised_equity": float(self.realised_equity),
            "gap": float(self.gap),
            "residual": float(self.residual),
            "tolerance": float(self.tolerance),
            "closes": bool(self.closes),
            **self.decomposition.as_record(),
        }


def close_tca_roundtrip(
    result: BacktestResult,
    *,
    tolerance_bps: float,
    strict: bool = True,
) -> TcaClosure:
    """Reken de roundtrip dicht tegen een schaduwboek op arrival prices.

    Parameters
    ----------
    result : een `backtest.engine.BacktestResult`.
    tolerance_bps : de VOORAF gedefinieerde numerieke tolerantie, als fractie
        van de startequity in basispunten. Komt uit `conf/tca/default.yaml`;
        er is geen default hier, want een tolerantie die de code kiest is een
        tolerantie die meebeweegt met de uitkomst.
    strict : crash wanneer de roundtrip niet sluit. Alleen `False` zetten voor
        diagnostiek.
    """
    _require(
        len(result.snapshots) > 0,
        "Een TCA-roundtrip op een run zonder boekstaat.",
        _DataContractError,
    )
    initial = float(result.audit["initial_equity"])
    shadow = _Ledger(initial)

    by_ts: dict[pd.Timestamp, list[ExecutionReport]] = {}
    for report in result.reports:
        fill = report.fill
        if fill is not None:
            by_ts.setdefault(fill.ts, []).append(report)

    # Replay: dezelfde fills, maar tegen arrival price en zonder fee.
    for snapshot in result.snapshots:
        for report in by_ts.get(snapshot.ts, []):
            assert report.fill is not None  # by construction of by_ts
            shadow.apply_fill(_replace(
                report.fill, price=float(report.arrival_price), fee=0.0))
        # Funding uit het ECHTE boek overnemen zou de positiegroottes moeten
        # matchen; die zijn identiek omdat de hoeveelheden identiek zijn. De
        # markprijzen zijn dat ook, dus funding valt in het verschil weg.
        shadow.mark(_marks_at(result, snapshot.ts), snapshot.ts)

    final_shadow = shadow.mark(_marks_at(result, result.snapshots[-1].ts),
                               result.snapshots[-1].ts)
    # Funding drukt op beide boeken gelijk; expliciet gelijkschakelen zodat het
    # verschil uitsluitend de EXECUTIEkosten bevat.
    arrival_equity = float(final_shadow.equity) - float(
        result.snapshots[-1].funding_paid)
    realised_equity = float(result.snapshots[-1].equity)
    gap = arrival_equity - realised_equity

    decomposition = CostDecomposition(
        spread=float(sum(r.spread_cost for r in result.reports)),
        impact=float(sum(r.impact_cost for r in result.reports)),
        fees=float(result.snapshots[-1].fees_paid),
        funding=float(result.snapshots[-1].funding_paid),
        timing=_timing_cost(result),
    )
    residual = gap - decomposition.execution_total
    tolerance = abs(float(tolerance_bps)) * 1e-4 * initial
    closes = abs(residual) <= tolerance

    if strict:
        _require(
            closes,
            "De TCA-roundtrip sluit niet: het verschil tussen het boek op "
            "arrival prices en het werkelijke boek is niet volledig verklaard "
            "door spread, impact en fees. Er is een kostenpost die in geen van "
            "beide administraties staat (fase-opdracht §13, §27).",
            _DataContractError,
            arrival_equity=arrival_equity,
            realised_equity=realised_equity,
            gap=gap,
            execution_total=decomposition.execution_total,
            residual=residual,
            tolerance=tolerance,
        )
    return TcaClosure(
        arrival_equity=arrival_equity, realised_equity=realised_equity, gap=gap,
        decomposition=decomposition, residual=residual, tolerance=tolerance,
        closes=closes,
    )


def _marks_at(result: BacktestResult, ts: pd.Timestamp) -> dict[str, float]:
    """De markprijzen die de engine op `ts` gebruikte, uit het auditspoor."""
    for snapshot, report_marks in zip(result.snapshots, result.audit["marks"],
                                      strict=True):
        if snapshot.ts == ts:
            return dict(report_marks)
    raise _DataContractError(f"Geen markprijzen vastgelegd voor {ts}.")


def _timing_cost(result: BacktestResult) -> float:
    """Opportuniteitskost van niet-gevulde hoeveelheid, tegen arrival price.

    Contrafeitelijk en daarom apart: het is de notional die we WILDEN handelen
    maar niet kregen. Nul wanneer alles vult.
    """
    return float(sum(
        abs(r.unfilled_qty) * r.arrival_price for r in result.reports))
