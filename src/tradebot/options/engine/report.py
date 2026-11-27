"""Backteststatistiek uit het journal en de equity-reeks van een replay.

PER STRATEGIE (alleen posities met een `position_closed`-regel)
  aantal gesloten posities, win rate (P&L > 0), gemiddelde en totale P&L, het grootste
  verlies (de laagste P&L, 0,0 als geen enkele positie verlies maakte) en de maximale
  drawdown in USD (`max_drawdown_usd`): de grootste daling vanaf de top van de cumulatieve
  gesloten P&L, in de volgorde van sluiten en vanaf 0 (dus ≥ 0, en 0,0 zonder daling).
  De P&L van een positie is de som van haar fills (`fill`: -prijs × 100 × aantal, ook de
  bijsturingen) min de commissie, plus de afrekening van vervallen, toegewezen of uitgeoefende
  legs (`settle_usd` op `expiry`/`assignment`/`exercise`), plus de laatste cumulatieve
  `hedge_realized_usd` (van `hedge_fill` of `position_closed`, dus inclusief het afbouwen van
  de aandelen na het sluiten van de opties). Een positie is gesloten bij `position_closed`; dat
  schrijft de engine ook als de laatste leg afrekent.

VOOR HET BOEK (equity per dag, met het startkapitaal als eerste punt)
  totaalrendement = laatste/eerste - 1; maximale drawdown = grootste daling vanaf de top,
  als fractie van die top; Sharpe = gem(r)/stdev(r) × √252 en Sortino = gem(r)/downside(r) ×
  √252 op de dagrendementen r, risicovrij 0, stdev met ddof=1, downside = √gem(min(r, 0)²).

RANDGEVALLEN (nooit een uitzondering, nooit NaN in de JSON)
  geen gesloten posities: lege `strategies`; één equity-punt: rendement en drawdown 0,0 en
  Sharpe/Sortino `null`; stdev 0 geeft Sharpe `null`; downside 0 geeft Sortino `null`.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..positions import MULTIPLIER

__all__ = ["BookStats", "ReplayStats", "StrategyStats", "book_stats", "buy_and_hold", "replay_stats",
           "strategy_stats"]

TRADING_DAYS = 252
#: Journal-events die een optieleg afrekenen; `settle_usd` is de waarde van die afrekening.
_SETTLEMENTS = ("expiry", "assignment", "exercise")
#: Onder deze waarde geldt een spreiding als nul (floatruis van een vlakke reeks).
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class StrategyStats:
    closed: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    largest_loss: float
    max_drawdown_usd: float

    def as_record(self) -> dict[str, Any]:
        return {"closed": self.closed, "win_rate": self.win_rate, "avg_pnl": self.avg_pnl,
                "total_pnl": self.total_pnl, "largest_loss": self.largest_loss,
                "max_drawdown_usd": self.max_drawdown_usd}


def _max_drawdown_usd(pnl: Sequence[float]) -> float:
    """Grootste daling vanaf de top van de cumulatieve P&L (start 0), in USD."""
    curve = np.concatenate(([0.0], np.cumsum(pnl)))
    return round(float((np.maximum.accumulate(curve) - curve).max()), 2)


@dataclass(frozen=True, slots=True)
class BookStats:
    total_return: float
    max_drawdown: float
    sharpe: float | None
    sortino: float | None

    def as_record(self) -> dict[str, Any]:
        return {"total_return": self.total_return, "max_drawdown": self.max_drawdown,
                "sharpe": self.sharpe, "sortino": self.sortino}


@dataclass(frozen=True, slots=True)
class ReplayStats:
    strategies: Mapping[str, StrategyStats]
    book: BookStats

    def as_record(self) -> dict[str, Any]:
        return {"strategies": {k: v.as_record() for k, v in sorted(self.strategies.items())},
                "book": self.book.as_record()}


def strategy_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, StrategyStats]:
    """Per strategie de statistiek van de gesloten posities in `rows` (het journal)."""
    strategy: dict[str, str] = {}
    pnl: dict[str, float] = defaultdict(float)
    hedge: dict[str, float] = {}               # laatste cumulatieve hedge-P&L per positie
    closed: list[str] = []
    for r in rows:
        event, pid = r["event"], r.get("position_id")
        if event == "position_opened":
            strategy[pid] = r["strategy"]
        elif event == "fill":
            pnl[pid] += -r["price"] * MULTIPLIER * r["quantity"] - r["commission"]
        elif event in _SETTLEMENTS and pid is not None:
            pnl[pid] += r["settle_usd"]
        elif event in ("hedge_fill", "position_closed") and "hedge_realized_usd" in r:
            hedge[pid] = r["hedge_realized_usd"]
        if event == "position_closed" and pid in strategy:
            closed.append(pid)
    by_strategy: dict[str, list[float]] = defaultdict(list)
    for pid in closed:
        by_strategy[strategy[pid]].append(pnl[pid] + hedge.get(pid, 0.0))
    return {name: StrategyStats(len(v), round(sum(x > 0 for x in v) / len(v), 4),
                                round(sum(v) / len(v), 2), round(sum(v), 2),
                                round(min(0.0, *v), 2), _max_drawdown_usd(v))
            for name, v in by_strategy.items()}


def _finite(x: float) -> float | None:
    return round(x, 6) if math.isfinite(x) else None


def book_stats(equity: Sequence[float]) -> BookStats:
    """Rendement, drawdown, Sharpe en Sortino van een equity-reeks (zie de moduledocstring).

    Rendement en drawdown volgen uit het eerste punt (> 0), ook als de equity daarna ≤ 0 komt;
    Sharpe en Sortino zijn `null` zodra er geen dagrendementen zijn (korter dan drie punten)
    of een rendement niet bestaat (een punt ≤ 0 vóór het laatste).
    """
    e = np.asarray(equity, dtype=float)
    if e.size < 2 or e[0] <= 0.0:
        return BookStats(0.0, 0.0, None, None)
    total = round(float(e[-1] / e[0] - 1.0), 6)
    peak = np.maximum.accumulate(e)
    drawdown = round(float(((peak - e) / peak).max()), 6)
    if e.size < 3 or bool((e[:-1] <= 0.0).any()):
        return BookStats(total, drawdown, None, None)
    r = e[1:] / e[:-1] - 1.0
    scale = math.sqrt(TRADING_DAYS)
    sd = float(r.std(ddof=1))
    down = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
    return BookStats(total, drawdown,
                     None if sd < _EPS else _finite(float(r.mean()) / sd * scale),
                     None if down < _EPS else _finite(float(r.mean()) / down * scale))


def replay_stats(rows: Sequence[Mapping[str, Any]], start_equity: float,
                 equity: Sequence[tuple[str, float]]) -> ReplayStats:
    return ReplayStats(strategy_stats(rows),
                       book_stats([start_equity, *(v for _, v in equity)]))


def buy_and_hold(bars: pd.DataFrame, dividends: Sequence[Any], days: Sequence[Any],
                 capital: float, symbol: str) -> dict[str, Any]:
    """De vergelijking naast een replay: `capital` op de eerste dag tegen de slotkoers in
    `symbol`, dividenden als kas op de ex-datum, en elke handelsdag op de slotkoers
    gewaardeerd. Zelfde conventie als het boek: het startkapitaal is het eerste punt."""
    close = bars["close"]
    shares = capital / float(close.loc[pd.Timestamp(days[0])])
    paid = {d.ex_date: float(d.amount) for d in dividends}
    cash, equity = 0.0, [capital]
    for d in days:
        cash += shares * paid.get(d, 0.0)
        equity.append(shares * float(close.loc[pd.Timestamp(d)]) + cash)
    return {"symbol": symbol, "end_equity": round(equity[-1], 2),
            **book_stats(equity).as_record()}
