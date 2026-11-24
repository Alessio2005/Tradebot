"""Posities, kandidaten en acties: wat strategieën lezen en teruggeven (detailontwerp 2.3, 7.1).

Tekenafspraken, overal gelijk:

* `qty` en `ratio`: + is long, − is short.
* `mark` van een combo: `Σ ratio × mid`, per eenheid en per aandeel. Een credit-combo heeft een
  negatieve mark (sluiten kost geld), een debit-combo een positieve.
* `net_credit` van een kandidaat: `−mark` bij de entry. Positief is ontvangen credit, negatief
  is betaald debit (per eenheid, per aandeel).
* `max_loss` is in USD, totaal, ≥ 0; `inf` alleen voor undefined risk.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

import pandas as pd

from ..utils.failfast import DataContractError, require
from .calendar import TradingCalendar
from .chain.builder import Chain
from .contracts import OptionContract
from .events import Events
from .iv_metrics import IVMetrics

__all__ = ["MULTIPLIER", "Action", "ActionKind", "Book", "Candidate", "Leg", "LegIntent",
           "MarketState", "PositionStatus", "StrategyPosition", "Trend", "Urgency", "mark",
           "position_greek", "trend_of"]

MULTIPLIER = 100


class Trend(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


def trend_of(daily_bars: pd.DataFrame, window: int = 50, flat_band: float = 0.01) -> Trend | None:
    """Slot tegen de SMA(`window`): binnen ±`flat_band` is FLAT. None bij te weinig dagen."""
    if len(daily_bars) < window:
        return None
    close = daily_bars["close"]
    sma = float(close.iloc[-window:].mean())
    dist = float(close.iloc[-1]) / sma - 1.0
    if abs(dist) < flat_band:
        return Trend.FLAT
    return Trend.UP if dist > 0 else Trend.DOWN


class PositionStatus(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class ActionKind(StrEnum):
    CLOSE = "close"
    ROLL = "roll"
    ADJUST = "adjust"
    CLOSE_LEG = "close_leg"


class Urgency(StrEnum):
    NORMAL = "normal"      # concessie tot 0,6 van mid naar de natuurlijke prijs
    STOP = "stop"          # concessie tot 1,0


@dataclass(frozen=True, slots=True)
class LegIntent:
    contract: OptionContract
    ratio: int             # per eenheid: + long, − short


@dataclass(frozen=True, slots=True)
class Leg:
    contract: OptionContract | None    # None: een aandelen-leg
    symbol: str
    qty: int                           # contracten of aandelen, + long, − short
    entry_price: float                 # per aandeel


@dataclass(frozen=True, slots=True)
class StrategyPosition:
    position_id: str
    strategy: str
    underlying: str
    legs: tuple[Leg, ...]
    opened_at: datetime
    units: int                         # aantal combo-eenheden
    entry_net: float                   # USD totaal: + ontvangen credit, − betaald debit
    max_loss: float
    adjustments: int = 0
    phase: str = ""                    # wheel: "csp" / "cc"
    variant: str = ""                  # long vol: "earnings" / "low_iv"
    status: PositionStatus = PositionStatus.OPEN

    def __post_init__(self) -> None:
        require(self.units > 0, "units moet positief zijn", DataContractError, units=self.units)
        require(self.max_loss >= 0, "max_loss moet ≥ 0 zijn", DataContractError)

    @property
    def option_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.contract is not None)

    def intents(self) -> tuple[LegIntent, ...]:
        """De optie-legs als verhoudingen per eenheid."""
        return tuple(LegIntent(leg.contract, leg.qty // self.units)  # type: ignore[arg-type]
                     for leg in self.option_legs)

    @property
    def entry_credit(self) -> float:
        """`entry_net` per eenheid en per aandeel (+ credit, − debit)."""
        return self.entry_net / (self.units * MULTIPLIER)

    def expiries(self) -> tuple[date, ...]:
        return tuple(sorted({leg.contract.expiry for leg in self.option_legs  # type: ignore[union-attr]
                             }))


@dataclass(frozen=True, slots=True)
class Candidate:
    strategy: str
    underlying: str
    legs: tuple[LegIntent, ...]
    quantity: int
    net_credit: float                  # per eenheid, per aandeel; + credit, − debit
    max_loss: float                    # USD totaal
    rank_score: float
    rule_trace: tuple[str, ...]
    variant: str = ""


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    position_id: str
    legs: tuple[LegIntent, ...]        # te sluiten (verhouding per eenheid van de positie)
    urgency: Urgency
    reason: str
    new_legs: tuple[LegIntent, ...] = ()   # te openen bij ROLL en ADJUST


@dataclass(frozen=True, slots=True)
class MarketState:
    """Bevroren momentopname, gemaakt door de engine. Ketens zijn al gefilterd en hebben
    Greeks van de beslissingsengine."""

    now: datetime
    calendar: TradingCalendar
    chains: Mapping[str, Chain]
    iv: Mapping[str, IVMetrics]
    trend: Mapping[str, Trend]
    events: Mapping[str, Events]
    capital_usd: float

    @property
    def today(self) -> date:
        return self.now.date()


@dataclass(frozen=True, slots=True)
class Book:
    positions: tuple[StrategyPosition, ...] = ()
    shares: Mapping[str, int] = field(default_factory=dict)
    cost_basis: Mapping[str, float] = field(default_factory=dict)   # per aandeel
    cash_usd: float = 0.0
    wheel_phase: Mapping[str, str] = field(default_factory=dict)    # "idle"/"csp"/"shares"/"cc"

    def open_max_loss(self, underlying: str) -> float:
        return sum(p.max_loss for p in self.positions
                   if p.underlying == underlying and p.status is not PositionStatus.CLOSED)


def mark(chain: Chain, legs: Iterable[LegIntent]) -> float | None:
    """`Σ ratio × mid` per eenheid en per aandeel; None als een leg geen mid heeft."""
    total = 0.0
    for leg in legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)
        if o is None or o.quote.mid is None:
            return None
        total += leg.ratio * o.quote.mid
    return total


def position_greek(chain: Chain, legs: Iterable[LegIntent], greek: str) -> float | None:
    """`Σ ratio × greek × 100` per eenheid (USD of aandelen); None als een leg geen Greeks heeft."""
    total = 0.0
    for leg in legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)
        if o is None or o.greeks is None:
            return None
        total += leg.ratio * getattr(o.greeks, greek) * MULTIPLIER
    return total
