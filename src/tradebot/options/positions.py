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

import math
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
from .vol_regime import VolState

__all__ = ["MULTIPLIER", "Action", "ActionKind", "Book", "Candidate", "Leg", "LegIntent",
           "MarketState", "PendingChange", "PositionStatus", "StrategyPosition", "Trend", "Urgency", "mark",
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
    # Gerealiseerde P&L van de aandelen-legs (USD totaal): wat afbouwen of omdraaien van de
    # hedge vastlegde, min de commissie van elke aandelenfill. Het open deel zit in de leg.
    hedge_realized_usd: float = 0.0

    def __post_init__(self) -> None:
        require(self.units > 0, "units moet positief zijn", DataContractError, units=self.units)
        require(self.max_loss >= 0, "max_loss moet ≥ 0 zijn", DataContractError)

    @property
    def option_legs(self) -> tuple[Leg, ...]:
        return tuple(leg for leg in self.legs if leg.contract is not None)

    @property
    def shares(self) -> int:
        """Het aantal aandelen in de aandelen-legs (de hedge), + long, − short."""
        return sum(leg.qty for leg in self.legs if leg.contract is None)

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
    vol: Mapping[str, VolState] = field(default_factory=dict)   # regime en RV-verwachting
    # Voor de drawdown-rem in de sizing (E1): de lopende equity en de piek uit
    # `EngineState.equity`. None (bijvoorbeeld in tests zonder engine): geen rem.
    equity_usd: float | None = None
    equity_peak_usd: float | None = None

    @property
    def today(self) -> date:
        return self.now.date()


@dataclass(frozen=True, slots=True)
class PendingChange:
    """Een lopende (nog niet gevulde) bijsturing of roll van een open positie.

    `legs` zijn de legs van de order in handelsrichting (per eenheid), `max_loss` het maximale
    verlies van de positie ná de order (USD totaal, oneindig bij undefined risk).
    """

    position_id: str
    legs: tuple[LegIntent, ...]
    quantity: int
    max_loss: float


@dataclass(frozen=True, slots=True)
class Book:
    positions: tuple[StrategyPosition, ...] = ()
    shares: Mapping[str, int] = field(default_factory=dict)
    cost_basis: Mapping[str, float] = field(default_factory=dict)   # per aandeel
    cash_usd: float = 0.0
    wheel_phase: Mapping[str, str] = field(default_factory=dict)    # "idle"/"csp"/"shares"/"cc"
    # Het deel van `shares` dat als aandelen-leg bij een positie hoort (de delta-hedge), ook bij
    # een gesloten positie waarvan de aandelen nog terug naar nul gaan. De rest van `shares`
    # komt uit assignment of uitoefening (de wheel).
    hedge_shares: Mapping[str, int] = field(default_factory=dict)
    # Lopende (nog niet gevulde) entry-orders, als positie met hun `max_loss` (oneindig bij
    # undefined risk). De risicopoort telt ze mee als open positie (aantallen, Greeks,
    # scenario's, stress) en `open_max_loss` ook: anders toetst elke entry van dezelfde
    # scanronde tegen hetzelfde boek en samen gaan ze over de grenzen (review 2026-11-27, 1).
    pending_entries: tuple[StrategyPosition, ...] = ()
    # Per gesloten positie die nog aandelen heeft: (onderliggende, aandelen). Daartegen toetst
    # de risicopoort een sluitorder, niet tegen het totaal van het symbool (`hedge_shares`).
    closing_shares: Mapping[str, tuple[str, int]] = field(default_factory=dict)
    # Lopende ADJUST- en ROLL-orders (review 2026-11-27, N10): de poort ziet het risico ervan
    # alvast, zoals bij `pending_entries`.
    pending_changes: tuple[PendingChange, ...] = ()

    @property
    def pending_undefined(self) -> tuple[StrategyPosition, ...]:
        """De lopende entries met undefined risk (`max_loss` oneindig)."""
        return tuple(p for p in self.pending_entries if math.isinf(p.max_loss))

    def open_max_loss(self, underlying: str) -> float:
        """Het maximale verlies op `underlying`: open posities plus lopende entries."""
        held = sum(p.max_loss for p in self.positions
                   if p.underlying == underlying and p.status is not PositionStatus.CLOSED)
        return held + sum(p.max_loss for p in self.pending_entries if p.underlying == underlying)


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
