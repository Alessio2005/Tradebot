"""Combo-orders, client-ids en de price-walk (detailontwerp 10.1-10.3).

Prijzen per eenheid en per aandeel, met het teken van de broker: `limit` > 0 is betalen
(debit), < 0 is ontvangen (credit). Er is geen market-order: een enkele optie is een combo
met één leg.

Price-walk:

```text
mid      = Σ ratio × mid_leg               (ratio + kopen, − verkopen)
natural  = Σ ratio × (ask bij kopen, bid bij verkopen)
limit_0  = mid, afgerond op de combo-tick in het voordeel van ons (de markt moet bewegen)
limit_k  = limit_0 + k × tick × richting   (richting: naar natural)
stop als |limit_k − mid| > max_concession × |natural − mid|
```
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from ..chain.builder import Chain
from ..positions import LegIntent, Urgency

__all__ = ["ComboOrder", "OrderStatus", "Purpose", "WalkStep", "client_id", "combo_prices",
           "position_id", "walk_limit"]


class Purpose(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"
    ROLL = "roll"
    ADJUST = "adjust"
    UNWIND = "unwind"


class OrderStatus(StrEnum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


TERMINAL = frozenset({OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED,
                      OrderStatus.EXPIRED})


@dataclass(frozen=True, slots=True)
class ComboOrder:
    client_id: str
    position_id: str
    legs: tuple[LegIntent, ...]
    quantity: int
    limit: float              # + betalen, − ontvangen, per eenheid per aandeel
    purpose: Purpose
    urgency: Urgency
    tif: str = "DAY"


def _h(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def position_id(strategy: str, underlying: str, legs: Iterable[LegIntent],
                opened: date) -> str:
    legs = list(legs)
    expiries = ",".join(sorted({leg.contract.expiry.isoformat() for leg in legs}))
    strikes = ",".join(f"{leg.contract.strike:g}{leg.contract.right.value}{leg.ratio:+d}"
                       for leg in sorted(legs, key=lambda x: x.contract.key()))
    return _h(strategy, underlying, expiries, strikes, opened.isoformat())[:12]


def client_id(pos_id: str, purpose: Purpose, trading_date: date, attempt: int) -> str:
    return _h(pos_id, purpose.value, trading_date.isoformat(), attempt)[:16]


def combo_prices(chain: Chain, legs: Sequence[LegIntent]) -> tuple[float, float] | None:
    """(mid, natural) als debit per eenheid; None als een leg geen bid of ask heeft."""
    mid = natural = 0.0
    for leg in legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)
        if o is None or o.quote.bid is None or o.quote.ask is None:
            return None
        mid += leg.ratio * 0.5 * (o.quote.bid + o.quote.ask)
        natural += leg.ratio * (o.quote.ask if leg.ratio > 0 else o.quote.bid)
    return mid, natural


@dataclass(frozen=True, slots=True)
class WalkStep:
    limit: float
    done: bool                # True: deze stap gaat verder dan de toegestane concessie


def walk_limit(mid: float, natural: float, step: int, tick: float,
               max_concession: float) -> WalkStep:
    """De limiet bij stap `step` (0 = mid), afgerond op `tick`."""
    direction = 1.0 if natural >= mid else -1.0
    base = (math.floor(mid / tick + 1e-9) if direction > 0 else math.ceil(mid / tick - 1e-9))
    limit = round((base + step * direction) * tick, 10)
    allowed = max_concession * abs(natural - mid)
    return WalkStep(limit, abs(limit - mid) > allowed + 1e-9 and step > 0)
