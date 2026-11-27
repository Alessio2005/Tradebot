"""Combo-orders, client-ids en de price-walk (detailontwerp 10.1-10.3).

Een aandelenorder (`StockOrder`, van de delta-hedger) is geen combo: één symbool, een aantal
met teken en een limiet per aandeel; hij loopt niet mee in de price-walk.

Prijzen per eenheid en per aandeel, met het teken van de broker: `limit` > 0 is betalen
(debit), < 0 is ontvangen (credit). Er is geen market-order: een enkele optie is een combo
met één leg.

Prijsstappen: `tick_for` (config `execution.ticks`): combo 0,01; één leg op SPY/QQQ/IWM 0,01;
één leg op andere onderliggenden 0,01 onder $3 en 0,05 erboven.

Price-walk:

```text
mid      = Σ ratio × mid_leg               (ratio + kopen, − verkopen)
natural  = Σ ratio × (ask bij kopen, bid bij verkopen)
limit_0  = mid, afgerond op de combo-tick in het voordeel van ons (de markt moet bewegen)
limit_k  = limit_0 + k × tick × richting   (richting: naar natural)
stop als |limit_k − mid| > max_concession × |natural − mid|
```

Fair-value-anker (`execution.limit_anchor: fair`, alleen ENTRY; exits, STOP, ROLL en ADJUST
houden de walk hierboven):

```text
fair_leg = QuantLib-waarde (scan-engine, `engine_for(Use.SCAN, ...)`) bij de vol van de
           gladde smile (`pricing.surface`) op die strike en expiratie; de mid van de leg als
           het oppervlak niet OK is of die expiratie geen smile heeft
fair     = Σ ratio × fair_leg
far      = 2 × mid − natural = Σ ratio × (bid bij kopen, ask bij verkopen)   (de andere kant)
richting = +1 als natural ≥ mid, anders −1 (met bid ≤ ask is natural altijd ≥ mid, ook bij
           een credit: natural is daar de kleinste ontvangst)
anker    = fair begrensd tot [min(far, natural), max(far, natural)]
limit_0  = anker afgerond op de combo-tick in ons voordeel, weg van natural: bij een debit
           betalen we minder (1.005 → 1.00), bij een credit ontvangen we meer (−1.005 → −1.01)
limit_k  = limit_0 + k × tick × richting
plafond  = fair + max_edge_frac × |natural − mid| × richting, maar nooit voorbij natural
stop als (limit_k − plafond) × richting > 0   (ook bij k = 0: ligt fair zo ver voorbij de
                                               andere kant dat zelfs limit_0 al voorbij het
                                               plafond ligt, dan gaat de entry niet uit)
```

Zo is de limiet nooit slechter dan natural en nooit beter dan de andere kant van het boek
(een gratis lunch die niemand geeft).
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from ...schemas.options import TicksConfig
from ..chain.builder import Chain
from ..positions import LegIntent, Urgency
from ..pricing.engines import EngineSettings, Use, engine_for
from ..pricing.surface import Surface, SurfaceStatus
from ..pricing.vanilla import OptionInputs, price

__all__ = ["ComboOrder", "OrderStatus", "Purpose", "StockOrder", "WalkStep", "client_id",
           "combo_fair", "combo_prices", "fair_walk_limit", "position_id", "tick_for",
           "walk_limit"]


class Purpose(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"
    ROLL = "roll"
    ADJUST = "adjust"
    UNWIND = "unwind"
    HEDGE = "hedge"


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


@dataclass(frozen=True, slots=True)
class StockOrder:
    """Een aandelenorder van de delta-hedger: `qty` + kopen, − verkopen; `limit` per aandeel."""

    client_id: str
    position_id: str
    symbol: str
    qty: int
    limit: float
    purpose: Purpose          # HEDGE, of EXIT als een gesloten positie naar nul gaat
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


def combo_fair(chain: Chain, surface: Surface | None, legs: Sequence[LegIntent],
               engines: EngineSettings) -> float | None:
    """Σ ratio × fair_leg als debit per eenheid (zie de moduledocstring); None als een leg
    geen bid of ask heeft."""
    smiles = ({s.expiry: s for s in surface.smiles}
              if surface is not None and surface.status is SurfaceStatus.OK else {})
    # Eén expiratie zonder smile (uit het oppervlak gevallen): alle legs op de mid, anders
    # vergelijkt de fair-limiet een modelwaarde met een mid (N27).
    use_model = all(leg.contract.expiry in smiles for leg in legs)
    fair = 0.0
    for leg in legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)
        if o is None or o.quote.bid is None or o.quote.ask is None:
            return None
        leg_fair = 0.5 * (o.quote.bid + o.quote.ask)
        smile = smiles.get(c.expiry)
        vol = (smile.vol(math.log(c.strike / smile.forward))
               if use_model and smile is not None else 0.0)
        if vol > 0.0:
            inp = OptionInputs(chain.spot, c.strike, c.right, c.expiry, chain.asof, chain.carry)
            method = engine_for(Use.SCAN, bool(inp.discrete_dividends), engines)
            leg_fair = price(inp, vol, method, engines)
        fair += leg.ratio * leg_fair
    return fair


def tick_for(ticks: TicksConfig, price: float, legs: Sequence[LegIntent]) -> float:
    """De prijsstap voor een order op `legs` bij een (absolute) prijs `price` per aandeel.

    * meer dan één leg (een echte combo): `ticks.combo`;
    * één leg op een penny-pilot-onderliggende (`ticks.penny_symbols`: SPY, QQQ, IWM):
      `ticks.penny` (0,01) op elk prijsniveau;
    * één leg op de rest: `ticks.default.below_3` onder $3, `ticks.default.above_3` vanaf $3.

    Een enkele optie is hier een combo met één leg, maar beurst als gewone order: de
    combo-tick (0,01) zou bij een niet-penny-optie boven $3 een niet-bestaande prijs sturen.
    """
    if len(legs) > 1:
        return ticks.combo
    if legs and legs[0].contract.underlying in ticks.penny_symbols:
        return ticks.penny
    return ticks.default.below_3 if abs(price) < 3.0 else ticks.default.above_3


def _favourable(x: float, tick: float, direction: float) -> int:
    """`x` in ticks, afgerond in ons voordeel: weg van natural (tegen `direction` in)."""
    return math.floor(x / tick + 1e-9) if direction > 0 else math.ceil(x / tick - 1e-9)


@dataclass(frozen=True, slots=True)
class WalkStep:
    limit: float
    done: bool                # True: deze stap gaat verder dan de toegestane concessie


def walk_limit(mid: float, natural: float, step: int, tick: float,
               max_concession: float) -> WalkStep:
    """De limiet bij stap `step` (0 = mid), afgerond op `tick`."""
    direction = 1.0 if natural >= mid else -1.0
    limit = round((_favourable(mid, tick, direction) + step * direction) * tick, 10)
    allowed = max_concession * abs(natural - mid)
    return WalkStep(limit, abs(limit - mid) > allowed + 1e-9 and step > 0)


def fair_walk_limit(mid: float, natural: float, fair: float, step: int, tick: float,
                    max_edge_frac: float) -> WalkStep:
    """De limiet bij stap `step` met het fair-value-anker (zie de moduledocstring)."""
    direction = 1.0 if natural >= mid else -1.0
    far = 2.0 * mid - natural
    anchor = min(max(fair, min(far, natural)), max(far, natural))
    limit = round((_favourable(anchor, tick, direction) + step * direction) * tick, 10)
    cap = fair + direction * max_edge_frac * abs(natural - mid)
    cap = min(cap, natural) if direction > 0 else max(cap, natural)
    return WalkStep(limit, (limit - cap) * direction > 1e-9)
