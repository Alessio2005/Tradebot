"""Van een besluit van het hefboomboek naar orders per been, en terug naar plat.

Een besluit (`Deltas`: `LeverStep` of het besluit van `carry.u1`) zegt per munt hoeveel van de equity er op spot (`da`) en op de
perp-short (`db`) bij of af moet. Hier wordt dat een order met een hoeveelheid:
`|d| · equity_pre / koers`, tegen de close van de besluitbar.

DE VOLGORDE IS EEN RISICOBESLUIT
================================
In een unified-margin-account is spot het onderpand van de short. Daarom:

1. Eerst alles wat risico AFBOUWT: perp-shorts terugkopen, dan spot verkopen. Zo daalt de
   onderhoudsmarge voordat het onderpand weggaat.
2. Daarna alles wat risico OPBOUWT: eerst spot kopen (onderpand erbij), dan pas de short
   openen.

Wie de volgorde omdraait, kan halverwege de herbalancering onder de marge zakken terwijl
het eindboek binnen de governor ligt.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from ..oms.order import OrderSide
from ..utils.failfast import DataContractError, require

__all__ = ["Deltas", "Leg", "LegOrder", "orders_for", "unwind_orders"]


class Deltas(Protocol):
    """Een besluit per munt: hoeveel van de equity er op spot (`da`) en op de perp-short
    (`db`) bij of af moet, en de equity waartegen dat rekent. `LeverStep` (v6-v8) en het
    besluit van het Kraken-boek (`carry.u1`) zijn er allebei een."""

    da: np.ndarray
    db: np.ndarray
    equity_pre: float


class Leg(str, Enum):
    SPOT = "spot"
    PERP = "perp"


@dataclass(frozen=True)
class LegOrder:
    """Eén marketorder op één been. `qty` in munten, altijd positief."""

    symbol: str
    leg: Leg
    side: OrderSide
    qty: float
    notional_usd: float
    ref_price: float
    reduce_only: bool

    def as_record(self) -> dict[str, object]:
        return {"symbol": self.symbol, "leg": self.leg.value, "side": self.side.value,
                "qty": self.qty, "notional_usd": self.notional_usd,
                "ref_price": self.ref_price, "reduce_only": self.reduce_only}


def _rank(o: LegOrder) -> int:
    """0 perp terugkopen, 1 spot verkopen, 2 spot kopen, 3 perp shorten."""
    if o.reduce_only:
        return 0 if o.leg is Leg.PERP else 1
    return 2 if o.leg is Leg.SPOT else 3


def _ordered(orders: list[LegOrder]) -> list[LegOrder]:
    return sorted(orders, key=lambda o: (_rank(o), o.symbol))


def _order(symbol: str, leg: Leg, delta_usd: float, price: float,
           qty: float | None = None) -> LegOrder:
    require(math.isfinite(price) and price > 0.0, "Een order zonder geldige koers.",
            DataContractError, symbol=symbol, leg=leg.value, price=price)
    # Spot: erbij = kopen. Perp: de short erbij = verkopen. Afbouwen is altijd reduce-only.
    grows = delta_usd > 0.0
    side = (OrderSide.BUY if grows else OrderSide.SELL) if leg is Leg.SPOT else (
        OrderSide.SELL if grows else OrderSide.BUY)
    notional = abs(float(delta_usd))
    return LegOrder(symbol=symbol, leg=leg, side=side,
                    qty=notional / float(price) if qty is None else float(qty),
                    notional_usd=notional, ref_price=float(price), reduce_only=not grows)


def orders_for(step: Deltas, symbols: Sequence[str], spot_price: np.ndarray,
               perp_price: np.ndarray) -> list[LegOrder]:
    """De orders van één besluit, in de margeveilige volgorde.

    `spot_price` en `perp_price` zijn de koersen waartegen het besluit rekent (de close van
    de besluitbar; voor een munt zonder koers die bar: de laatst bekende). Een munt zonder
    enige koers met een trade is een contractbreuk, geen order tegen prijs 0."""
    require(len(symbols) == step.da.size == step.db.size == spot_price.size == perp_price.size,
            "Besluit, symbolen en koersen hebben niet dezelfde lengte.", DataContractError)
    out: list[LegOrder] = []
    for i, sym in enumerate(symbols):
        if step.da[i] != 0.0:
            out.append(_order(sym, Leg.SPOT, step.da[i] * step.equity_pre, spot_price[i]))
        if step.db[i] != 0.0:
            out.append(_order(sym, Leg.PERP, step.db[i] * step.equity_pre, perp_price[i]))
    return _ordered(out)


def unwind_orders(spot_qty: Mapping[str, float], perp_short_qty: Mapping[str, float],
                  spot_price: Mapping[str, float],
                  perp_price: Mapping[str, float]) -> list[LegOrder]:
    """Het hele boek plat, gehedged: eerst elke short terug, dan elke spotpositie weg.

    Voor de kill switch. De hoeveelheden zijn die van de beurs, niet die van het model."""
    out: list[LegOrder] = []
    for sym, q in perp_short_qty.items():
        if q > 0.0:
            out.append(_order(sym, Leg.PERP, -q * perp_price[sym], perp_price[sym], q))
    for sym, q in spot_qty.items():
        if q > 0.0:
            out.append(_order(sym, Leg.SPOT, -q * spot_price[sym], spot_price[sym], q))
    return _ordered(out)
