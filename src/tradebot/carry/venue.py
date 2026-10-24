"""Het contract met een beurs, en de paper-venue.

Wat een echte adapter (Binance spot + USDⓈ-M + portfolio margin) moet leveren, staat in
`Venue`. Hij staat hier niet: zonder toegang tot de beurs en een testnet is een adapter
ongetoetste code die er af uitziet. Dit zijn de eisen die hij moet halen:

* `book()` geeft de WERKELIJKE stand: spot per munt, de perp-short per munt (positief)
  en de equity van het account in USD (inclusief lening, ongerealiseerde P&L, funding).
* `execute()` voert marketorders uit IN DE GEGEVEN VOLGORDE (die is margeveilig, zie
  `orders`), wacht per order op een eindstatus, en geeft de werkelijk uitgevoerde
  hoeveelheid en prijs terug. Een order zonder uitvoering is een fout, geen fill.
* Reduce-only waar de order dat zegt.

De paper-venue is het model zelf: zijn stand is wat het boek na elke stap verwacht. Dat
maakt het shadowboek per constructie gelijk aan de backtest; het verschil met de
werkelijkheid meet pas een echte venue.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..utils.failfast import TradebotContractError
from .orders import Leg, LegOrder

__all__ = ["LegFill", "PaperVenue", "ReconciliationBreak", "Venue", "VenueBook",
           "reconcile_book"]


class ReconciliationBreak(TradebotContractError):
    """De beurs houdt iets anders dan het boek verwacht: een gemiste fill, een handmatige
    ingreep, een liquidatie of auto-deleveraging. Het boek handelt dan niet door."""


@dataclass(frozen=True)
class VenueBook:
    """Wat de beurs houdt. Hoeveelheden in munten; de short als positieve hoeveelheid."""

    spot: Mapping[str, float]
    perp_short: Mapping[str, float]
    equity_usd: float


@dataclass(frozen=True)
class LegFill:
    order: LegOrder
    qty: float
    price: float
    fee_usd: float


class Venue(Protocol):
    """Het contract met een beurs (zie de moduledocstring)."""

    is_paper: bool

    def book(self) -> VenueBook: ...

    def execute(self, orders: Sequence[LegOrder]) -> list[LegFill]: ...


@dataclass
class PaperVenue:
    """Het model als beurs: elke order volledig tegen zijn referentiekoers, zonder fee
    (het boek rekent de kosten zelf). `sync` zet de stand op wat het model verwacht."""

    spot: dict[str, float] = field(default_factory=dict)
    perp_short: dict[str, float] = field(default_factory=dict)
    equity_usd: float = 0.0
    fills: list[LegFill] = field(default_factory=list)
    is_paper: bool = True

    def book(self) -> VenueBook:
        return VenueBook(spot=dict(self.spot), perp_short=dict(self.perp_short),
                         equity_usd=self.equity_usd)

    def execute(self, orders: Sequence[LegOrder]) -> list[LegFill]:
        out = [LegFill(order=o, qty=o.qty, price=o.ref_price, fee_usd=0.0) for o in orders]
        self.fills.extend(out)
        return out

    def sync(self, spot: Mapping[str, float], perp_short: Mapping[str, float],
             equity_usd: float) -> None:
        self.spot, self.perp_short = dict(spot), dict(perp_short)
        self.equity_usd = float(equity_usd)


def reconcile_book(expected_spot: Mapping[str, float], expected_perp: Mapping[str, float],
                   venue: VenueBook, *, rel_tol: float = 0.01,
                   abs_tol_usd: float = 10.0,
                   prices: Mapping[str, float] | None = None) -> list[str]:
    """Leg de beurs tegen wat het boek verwacht. Geeft de breuken terug (leeg = in orde).

    Een been breekt als het verschil groter is dan `rel_tol` van het verwachte EN groter
    dan `abs_tol_usd` (tegen `prices`, zonder koers: elke afwijking boven `rel_tol`). De
    UNIE van beide boeken telt: een positie die alleen de beurs kent, is ook een breuk."""
    breaks: list[str] = []
    for leg, want, have in ((Leg.SPOT, expected_spot, venue.spot),
                            (Leg.PERP, expected_perp, venue.perp_short)):
        for sym in sorted(set(want) | set(have)):
            w, h = float(want.get(sym, 0.0)), float(have.get(sym, 0.0))
            diff = abs(h - w)
            if diff <= rel_tol * abs(w):
                continue
            px = None if prices is None else prices.get(sym)
            if px is not None and math.isfinite(px) and diff * px <= abs_tol_usd:
                continue
            breaks.append(f"{sym} {leg.value}: expected {w:.8g}, venue {h:.8g}")
    return breaks
