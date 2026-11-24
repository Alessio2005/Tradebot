"""De wheel: CSP en covered call als toestandsmachine per onderliggende (detailontwerp 7.4).

Standaard uit (open vraag C). De toestand staat in het boek (`Book.wheel_phase`); de engine
past hem aan met `transition` op elke gebeurtenis. Er bestaan alleen de overgangen uit
`TRANSITIONS`: een andere combinatie is een fout, geen stille no-op.
"""
from __future__ import annotations

import dataclasses
from enum import StrEnum

from ...utils.failfast import DataContractError
from ..positions import Action, Book, Candidate, MarketState, StrategyPosition
from .income import CashSecuredPut, CoveredCall

__all__ = ["TRANSITIONS", "Wheel", "WheelEvent", "WheelPhase", "transition"]


class WheelPhase(StrEnum):
    IDLE = "idle"
    CSP = "csp"
    SHARES = "shares"
    CC = "cc"


class WheelEvent(StrEnum):
    CSP_OPENED = "csp_opened"
    CSP_CLOSED = "csp_closed"          # gesloten of verlopen zonder assignment
    ASSIGNED = "assigned"              # aandelen ontvangen
    CC_OPENED = "cc_opened"
    CC_CLOSED = "cc_closed"            # gesloten of verlopen
    CALLED_AWAY = "called_away"        # aandelen weg
    UNWIND = "unwind"                  # alles sluiten, vanuit elke toestand


TRANSITIONS: dict[tuple[WheelPhase, WheelEvent], WheelPhase] = {
    (WheelPhase.IDLE, WheelEvent.CSP_OPENED): WheelPhase.CSP,
    (WheelPhase.CSP, WheelEvent.CSP_CLOSED): WheelPhase.IDLE,
    (WheelPhase.CSP, WheelEvent.ASSIGNED): WheelPhase.SHARES,
    (WheelPhase.SHARES, WheelEvent.CC_OPENED): WheelPhase.CC,
    (WheelPhase.CC, WheelEvent.CC_CLOSED): WheelPhase.SHARES,
    (WheelPhase.CC, WheelEvent.CALLED_AWAY): WheelPhase.IDLE,
    **{(p, WheelEvent.UNWIND): WheelPhase.IDLE for p in WheelPhase},
}


def transition(phase: WheelPhase, event: WheelEvent) -> WheelPhase:
    try:
        return TRANSITIONS[(phase, event)]
    except KeyError:
        raise DataContractError(
            f"wheel: geen overgang van {phase.value} bij {event.value}") from None


class Wheel:
    name = "wheel"

    def __init__(self, csp: CashSecuredPut, cc: CoveredCall) -> None:
        self.csp, self.cc = csp, cc

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        out = []
        for phase, inner, variant in ((WheelPhase.IDLE, self.csp, "csp"),
                                      (WheelPhase.SHARES, self.cc, "cc")):
            for c in inner.scan(market, book):
                if WheelPhase(book.wheel_phase.get(c.underlying, "idle")) is phase:
                    out.append(dataclasses.replace(c, strategy=self.name, variant=variant))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        return (self.csp if pos.phase == "csp" else self.cc).manage(pos, market)
