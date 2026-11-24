"""Optiecontracten en quotes: de onveranderlijke bouwstenen van de options-setup.

Posities, legs en orders komen in latere fasen (detailontwerp 2.3 en 10.1); dit module bevat
alleen wat de pricer en de ketens nodig hebben.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from ..utils.failfast import DataContractError, require

__all__ = ["Exercise", "OptionContract", "Quote", "Right"]


class Right(StrEnum):
    CALL = "C"
    PUT = "P"


class Exercise(StrEnum):
    AMERICAN = "american"
    EUROPEAN = "european"


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Eén optieserie. `expiry` is de laatste handelsdag (expiratie 16:00 New York)."""

    underlying: str
    expiry: date
    strike: float
    right: Right
    multiplier: int = 100
    exercise: Exercise = Exercise.AMERICAN
    con_id: int | None = None

    def __post_init__(self) -> None:
        require(bool(self.underlying), "onderliggende ontbreekt", DataContractError)
        require(self.strike > 0, "strike moet positief zijn", DataContractError,
                strike=self.strike)
        require(self.multiplier > 0, "multiplier moet positief zijn", DataContractError,
                multiplier=self.multiplier)

    def key(self) -> str:
        """Deterministische sleutel, bv. `SPY 2026-12-18 600 C`."""
        return f"{self.underlying} {self.expiry.isoformat()} {self.strike:g} {self.right.value}"


@dataclass(frozen=True, slots=True)
class Quote:
    """Een momentopname van het boek van één contract. `None` is: niet geleverd."""

    bid: float | None
    ask: float | None
    ts: datetime
    last: float | None = None
    bid_size: int = 0
    ask_size: int = 0
    volume: int = 0
    open_interest: int = 0
    delayed: bool = False

    def __post_init__(self) -> None:
        require(self.ts.tzinfo is not None, "quote-tijd moet een tijdzone hebben",
                DataContractError)
        if self.bid is not None and self.ask is not None:
            require(0 <= self.bid <= self.ask, "bid moet tussen 0 en ask liggen",
                    DataContractError, bid=self.bid, ask=self.ask)

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return 0.5 * (self.bid + self.ask)
