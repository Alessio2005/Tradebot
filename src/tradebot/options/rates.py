"""Rente en dividenden: de carry-invoer van de pricer.

In v1 is de risicovrije rente een vlak getal uit de config (`pricing.risk_free_rate`). Een
fout van 50 bp verschuift de prijs van een 45-daagse ATM-optie met ongeveer 0,06 % van de
spot, klein ten opzichte van de spread (detailontwerp 5.1).

Dividenden zijn discreet (`CashDividend`: ex-datum en bedrag) óf een continue yield. Een
discreet dividend binnen de looptijd vraagt de FD-engine; zie `pricing.engines.engine_for`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..utils.failfast import DataContractError, require

__all__ = ["CashDividend", "Carry"]


@dataclass(frozen=True, slots=True)
class CashDividend:
    ex_date: date
    amount: float

    def __post_init__(self) -> None:
        require(self.amount > 0, "een dividend moet positief zijn", DataContractError,
                amount=self.amount)


@dataclass(frozen=True, slots=True)
class Carry:
    """Rente (continu, per jaar), dividendyield (continu, per jaar) en discrete dividenden."""

    rate: float
    div_yield: float = 0.0
    dividends: tuple[CashDividend, ...] = ()

    def __post_init__(self) -> None:
        require(-0.05 <= self.rate <= 0.25, "rente buiten [-5 %, 25 %]: waarschijnlijk een "
                "eenheidsfout (procent in plaats van fractie)", DataContractError, rate=self.rate)
        require(0.0 <= self.div_yield <= 0.25, "dividendyield buiten [0, 25 %]",
                DataContractError, div_yield=self.div_yield)
        require(not (self.div_yield and self.dividends),
                "kies een continue yield óf discrete dividenden, niet beide", DataContractError)

    def dividends_in(self, after: date, until: date) -> tuple[CashDividend, ...]:
        """De discrete dividenden met ex-datum in `(after, until]`."""
        return tuple(d for d in self.dividends if after < d.ex_date <= until)
