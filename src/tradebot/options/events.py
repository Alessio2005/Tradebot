"""Kalendergebeurtenissen per onderliggende: earnings en dividenden, met hun ouderdom.

De bron (in v1 `yfinance`, al een harde dependency) is niet altijd betrouwbaar. De regel is
daarom streng: ontbreekt de data of is ze ouder dan `max_calendar_data_age_days`, dan is er
geen nieuwe entry op die onderliggende (detailontwerp 4.1). Dit module zegt alleen of de data
bruikbaar is en of er een gebeurtenis in een venster valt; de beslissing ligt bij de strategie.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .rates import CashDividend

__all__ = ["Events"]


@dataclass(frozen=True, slots=True)
class Events:
    underlying: str
    fetched: date | None                 # None: nooit opgehaald
    earnings: tuple[date, ...] = ()
    dividends: tuple[CashDividend, ...] = ()

    def usable(self, today: date, max_age_days: int) -> bool:
        return self.fetched is not None and (today - self.fetched).days <= max_age_days

    def earnings_between(self, after: date, until: date) -> bool:
        """Valt er een earningsdatum in `(after, until]`?"""
        return any(after < d <= until for d in self.earnings)

    def next_earnings(self, after: date) -> date | None:
        future = [d for d in self.earnings if d > after]
        return min(future) if future else None
