"""De handelskalender van de NYSE, sessietijden en looptijden op seconden.

Feestdagen komen uit QuantLib (`ql.UnitedStates(ql.UnitedStates.NYSE)`); dat scheelt een
extra kalenderpakket. De halve handelsdagen (sluiting 13:00) staan in de config
(`universe.half_days`), omdat QuantLib ze niet kent.

De looptijd van een optie loopt tot het exacte expiratiemoment, 16:00 New York op de
expiratiedag, en telt op seconden (Actual/365 Fixed). Bij korte looptijden is het verschil
met hele dagen groot: een optie die over 2 dagen en 3 uur afloopt, heeft `T = 51 / 8760`,
niet `2 / 365`.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import QuantLib as ql

from ..utils.failfast import DataContractError, require

__all__ = ["NY", "SECONDS_PER_YEAR", "TradingCalendar", "expiry_moment", "year_fraction"]

NY = ZoneInfo("America/New_York")
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
_OPEN = time(9, 30)
_CLOSE = time(16, 0)
_HALF_DAY_CLOSE = time(13, 0)
_NYSE = ql.UnitedStates(ql.UnitedStates.NYSE)


def _aware(ts: datetime) -> None:
    require(ts.tzinfo is not None, "tijdstip moet een tijdzone hebben", DataContractError,
            ts=ts.isoformat())


def expiry_moment(expiry: date) -> datetime:
    """Het expiratiemoment van een US-aandelenoptie: 16:00 New York op de expiratiedag."""
    return datetime.combine(expiry, _CLOSE, tzinfo=NY)


def year_fraction(start: datetime, end: datetime) -> float:
    """`(end - start)` in jaren van 365 dagen, op seconden. Negatief als `end` voorbij is."""
    _aware(start)
    _aware(end)
    return (end - start).total_seconds() / SECONDS_PER_YEAR


class TradingCalendar:
    """NYSE-handelsdagen en -sessies, met de halve dagen uit de config."""

    def __init__(self, half_days: Iterable[date] = ()) -> None:
        self.half_days = frozenset(half_days)
        for d in self.half_days:
            require(self.is_trading_day(d), "een halve dag moet een handelsdag zijn",
                    DataContractError, day=d.isoformat())

    @staticmethod
    def is_trading_day(d: date) -> bool:
        return bool(_NYSE.isBusinessDay(ql.Date(d.day, d.month, d.year)))

    def session(self, d: date) -> tuple[datetime, datetime]:
        """Opening en sluiting (New York) van handelsdag `d`."""
        require(self.is_trading_day(d), "geen handelsdag", DataContractError,
                day=d.isoformat())
        close = _HALF_DAY_CLOSE if d in self.half_days else _CLOSE
        return datetime.combine(d, _OPEN, tzinfo=NY), datetime.combine(d, close, tzinfo=NY)

    def is_open(self, now: datetime) -> bool:
        _aware(now)
        local = now.astimezone(NY)
        if not self.is_trading_day(local.date()):
            return False
        start, end = self.session(local.date())
        return start <= local < end

    def next_trading_day(self, d: date) -> date:
        d += timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def previous_trading_day(self, d: date) -> date:
        d -= timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def trading_days_between(self, start: date, end: date) -> int:
        """Aantal handelsdagen in `(start, end]`; 0 als `end <= start`."""
        n, d = 0, start
        while d < end:
            d += timedelta(days=1)
            n += self.is_trading_day(d)
        return n
