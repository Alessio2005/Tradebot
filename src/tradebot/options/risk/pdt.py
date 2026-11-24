"""De pattern-day-trader-teller (detailontwerp 8.5).

Een daytrade is een positie die op dezelfde handelsdag geopend en (deels) gesloten wordt. Is de
equity onder 25.000 USD en zijn er in de laatste 5 handelsdagen al 3, dan weigert de poort een
entry na `PDT_CUTOFF` (15:00): die zou vandaag nog gesloten kunnen moeten worden. Een sluitorder
voor een stop wordt nooit tegengehouden; de engine schrijft dan een waarschuwing.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time

from ..calendar import TradingCalendar

__all__ = ["PDT_CUTOFF", "PDT_EQUITY", "day_trades_in_window", "pdt_blocks_entry"]

PDT_EQUITY = 25_000.0
PDT_CUTOFF = time(15, 0)
WINDOW = 5
MAX_DAY_TRADES = 3


def day_trades_in_window(days: Iterable[date], today: date, cal: TradingCalendar) -> int:
    """Aantal daytrades in de laatste 5 handelsdagen, vandaag meegeteld."""
    start = today
    for _ in range(WINDOW - 1):
        start = cal.previous_trading_day(start)
    return sum(start <= d <= today for d in days)


def pdt_blocks_entry(days: Iterable[date], now: datetime, equity: float,
                     cal: TradingCalendar) -> bool:
    return (equity < PDT_EQUITY and now.time() >= PDT_CUTOFF
            and day_trades_in_window(days, now.date(), cal) >= MAX_DAY_TRADES)
