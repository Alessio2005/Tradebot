"""De pattern-day-trader-teller (detailontwerp 8.5).

Een daytrade is een positie die op dezelfde handelsdag geopend en (deels) gesloten wordt. Is de
equity onder 25.000 USD en zijn daytrades (laatste 5 handelsdagen) plus entries van vandaag
(gevuld en lopend) samen al 3, dan weigert de poort elke entry, op elk uur: een stop op de dag
van de entry maakt van die entry een daytrade (review 2026-11-27, N23).

Tot 2026-11-27 gold dit pas vanaf 15:00 (detailontwerp 8.5), maar de scans draaien om 10:00 en
13:30, dus de guard greep nooit in (review 2026-11-27, punt 4). Een sluitorder voor een stop
wordt nooit tegengehouden; de engine schrijft dan een waarschuwing.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from ..calendar import TradingCalendar

__all__ = ["PDT_EQUITY", "day_trades_in_window", "pdt_blocks_entry"]

PDT_EQUITY = 25_000.0
WINDOW = 5
MAX_DAY_TRADES = 3


def day_trades_in_window(days: Iterable[date], today: date, cal: TradingCalendar) -> int:
    """Aantal daytrades in de laatste 5 handelsdagen, vandaag meegeteld."""
    start = today
    for _ in range(WINDOW - 1):
        start = cal.previous_trading_day(start)
    return sum(start <= d <= today for d in days)


def pdt_blocks_entry(days: Iterable[date], now: datetime, equity: float,
                     cal: TradingCalendar, entries_today: int = 0) -> bool:
    """Weigert een entry als de daytrades plus de entries van vandaag (gevuld én lopend) de
    grens al halen: elke entry van vandaag kan met een stop dezelfde dag een daytrade worden
    (review 2026-11-27, N23). `entries_today` telt de entries die er al zijn, zonder de nieuwe."""
    return (equity < PDT_EQUITY
            and day_trades_in_window(days, now.date(), cal) + entries_today >= MAX_DAY_TRADES)
