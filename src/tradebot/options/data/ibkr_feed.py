"""Marktdata van IBKR voor de paper-dienst (detailontwerp 12.1), met vertraagde data.

`refresh(now)` haalt per onderliggende de koers, de optieketen (expiraties binnen
`MAX_DTE` dagen, strikes binnen `strike_range` van spot) en de dagbars op, en bewaart een
`Snapshot`. De engine leest die daarna synchroon via het `MarketFeed`-protocol. Een paper-
login mag maar ongeveer 100 marktdatalijnen tegelijk: de opties gaan in pakketten van
`BATCH`.

Earnings en dividenden komen van `yfinance` (detailontwerp 4.1); ontbreekt die data, dan
blokkeren de strategieën zelf nieuwe entries (`events:stale`).
"""
from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import Callable
from datetime import date, datetime, timedelta
from itertools import pairwise
from typing import Any

import pandas as pd
import yfinance as yf
from ib_async import Option, Stock

from ..calendar import NY
from ..chain.builder import RawOption
from ..contracts import OptionContract, Quote, Right
from ..events import Events
from ..rates import CashDividend
from .store import Snapshot

__all__ = ["BATCH", "MAX_DTE", "IBKRFeed", "yfinance_events"]

BATCH = 90
MAX_DTE = 100
WAIT_SECONDS = 4.0
EventsSource = Callable[[str, date], Events]
_COLUMNS = ["open", "high", "low", "close"]


def _no_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS, index=pd.DatetimeIndex([], name="date"))


def _px(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _int(v: Any) -> int:
    x = _px(v)
    return int(x) if x is not None else 0


#: Standaardcadans (dagen) zonder bruikbare historie: kwartaaldividend.
DIVIDEND_CADENCE_DAYS = 91
#: Hoe ver vooruit het dividendschema geprojecteerd wordt (de looptijden zijn hooguit `MAX_DTE`).
DIVIDEND_HORIZON_DAYS = 365


def project_dividends(announced: date | None, history: list[tuple[date, float]],
                      today: date, horizon_days: int = DIVIDEND_HORIZON_DAYS
                      ) -> tuple[CashDividend, ...]:
    """Het dividendschema tot `today + horizon_days`: het aangekondigde dividend (als er een
    is) en daarna het laatste bedrag op de vaste cadans uit de historie (mediaan van de laatste
    vier tussenpozen, anders kwartaal), met `estimated=True` voor alles wat geprojecteerd is.

    Alleen het eerstvolgende dividend meenemen laat bij looptijden over meerdere ex-datums de
    latere dividenden uit de prijs, en daarmee de put te duur en de call te goedkoop. Zonder
    historie is er geen betaler en dus geen schema.
    """
    if not history:
        return ()
    hist = sorted(history)
    amount = hist[-1][1]
    recent = [d for d, _ in hist[-5:]]
    gaps = [(b - a).days for a, b in pairwise(recent)]
    cadence = max(int(statistics.median(gaps)), 25) if gaps else DIVIDEND_CADENCE_DAYS
    until = today + timedelta(days=horizon_days)
    out: list[CashDividend] = []
    if announced is not None and announced >= today:
        out.append(CashDividend(announced, amount))
        nxt = announced + timedelta(days=cadence)
    else:
        nxt = hist[-1][0] + timedelta(days=cadence)
        while nxt < today:
            nxt += timedelta(days=cadence)
    while nxt <= until:
        out.append(CashDividend(nxt, amount, estimated=True))
        nxt += timedelta(days=cadence)
    return tuple(out)


def yfinance_events(symbol: str, today: date) -> Events:
    t = yf.Ticker(symbol)
    cal = t.calendar or {}
    earnings = tuple(sorted(d for d in cal.get("Earnings Date", []) if isinstance(d, date)))
    ex = cal.get("Ex-Dividend Date")
    hist = t.dividends
    history = ([(pd.Timestamp(d).date(), float(a)) for d, a in hist.items()]
               if isinstance(hist.index, pd.DatetimeIndex) else [])
    if len(hist) and not history:                  # zonder datums: alleen het bedrag bekend
        history = [(today - timedelta(days=DIVIDEND_CADENCE_DAYS), float(hist.iloc[-1]))]
    divs = project_dividends(ex if isinstance(ex, date) else None, history, today)
    return Events(symbol, today, earnings, divs)


class IBKRFeed:
    def __init__(self, ib: Any, underlyings: tuple[str, ...], rate: float,
                 strike_range: float = 0.15, events_source: EventsSource = yfinance_events,
                 wait_seconds: float = WAIT_SECONDS) -> None:
        self.ib, self._names, self.rate = ib, tuple(sorted(underlyings)), rate
        self.strike_range, self.events_source, self.wait = strike_range, events_source, wait_seconds
        self._snaps: dict[str, Snapshot] = {}
        self._bars: dict[str, pd.DataFrame] = {}
        self._events: dict[str, Events] = {}
        self._bars_day: date | None = None

    # --- MarketFeed ---------------------------------------------------------------------

    def underlyings(self) -> tuple[str, ...]:
        return self._names

    def snapshot(self, underlying: str, now: datetime) -> Snapshot | None:
        s = self._snaps.get(underlying)
        return s if s is not None and s.asof <= now and s.asof.date() == now.date() else None

    def bars_before(self, underlying: str, day: date) -> pd.DataFrame:
        b = self._bars.get(underlying)
        if b is None:
            return _no_bars()
        return b[b.index < pd.Timestamp(day)]

    def events(self, underlying: str) -> Events | None:
        return self._events.get(underlying)

    # --- ophalen ------------------------------------------------------------------------

    async def _ticks(self, contracts: list[Any], generic: str) -> list[Any]:
        out: list[Any] = []
        for i in range(0, len(contracts), BATCH):
            part = contracts[i:i + BATCH]
            ticks = [self.ib.reqMktData(c, generic) for c in part]
            await asyncio.sleep(self.wait)
            for c in part:
                self.ib.cancelMktData(c)
            out += ticks
        return out

    async def _daily(self, stock: Any, name: str, today: date) -> None:
        bars = await self.ib.reqHistoricalDataAsync(stock, "", "1 Y", "1 day", "TRADES", True)
        df = pd.DataFrame([{"date": pd.Timestamp(b.date), "open": b.open, "high": b.high,
                            "low": b.low, "close": b.close} for b in bars])
        self._bars[name] = df.set_index("date").sort_index() if len(df) else _no_bars()
        self._events[name] = self.events_source(name, today)

    async def refresh(self, now: datetime) -> None:
        today = now.date()
        new_day = self._bars_day != today
        for name in self._names:
            (stock,) = await self.ib.qualifyContractsAsync(Stock(name, "SMART", "USD"))
            if new_day:
                await self._daily(stock, name, today)
            (st,) = await self._ticks([stock], "")
            spot = _px(st.last) or _px(st.close)
            if spot is None:
                continue
            chains = await self.ib.reqSecDefOptParamsAsync(name, "", "STK", stock.conId)
            smart = next((c for c in chains if c.exchange == "SMART"), None)
            if smart is None:
                continue
            last = today + timedelta(days=MAX_DTE)
            expiries = sorted(e for e in (datetime.strptime(x, "%Y%m%d").date()
                                          for x in smart.expirations) if today < e <= last)
            lo, hi = spot * (1 - self.strike_range), spot * (1 + self.strike_range)
            strikes = sorted(k for k in smart.strikes if lo <= k <= hi)
            wanted = [Option(name, e.strftime("%Y%m%d"), k, r, "SMART", multiplier="100",
                             currency="USD", tradingClass=smart.tradingClass)
                      for e in expiries for k in strikes for r in ("P", "C")]
            q = [c for c in await self.ib.qualifyContractsAsync(*wanted)
                 if c is not None and c.conId]
            rows = []
            for t in await self._ticks(q, "100,101"):
                c = t.contract
                right = Right.CALL if c.right == "C" else Right.PUT
                oi = t.callOpenInterest if right is Right.CALL else t.putOpenInterest
                exp = datetime.strptime(c.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
                rows.append(RawOption(OptionContract(name, exp, float(c.strike), right),
                                      Quote(_px(t.bid), _px(t.ask), now, _px(t.last),
                                            _int(t.bidSize), _int(t.askSize), _int(t.volume),
                                            _int(oi), delayed=True)))
            self._snaps[name] = Snapshot(name, now.astimezone(NY), spot, tuple(rows))
        self._bars_day = today
