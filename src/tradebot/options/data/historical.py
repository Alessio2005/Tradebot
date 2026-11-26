"""Een historische markt voor replay: echte koersen en echte IV, ketens uit het model.

Er is geen historische ketendata (detailontwerp Q6). Deze bouwer neemt per onderliggende de
echte dagbars en IBKR's dagelijkse 30-daagse implied volatility, en reconstrueert daaruit de
ketens met QuantLib, op dezelfde manier als de synthetische markt:

* Spot en IV op een snapshot-tijd liggen lineair tussen de opening en de slotkoers van die dag,
  naar het deel van de handelsdag dat verstreken is (10:00 ligt dicht bij de opening).
* De smile is `σ(k) = iv + curvature·k² + slope·k` in `k = ln(K/S)`: het niveau is echt, de
  vorm is vast. Quotes, spreads, open interest en volume komen uit het model (`synthetic`).
* Dividenden en earnings komen uit de bron (yfinance). `Events.fetched` is de laatste dag:
  de kalender is in een historische run volledig bekend, net als in de dienst vooraf.
* De IBKR-IV van vóór de eerste dag gaat als `Market.iv_history` mee, zodat IVR vanaf dag één
  bruikbaar is.

De uitkomst meet dus de REGELS op echte koers- en IV-paden, niet echte fills. Het rapport moet
dat zo zeggen.

`python -I -m tradebot.options historical --out DIR --start D --end D` haalt de data bij IB
Gateway (alleen lezen) en schrijft de markt; `replay --market DIR` speelt hem af.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from ib_async import Stock

from ...utils.failfast import DataContractError, require
from ..calendar import NY, TradingCalendar
from ..chain.builder import RawOption
from ..contracts import OptionContract, Right
from ..events import Events
from ..pricing.engines import Method
from ..pricing.ql_env import scaled_time
from ..pricing.vanilla import OptionInputs, price
from ..rates import Carry, CashDividend
from .store import Market, Snapshot
from .synthetic import _expiries, _quote

__all__ = ["HistoricalSeries", "HistoricalSpec", "build_market", "fetch_history",
           "session_fraction", "strike_step", "yfinance_history"]

_OPEN, _CLOSE = time(9, 30), time(16, 0)


@dataclass(frozen=True, slots=True)
class HistoricalSeries:
    """De ruwe historie van één onderliggende. `bars` en `iv` hebben een datumindex."""
    name: str
    bars: pd.DataFrame            # open, high, low, close, volume
    iv: pd.DataFrame              # open, close: IBKR's 30-daagse IV als fractie
    dividends: tuple[CashDividend, ...] = ()
    earnings: tuple[date, ...] = ()


@dataclass(frozen=True, slots=True)
class HistoricalSpec:
    start: date
    end: date
    rate: float = 0.04
    snapshot_times: tuple[time, ...] = (time(10, 0), time(13, 30), time(15, 0))
    strike_range: float = 0.12
    weekly_days: int = 21
    max_days: int = 60
    smile_curvature: float = 0.10
    smile_slope: float = -0.05
    min_spread: float = 0.02
    spread_frac: float = 0.03
    open_interest: int = 5000
    volume: int = 1000
    thin_wing_k: float = 0.10
    half_days: tuple[date, ...] = field(default=())


def session_fraction(t: time) -> float:
    """Het deel van de gewone handelsdag dat om `t` verstreken is, in [0, 1]."""
    total = (_CLOSE.hour * 60 + _CLOSE.minute) - (_OPEN.hour * 60 + _OPEN.minute)
    done = (t.hour * 60 + t.minute) - (_OPEN.hour * 60 + _OPEN.minute)
    return min(max(done / total, 0.0), 1.0)


def strike_step(spot: float) -> float:
    """Een vaste, gangbare strike-stap bij deze koers."""
    if spot < 100:
        return 1.0
    return 2.5 if spot < 300 else 5.0


def _between(row: pd.Series, w: float) -> float:
    return float(row["open"]) + w * (float(row["close"]) - float(row["open"]))


def build_market(series: Iterable[HistoricalSeries], spec: HistoricalSpec) -> Market:
    cal = TradingCalendar(spec.half_days)
    series = sorted(series, key=lambda s: s.name)
    require(bool(series), "geen onderliggende voor de historische markt", DataContractError)
    common = set.intersection(*({d.date() for d in pd.to_datetime(s.bars.index)} &
                                {d.date() for d in pd.to_datetime(s.iv.index)} for s in series))
    days = tuple(sorted(d for d in common if spec.start <= d <= spec.end and cal.is_trading_day(d)))
    require(bool(days), "geen handelsdagen met koers en IV in de periode", DataContractError,
            start=spec.start.isoformat(), end=spec.end.isoformat())
    bars, events, ivh, snapshots = {}, {}, {}, []
    for s in series:
        b = s.bars.copy()
        b.index = pd.to_datetime(b.index)
        iv = s.iv.copy()
        iv.index = pd.to_datetime(iv.index)
        require(bool((iv[["open", "close"]] > 0).all().all()), "IV moet positief zijn",
                DataContractError, underlying=s.name)
        bars[s.name] = b[b.index <= pd.Timestamp(spec.end)][["open", "high", "low", "close",
                                                              "volume"]]
        events[s.name] = Events(s.name, spec.end, tuple(sorted(s.earnings)),
                                tuple(sorted(s.dividends, key=lambda d: d.ex_date)))
        ivh[s.name] = tuple((d.date().isoformat(), float(v)) for d, v in iv["close"].items()
                            if d.date() < days[0])
        carry = Carry(spec.rate, dividends=events[s.name].dividends)
        for day in days:
            bar, ivd = b.loc[pd.Timestamp(day)], iv.loc[pd.Timestamp(day)]
            for t in spec.snapshot_times:
                w = session_fraction(t)
                spot, vol = _between(bar, w), _between(ivd, w)
                ts = datetime.combine(day, t, tzinfo=NY)
                snapshots.append(Snapshot(s.name, ts, spot,
                                          tuple(_chain(spec, cal, s.name, spot, vol, ts, carry))))
    snapshots.sort(key=lambda x: (x.asof, x.underlying))
    return Market(spec.rate, days, bars, tuple(snapshots), events, ivh)


def _chain(spec: HistoricalSpec, cal: TradingCalendar, name: str, spot: float, atm: float,
           ts: datetime, carry: Carry) -> list[RawOption]:
    step = strike_step(spot)
    lo = math.ceil(spot * (1 - spec.strike_range) / step) * step
    strikes = np.arange(lo, spot * (1 + spec.strike_range) + 1e-9, step)
    out = []
    for expiry in _expiries(cal, ts.date(), spec):
        start = scaled_time(ts, expiry).eval_date      # hetzelfde venster als de pricer
        method = Method.FD if carry.dividends_in(start, expiry) else Method.BAW
        for k in strikes:
            m = math.log(float(k) / spot)
            vol = max(atm + spec.smile_curvature * m * m + spec.smile_slope * m, 0.05)
            thin = abs(m) > spec.thin_wing_k
            for right in (Right.PUT, Right.CALL):
                theo = price(OptionInputs(spot, float(k), right, expiry, ts, carry), vol, method)
                c = OptionContract(name, expiry, float(round(k, 2)), right)
                out.append(RawOption(c, _quote(theo, spec, ts, thin)))
    return out


# --- ophalen (netwerk) ---------------------------------------------------------------------

EventsHistory = Callable[[str, date, date], tuple[tuple[CashDividend, ...], tuple[date, ...]]]


def yfinance_history(symbol: str, start: date,
                     end: date) -> tuple[tuple[CashDividend, ...], tuple[date, ...]]:
    """Dividenden (ex-datum, bedrag) en earningsdata tussen `start` en `end`, uit yfinance."""
    t = yf.Ticker(symbol)
    divs = tuple(CashDividend(d.date(), float(a)) for d, a in t.dividends.items()
                 if start <= d.date() <= end + timedelta(days=120) and float(a) > 0)
    ed = t.get_earnings_dates(limit=16)
    earnings = tuple(sorted({d.date() for d in (ed.index if ed is not None else [])
                             if start <= d.date() <= end + timedelta(days=120)}))
    return divs, earnings


def _frame(bars: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([{"date": pd.Timestamp(b.date), "open": float(b.open),
                          "high": float(b.high), "low": float(b.low), "close": float(b.close),
                          "volume": float(b.volume)} for b in bars]).set_index("date")


async def fetch_history(ib: Any, names: Iterable[str], start: date, end: date,
                        events_history: EventsHistory = yfinance_history,
                        duration: str = "2 Y") -> list[HistoricalSeries]:
    """Dagbars en dagelijkse IV per naam bij IB Gateway (alleen lezen), plus de kalender."""
    out = []
    for name in sorted(names):
        c = (await ib.qualifyContractsAsync(Stock(name, "SMART", "USD")))[0]
        require(c is not None, "onbekende onderliggende bij IBKR", DataContractError,
                underlying=name)
        frames = {}
        for what in ("TRADES", "OPTION_IMPLIED_VOLATILITY"):
            rows = await ib.reqHistoricalDataAsync(c, end.strftime("%Y%m%d 23:59:59 US/Eastern"),
                                                   duration, "1 day", what, True)
            require(bool(rows), "geen historische data van IBKR", DataContractError,
                    underlying=name, what=what)
            frames[what] = _frame(rows)
        divs, earnings = events_history(name, start - timedelta(days=150), end)
        out.append(HistoricalSeries(name, frames["TRADES"],
                                    frames["OPTION_IMPLIED_VOLATILITY"][["open", "close"]],
                                    divs, earnings))
    return out
