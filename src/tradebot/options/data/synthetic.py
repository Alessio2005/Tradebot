"""Synthetische markten voor replay en tests: deterministisch, met een vast seed.

Er is geen historische ketendata. Deze generator maakt een kleine, volledig vastgelegde markt
om de ENGINE en de REGELS te testen, niet om rendement te meten (detailontwerp 15).

* Onderliggende: GBM met vaste parameters per naam, plus optioneel één geplande sprong en één
  dividend. Een opwarmperiode met dagbars (voor RV20 en SMA50) gaat aan de start vooraf.
* Ketens: op vaste momenten per handelsdag; strikes in vaste stappen rond spot (per naam
  instelbaar); expiraties op
  vrijdagen (wekelijks tot `weekly_days`, daarna de derde vrijdag van de maand tot
  `max_days`), naar donderdag als de vrijdag geen handelsdag is.
* Prijzen: QuantLib (BAW, of FD bij een dividend in de looptijd) met een vaste smile
  `σ(k) = base + curvature·k² + slope·k` in `k = ln(K/S)`.
* Quotes: theo ± een halve spread (`max(min_spread, spread_frac × theo)`), op centen; bid
  niet onder 0. Open interest en volume vast, behalve in de verre vleugels (die vallen in de
  filters af).

`python -I -m tradebot.options.data.synthetic --out <map>` schrijft de standaardmarkt naar
schijf (parquet + JSON), in het formaat dat `data.store` leest.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..calendar import NY, TradingCalendar
from ..chain.builder import RawOption
from ..contracts import OptionContract, Quote, Right
from ..events import Events
from ..pricing.engines import Method
from ..pricing.ql_env import scaled_time
from ..pricing.vanilla import OptionInputs, price
from ..rates import Carry, CashDividend
from .store import Market, Snapshot, write_market

__all__ = ["SyntheticUnderlying", "SyntheticSpec", "default_spec", "generate", "main"]


@dataclass(frozen=True, slots=True)
class SyntheticUnderlying:
    name: str
    spot: float
    drift: float = 0.05
    vol: float = 0.22
    jump_day: int | None = None          # index van de handelsdag met een sprong bij de opening
    jump: float = 0.0
    dividend_day: int | None = None      # index van de ex-dividenddag
    dividend: float = 0.0
    earnings_day: int | None = None
    strike_step: float | None = None     # None: `SyntheticSpec.strike_step`


@dataclass(frozen=True, slots=True)
class SyntheticSpec:
    underlyings: tuple[SyntheticUnderlying, ...]
    start: date
    n_days: int
    seed: int = 20261124
    rate: float = 0.04
    snapshot_times: tuple[time, ...] = (time(10, 0), time(13, 0), time(15, 0))
    strike_step: float = 2.0
    strike_range: float = 0.15
    weekly_days: int = 21
    max_days: int = 100
    smile_base: float = 0.22
    smile_curvature: float = 0.10
    smile_slope: float = -0.05
    min_spread: float = 0.02
    spread_frac: float = 0.03
    open_interest: int = 5000
    volume: int = 1000
    thin_wing_k: float = 0.12
    warmup_days: int = 60
    half_days: tuple[date, ...] = field(default=(date(2026, 11, 27), date(2026, 12, 24)))


def default_spec() -> SyntheticSpec:
    """De standaardset: 15 handelsdagen vanaf 2026-12-01, twee namen."""
    return SyntheticSpec(
        underlyings=(
            SyntheticUnderlying("TSTA", 100.0, jump_day=9, jump=-0.08, earnings_day=None,
                                dividend_day=6, dividend=0.60),
            SyntheticUnderlying("TSTB", 50.0, vol=0.30, earnings_day=12, strike_step=1.0),
        ),
        start=date(2026, 12, 1), n_days=15)


def _trading_days(cal: TradingCalendar, start: date, n: int) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if cal.is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def _expiries(cal: TradingCalendar, today: date, spec: SyntheticSpec) -> list[date]:
    out = []
    d = today + timedelta(days=1)
    while (d - today).days <= spec.max_days:
        if d.weekday() == 4:
            third_friday = 15 <= d.day <= 21
            if (d - today).days <= spec.weekly_days or third_friday:
                e = d if cal.is_trading_day(d) else cal.previous_trading_day(d)
                out.append(e)
        d += timedelta(days=1)
    return out


def _smile_vol(spec: SyntheticSpec, spot: float, strike: float) -> float:
    k = math.log(strike / spot)
    return spec.smile_base + spec.smile_curvature * k * k + spec.smile_slope * k


def _quote(theo: float, spec: SyntheticSpec, ts: datetime, thin: bool) -> Quote:
    half = 0.5 * max(spec.min_spread, spec.spread_frac * theo)
    bid = max(round(theo - half, 2), 0.0)
    ask = max(round(theo + half, 2), round(bid + 0.01, 2))
    oi, vol = (100, 10) if thin else (spec.open_interest, spec.volume)
    return Quote(bid, ask, ts, volume=vol, open_interest=oi)


def generate(spec: SyntheticSpec) -> Market:
    cal = TradingCalendar(spec.half_days)
    rng = np.random.default_rng(spec.seed)
    days = _trading_days(cal, spec.start, spec.n_days)
    warm, d = [], spec.start - timedelta(days=1)
    while len(warm) < spec.warmup_days:
        if cal.is_trading_day(d):
            warm.append(d)
        d -= timedelta(days=1)
    warm.reverse()

    bars: dict[str, pd.DataFrame] = {}
    snapshots: list[Snapshot] = []
    events: dict[str, Events] = {}
    n_steps = len(spec.snapshot_times)
    for u in spec.underlyings:
        divs = ((CashDividend(days[u.dividend_day], u.dividend),)
                if u.dividend_day is not None else ())
        carry = Carry(spec.rate, dividends=divs)
        events[u.name] = Events(u.name, spec.start,
                                (days[u.earnings_day],) if u.earnings_day is not None else (),
                                divs)
        spot, rows = u.spot, []
        dt = 1.0 / 252.0
        for i, day in enumerate([*warm, *days]):
            live = i - len(warm)
            if live >= 0 and live == u.jump_day:
                spot *= 1.0 + u.jump
            if live >= 0 and live == u.dividend_day:
                spot -= u.dividend
            path = []
            for _ in range(n_steps):
                z = rng.standard_normal()
                spot *= math.exp((u.drift - 0.5 * u.vol**2) * dt / n_steps
                                 + u.vol * math.sqrt(dt / n_steps) * z)
                path.append(spot)
            o = path[0]
            rows.append({"date": pd.Timestamp(day), "open": o, "high": max(path) * 1.002,
                         "low": min(path) * 0.998, "close": path[-1], "volume": 5_000_000})
            if live < 0:
                continue
            for t, s in zip(spec.snapshot_times, path, strict=True):
                ts = datetime.combine(day, t, tzinfo=NY)
                snapshots.append(Snapshot(u.name, ts, s, tuple(_chain(spec, cal, u, s, ts,
                                                                      carry))))
        bars[u.name] = pd.DataFrame(rows).set_index("date")
    snapshots.sort(key=lambda x: (x.asof, x.underlying))
    return Market(spec.rate, tuple(days), bars, tuple(snapshots), events)


def _chain(spec: SyntheticSpec, cal: TradingCalendar, u: SyntheticUnderlying, spot: float,
           ts: datetime, carry: Carry) -> list[RawOption]:
    step = u.strike_step or spec.strike_step
    lo = math.ceil(spot * (1 - spec.strike_range) / step) * step
    hi = spot * (1 + spec.strike_range)
    strikes = np.arange(lo, hi + 1e-9, step)
    out = []
    for expiry in _expiries(cal, ts.date(), spec):
        start = scaled_time(ts, expiry).eval_date      # hetzelfde venster als de pricer
        method = Method.FD if carry.dividends_in(start, expiry) else Method.BAW
        for k in strikes:
            vol = _smile_vol(spec, spot, float(k))
            thin = abs(math.log(k / spot)) > spec.thin_wing_k
            for right in (Right.PUT, Right.CALL):
                theo = price(OptionInputs(spot, float(k), right, expiry, ts, carry), vol, method)
                c = OptionContract(u.name, expiry, float(round(k, 2)), right)
                out.append(RawOption(c, _quote(theo, spec, ts, thin)))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot.options.data.synthetic",
                                 description="Schrijf de synthetische standaardmarkt naar schijf.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    write_market(generate(default_spec()), args.out)
    print(f"synthetische markt geschreven naar {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI-ingang, getest via main()
    raise SystemExit(main())
