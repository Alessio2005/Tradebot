"""Een markt uit ECHTE dagkoersen en ECHTE implied vol, met MODELKETENS (historische simulatie).

Er is geen historische ketendata (detailontwerp 15, Q6). Wel zijn per dag de koers (OHLC) en de
30-daagse IV van de onderliggende te krijgen. Hieruit bouwt dit bestand een `Market` voor de
bestaande replay:

* de onderliggende volgt de echte open en slot; tussen die twee loopt het pad op de
  snapshotmomenten lineair in ln(koers), naar het deel van de handelsdag dat dan voorbij is;
* de ketens komen uit dezelfde QuantLib-modelketen als de synthetische markt, met de smile
  gecentreerd op de echte IV van die dag, en dezelfde spreads en strikestappen;
* dividenden en earnings zitten er niet in (geen bron): de strategieën zien dus geen
  earnings-filter.

Dit test de REGELS en de ENGINE op echte koers- en vol-paden. Het zegt niets over echte fills,
skew of spreads: het rendement is gemodelleerd, niet gemeten.

Invoer: een map met `<NAAM>.csv` per onderliggende, kolommen `date,open,high,low,close,volume,iv`
(`iv` als fractie, 0,22 = 22 %). `python -I -m tradebot.options.data.historical --in DIR --out
DIR --start 2026-06-01 --end 2026-11-25` schrijft een markt in het formaat van `data.store`.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from ...utils.failfast import DataContractError, require
from ..calendar import NY, TradingCalendar
from ..events import Events
from ..rates import Carry
from .store import Market, Snapshot, write_market
from .synthetic import SyntheticSpec, SyntheticUnderlying, chain_at

__all__ = ["build_market", "load_inputs", "main", "strike_step"]

_OPEN, _CLOSE = time(9, 30), time(16, 0)
_COLUMNS = ("open", "high", "low", "close", "volume", "iv")


def strike_step(spot: float) -> float:
    """De strikestap van een gewone aandelenketen, naar koersniveau."""
    return 1.0 if spot < 50 else 2.5 if spot < 200 else 5.0


def _fraction(t: time) -> float:
    """Het deel van de handelsdag (9:30-16:00) dat op `t` voorbij is."""
    minutes = (t.hour - _OPEN.hour) * 60 + t.minute - _OPEN.minute
    return min(max(minutes / 390.0, 0.0), 1.0)


def load_inputs(src: Path) -> dict[str, pd.DataFrame]:
    """Lees `<NAAM>.csv` uit `src`; elke naam met de kolommen van `_COLUMNS`."""
    files = sorted(src.glob("*.csv"))
    require(bool(files), "geen CSV-bestanden in de invoermap", DataContractError, path=str(src))
    out = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
        require(set(_COLUMNS) <= set(df.columns), "CSV mist kolommen", DataContractError,
                bestand=f.name, nodig=_COLUMNS)
        out[f.stem] = df[list(_COLUMNS)].astype(float)
    return out


def build_market(data: Mapping[str, pd.DataFrame], start: date, end: date, rate: float = 0.04,
                 spec: SyntheticSpec | None = None,
                 snapshot_times: Sequence[time] = (time(10, 0), time(13, 30)),
                 half_days: Sequence[date] = ()) -> Market:
    """De markt van `start` t/m `end`; de dagen vóór `start` blijven als opwarmgeschiedenis."""
    require(bool(data), "geen onderliggenden", DataContractError)
    base = spec or SyntheticSpec(underlyings=(), start=start, n_days=0, rate=rate,
                                 snapshot_times=tuple(snapshot_times))
    cal = TradingCalendar(tuple(half_days))
    for name, df in data.items():
        require(bool(((df["iv"] > 0.02) & (df["iv"] < 3.0)).all()),
                "iv hoort een fractie tussen 0,02 en 3 te zijn", DataContractError, naam=name)
        require(bool((df[["open", "high", "low", "close"]] > 0).all().all()),
                "koersen moeten positief zijn", DataContractError, naam=name)
    common = sorted(set.intersection(*(set(df.index) for df in data.values())))
    days = tuple(d.date() for d in common
                 if start <= d.date() <= end and cal.is_trading_day(d.date()))
    require(bool(days), "geen gemeenschappelijke handelsdagen in de periode", DataContractError,
            start=str(start), end=str(end))
    carry = Carry(rate)
    bars = {n: df[["open", "high", "low", "close", "volume"]] for n, df in data.items()}
    snapshots = []
    for name, df in data.items():
        for day in days:
            row = df.loc[pd.Timestamp(day)]
            day_spec = replace(base, smile_base=float(row["iv"]))
            for t in snapshot_times:
                spot = float(row["open"] * math.exp(
                    math.log(row["close"] / row["open"]) * _fraction(t)))
                ts = datetime.combine(day, t, tzinfo=NY)
                u = SyntheticUnderlying(name, spot, strike_step=strike_step(spot))
                snapshots.append(Snapshot(name, ts, spot,
                                          tuple(chain_at(day_spec, cal, u, spot, ts, carry))))
    snapshots.sort(key=lambda x: (x.asof, x.underlying))
    events = {n: Events(n, start) for n in data}
    return Market(rate, days, bars, tuple(snapshots), events)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot.options.data.historical",
                                 description="Bouw een historische simulatiemarkt.")
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    args = ap.parse_args(argv)
    market = build_market(load_inputs(args.src), args.start, args.end)
    write_market(market, args.out)
    print(f"{len(market.days)} dagen, {len(market.snapshots)} snapshots naar {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI-ingang, getest via main()
    raise SystemExit(main())
