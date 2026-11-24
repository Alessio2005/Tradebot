"""Opslag van markten: ketensnapshots (parquet), dagbars (parquet) en gebeurtenissen (JSON).

Hetzelfde formaat dient voor de synthetische replay-markt en, later, voor het ketenarchief
dat de dienst bij elke scan wegschrijft (detailontwerp 11.3). Er staan alleen ruwe quotes in,
geen IV: die rekent de engine bij het afspelen opnieuw uit, met de pricer van dat moment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ...utils.failfast import DataContractError, require
from ..chain.builder import RawOption
from ..contracts import OptionContract, Quote, Right
from ..events import Events
from ..rates import CashDividend

__all__ = ["Market", "Snapshot", "read_market", "write_market"]

_SNAPSHOTS = "chains.parquet"
_BARS = "bars.parquet"
_META = "market.json"


@dataclass(frozen=True, slots=True)
class Snapshot:
    underlying: str
    asof: datetime
    spot: float
    options: tuple[RawOption, ...]


@dataclass(frozen=True, slots=True)
class Market:
    rate: float
    days: tuple[date, ...]
    bars: dict[str, pd.DataFrame]
    snapshots: tuple[Snapshot, ...]
    events: dict[str, Events]

    def bars_until(self, underlying: str, day: date) -> pd.DataFrame:
        """Dagbars tot en met `day` (causaal: niets van na `day`)."""
        b = self.bars[underlying]
        return b[b.index <= pd.Timestamp(day)]


def write_market(m: Market, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"underlying": s.underlying, "asof": s.asof.isoformat(), "spot": s.spot,
             "expiry": r.contract.expiry.isoformat(), "strike": r.contract.strike,
             "right": r.contract.right.value, "bid": r.quote.bid, "ask": r.quote.ask,
             "volume": r.quote.volume, "open_interest": r.quote.open_interest}
            for s in m.snapshots for r in s.options]
    pd.DataFrame(rows).to_parquet(out / _SNAPSHOTS, index=False)
    pd.concat({k: v for k, v in sorted(m.bars.items())}, names=["underlying", "date"]) \
        .to_parquet(out / _BARS)
    meta = {"rate": m.rate, "days": [d.isoformat() for d in m.days],
            "events": {k: {"fetched": e.fetched.isoformat() if e.fetched else None,
                           "earnings": [d.isoformat() for d in e.earnings],
                           "dividends": [[d.ex_date.isoformat(), d.amount] for d in e.dividends]}
                       for k, e in sorted(m.events.items())}}
    (out / _META).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def read_market(src: Path) -> Market:
    for name in (_SNAPSHOTS, _BARS, _META):
        require((src / name).is_file(), "marktbestand ontbreekt", DataContractError,
                path=str(src / name))
    meta = json.loads((src / _META).read_text(encoding="utf-8"))
    df = pd.read_parquet(src / _SNAPSHOTS)
    snapshots = []
    for (und, asof), g in df.groupby(["underlying", "asof"], sort=True):
        ts = datetime.fromisoformat(str(asof))
        opts = tuple(
            RawOption(OptionContract(str(und), date.fromisoformat(r.expiry), float(r.strike),
                                     Right(r.right)),
                      Quote(None if pd.isna(r.bid) else float(r.bid),
                            None if pd.isna(r.ask) else float(r.ask), ts,
                            volume=int(r.volume), open_interest=int(r.open_interest)))
            for r in g.itertuples(index=False))
        snapshots.append(Snapshot(str(und), ts, float(g["spot"].iloc[0]), opts))
    snapshots.sort(key=lambda s: (s.asof, s.underlying))
    bars_all = pd.read_parquet(src / _BARS)
    bars = {str(k): g.droplevel("underlying") for k, g in bars_all.groupby(level="underlying")}
    events = {k: Events(k, date.fromisoformat(v["fetched"]) if v["fetched"] else None,
                        tuple(date.fromisoformat(d) for d in v["earnings"]),
                        tuple(CashDividend(date.fromisoformat(d), float(a))
                              for d, a in v["dividends"]))
              for k, v in meta["events"].items()}
    return Market(float(meta["rate"]), tuple(date.fromisoformat(d) for d in meta["days"]),
                  bars, tuple(snapshots), events)
