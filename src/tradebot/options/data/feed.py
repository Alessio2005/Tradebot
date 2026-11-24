"""Waar de engine zijn marktdata vandaan haalt: een `MarketFeed` (detailontwerp 12).

`ReplayFeed` speelt een opgeslagen markt (`data.store.Market`) af: bij elk tijdstip de laatste
ketensnapshot van die dag tot en met dat moment, de dagbars tot en met gisteren en de
gebeurtenissen. Niets van na het tijdstip (causaal). De IBKR-feed (fase 7) heeft dezelfde vorm.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime
from typing import Protocol

import pandas as pd

from ..events import Events
from .store import Market, Snapshot

__all__ = ["MarketFeed", "ReplayFeed"]


class MarketFeed(Protocol):
    rate: float

    def underlyings(self) -> tuple[str, ...]: ...

    def snapshot(self, underlying: str, now: datetime) -> Snapshot | None: ...

    def bars_before(self, underlying: str, day: date) -> pd.DataFrame: ...

    def events(self, underlying: str) -> Events | None: ...


class ReplayFeed:
    def __init__(self, market: Market) -> None:
        self.market = market
        self.rate = market.rate
        self._by: dict[str, list[Snapshot]] = {}
        for s in market.snapshots:
            self._by.setdefault(s.underlying, []).append(s)
        self._times = {k: [s.asof for s in v] for k, v in self._by.items()}

    def underlyings(self) -> tuple[str, ...]:
        return tuple(sorted(self._by))

    def times(self) -> tuple[datetime, ...]:
        return tuple(sorted({s.asof for s in self.market.snapshots}))

    def snapshot(self, underlying: str, now: datetime) -> Snapshot | None:
        times = self._times.get(underlying, [])
        i = bisect_right(times, now)
        if i == 0:
            return None
        s = self._by[underlying][i - 1]
        return s if s.asof.date() == now.date() else None

    def bars_before(self, underlying: str, day: date) -> pd.DataFrame:
        b = self.market.bars[underlying]
        return b[b.index < pd.Timestamp(day)]

    def events(self, underlying: str) -> Events | None:
        return self.market.events.get(underlying)
