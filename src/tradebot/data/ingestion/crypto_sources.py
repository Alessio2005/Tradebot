"""Crypto-bronnen voor het ingestion-contract — Phase 1, deliverables 3/4/5.

Drie bronnen, elk uitsluitend `fetch` + `normalise`; de volgorde, de validatie,
het hashen en het persisteren komen uit `contract.run_ingestion` en zijn niet
per bron te omzeilen.

FUNDING-SETTLEMENTSEMANTIEK (stap 6)
------------------------------------
Een funding rate die om 08:00 UTC settelt is **pas op 08:00 UTC bekend**, niet
om 00:00 van diezelfde dag. `asof_ts_ns` is daarom gelijk aan het
settlement-moment, niet aan het begin van de fundingperiode. Wie dat omdraait
bouwt een carry-signaal dat de rate acht uur te vroeg kent — een lookahead-lek
dat in de backtest als alpha verschijnt.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...utils.failfast import DataContractError, require
from .bybit import MS_TO_NS, BybitV5Client
from .contract import IngestionSource, IngestionSpec

__all__ = ["FundingSource", "OhlcvSource", "OpenInterestSource", "BYBIT_INTERVAL"]

#: Granulariteit uit conf/ -> Bybit kline-interval.
BYBIT_INTERVAL: dict[str, str] = {
    "1d": "D", "4h": "240", "1h": "60", "15m": "15", "5m": "5", "1m": "1",
}

#: Granulariteit -> Bybit open-interest intervalTime.
BYBIT_OI_INTERVAL: dict[str, str] = {
    "1d": "1d", "4h": "4h", "1h": "1h", "15m": "15min", "5m": "5min",
}


@dataclass
class _WindowedSource(IngestionSource):
    """Basis: elke bron haalt een expliciet [start, end]-venster op."""

    client: BybitV5Client
    start_ms: int
    end_ms: int

    def _require_window(self) -> None:
        require(self.end_ms > self.start_ms,
                "Ingestion-venster is leeg of omgekeerd.",
                DataContractError, start=self.start_ms, end=self.end_ms)


class OhlcvSource(_WindowedSource):
    """Spot/perp OHLCV-bars (deliverable 3).

    `event_ts_ns` = OPENINGStijd van de bar.
    `asof_ts_ns`  = SLUITINGStijd van de bar. Een daily bar over 3 januari is
    pas op 4 januari 00:00 UTC compleet; hem op zijn openingstijd als bekend
    markeren zou de close van vandaag vandaag al beschikbaar maken.
    """

    def fetch(self, spec: IngestionSpec) -> pd.DataFrame:
        self._require_window()
        require(spec.granularity in BYBIT_INTERVAL,
                "Granulariteit niet ondersteund door de Bybit kline-endpoint.",
                DataContractError, granularity=spec.granularity,
                known=sorted(BYBIT_INTERVAL))
        return self.client.kline(spec.symbol, BYBIT_INTERVAL[spec.granularity],
                                 self.start_ms, self.end_ms)

    def normalise(self, raw: pd.DataFrame, spec: IngestionSpec) -> pd.DataFrame:
        from ..validation.gaps import cadence_ns

        step = cadence_ns(spec.granularity)
        ev = raw["start_ms"].astype("int64") * MS_TO_NS
        out = pd.DataFrame({
            "event_ts_ns": ev.astype("int64"),
            # De bar is pas compleet op zijn sluitingstijd.
            "asof_ts_ns": (ev + step).astype("int64"),
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "volume": raw["volume"].astype(float),
            "turnover": raw["turnover"].astype(float),
            "symbol": spec.symbol,
        })
        out = out.drop_duplicates(subset=["event_ts_ns"], keep="last")
        out = out[(out["event_ts_ns"] >= self.start_ms * MS_TO_NS)
                  & (out["event_ts_ns"] <= self.end_ms * MS_TO_NS)]
        return out.sort_values("event_ts_ns").reset_index(drop=True)


class FundingSource(_WindowedSource):
    """Perpetual funding rates (deliverable 4).

    `event_ts_ns` = `asof_ts_ns` = settlement-moment. De rate is op precies dat
    moment bekend en geen seconde eerder.
    """

    funding_interval_hours: int = 8

    def __init__(self, client: BybitV5Client, start_ms: int, end_ms: int,
                 funding_interval_hours: int) -> None:
        super().__init__(client, start_ms, end_ms)
        self.funding_interval_hours = int(funding_interval_hours)

    def fetch(self, spec: IngestionSpec) -> pd.DataFrame:
        self._require_window()
        return self.client.funding_history(spec.symbol, self.start_ms, self.end_ms)

    def normalise(self, raw: pd.DataFrame, spec: IngestionSpec) -> pd.DataFrame:
        settle = raw["settle_ms"].astype("int64") * MS_TO_NS
        out = pd.DataFrame({
            "event_ts_ns": settle.astype("int64"),
            # Settlement-semantiek: bekend OP het settlement-moment.
            "asof_ts_ns": settle.astype("int64"),
            "funding_rate": raw["funding_rate"].astype(float),
            "funding_interval_hours": self.funding_interval_hours,
            "symbol": spec.symbol,
        })
        out = out.drop_duplicates(subset=["event_ts_ns"], keep="last")
        return out.sort_values("event_ts_ns").reset_index(drop=True)


class OpenInterestSource(_WindowedSource):
    """Open interest per symbool (deliverable 5)."""

    def fetch(self, spec: IngestionSpec) -> pd.DataFrame:
        self._require_window()
        require(spec.granularity in BYBIT_OI_INTERVAL,
                "Granulariteit niet ondersteund door de open-interest endpoint.",
                DataContractError, granularity=spec.granularity,
                known=sorted(BYBIT_OI_INTERVAL))
        return self.client.open_interest(
            spec.symbol, BYBIT_OI_INTERVAL[spec.granularity],
            self.start_ms, self.end_ms)

    def normalise(self, raw: pd.DataFrame, spec: IngestionSpec) -> pd.DataFrame:
        ts = raw["ts_ms"].astype("int64") * MS_TO_NS
        out = pd.DataFrame({
            "event_ts_ns": ts.astype("int64"),
            # Een OI-snapshot is bekend op het moment van de snapshot zelf.
            "asof_ts_ns": ts.astype("int64"),
            "open_interest": raw["open_interest"].astype(float),
            "symbol": spec.symbol,
        })
        out = out.drop_duplicates(subset=["event_ts_ns"], keep="last")
        return out.sort_values("event_ts_ns").reset_index(drop=True)
