"""Gap-detectie en gap-ledger — Phase 1, deliverable 7b.

DOCTRINE (Phase 1, regels)
--------------------------
> *"Geen stilzwijgende interpolatie. `fillna(method="ffill")` over een gat is een
> keuze met statistische gevolgen. Zulke keuzes staan in de config, worden
> geregistreerd in de gap-ledger, en worden nooit impliciet in een
> ingestion-functie genomen."*

Deze module interpoleert dus niets. Hij **meet** ontbrekende bars, schrijft ze
naar een inspecteerbare ledger, en laat de `gap_policy` uit `conf/data/`
bepalen of een gat de ingestion breekt (`reject`) of alleen wordt vastgelegd
(`register`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ...utils.failfast import DataContractError, require

__all__ = ["GapLedger", "GapRecord", "detect_gaps", "enforce_gap_policy"]

#: Bar-cadans in nanoseconden per granulariteit.
_CADENCE_NS: dict[str, int] = {
    "1m": 60 * 10**9,
    "5m": 300 * 10**9,
    "15m": 900 * 10**9,
    "1h": 3600 * 10**9,
    "4h": 14400 * 10**9,
    "8h": 28800 * 10**9,
    "1d": 86400 * 10**9,
}


@dataclass(frozen=True)
class GapRecord:
    """Eén aaneengesloten reeks ontbrekende bars."""

    asset_class: str
    symbol: str
    granularity: str
    #: laatste aanwezige bar vóór het gat
    gap_start_ns: int
    #: eerste aanwezige bar ná het gat
    gap_end_ns: int
    #: aantal ontbrekende bars daartussen
    n_missing: int

    @property
    def gap_start_iso(self) -> str:
        return str(pd.Timestamp(self.gap_start_ns, unit="ns", tz="UTC"))

    @property
    def gap_end_iso(self) -> str:
        return str(pd.Timestamp(self.gap_end_ns, unit="ns", tz="UTC"))


def cadence_ns(granularity: str) -> int:
    require(granularity in _CADENCE_NS,
            "Onbekende granulariteit; de bar-cadans kan niet worden bepaald en "
            "gaps zijn dus niet meetbaar.",
            DataContractError, granularity=granularity,
            known=sorted(_CADENCE_NS))
    return _CADENCE_NS[granularity]


def detect_gaps(
    df: pd.DataFrame,
    *,
    asset_class: str,
    symbol: str,
    granularity: str,
) -> list[GapRecord]:
    """Meet ontbrekende bars in een oplopende `event_ts_ns`-reeks.

    Retourneert een lijst van `GapRecord`. Een LEGE lijst betekent: geen enkele
    ontbrekende bar. Deze functie muteert `df` niet en vult niets aan.
    """
    require("event_ts_ns" in df.columns,
            "detect_gaps vereist een event_ts_ns-kolom.",
            DataContractError, columns=list(df.columns)[:15])
    require(df["event_ts_ns"].is_monotonic_increasing,
            "detect_gaps vereist een oplopend gesorteerde reeks.",
            DataContractError, symbol=symbol)

    step = cadence_ns(granularity)
    ev = df["event_ts_ns"].to_numpy()
    if ev.size < 2:
        return []

    deltas = ev[1:] - ev[:-1]
    out: list[GapRecord] = []
    for i, d in enumerate(deltas):
        if d > step:
            n_missing = int(d // step) - 1
            if n_missing > 0:
                out.append(GapRecord(
                    asset_class=asset_class, symbol=symbol,
                    granularity=granularity,
                    gap_start_ns=int(ev[i]), gap_end_ns=int(ev[i + 1]),
                    n_missing=n_missing,
                ))
    return out


class GapLedger:
    """Append-only, inspecteerbaar register van elke ontbrekende bar."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, records: list[GapRecord]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for r in records:
                row = asdict(r)
                row["gap_start_iso"] = r.gap_start_iso
                row["gap_end_iso"] = r.gap_end_iso
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def read_all(self) -> pd.DataFrame:
        if not self.path.is_file():
            return pd.DataFrame(columns=[
                "asset_class", "symbol", "granularity", "gap_start_ns",
                "gap_end_ns", "n_missing", "gap_start_iso", "gap_end_iso"])
        rows = [json.loads(ln) for ln in
                self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        df = self.read_all()
        if df.empty:
            return df
        return (df.groupby(["asset_class", "symbol", "granularity"], as_index=False)
                  .agg(n_gaps=("n_missing", "size"),
                       n_missing_bars=("n_missing", "sum")))


def enforce_gap_policy(
    records: list[GapRecord],
    *,
    policy: str,
    ledger: GapLedger,
    context: str = "",
) -> None:
    """Pas de `gap_policy` uit `conf/data/` toe.

    `reject`    - elk gat breekt de ingestion (default; sectie 7.2).
    `register`  - het gat wordt vastgelegd en de ingestion gaat door. Dit is
                  uitsluitend legitiem wanneer de gebruikende research track
                  EXPLICIET verklaart hoe hij met het gat omgaat; die verklaring
                  hoort in `docs/DATA_REGISTER.md`.

    In beide gevallen wordt het gat WEGGESCHREVEN. Er bestaat geen pad waarin
    een gat verdwijnt zonder spoor.
    """
    ledger.append(records)
    require(policy in ("reject", "register"),
            "Onbekende gap_policy.", DataContractError, policy=policy)
    if policy == "reject" and records:
        total = sum(r.n_missing for r in records)
        first = records[0]
        raise DataContractError(
            f"{len(records)} gat(en), samen {total} ontbrekende bar(s) "
            f"{('in ' + context) if context else ''}. gap_policy='reject'. "
            f"Eerste gat: {first.gap_start_iso} -> {first.gap_end_iso} "
            f"({first.n_missing} bars). Er wordt NIET geinterpoleerd: dat zou "
            f"een statistische keuze zijn die stilzwijgend in de ingestion "
            f"wordt genomen. Alle gaten staan in {ledger.path}."
        )
