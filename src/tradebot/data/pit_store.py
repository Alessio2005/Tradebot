"""Append-only Point-in-Time store — Phase 1, deliverable 1.

De onveranderlijke opslaglaag waarop elke latere statistische claim rust.

CONTRACT
--------
1. **Partitionering** op `(asset_class, symbol, granularity, date)`. Het pad is
   de sleutel; er bestaat geen index-bestand dat uit sync kan raken.
2. **Append-only.** Een tweede schrijfactie op dezelfde partitie met AFWIJKENDE
   inhoud crasht met `DataContractError`. Een schrijfactie met identieke inhoud
   is een no-op — zodat een herstart van een ingestion-run veilig is.
3. **Tijd** in UTC Unix nanoseconden (`event_ts_ns`), plus de
   beschikbaarheidskolom `asof_ts_ns`. Beide int64. Zie `utils/time.py`.
4. **`data_hash`** over de GESORTEERDE INHOUD, niet over bestandsmetadata: twee
   onafhankelijke ingestion-runs op dezelfde bronperiode moeten dezelfde hash
   opleveren, ongeacht schrijfvolgorde of machine.
5. **Correcties zijn nieuwe versies**, geen edits. Een gecorrigeerde partitie
   krijgt een nieuwe `data_hash` en wordt naast de oude geregistreerd.

Er bestaat geen `overwrite=True`. Dat is opzet: onveranderlijkheid boven gemak.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ..utils.failfast import DataContractError, require
from ..utils.hashing import dataframe_content_hash

__all__ = [
    "PARTITION_KEYS",
    "PitStore",
    "PartitionRef",
]

PARTITION_KEYS = ("asset_class", "symbol", "granularity", "date")

#: Verplichte kolommen in elke PIT-partitie.
REQUIRED_COLUMNS = ("event_ts_ns", "asof_ts_ns")

AssetClass = Literal["crypto", "fx", "macro", "commodities", "equities"]


@dataclass(frozen=True)
class PartitionRef:
    """Onveranderlijke verwijzing naar één partitie in de store."""

    asset_class: str
    symbol: str
    granularity: str
    date: str  # ISO yyyy-mm-dd

    def relpath(self) -> Path:
        return Path(
            f"asset_class={self.asset_class}",
            f"symbol={self.symbol}",
            f"granularity={self.granularity}",
            f"date={self.date}",
        )

    def __str__(self) -> str:
        return "/".join(self.relpath().parts)


class PitStore:
    """Append-only Parquet-store met een expliciet onveranderlijkheidscontract."""

    _DATA_FILE = "data.parquet"
    _META_FILE = "_meta.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ paths
    def partition_dir(self, ref: PartitionRef) -> Path:
        return self.root / ref.relpath()

    def data_path(self, ref: PartitionRef) -> Path:
        return self.partition_dir(ref) / self._DATA_FILE

    def meta_path(self, ref: PartitionRef) -> Path:
        return self.partition_dir(ref) / self._META_FILE

    def exists(self, ref: PartitionRef) -> bool:
        return self.data_path(ref).is_file()

    # ------------------------------------------------------------- validation
    @staticmethod
    def _validate(df: pd.DataFrame, ref: PartitionRef) -> None:
        require(
            not df.empty,
            "Weigering om een LEGE partitie te schrijven. Een lege dataset is "
            "geen geldig resultaat maar een symptoom van een gefaalde fetch.",
            DataContractError, partition=str(ref),
        )
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        require(
            not missing,
            "Partitie mist verplichte tijdkolommen.",
            DataContractError, partition=str(ref), missing=missing,
        )
        for col in REQUIRED_COLUMNS:
            require(
                pd.api.types.is_integer_dtype(df[col]),
                f"{col} moet int64 UTC Unix nanoseconden zijn (sectie 7.2), "
                f"geen datetime en geen float.",
                DataContractError, partition=str(ref), dtype=str(df[col].dtype),
            )
            require(
                df[col].notna().all(),
                f"{col} bevat NaN.",
                DataContractError, partition=str(ref),
            )
        require(
            df["event_ts_ns"].is_monotonic_increasing,
            "event_ts_ns is niet oplopend gesorteerd. Een ongesorteerde partitie "
            "maakt elke merge_asof stil onbetrouwbaar.",
            DataContractError, partition=str(ref),
        )
        require(
            (df["asof_ts_ns"] >= df["event_ts_ns"]).all(),
            "asof_ts_ns ligt VOOR event_ts_ns: de waarde zou bekend zijn "
            "geweest voordat de gebeurtenis plaatsvond. Dat is per definitie "
            "een lookahead-lek.",
            DataContractError, partition=str(ref),
            n_violations=int((df["asof_ts_ns"] < df["event_ts_ns"]).sum()),
        )

    # ------------------------------------------------------------------ write
    def write(
        self,
        df: pd.DataFrame,
        ref: PartitionRef,
        *,
        source: str,
        extra_meta: dict | None = None,
    ) -> str:
        """Schrijf één partitie. Retourneert de `data_hash`.

        Bestaat de partitie al met IDENTIEKE inhoud, dan is dit een no-op en
        wordt de bestaande hash teruggegeven (zodat een herstarte ingestion-run
        veilig is). Bestaat hij met AFWIJKENDE inhoud, dan crasht deze methode.
        """
        self._validate(df, ref)
        df = df.reset_index(drop=True)
        new_hash = dataframe_content_hash(df)

        if self.exists(ref):
            existing = self.read(ref)
            existing_hash = dataframe_content_hash(existing.reset_index(drop=True))
            require(
                existing_hash == new_hash,
                "PIT-store is APPEND-ONLY. Er bestaat al een partitie met "
                "afwijkende inhoud. Een correctie op historische data is een "
                "NIEUWE versie met een nieuwe data_hash, nooit een edit.",
                DataContractError,
                partition=str(ref),
                existing_hash=existing_hash,
                new_hash=new_hash,
                existing_rows=len(existing),
                new_rows=len(df),
            )
            return existing_hash

        d = self.partition_dir(ref)
        d.mkdir(parents=True, exist_ok=True)
        tmp = self.data_path(ref).with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, compression="snappy")
        tmp.replace(self.data_path(ref))

        meta = {
            "asset_class": ref.asset_class,
            "symbol": ref.symbol,
            "granularity": ref.granularity,
            "date": ref.date,
            "source": source,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "data_hash": new_hash,
            "event_ts_ns_min": int(df["event_ts_ns"].min()),
            "event_ts_ns_max": int(df["event_ts_ns"].max()),
            "written_at_ns": int(pd.Timestamp.now(tz="UTC").value),
        }
        if extra_meta:
            meta.update(extra_meta)
        self.meta_path(ref).write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return new_hash

    # ------------------------------------------------------------------- read
    def read(self, ref: PartitionRef) -> pd.DataFrame:
        require(
            self.exists(ref),
            "Partitie bestaat niet in de PIT-store.",
            DataContractError, partition=str(ref), root=str(self.root),
        )
        return pd.read_parquet(self.data_path(ref))

    def read_meta(self, ref: PartitionRef) -> dict:
        p = self.meta_path(ref)
        require(
            p.is_file(),
            "Partitie heeft geen metadata; provenance is niet vast te stellen.",
            DataContractError, partition=str(ref),
        )
        return json.loads(p.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------- scan
    def partitions(
        self,
        asset_class: str | None = None,
        symbol: str | None = None,
        granularity: str | None = None,
    ) -> list[PartitionRef]:
        """Alle partities in de store, optioneel gefilterd."""
        if not self.root.is_dir():
            return []
        out: list[PartitionRef] = []
        for p in sorted(self.root.rglob(self._DATA_FILE)):
            parts = p.relative_to(self.root).parts
            if len(parts) < 5:
                continue
            kv = {}
            for seg in parts[:4]:
                k, _, v = seg.partition("=")
                kv[k] = v
            if set(kv) != set(PARTITION_KEYS):
                continue
            ref = PartitionRef(kv["asset_class"], kv["symbol"],
                               kv["granularity"], kv["date"])
            if asset_class and ref.asset_class != asset_class:
                continue
            if symbol and ref.symbol != symbol:
                continue
            if granularity and ref.granularity != granularity:
                continue
            out.append(ref)
        return out

    def load(
        self,
        asset_class: str,
        symbol: str,
        granularity: str,
    ) -> pd.DataFrame:
        """Concateneer elke partitie van één reeks tot één oplopende frame."""
        refs = self.partitions(asset_class, symbol, granularity)
        require(
            refs,
            "Geen enkele partitie gevonden voor deze reeks. Er wordt GEEN lege "
            "DataFrame teruggegeven: dat zou stilzwijgend als 'geen signaal' "
            "worden geinterpreteerd in plaats van als ontbrekende data.",
            DataContractError,
            asset_class=asset_class, symbol=symbol, granularity=granularity,
            root=str(self.root),
        )
        frames = [self.read(r) for r in refs]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("event_ts_ns", kind="stable").reset_index(drop=True)
        return df

    def series_hash(self, asset_class: str, symbol: str, granularity: str) -> str:
        """Deterministische `data_hash` over de volledige, gesorteerde reeks."""
        return dataframe_content_hash(self.load(asset_class, symbol, granularity))

    # ----------------------------------------------------------------- delete
    def drop_partition(self, ref: PartitionRef, *, i_understand: bool = False) -> None:
        """Verwijder een partitie. Uitsluitend voor testopruiming.

        Vereist `i_understand=True`. In productie bestaat er geen legitieme
        reden om een PIT-partitie te verwijderen: de store is append-only.
        """
        require(
            i_understand,
            "drop_partition() is geen productie-operatie. De PIT-store is "
            "append-only; een correctie is een nieuwe versie, geen verwijdering.",
            DataContractError, partition=str(ref),
        )
        d = self.partition_dir(ref)
        if d.is_dir():
            shutil.rmtree(d)
