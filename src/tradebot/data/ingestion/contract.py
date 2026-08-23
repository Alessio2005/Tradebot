"""Het generieke ingestion-contract — Phase 1, deliverable 2.

    fetch -> validate -> normalise -> hash -> persist

**Geen enkele bron mag deze volgorde overslaan.** Dat is geen stijlregel: elke
overgeslagen stap is een dataset die later in de ledger verschijnt met een
`data_hash` die niets garandeert.

Een bron implementeert `IngestionSource` en levert uitsluitend de *fetch* en de
*normalise*. De volgorde, de validatie, het hashen en het wegschrijven zitten in
`run_ingestion` en zijn daarmee niet per bron te omzeilen.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from ...utils.failfast import DataContractError, require
from ...utils.hashing import dataframe_content_hash
from ..pit_store import PartitionRef, PitStore
from ..validation import (
    GapLedger,
    SeriesSpec,
    detect_gaps,
    detect_outliers,
    enforce_gap_policy,
    enforce_outlier_policy,
    validate_schema,
)

__all__ = ["IngestionResult", "IngestionSource", "IngestionSpec", "run_ingestion"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionSpec:
    """Alles wat één ingestion-run nodig heeft. Geen enkel veld heeft een default
    dat een statistische keuze impliceert."""

    asset_class: str
    #: "ohlcv" | "funding" | "open_interest" - onderdeel van de partitiesleutel
    dataset: str
    symbol: str
    granularity: str
    #: schema waartegen de genormaliseerde frame wordt gevalideerd
    series_spec: SeriesSpec
    #: "reject" | "register" — uit conf/data/
    gap_policy: str
    #: drempel op |log-return| — uit conf/data/
    max_abs_log_return: float
    #: of gemeten prijssprongen als echte marktgebeurtenissen zijn geaccepteerd
    allow_price_jumps: bool
    #: herkomst, gaat mee in de partitie-metadata
    source_name: str
    extra_meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    spec: IngestionSpec
    n_rows: int
    n_partitions: int
    data_hash: str
    first_event_ns: int
    last_event_ns: int
    n_gaps: int
    n_missing_bars: int

    @property
    def first_iso(self) -> str:
        return str(pd.Timestamp(self.first_event_ns, unit="ns", tz="UTC"))

    @property
    def last_iso(self) -> str:
        return str(pd.Timestamp(self.last_event_ns, unit="ns", tz="UTC"))


class IngestionSource(ABC):
    """Een databron levert uitsluitend `fetch` en `normalise`."""

    @abstractmethod
    def fetch(self, spec: IngestionSpec) -> pd.DataFrame:
        """Haal de ruwe respons op. Crasht bij een gefaalde of lege fetch."""

    @abstractmethod
    def normalise(self, raw: pd.DataFrame, spec: IngestionSpec) -> pd.DataFrame:
        """Zet de ruwe respons om naar het canonieke schema.

        Verplicht in de output: `event_ts_ns` en `asof_ts_ns` als int64 UTC Unix
        nanoseconden, oplopend gesorteerd. Zie `utils/time.to_utc_ns_series`.
        """


class _HasPartitionDate(Protocol):
    def __call__(self, event_ts_ns: int) -> str: ...


def _partition_date(event_ts_ns: int, granularity: str) -> str:
    """Partitiedatum voor één bar.

    Daily bars krijgen één partitie per JAAR — een partitie per dag zou voor
    zes symbolen over acht jaar ruim 17.000 mappen met elk vijf rijen opleveren,
    wat het lezen domineert met filesystem-overhead. Intraday krijgt een
    partitie per DAG.
    """
    ts = pd.Timestamp(event_ts_ns, unit="ns", tz="UTC")
    if granularity in ("1d", "8h", "4h"):
        return f"{ts.year:04d}-01-01"
    return ts.strftime("%Y-%m-%d")


def run_ingestion(
    source: IngestionSource,
    spec: IngestionSpec,
    store: PitStore,
    gap_ledger: GapLedger,
) -> IngestionResult:
    """Draai het volledige contract. Elke stap crasht bij schending.

    De volgorde is hier vastgelegd en niet per bron te wijzigen — dat is het
    hele punt van het contract.
    """
    # ---- 1. FETCH ---------------------------------------------------------
    raw = source.fetch(spec)
    require(isinstance(raw, pd.DataFrame) and not raw.empty,
            "fetch() leverde geen rijen op. Een lege respons is een gefaalde "
            "fetch, geen geldig resultaat; er wordt niets weggeschreven.",
            DataContractError, symbol=spec.symbol,
            granularity=spec.granularity, source=spec.source_name)

    # ---- 2. NORMALISE -----------------------------------------------------
    df = source.normalise(raw, spec)
    require(isinstance(df, pd.DataFrame) and not df.empty,
            "normalise() leverde een lege frame op.",
            DataContractError, symbol=spec.symbol)
    df = df.sort_values("event_ts_ns", kind="stable").reset_index(drop=True)

    # ---- 2b. DROP NOG-NIET-KENBARE RIJEN ---------------------------------
    # Universele PIT-regel: een rij waarvan `asof_ts_ns` in de TOEKOMST ligt is
    # nog niet kenbaar en dus geen observatie.
    #
    # Concreet: de bar van vandaag is nog niet gesloten. Zijn "close" is de
    # laatste prijs tot NU, niet de dagslot-prijs. Hem wegschrijven levert twee
    # problemen op die allebei stil zijn:
    #   1. de store wordt niet-reproduceerbaar - dezelfde ingestion-run levert
    #      een uur later een andere data_hash op (aangetroffen tijdens de eerste
    #      volledige ingestion; de append-only guard van de PIT-store ving het);
    #   2. een backtest zou een gedeeltelijke dagslot-prijs als definitief
    #      behandelen.
    now_ns = int(pd.Timestamp.now(tz="UTC").value)
    n_future = int((df["asof_ts_ns"] > now_ns).sum())
    if n_future:
        logger.info("[%s %s] %d nog niet gesloten bar(s) overgeslagen "
                    "(asof in de toekomst).", spec.symbol, spec.granularity,
                    n_future)
        df = df[df["asof_ts_ns"] <= now_ns].reset_index(drop=True)
    require(not df.empty,
            "Na het verwijderen van nog niet gesloten bars bleef er niets over.",
            DataContractError, symbol=spec.symbol, n_future=n_future)

    # ---- 3. VALIDATE ------------------------------------------------------
    ctx = f"{spec.symbol} {spec.granularity}"
    validate_schema(df, spec.series_spec, context=ctx)

    gaps = detect_gaps(df, asset_class=spec.asset_class, symbol=spec.symbol,
                       granularity=spec.granularity)
    enforce_gap_policy(gaps, policy=spec.gap_policy, ledger=gap_ledger,
                       context=ctx)

    if "close" in df.columns:
        report = detect_outliers(df, symbol=spec.symbol,
                                 granularity=spec.granularity,
                                 max_abs_log_return=spec.max_abs_log_return)
        enforce_outlier_policy(report, allow_price_jumps=spec.allow_price_jumps)

    # ---- 4. HASH ----------------------------------------------------------
    series_hash = dataframe_content_hash(df)

    # ---- 5. PERSIST -------------------------------------------------------
    df = df.assign(_pdate=[_partition_date(int(t), spec.granularity)
                           for t in df["event_ts_ns"]])
    n_parts = 0
    for pdate, chunk in df.groupby("_pdate", sort=True):
        ref = PartitionRef(spec.asset_class, spec.dataset, spec.symbol,
                           spec.granularity, str(pdate))
        store.write(
            chunk.drop(columns=["_pdate"]).reset_index(drop=True),
            ref,
            source=spec.source_name,
            extra_meta={**spec.extra_meta, "series_data_hash": series_hash},
        )
        n_parts += 1

    out = df.drop(columns=["_pdate"])
    logger.info("[%s %s] %d rijen, %d partities, hash=%s",
                spec.symbol, spec.granularity, len(out), n_parts, series_hash)
    return IngestionResult(
        spec=spec,
        n_rows=len(out),
        n_partitions=n_parts,
        data_hash=series_hash,
        first_event_ns=int(out["event_ts_ns"].iloc[0]),
        last_event_ns=int(out["event_ts_ns"].iloc[-1]),
        n_gaps=len(gaps),
        n_missing_bars=sum(g.n_missing for g in gaps),
    )
