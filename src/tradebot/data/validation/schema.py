"""Schemavalidatie voor PIT-datasets — Phase 1, deliverable 7a.

Kolomtypen, verplichte velden, monotone timestamps.

Elke validator **raiset** bij schending en retourneert géén boolean. Een
aanroeper mag de uitkomst niet kunnen negeren; dat is precies het patroon dat
in Phase 0 uit de hele codebase is verwijderd.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ...utils.failfast import DataContractError, require

__all__ = ["OHLCV_COLUMNS", "SeriesSpec", "validate_schema"]

#: Verplichte OHLCV-kolommen, naast de tijdkolommen.
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class SeriesSpec:
    """Verwacht schema van één PIT-reeks."""

    name: str
    numeric_columns: tuple[str, ...]
    #: kolommen die aanwezig moeten zijn maar niet-numeriek mogen zijn
    other_columns: tuple[str, ...] = field(default=())
    #: kolommen die strikt positief moeten zijn (prijzen)
    positive_columns: tuple[str, ...] = field(default=())
    #: kolommen die niet-negatief moeten zijn (volumes, open interest)
    non_negative_columns: tuple[str, ...] = field(default=())


OHLCV_SPEC = SeriesSpec(
    name="ohlcv",
    numeric_columns=OHLCV_COLUMNS,
    positive_columns=("open", "high", "low", "close"),
    non_negative_columns=("volume",),
)

FUNDING_SPEC = SeriesSpec(
    name="funding",
    numeric_columns=("funding_rate",),
    other_columns=("funding_interval_hours",),
)

OPEN_INTEREST_SPEC = SeriesSpec(
    name="open_interest",
    numeric_columns=("open_interest",),
    non_negative_columns=("open_interest",),
)


def validate_schema(df: pd.DataFrame, spec: SeriesSpec, *, context: str = "") -> None:
    """Dwing `spec` af op `df`. Raiset `DataContractError` bij elke schending.

    Gecontroleerd wordt:
      * aanwezigheid van `event_ts_ns` en `asof_ts_ns` als int64;
      * strikt monotoon oplopende `event_ts_ns` (geen duplicaten);
      * `asof_ts_ns >= event_ts_ns` (niets is bekend vóór het gebeurde);
      * aanwezigheid en numeriek dtype van elke verplichte kolom;
      * afwezigheid van NaN/inf in de numerieke kolommen;
      * positiviteit respectievelijk niet-negativiteit waar dat geldt;
      * de OHLC-ordening `low <= min(open, close) <= max(open, close) <= high`.
    """
    where = f" [{context}]" if context else ""

    require(not df.empty, f"Lege dataset{where}: dit is geen resultaat maar een "
                          f"gefaalde fetch.", DataContractError, spec=spec.name)

    for col in ("event_ts_ns", "asof_ts_ns"):
        require(col in df.columns, f"Kolom {col!r} ontbreekt{where}.",
                DataContractError, spec=spec.name, columns=list(df.columns)[:15])
        require(pd.api.types.is_integer_dtype(df[col]),
                f"{col} moet int64 UTC Unix nanoseconden zijn{where} (sectie 7.2).",
                DataContractError, spec=spec.name, dtype=str(df[col].dtype))

    ev = df["event_ts_ns"]
    require(ev.is_monotonic_increasing,
            f"event_ts_ns is niet oplopend gesorteerd{where}.",
            DataContractError, spec=spec.name)
    n_dup = int(ev.duplicated().sum())
    require(n_dup == 0,
            f"event_ts_ns bevat duplicaten{where}. Een dubbele bar telt dubbel "
            f"mee in elke aggregatie en in elke return-reeks.",
            DataContractError, spec=spec.name, n_duplicates=n_dup)
    n_ahead = int((df["asof_ts_ns"] < ev).sum())
    require(n_ahead == 0,
            f"asof_ts_ns ligt vóór event_ts_ns{where}: de waarde zou bekend zijn "
            f"geweest voordat de gebeurtenis plaatsvond.",
            DataContractError, spec=spec.name, n_violations=n_ahead)

    for col in spec.numeric_columns:
        require(col in df.columns, f"Verplichte kolom {col!r} ontbreekt{where}.",
                DataContractError, spec=spec.name, columns=list(df.columns)[:15])
        require(pd.api.types.is_numeric_dtype(df[col]),
                f"Kolom {col!r} is niet numeriek{where}.",
                DataContractError, spec=spec.name, dtype=str(df[col].dtype))
        finite = pd.Series(df[col]).replace([float("inf"), float("-inf")], pd.NA)
        n_bad = int(finite.isna().sum())
        require(n_bad == 0,
                f"Kolom {col!r} bevat {n_bad} NaN/inf-waarde(n){where}. Gaten "
                f"worden geregistreerd in de gap-ledger, nooit stilzwijgend "
                f"geinterpoleerd.",
                DataContractError, spec=spec.name, column=col, n_bad=n_bad)

    for col in spec.other_columns:
        require(col in df.columns, f"Verplichte kolom {col!r} ontbreekt{where}.",
                DataContractError, spec=spec.name)

    for col in spec.positive_columns:
        n_bad = int((df[col] <= 0).sum())
        require(n_bad == 0,
                f"Kolom {col!r} bevat {n_bad} niet-positieve waarde(n){where}. "
                f"Een prijs van nul of lager is geen observatie maar corruptie.",
                DataContractError, spec=spec.name, column=col, n_bad=n_bad)

    for col in spec.non_negative_columns:
        n_bad = int((df[col] < 0).sum())
        require(n_bad == 0,
                f"Kolom {col!r} bevat {n_bad} negatieve waarde(n){where}.",
                DataContractError, spec=spec.name, column=col, n_bad=n_bad)

    if set(OHLCV_COLUMNS).issubset(df.columns):
        body_hi = df[["open", "close"]].max(axis=1)
        body_lo = df[["open", "close"]].min(axis=1)
        n_hi = int((df["high"] < body_hi).sum())
        n_lo = int((df["low"] > body_lo).sum())
        require(n_hi == 0 and n_lo == 0,
                f"OHLC-ordening geschonden{where}: high < max(open, close) of "
                f"low > min(open, close). Zulke bars breken elke "
                f"range-gebaseerde volatiliteitsschatter (Parkinson, "
                f"Garman-Klass, Yang-Zhang) zonder dat die dat merkt.",
                DataContractError, spec=spec.name,
                n_high_violations=n_hi, n_low_violations=n_lo)
