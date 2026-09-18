# src/tradebot/utils/time.py
"""UTC-only time utilities for the live trading stack.

All timestamps in tradebot are UTC-aware pd.Timestamp.  This module
provides conversion helpers that enforce that invariant at system boundaries
(user input, exchange API responses, config files).

PHASE 1 - TIJDSTANDAARD (audit sectie 7.2)
------------------------------------------
De opslagstandaard van het platform is **UTC Unix nanoseconden** (int64).
`to_utc_ns` en `to_utc_ns_series` zijn de enige toegestane conversie naar die
representatie, en zij WIJZEN NAIEVE TIMESTAMPS HARD AF met `DataContractError`.

Dat verschilt bewust van `to_utc`, dat een naieve timestamp als UTC aanneemt.
Die aanname is verdedigbaar aan de LIVE-kant (exchange-responses zijn per
conventie UTC) maar onaanvaardbaar aan de INGESTION-kant: daar is een naieve
timestamp een symptoom van een bron waarvan de tijdzone niet is vastgesteld, en
een verkeerde aanname verschuift de hele reeks stilzwijgend met uren.

`asof_join` behoudt zijn `merge_asof(direction="backward")`-kern ongewijzigd
(RETAIN-item, audit sectie 24). Toegevoegd zijn uitsluitend contract-guards:
een UTC-assertie op beide zijden, een sorteer-assertie, en een VERPLICHTE
`tolerance`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .failfast import DataContractError, require

__all__ = [
    "now_utc",
    "to_utc",
    "to_utc_ns",
    "to_utc_ns_series",
    "from_utc_ns",
    "assert_utc_index",
    "bar_cadence_seconds",
    "align_to_bar",
    "seconds_since",
    "asof_join",
]

#: Naam van de canonieke tijdkolom in de PIT-store.
EVENT_TS = "event_ts"
#: Naam van de canonieke beschikbaarheidskolom (wanneer werd het BEKEND).
ASOF_TS = "asof_ts"

_BAR_CADENCES: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def now_utc() -> pd.Timestamp:
    """Return the current UTC time as a timezone-aware pd.Timestamp."""
    return pd.Timestamp.now(tz="UTC")


def to_utc(ts: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert any timestamp representation to a UTC-aware pd.Timestamp.

    Raises ValueError if the input has no timezone and cannot be assumed UTC.
    Naive datetime inputs are assumed UTC (exchange convention).
    """
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return pd.Timestamp(ts, tz="UTC")
        return pd.Timestamp(ts).tz_convert("UTC")

    # string — let pandas parse, then enforce UTC
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def to_utc_ns(ts: str | datetime | pd.Timestamp) -> int:
    """Converteer naar UTC Unix nanoseconden (int64). Wijst naieve input AF.

    Dit is de centrale conversie uit sectie 7.2. Elke ingestion-bron passeert
    hem. Anders dan `to_utc` wordt een timezone-loze timestamp NIET als UTC
    aangenomen: dat zou een reeks stilzwijgend met uren kunnen verschuiven.

    Raises
    ------
    DataContractError
        Bij een naieve timestamp of een niet-parseerbare waarde.
    """
    if isinstance(ts, str):
        parsed = pd.Timestamp(ts)
    elif isinstance(ts, pd.Timestamp):
        parsed = ts
    elif isinstance(ts, datetime):
        parsed = pd.Timestamp(ts)
    else:
        raise DataContractError(
            f"to_utc_ns kreeg {type(ts).__name__}; verwacht str, datetime of "
            f"pd.Timestamp."
        )
    require(
        parsed.tzinfo is not None,
        "Naieve (timezone-loze) timestamp geweigerd aan de ingestion-grens. "
        "De tijdzone van de bron moet expliciet zijn vastgesteld; hem als UTC "
        "aannemen kan de hele reeks met uren verschuiven.",
        DataContractError,
        value=str(ts),
    )
    return int(parsed.tz_convert("UTC").value)


def to_utc_ns_series(s: pd.Series) -> pd.Series:
    """Vectorvariant van `to_utc_ns`. Wijst een naieve reeks in zijn geheel af."""
    require(
        pd.api.types.is_datetime64_any_dtype(s),
        "to_utc_ns_series verwacht een datetime-reeks.",
        DataContractError,
        dtype=str(s.dtype),
    )
    require(
        getattr(s.dt, "tz", None) is not None,
        "Naieve (timezone-loze) datetime-reeks geweigerd aan de "
        "ingestion-grens. Stel de tijdzone van de bron expliciet vast.",
        DataContractError,
        name=str(s.name),
    )
    return s.dt.tz_convert("UTC").astype("int64")


def from_utc_ns(
    ns: int | pd.Series | npt.NDArray[np.integer[Any]],
) -> pd.Timestamp | pd.Series:
    """Inverse van `to_utc_ns`: UTC Unix nanoseconden terug naar UTC-aware tijd."""
    if isinstance(ns, (int, np.integer)):
        return pd.Timestamp(int(ns), unit="ns", tz="UTC")
    return pd.to_datetime(pd.Series(ns), unit="ns", utc=True)


def assert_utc_index(df: pd.DataFrame, *, name: str = "frame",
                     require_monotonic: bool = True) -> None:
    """Dwing af dat `df` een UTC-aware, oplopend gesorteerde DatetimeIndex heeft."""
    require(
        isinstance(df.index, pd.DatetimeIndex),
        f"{name} moet een DatetimeIndex hebben.",
        DataContractError,
        index_type=type(df.index).__name__,
    )
    require(
        df.index.tz is not None,
        f"{name} heeft een naieve DatetimeIndex; UTC is verplicht (sectie 7.2).",
        DataContractError,
    )
    require(
        str(df.index.tz) in ("UTC", "utc"),
        f"{name} staat niet in UTC.",
        DataContractError,
        tz=str(df.index.tz),
    )
    if require_monotonic:
        require(
            df.index.is_monotonic_increasing,
            f"{name} is niet oplopend gesorteerd; merge_asof levert dan stille "
            f"onzin op in plaats van een fout.",
            DataContractError,
        )


def bar_cadence_seconds(interval: str) -> int:
    """Return the number of seconds in a bar interval string (e.g. ``"1h"``).

    Raises KeyError for unknown interval strings.
    """
    if interval not in _BAR_CADENCES:
        raise KeyError(
            f"Unknown interval {interval!r}. Known: {sorted(_BAR_CADENCES)}"
        )
    return _BAR_CADENCES[interval]


def align_to_bar(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Floor a timestamp to the nearest bar boundary for the given interval."""
    cadence = bar_cadence_seconds(interval)
    epoch_s = int(ts.timestamp())
    floored_s = (epoch_s // cadence) * cadence
    return pd.Timestamp(floored_s, unit="s", tz="UTC")


def seconds_since(ts: pd.Timestamp) -> float:
    """Return the number of seconds elapsed since ``ts`` (UTC)."""
    return float((now_utc() - ts).total_seconds())


def asof_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    asof_col: str = "asof_ts",
    by: str | None = None,
    suffix: str = "",
    *,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Join ``right`` onto ``left`` using only information AVAILABLE at t.

    For every row of ``left`` (UTC DatetimeIndex = decision time t), take the
    most recent row of ``right`` whose ``asof_col`` (availability moment,
    Mandate v3 §7) is STRICTLY <= t. Rows of ``right`` published after t can
    never leak in — this is the single sanctioned way to merge PIT sources
    (R-1; tested in tests/lookahead/).

    Parameters
    ----------
    left  : decision-time frame, tz-aware UTC DatetimeIndex.
    right : PIT frame with ``asof_col`` tz-aware UTC column.
    by    : optional key column (e.g. ``symbol``) present in both frames for
            grouped as-of joins; ``left`` must then carry it as a column.
    suffix: appended to right-hand column names on collision.
    tolerance : VERPLICHT, keyword-only. Maximale ouderdom van de rechter-rij.
        Zonder bovengrens draagt `merge_asof` een waarde onbeperkt vooruit: een
        funding rate uit 2021 zou dan nog aan een bar uit 2026 worden gekoppeld,
        zonder enige melding. De waarde is een BELEIDSKEUZE en hoort daarom in
        `conf/data/` (`asof_tolerance_seconds`), niet als default in deze functie.

    Phase 1: de `merge_asof(direction="backward")`-kern is ONGEWIJZIGD gebleven
    (RETAIN-item, audit sectie 24). Toegevoegd zijn uitsluitend contract-guards:
    UTC-assertie op beide zijden, sorteer-assertie, en de verplichte tolerance.
    """
    assert_utc_index(left, name="asof_join(left=...)")
    require(
        asof_col in right.columns,
        f"asof_join: rechterframe mist de beschikbaarheidskolom {asof_col!r}.",
        DataContractError,
        columns=list(right.columns)[:12],
    )
    rs = right[asof_col]
    require(
        pd.api.types.is_datetime64_any_dtype(rs),
        f"right[{asof_col!r}] is geen datetime-kolom.",
        DataContractError,
        dtype=str(rs.dtype),
    )
    require(
        getattr(rs.dt, "tz", None) is not None,
        f"right[{asof_col!r}] is naief; UTC is verplicht (sectie 7.2).",
        DataContractError,
    )
    require(
        isinstance(tolerance, pd.Timedelta) and tolerance > pd.Timedelta(0),
        "asof_join vereist een expliciete, positieve tolerance (pd.Timedelta). "
        "Zonder bovengrens draagt merge_asof een waarde onbeperkt vooruit.",
        DataContractError,
        tolerance=str(tolerance),
    )

    lf = left.reset_index().rename(columns={left.index.name or "index": "_t"})
    rf = right.sort_values(asof_col, kind="stable")
    overlap = (set(lf.columns) & set(rf.columns)) - {asof_col, by}
    if overlap:
        rf = rf.rename(columns={c: f"{c}{suffix or '_r'}" for c in overlap})
    merged = pd.merge_asof(
        lf.sort_values("_t", kind="stable"),
        rf,
        left_on="_t",
        right_on=asof_col,
        by=by,
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )
    return merged.set_index("_t").rename_axis(left.index.name)
