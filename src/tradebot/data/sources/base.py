# src/tradebot/data/sources/base.py
"""Shared contract for all point-in-time data sources.

The invariant enforced here (and tested in tests/lookahead/):

    every row carries ``asof_ts`` — the UTC moment the row became KNOWABLE
    (publication / filing / settlement time + explicit lag), and
    ``asof_ts >= event_ts`` for every row.

Downstream merges must go through ``tradebot.utils.time.asof_join`` which
conditions on ``asof_ts``, never on the event timestamp.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "ASOF_COL",
    "EVENT_COL",
    "SourceMeta",
    "http_get_text",
    "stamp_asof",
    "validate_pit",
    "write_market_parquet",
]

ASOF_COL = "asof_ts"
EVENT_COL = "event_ts"

# EDGAR requires a descriptive UA; harmless elsewhere.
_UA = {"User-Agent": "tradebot-research algulizia@gmail.com"}


@dataclass(frozen=True)
class SourceMeta:
    """G8 data-register row, kept next to the code that fetches it."""

    name: str
    url: str
    license: str
    publication_lag: str
    coverage: str
    survivorship: str
    quality_notes: str = ""

    def register_row(self) -> str:
        return (
            f"| {self.name} | {self.url} | {self.license} | "
            f"{self.publication_lag} | {self.coverage} | "
            f"{self.survivorship} | {self.quality_notes} |"
        )


def http_get_text(
    url: str, timeout: int = 60, headers: dict[str, str] | None = None
) -> str:
    """GET a text resource. Retries with exponential backoff; raises for
    HTTP errors.

    ``headers`` overrides the default UA — some hosts (Stooq) refuse
    non-browser UAs while others (SEC) REQUIRE a contact UA. The backoff
    matters under host rate-limiting (FRED throttles after ~100 rapid
    requests; immediate retries then fail 3/3 — observed 2026-06-11).
    """
    import time

    import requests

    for attempt in (1, 2, 3, 4):
        try:
            resp = requests.get(url, headers=headers or _UA, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2.0 ** attempt)  # 2s, 4s, 8s
    raise RuntimeError("unreachable")


def read_csv_text(text: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), **kwargs)


def stamp_asof(
    df: pd.DataFrame,
    lag: pd.Timedelta,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """Add ``asof_ts`` = event timestamp + explicit publication lag.

    Use ONLY when the true publication moment is not in the data itself
    (e.g. EOD prices). When the source carries a real publication timestamp
    (EDGAR acceptance time), set ``asof_ts`` from that instead — never from
    a guessed lag.
    """
    if lag < pd.Timedelta(0):
        raise ValueError(f"publication lag must be >= 0, got {lag}")
    out = df.copy()
    out[ASOF_COL] = pd.to_datetime(out[event_col], utc=True) + lag
    return out


def validate_pit(df: pd.DataFrame, required: tuple[str, ...] = ()) -> pd.DataFrame:
    """Hard PIT contract: asof/event present, UTC, asof >= event, sorted.

    Raises on violation — a silent fix here would be a hidden lookahead.
    """
    for col in (EVENT_COL, ASOF_COL, *required):
        if col not in df.columns:
            raise ValueError(f"PIT contract violation: missing column {col!r}")
    for col in (EVENT_COL, ASOF_COL):
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s) or s.dt.tz is None:
            raise ValueError(f"{col} must be tz-aware UTC datetimes")
        if str(s.dt.tz) not in {"UTC", "tzutc()", "UTC+00:00"}:
            raise ValueError(f"{col} must be UTC, got {s.dt.tz}")
        if s.isna().any():
            raise ValueError(f"{col} contains NaT")
    if (df[ASOF_COL] < df[EVENT_COL]).any():
        n = int((df[ASOF_COL] < df[EVENT_COL]).sum())
        raise ValueError(
            f"PIT contract violation: {n} rows with asof_ts < event_ts "
            "(data 'known' before it happened — lookahead)."
        )
    out = df.sort_values(ASOF_COL, kind="stable").reset_index(drop=True)
    return out


def write_market_parquet(
    df: pd.DataFrame, market: str, name: str, root: Path | str = "market_data_parquet"
) -> Path:
    """Persist a validated PIT frame under market_data_parquet/<market>/."""
    df = validate_pit(df)
    dest = Path(root) / market
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
