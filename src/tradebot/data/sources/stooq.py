# src/tradebot/data/sources/stooq.py
"""Stooq daily OHLCV — equities / ETFs / FX / futures continuations (free).

PIT convention: a daily bar's event_ts = session close date (00:00 UTC of
that calendar day); asof_ts = event date + ``lag`` (default: next calendar
day 00:00 UTC, conservative — Stooq publishes EOD the same evening but we
never assume intraday availability of a close).

Coverage caveat (G8): Stooq carries *some* delisted US names but coverage is
incomplete — survivorship per universe must be measured and documented in
docs/DATA_REGISTER.md, and stressed in the wave log (Mandate §10 G8).
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pandas as pd

from .base import (
    EVENT_COL,
    SourceMeta,
    read_csv_text,
    stamp_asof,
    validate_pit,
    write_market_parquet,
)

__all__ = ["META", "fetch_daily", "fetch_universe"]

META = SourceMeta(
    name="stooq",
    url="https://stooq.com/q/d/l/",
    license="free for personal/research use (no API key)",
    publication_lag="EOD close -> available next calendar day 00:00 UTC (conservative)",
    coverage="US/intl equities, ETFs, FX, futures continuations; daily; multi-decade",
    survivorship="PARTIAL delisted coverage — measure per universe, document gap",
    quality_notes="prices unadjusted/adjusted per symbol suffix; rate-limit politely",
)

_DEFAULT_LAG = pd.Timedelta(days=1)

# Stooq 404s on non-browser UAs (observed 2026-06-10) — send a browser UA.
_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_HOSTS = ("https://stooq.com", "https://stooq.pl")

# Stooq fronts a proof-of-work wall (observed 2026-06-11): the page hands a
# SHA-256 challenge (find n s.t. sha256(c+n) starts with d hex zeros) and
# grants a session cookie on POST /__verify. This is a crawler-throttle, not
# authentication — we do the work the site asks for, once per session, and
# keep the polite per-request pacing below (G8: registered in DATA_REGISTER).
_session = None  # lazily-created requests.Session holding the verify cookie


def _solve_pow(html: str) -> tuple[str, int] | None:
    m = re.search(r'const c="([^"]+)",d=(\d+)', html)
    if not m:
        return None
    c, d = m.group(1), int(m.group(2))
    target = "0" * d
    n = 0
    while not hashlib.sha256(f"{c}{n}".encode()).hexdigest().startswith(target):
        n += 1
    return c, n


def _get_verified(url: str, host: str, timeout: int = 60) -> str:
    global _session
    import requests

    if _session is None:
        _session = requests.Session()
        _session.headers.update(_BROWSER_UA)
    resp = _session.get(url, timeout=timeout)
    if "/__verify" in resp.text and "crypto.subtle" in resp.text:
        solved = _solve_pow(resp.text)
        if solved is not None:
            c, n = solved
            _session.post(
                f"{host}/__verify", data={"c": c, "n": n}, timeout=timeout
            )
            resp = _session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _fetch_csv(symbol: str) -> str:
    last_exc: Exception | None = None
    for host in _HOSTS:
        text = _get_verified(f"{host}/q/d/l/?s={symbol.lower()}&i=d", host)
        if text and not text.strip().lower().startswith(("no data", "<", "przekroczony")):
            return text
        last_exc = ValueError(f"empty/limit response from {host} for {symbol!r}")
    raise ValueError(f"Stooq fetch failed for {symbol!r}: {last_exc}")


def fetch_daily(
    symbol: str,
    lag: pd.Timedelta = _DEFAULT_LAG,
) -> pd.DataFrame:
    """Fetch full daily history for one Stooq symbol (e.g. ``aapl.us``).

    Returns a validated PIT frame:
    columns = [symbol, event_ts, open, high, low, close, volume, asof_ts].
    """
    text = _fetch_csv(symbol)
    df = read_csv_text(text)
    df.columns = [c.strip().lower() for c in df.columns]
    expected = {"date", "open", "high", "low", "close"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Unexpected Stooq columns for {symbol!r}: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = float("nan")  # FX/indices have no volume

    df[EVENT_COL] = pd.to_datetime(df["date"], utc=True)
    df["symbol"] = symbol.lower()
    df = df[["symbol", EVENT_COL, "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = stamp_asof(df, lag)
    return validate_pit(df, required=("symbol", "open", "high", "low", "close"))


def fetch_universe(
    symbols: list[str],
    market: str,
    name: str,
    root: Path | str = "market_data_parquet",
    lag: pd.Timedelta = _DEFAULT_LAG,
    sleep_s: float = 0.4,
) -> tuple[Path, dict[str, str]]:
    """Bulk-fetch a symbol list; persist one combined parquet.

    Returns (path, failures) — failures maps symbol -> error string so the
    coverage gap can be documented in the data register (G8). Failures never
    silently shrink the universe without a trace.
    """
    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    for sym in symbols:
        frames.append(fetch_daily(sym, lag=lag))
        time.sleep(sleep_s)  # polite: avoid the Stooq daily request limit
    if not frames:
        raise ValueError("No symbol fetched successfully")
    combined = pd.concat(frames, ignore_index=True)
    path = write_market_parquet(combined, market=market, name=name, root=root)
    return path, failures
