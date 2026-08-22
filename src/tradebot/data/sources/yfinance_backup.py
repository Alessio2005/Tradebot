# src/tradebot/data/sources/yfinance_backup.py
"""yfinance EOD fallback — SUPPLEMENT ONLY (Mandate §5.2).

Primary equity OHLCV source is Stooq; this module exists because Stooq
404'd wholesale on 2026-06-10 (two hosts, two networks). ToS caveat is
registered in docs/DATA_REGISTER.md: Yahoo data is for personal use, no
redistribution — acceptable for in-house research, never for distribution.

Same PIT convention as Stooq: event_ts = session date 00:00 UTC,
asof_ts = event + 1 calendar day (conservative EOD availability).
Requires: pip install yfinance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import EVENT_COL, SourceMeta, stamp_asof, validate_pit, write_market_parquet

__all__ = ["META", "fetch_universe_yf", "to_yf_symbol"]

META = SourceMeta(
    name="yfinance_backup",
    url="https://finance.yahoo.com (via yfinance)",
    license="gratis; Yahoo ToS: persoonlijk gebruik, geen redistributie (geregistreerde caveat)",
    publication_lag="EOD close -> volgende kalenderdag 00:00 UTC (conservatief)",
    coverage="US equities daily, splits/div-adjusted optional; delisted GROTENDEELS AFWEZIG",
    survivorship="delisted names largely missing -> gap meten zoals bij Stooq (G8)",
    quality_notes="fallback wanneer Stooq blokkeert; auto_adjust=True (split-safe, dividends ingebakken = total-return-achtig)",
)

_CHUNK = 50


def to_yf_symbol(ticker: str) -> str:
    """'BRK.B' -> 'BRK-B' (Yahoo class-share convention)."""
    return ticker.strip().upper().replace(".", "-")


def fetch_universe_yf(
    tickers: list[str],
    market: str,
    name: str,
    root: Path | str = "market_data_parquet",
    lag: pd.Timedelta = pd.Timedelta(days=1),
) -> tuple[Path, dict[str, str]]:
    """Bulk EOD fetch via yfinance; same output contract as stooq.fetch_universe.

    Returns (parquet_path, failures). Symbols in the output keep the
    ORIGINAL ticker spelling (e.g. 'BRK.B'), matching membership events.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance fallback requires: pip install yfinance") from exc

    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    yf_map = {to_yf_symbol(t): t for t in tickers}

    for i in range(0, len(tickers), _CHUNK):
        chunk = [to_yf_symbol(t) for t in tickers[i : i + _CHUNK]]
        try:
            # auto_adjust=True is MANDATORY: raw close turns every split
            # into a fake +/-90% "return" (observed: +16078% year-2012
            # artifact in the first W22 run). Adjusted OHLC = split-safe,
            # dividends baked in (total-return-ish; registered in G8).
            raw = yf.download(
                tickers=chunk, period="max", interval="1d",
                group_by="ticker", auto_adjust=True, actions=False,
                progress=False, threads=True,
            )
        except Exception as exc:  # noqa: BLE001
            for s in chunk:
                failures[yf_map[s]] = f"chunk download failed: {exc}"
            continue
        for s in chunk:
            try:
                sub = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = sub.dropna(subset=["Close"])
                if sub.empty:
                    raise ValueError("empty history")
                df = pd.DataFrame(
                    {
                        "symbol": yf_map[s],
                        EVENT_COL: pd.to_datetime(sub.index, utc=True).normalize(),
                        "open": sub["Open"].to_numpy(float),
                        "high": sub["High"].to_numpy(float),
                        "low": sub["Low"].to_numpy(float),
                        "close": sub["Close"].to_numpy(float),
                        "volume": sub["Volume"].to_numpy(float),
                    }
                )
                df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
                frames.append(stamp_asof(df, lag))
            except Exception as exc:  # noqa: BLE001 — recorded per symbol
                failures[yf_map[s]] = str(exc)

    if not frames:
        raise ValueError("yfinance: no symbol fetched successfully")
    combined = pd.concat(frames, ignore_index=True)
    combined = validate_pit(combined, required=("symbol", "close"))
    path = write_market_parquet(combined, market=market, name=name, root=root)
    return path, failures
