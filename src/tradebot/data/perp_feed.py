"""perp_feed.py — Bybit linear (USDT) perpetual market-data feed (public REST).

Used by the market-neutral paper trader for warm-start and daily updates.
No API key required (public endpoints). All timestamps UTC.

NOTE (venue migration 2026-06-14): migrated Binance USDⓈ-M (``fapi.binance.com``)
→ **Bybit V5** (``api.bybit.com``, ``category=linear``).  Endpoint mapping:
  exchangeInfo        → /v5/market/instruments-info
  klines (1d)         → /v5/market/kline  (interval="D", DESC, limit<=1000)
  fundingRate         → /v5/market/funding/history
Bybit kline/funding responses are DESCENDING in time and capped at 1000/200
rows per call respectively, so the walkers below page backwards via the
``end`` / ``endTime`` cursor and sort ascending at the end.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BYBIT = "https://api.bybit.com"
CATEGORY = "linear"


def _get(url: str, timeout: int = 30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def _get_v5(endpoint: str, params: dict, timeout: int = 30) -> dict:
    """GET a Bybit V5 endpoint and return ``result`` (raises on retCode != 0)."""
    qs = urllib.parse.urlencode(params)
    payload = _get(f"{BYBIT}{endpoint}?{qs}", timeout=timeout)
    if payload.get("retCode", -1) != 0:
        raise RuntimeError(
            f"Bybit {endpoint} retCode={payload.get('retCode')} "
            f"msg={payload.get('retMsg')}"
        )
    return payload.get("result") or {}


def fetch_universe(onboard_before: str = "2021-07-01") -> list[str]:
    """Established USDT linear perpetuals (launched before cutoff) — full
    multi-regime history, excludes brand-new listings.

    Bybit ``contractType`` for USDT perps is ``"LinearPerpetual"``; the launch
    epoch is ``launchTime`` (ms).  Pages via ``nextPageCursor``.
    """
    cutoff_ms = int(pd.Timestamp(onboard_before, tz="UTC").timestamp() * 1000)
    out: list[str] = []
    cursor = ""
    for _ in range(20):  # safety cap on pagination
        params = {"category": CATEGORY, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        result = _get_v5("/v5/market/instruments-info", params)
        for s in result.get("list", []):
            try:
                launch_ms = int(s.get("launchTime", 0) or 0)
            except (TypeError, ValueError):
                launch_ms = 0
            if (
                s.get("contractType") == "LinearPerpetual"
                and s.get("quoteCoin") == "USDT"
                and s.get("status") == "Trading"
                and 0 < launch_ms < cutoff_ms
            ):
                out.append(s["symbol"])
        cursor = result.get("nextPageCursor") or ""
        if not cursor:
            break
    return sorted(set(out))


def fetch_daily_closes(symbol: str, start_ms: int) -> pd.Series | None:
    """Daily close series for ``symbol`` from ``start_ms`` to now.

    Bybit kline returns DESC, max 1000 rows; each row is
    ``[start, open, high, low, close, volume, turnover]`` (strings).  We walk
    backwards via the ``end`` cursor until we cover ``start_ms``.
    """
    rows: list[tuple[int, float]] = []
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

    for _ in range(12):
        try:
            result = _get_v5(
                "/v5/market/kline",
                {
                    "category": CATEGORY,
                    "symbol": symbol,
                    "interval": "D",
                    "start": start_ms,
                    "end": end_ms,
                    "limit": 1000,
                },
            )
        except Exception:
            break
        klist = result.get("list", [])
        if not klist:
            break
        rows += [(int(k[0]), float(k[4])) for k in klist]
        oldest_ms = int(klist[-1][0])  # DESC → last is oldest
        if oldest_ms <= start_ms or len(klist) < 1000:
            break
        end_ms = oldest_ms - 1

    if not rows:
        return None
    s = pd.Series({pd.to_datetime(t, unit="ms", utc=True): c for t, c in rows}).sort_index()
    return s[~s.index.duplicated()]


def build_or_update_panel(cache: Path, onboard_before: str = "2021-07-01",
                          start: str = "2021-06-01", sleep: float = 0.05) -> pd.DataFrame:
    """Load cached daily-close panel and extend it to the latest bar; build fresh
    if no cache. Returns wide DataFrame [date x symbol]."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    existing = pd.read_parquet(cache) if cache.exists() else None
    syms = fetch_universe(onboard_before)
    cols = {}
    last_ts = existing.index.max() if existing is not None and len(existing) else None
    fetch_from = int(last_ts.timestamp() * 1000) if last_ts is not None else start_ms
    for _i, sym in enumerate(syms):
        try:
            s = fetch_daily_closes(sym, fetch_from)
            if s is not None and len(s):
                cols[sym] = s
        except Exception:
            pass
        time.sleep(sleep)
    new = pd.DataFrame(cols)
    if existing is not None:
        panel = existing.combine_first(new)
        # overwrite overlapping tail with fresh values
        for c in new.columns:
            panel.loc[new.index, c] = new[c]
        panel = panel.reindex(columns=sorted(set(existing.columns) | set(new.columns)))
    else:
        panel = new
    panel = panel.sort_index()
    panel = panel.resample("1D").last()
    cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache)
    return panel


def fetch_recent_funding(symbols: list[str], lookback: int = 10) -> pd.DataFrame:
    """Recent realised funding per symbol -> daily funding paid by a LONG.
    Returns wide DataFrame [date x symbol]. (Live-only; deep history via
    crypto_macro.CryptoMacroFetcher.)"""
    out = {}
    for sym in symbols:
        try:
            result = _get_v5(
                "/v5/market/funding/history",
                {"category": CATEGORY, "symbol": sym, "limit": min(lookback, 200)},
            )
            rows = result.get("list", [])
            s = pd.Series({
                pd.to_datetime(int(x["fundingRateTimestamp"]), unit="ms", utc=True):
                    float(x["fundingRate"])
                for x in rows
            })
            out[sym] = s.resample("1D").sum()
        except Exception:
            pass
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()
