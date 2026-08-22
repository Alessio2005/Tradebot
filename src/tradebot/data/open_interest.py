"""data/open_interest.py — Bybit open-interest fetcher + per-bar loader.

Open interest (OI) is the total notional of outstanding perpetual contracts.
Unlike funding (a flow paid at the 8h boundary) OI is a STOCK sampled at a
fixed interval; the per-bar loader therefore forward-fills the last observed
level onto the bar grid (causal as-of join), it does NOT place-and-zero.

Economic thesis (pre-committed, single hypothesis — see CHIEF data review
2026-06-14): OI combined with price and funding identifies positioning
regimes:
  • price↑ + OI↑  → fresh longs / trend confirmation
  • price↓ + OI↑  → fresh shorts (build-up of downside positioning)
  • price↑ + OI↓  → short covering (less durable)
  • price↓ + OI↓  → long liquidation / deleveraging
Crowded carry + rising OI precedes deleveraging cascades — a crypto-native
positioning signal absent from price/volume alone.

Source: Bybit V5 ``/v5/market/open-interest`` (public, no key).  Results are
DESCENDING in time, capped at 200/call, paged via ``nextPageCursor``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import requests

from ..utils.failfast import DataContractError

logger = logging.getLogger("data.open_interest")

_BASE_URL = "https://api.bybit.com"
_VALID_INTERVALS = {"5min", "15min", "30min", "1h", "4h", "1d"}


class OpenInterestFetcher:
    """Single-symbol Bybit open-interest fetcher.

    Mirrors :class:`tradebot.data.crypto_macro.CryptoMacroFetcher` so the
    research and live pipelines stay structurally identical.
    """

    def __init__(
        self,
        data_dir: str = "market_data_parquet/macro",
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        fetch_years: int = 5,
    ):
        if interval not in _VALID_INTERVALS:
            raise ValueError(
                f"interval {interval!r} not in {sorted(_VALID_INTERVALS)}"
            )
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.interval = interval
        self.category = "linear"
        self.start_ts = int(
            (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=fetch_years))
            .timestamp() * 1000
        )

    def _paginate_backward(self, limit: int = 200) -> list:
        """Page Bybit OI history backward (via nextPageCursor) to ``start_ts``."""
        endpoint = "/v5/market/open-interest"
        all_data: list = []
        cursor = ""

        for _guard in range(5000):  # hard cap
            params = {
                "category": self.category,
                "symbol": self.symbol,
                "intervalTime": self.interval,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                r = requests.get(f"{_BASE_URL}{endpoint}", params=params, timeout=10)
                r.raise_for_status()
                payload = r.json()

                if payload.get("retCode", -1) != 0:
                    logger.error(
                        "[%s] Bybit open-interest retCode=%s msg=%s",
                        self.symbol, payload.get("retCode"), payload.get("retMsg"),
                    )
                    break

                result = payload.get("result") or {}
                rows = result.get("list") or []
                if not rows:
                    break

                all_data.extend(rows)

                oldest_ts = int(rows[-1]["timestamp"])  # DESC → last is oldest
                cursor = result.get("nextPageCursor") or ""
                if not cursor or oldest_ts <= self.start_ts:
                    break
                time.sleep(0.1)

            except Exception as e:
                # Phase 0: zie crypto_macro.py - geen stille gedeeltelijke reeks.
                raise DataContractError(
                    f"[{self.symbol}] Bybit open-interest-pagination afgebroken "
                    f"op {endpoint} na {len(all_data)} rijen: {e}."
                ) from e

        return all_data

    def fetch_open_interest(self) -> pd.DataFrame:
        """Return a UTC-indexed DataFrame with a single ``openInterest`` column."""
        data = self._paginate_backward(limit=200)
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype("int64"), unit="ms", utc=True
        )
        df["openInterest"] = df["openInterest"].astype(float)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df[["openInterest"]]

    def run_pipeline(self) -> pd.DataFrame:
        """Fetch and persist ``oi_crypto_{symbol}.parquet``."""
        logger.info("[%s] Downloading Bybit open-interest (interval=%s)...",
                    self.symbol, self.interval)
        df_oi = self.fetch_open_interest()
        if df_oi.empty:
            logger.error("[%s] Open-interest stream empty. Abort.", self.symbol)
            return pd.DataFrame()

        out_path = self.data_dir / f"oi_crypto_{self.symbol}.parquet"
        out_df = df_oi.reset_index()
        out_df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        logger.info("[%s] %d OI records -> %s", self.symbol, len(out_df), out_path.name)
        return df_oi


def load_per_bar_open_interest(
    symbol: str,
    bar_index: pd.DatetimeIndex,
    oi_dir: Path,
) -> np.ndarray:
    """Load per-bar open-interest LEVEL on the ``bar_index`` grid (causal as-of).

    For each bar ``t`` returns the most recent OI observation with
    ``obs_ts <= t`` (backward as-of / forward-fill).  OI is a stock, so this
    is the correct alignment (no place-and-zero).  Missing file / parse error
    → array of zeros + a warning (never raises; backward-compatible with the
    funding loader's contract).
    """
    n = len(bar_index)
    out = np.zeros(n, dtype=np.float64)

    fpath = Path(oi_dir) / f"oi_crypto_{symbol}.parquet"
    if not fpath.exists():
        logger.warning(
            "[%s] Open-interest parquet not found (%s) — features set to 0.",
            symbol, fpath.name,
        )
        return out

    df = pd.read_parquet(fpath)
    if "openInterest" not in df.columns:
        logger.warning(
            "[%s] No 'openInterest' column in %s — set to 0.",
            symbol, fpath.name,
        )
        return out

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    oi = df["openInterest"].dropna()
    oi_idx = cast(pd.DatetimeIndex, oi.index)
    if oi_idx.tz is None:
        oi.index = oi_idx.tz_localize("UTC")

    bar_idx_dt = cast(pd.DatetimeIndex, pd.DatetimeIndex(bar_index))
    if bar_idx_dt.tz is None:
        bar_idx_dt = cast(pd.DatetimeIndex, bar_idx_dt.tz_localize("UTC"))

    # Causal as-of: each bar gets the last OI level observed at/<= its ts.
    aligned = oi.reindex(
        oi.index.union(bar_idx_dt)
    ).sort_index().ffill().reindex(bar_idx_dt)
    out = aligned.to_numpy(dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0)
    return out


__all__ = ["OpenInterestFetcher", "load_per_bar_open_interest"]
