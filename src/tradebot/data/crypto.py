# src/tradebot/data/crypto.py
"""Asset-agnostic Bybit public-archive data ingestion engine.

Extracted from ``data_crypto.py``; venue migrated Binance → Bybit 2026-06-14.

``BybitPublicClient`` and ``ParquetStorage`` work for any Bybit linear (USDT)
perpetual symbol (BTCUSDT, ETHUSDT, SOLUSDT, …).  ``CryptoIngestionEngine``
accepts a ``training_universe`` list from config and synchronises all assets in
one run sequentially.

Source archive (no API key, public): ``https://public.bybit.com/trading/{SYM}/``
serves one gzip CSV per UTC day: ``{SYM}{YYYY-MM-DD}.csv.gz`` with columns
``timestamp, symbol, side, size, price, tickDirection, trdMatchID, grossValue,
homeNotional, foreignNotional``.  ``timestamp`` is Unix seconds (fractional),
``side`` is the taker-aggressor side (Buy/Sell), ``size`` is base-asset
quantity.  This is the linear-perp trade tape, so the resulting 5s bars carry
the SAME taker_buy/sell volume distribution as the live futures feed (CVD
parity) — the previous Binance path used SPOT aggTrades, a known covariate
shift.

What is multi-asset worthy here:
  * One ingestion engine for N symbols with sequential sync (rate-limit safe).
  * Per-symbol resume: get_last_timestamp() per symbol is respected.
  * Atomic parquet writes with .tmp -> .replace so a crash mid-sync never
    corrupts the existing monthly partition.
  * Throttling between symbols via ``inter_symbol_sleep``.
"""
from __future__ import annotations

import asyncio
import gzip
import io
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("data.crypto")

# Bybit public trade-tape archive (one gzip CSV per UTC day, per symbol).
BYBIT_ARCHIVE_BASE = "https://public.bybit.com/trading"


# =============================================================================
# BYBIT PUBLIC-ARCHIVE DOWNLOADER (DAILY TRADE-TAPE → 5s OHLCV)
# =============================================================================
class BybitPublicClient:
    """Downloads daily trade-tape CSVs from the Bybit public archive.

    Computes pure OHLCV, Tick Volume and Taker Buy/Sell pressure.
    Fully asset-agnostic: pass ``symbol_override`` to fetch for a symbol
    different from the config default.
    """

    def __init__(self, cfg: dict[str, Any], symbol_override: str | None = None):
        cfg_data = cfg.get("data", {}) or {}
        self.symbol: str = (
            symbol_override
            if symbol_override is not None
            else (cfg_data.get("bybit_symbol") or "BTCUSDT")
        )

        self.temp_dir = (
            Path(cfg.get("machine", {}).get("data_dir", "market_data_parquet")) / "temp"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.cpu_executor = ThreadPoolExecutor(max_workers=4)

    async def fetch_period_processed(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        # ThreadedResolver avoids aiodns which requires SelectorEventLoop (breaks on Windows ProactorEventLoop)
        connector = aiohttp.TCPConnector(limit=5, ssl=False, resolver=aiohttp.ThreadedResolver())
        timeout = aiohttp.ClientTimeout(total=600)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Bybit archive is daily-only (no monthly bundles).
            df = await self._process_days_in_range(session, start_dt, end_dt)

        if df.empty:
            return df

        # CHIEF AUDIT 2026-05-23 (FIX 2 / partial-bar leak): when end_dt is
        # close to wall-clock "now", the last 5-second bar may not yet be
        # complete.  For historical backfills (end_dt clearly in the past)
        # keep inclusive ``<=``; only switch to exclusive ``<`` when tailing
        # the live tip (within one resample interval of now).
        now_utc = datetime.now(timezone.utc)
        resample_interval = timedelta(seconds=5)  # matches RESAMPLE_TF in _format_and_resample
        if end_dt > (now_utc - resample_interval):
            df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]
        else:
            df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
        return df

    async def _process_days_in_range(
        self, session: aiohttp.ClientSession, start_dt: datetime, end_dt: datetime
    ) -> pd.DataFrame:
        current_day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        tasks = []

        while current_day <= end_dt:
            tasks.append(self._process_single_day(session, current_day))
            current_day += timedelta(days=1)

        if not tasks:
            return pd.DataFrame()
        results = await asyncio.gather(*tasks)

        valid_dfs = [d for d in results if not d.empty]
        if valid_dfs:
            combined = pd.concat(valid_dfs)
            combined.sort_values("timestamp", inplace=True)
            return combined
        return pd.DataFrame()

    async def _process_single_day(
        self, session: aiohttp.ClientSession, day_dt: datetime
    ) -> pd.DataFrame:
        date_str = day_dt.strftime("%Y-%m-%d")
        file_name = f"{self.symbol}{date_str}.csv.gz"
        url_daily = f"{BYBIT_ARCHIVE_BASE}/{self.symbol}/{file_name}"
        return await self._download_and_parse_to_disk(session, url_daily, file_name)

    async def _download_and_parse_to_disk(
        self, session: aiohttp.ClientSession, url: str, file_name: str
    ) -> pd.DataFrame:
        temp_file = self.temp_dir / file_name

        for _attempt in range(3):
            try:
                async with session.get(url) as resp:
                    if resp.status in [404, 403]:
                        return pd.DataFrame()
                    elif resp.status == 200:
                        with open(temp_file, "wb") as f:
                            while True:
                                chunk = await resp.content.read(4 * 1024 * 1024)
                                if not chunk:
                                    break
                                f.write(chunk)

                        loop = asyncio.get_running_loop()
                        df = await loop.run_in_executor(
                            self.cpu_executor, self._process_gz_chunked, temp_file
                        )

                        if temp_file.exists():
                            temp_file.unlink()
                        return df
                    else:
                        await asyncio.sleep(2)
            except Exception:
                await asyncio.sleep(2)

        if temp_file.exists():
            temp_file.unlink()
        return pd.DataFrame()

    def _process_gz_chunked(self, file_path: Path) -> pd.DataFrame:
        """Parse a Bybit daily trade-tape gzip CSV into 5s OHLCV bars.

        Bybit CSVs always ship a header row:
            timestamp, symbol, side, size, price, tickDirection, ...
        ``side`` is the taker-aggressor side (Buy/Sell); ``size`` is base-asset
        quantity; ``timestamp`` is Unix seconds (fractional).
        """
        resampled_chunks: list[pd.DataFrame] = []
        try:
            # Decompress to memory once, then chunk the CSV.  Daily perp tapes
            # are typically tens of MB decompressed — fine for streaming chunks.
            with gzip.open(file_path, "rb") as gz:
                raw = gz.read()

            for chunk in pd.read_csv(
                io.BytesIO(raw),
                usecols=lambda c: str(c).lower() in ("timestamp", "side", "size", "price"),
                chunksize=1_000_000,
            ):
                cols_lower = {str(c).lower(): c for c in chunk.columns}
                ts_col   = cols_lower.get("timestamp")
                side_col = cols_lower.get("side")
                size_col = cols_lower.get("size")
                price_col = cols_lower.get("price")
                if None in (ts_col, side_col, size_col, price_col):
                    logger.error("Unexpected Bybit CSV columns: %s", list(chunk.columns))
                    return pd.DataFrame()

                sub = chunk[[ts_col, side_col, size_col, price_col]].copy()
                sub.columns = ["transact_time", "side", "quantity", "price"]
                resampled_chunks.append(self._format_and_resample(sub))

            if not resampled_chunks:
                return pd.DataFrame()

            combined = pd.concat(resampled_chunks)
            final_df = (
                combined.groupby(combined.index)
                .agg(
                    {
                        "open":              "first",
                        "high":              "max",
                        "low":               "min",
                        "close":             "last",
                        "tick_volume":       "sum",
                        "real_volume":       "sum",
                        "taker_buy_volume":  "sum",
                        "taker_sell_volume": "sum",
                    }
                )
                .dropna()
            )

            final_df.reset_index(inplace=True)
            final_df.rename(columns={"time": "timestamp", "index": "timestamp"}, inplace=True)
            return final_df

        except Exception as e:
            logger.error("Error parsing %s: %s", file_path, e)
            return pd.DataFrame()

    def _format_and_resample(self, chunk: pd.DataFrame) -> pd.DataFrame:
        # Bybit ``side`` is the taker-aggressor side: Buy → taker buy.
        side_str = chunk["side"].astype(str).str.lower()
        is_taker_buy = side_str == "buy"

        # timestamp is Unix seconds (fractional). Guard against ms-encoded files.
        t0 = float(chunk["transact_time"].iloc[0])
        unit = "ms" if t0 > 1e12 else "s"
        chunk["time"] = pd.to_datetime(chunk["transact_time"], unit=unit, utc=True)

        chunk = chunk.sort_values("transact_time", kind="mergesort")
        is_taker_buy = is_taker_buy.loc[chunk.index]

        chunk["taker_buy_volume"] = np.where(
            is_taker_buy, chunk["quantity"], 0
        ).astype("float32")
        chunk["taker_sell_volume"] = np.where(
            ~is_taker_buy, chunk["quantity"], 0
        ).astype("float32")

        chunk["price"] = chunk["price"].astype("float64")
        chunk.set_index("time", inplace=True)
        chunk = chunk.sort_index(kind="mergesort")

        RESAMPLE_TF = "5s"
        grouper = chunk.groupby(pd.Grouper(freq=RESAMPLE_TF, label="right", closed="right"))

        ohlc       = grouper["price"].ohlc()
        tick_vol   = grouper["price"].count().rename("tick_volume")
        real_vol   = grouper["quantity"].sum().rename("real_volume")
        t_buy_vol  = grouper["taker_buy_volume"].sum()
        t_sell_vol = grouper["taker_sell_volume"].sum()

        return pd.concat([ohlc, tick_vol, real_vol, t_buy_vol, t_sell_vol], axis=1)


# =============================================================================
# PARQUET STORAGE CLIENT (ATOMIC WRITES)
# =============================================================================
class ParquetStorage:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str, dt: datetime) -> Path:
        year_str  = str(dt.year)
        month_str = f"{dt.year}-{dt.month:02d}"
        symbol_dir = self.base_dir / symbol / year_str
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{symbol}_{month_str}.parquet"

    async def get_last_timestamp(self, symbol: str) -> datetime | None:
        symbol_dir = self.base_dir / symbol
        if not symbol_dir.exists():
            return None

        all_files = sorted(list(symbol_dir.glob("**/*.parquet")))
        if not all_files:
            return None

        last_file = all_files[-1]
        try:
            df_tail = pd.read_parquet(last_file, columns=["timestamp"]).iloc[-1:]
            last_ts = pd.to_datetime(df_tail["timestamp"].iloc[0])
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            return last_ts
        except Exception:
            return None

    async def save_chunk(self, df: pd.DataFrame, symbol: str) -> None:
        if df.empty:
            return

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        elif str(df["timestamp"].dt.tz) != "UTC":
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        df["month_period"] = df["timestamp"].dt.strftime("%Y-%m")

        for _period, group in df.groupby("month_period"):
            ref_date  = group["timestamp"].iloc[0]
            file_path = self._get_file_path(symbol, ref_date)
            temp_path = file_path.with_suffix(".parquet.tmp")

            data_to_write = group.drop(columns=["month_period"]).copy()

            f32_cols = ["tick_volume", "real_volume", "taker_buy_volume", "taker_sell_volume"]
            for c in f32_cols:
                if c in data_to_write.columns:
                    data_to_write[c] = data_to_write[c].astype("float32")

            if file_path.exists():
                try:
                    existing_df  = pd.read_parquet(file_path)
                    combined_df  = pd.concat([existing_df, data_to_write])
                    combined_df  = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
                    combined_df.sort_values("timestamp", inplace=True)
                    combined_df.reset_index(drop=True, inplace=True)
                    self._write_parquet(combined_df, temp_path)
                except Exception as e:
                    logger.warning(
                        "Failed merging existing parquet %s: %s. Overwriting.", file_path, e
                    )
                    self._write_parquet(data_to_write, temp_path)
            else:
                self._write_parquet(data_to_write, temp_path)

            # Atomic replace to prevent corruption on crash.
            temp_path.replace(file_path)

    def _write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)

    async def load_data(self, symbol: str) -> pd.DataFrame:
        symbol_dir = self.base_dir / symbol
        if not symbol_dir.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(symbol_dir, engine="pyarrow")
            df.sort_values("timestamp", inplace=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception:
            return pd.DataFrame()

    def prune_old_data(self, symbol: str, retention_years: int = 5) -> None:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=365 * retention_years)
        symbol_dir  = self.base_dir / symbol
        if not symbol_dir.exists():
            return

        deleted_count = 0
        for year_dir in symbol_dir.iterdir():
            if not year_dir.is_dir():
                continue
            try:
                if int(year_dir.name) < (cutoff_date.year - 1):
                    shutil.rmtree(year_dir)
                    deleted_count += 1
                    continue
            except ValueError:
                continue

            for pq_file in year_dir.glob("*.parquet"):
                try:
                    date_part = pq_file.stem[-7:]
                    file_date = datetime.strptime(date_part, "%Y-%m").replace(
                        tzinfo=timezone.utc
                    )
                    if file_date < cutoff_date.replace(day=1):
                        pq_file.unlink()
                        deleted_count += 1
                except Exception:
                    continue

        if deleted_count > 0:
            logger.info("Cleanup complete. Removed %d old files/directories.", deleted_count)


# =============================================================================
# MULTI-ASSET INGESTION ENGINE
# =============================================================================
class CryptoIngestionEngine:
    """Multi-asset ingestion engine for Bybit public-archive linear-perp data.

    Accepts a list of symbols (BTCUSDT/ETHUSDT/SOLUSDT/...) and synchronises
    them sequentially — each with its own retention and per-symbol resume from
    last-timestamp.

    Backwards compat: when ``training_universe`` is absent it falls back to
    ``data.bybit_symbol``.
    """

    DEFAULT_UNIVERSE: list[str] = [  # noqa: RUF012
        "BTCUSDT", "ETHUSDT", "SOLUSDT",
        "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    ]

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        data_dir     = cfg.get("machine", {}).get("data_dir", "market_data_parquet")
        self.storage = ParquetStorage(data_dir)

        cfg_data  = cfg.get("data", {}) or {}
        cfg_train = cfg.get("training", {}) or {}

        single   = cfg_data.get("bybit_symbol") or "BTCUSDT"
        universe = cfg_train.get("training_universe") or [single]

        crypto_suffixes = ("USDT", "USDC")
        self.symbols: list[str] = [
            s for s in universe
            if isinstance(s, str) and s.upper().endswith(crypto_suffixes)
        ]
        if not self.symbols:
            self.symbols = [single]
        if not self.symbols:
            self.symbols = ["BTCUSDT"]

        self.target_symbol        = single
        self.fetch_symbol         = single
        self.retention_years      = int(cfg_data.get("history_fetch_years", 5))
        self.inter_symbol_sleep: float = float(cfg_data.get("inter_symbol_sleep", 0.5))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def sync_raw_data(self, symbol_override: str | None = None) -> None:
        """Sync one symbol. Preserves legacy signature from data_btc.py.

        When ``symbol_override`` is None and ``training_universe`` contains
        multiple symbols: iterate over all. Otherwise sync one symbol.
        """
        if symbol_override is not None:
            await self._sync_one(symbol_override)
            return

        targets = self.symbols if len(self.symbols) > 1 else [self.target_symbol]
        for sym in targets:
            try:
                await self._sync_one(sym)
            except Exception as e:
                logger.error("[%s] Sync failed: %s", sym, e)
            await asyncio.sleep(self.inter_symbol_sleep)

    async def _sync_one(self, symbol: str) -> None:
        logger.info("=== [%s] BYBIT Sync (MONTHLY PARQUET ENGINE) ===", symbol)

        self.storage.prune_old_data(symbol, retention_years=self.retention_years)
        last_ts  = await self.storage.get_last_timestamp(symbol)
        now_utc  = datetime.now(timezone.utc)

        start_date = now_utc - timedelta(days=365 * self.retention_years)
        if last_ts:
            start_date = last_ts + timedelta(seconds=1)

        if start_date >= now_utc:
            logger.info("[%s] Data is already up to date.", symbol)
            return

        logger.info("[%s] Fetching range: %s -> %s", symbol, start_date.date(), now_utc.date())
        await self._sync_bybit_chunked(symbol, symbol, start_date, now_utc)

    async def _sync_bybit_chunked(
        self,
        fetch_sym: str,
        save_as_sym: str,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        client = BybitPublicClient(self.cfg, symbol_override=fetch_sym)

        current_month = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_month     = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        while current_month <= end_month:
            if current_month.month == 12:
                next_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                next_month = current_month.replace(month=current_month.month + 1)

            chunk_start = max(start_date, current_month)
            chunk_end   = min(end_date, next_month - timedelta(seconds=1))

            logger.info("[%s] Fetching Month: %s", fetch_sym, current_month.strftime("%Y-%m"))

            t0       = time.time()
            df_chunk = await client.fetch_period_processed(chunk_start, chunk_end)
            fetch_time = time.time() - t0

            if not df_chunk.empty:
                await self.storage.save_chunk(df_chunk, save_as_sym)
                logger.info(
                    " -> [%s] Saved %d S5 bars to Parquet. Speed: %.2fs",
                    fetch_sym, len(df_chunk), fetch_time,
                )
            else:
                logger.info(
                    " -> [%s] No data found for %s.",
                    fetch_sym, current_month.strftime("%Y-%m"),
                )

            current_month = next_month
            await asyncio.sleep(0.01)


__all__ = [
    "BybitPublicClient",
    "CryptoIngestionEngine",
    "ParquetStorage",
]
