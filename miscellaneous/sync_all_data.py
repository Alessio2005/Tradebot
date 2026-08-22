"""sync_all_data.py — Download aggTrade data for ETH, SOL and BTC backfill.

Runs CryptoIngestionEngine directly (no Hydra). Syncs sequentially to respect
Binance Vision rate limits. BTC uses retention_years=6 to cover 2020 gap.

Usage: python sync_all_data.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sync_all_data.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("sync_all_data")

from tradebot.data.crypto import CryptoIngestionEngine


BASE_CFG = {
    "machine": {"data_dir": "market_data_parquet"},
    "data": {
        "binance_symbol": "BTCUSDT",
        "history_fetch_years": 5,
        "inter_symbol_sleep": 1.0,
    },
    "training": {"training_universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
}

BTC_BACKFILL_CFG = {
    "machine": {"data_dir": "market_data_parquet"},
    "data": {
        "binance_symbol": "BTCUSDT",
        "history_fetch_years": 6,   # 6 years back = 2020-05 from today 2026-05
        "inter_symbol_sleep": 1.0,
    },
    "training": {"training_universe": ["BTCUSDT"]},
}


async def main() -> None:
    logger.info("=== SYNC START ===")

    # 1. ETH aggTrade (2021-05 → now, ~60 months)
    logger.info("--- Syncing ETHUSDT ---")
    eth_engine = CryptoIngestionEngine(BASE_CFG)
    await eth_engine.sync_raw_data(symbol_override="ETHUSDT")
    logger.info("--- ETHUSDT done ---")

    await asyncio.sleep(2)

    # 2. SOL aggTrade (2021-08 listed → now, ~57 months)
    logger.info("--- Syncing SOLUSDT ---")
    sol_engine = CryptoIngestionEngine(BASE_CFG)
    await sol_engine.sync_raw_data(symbol_override="SOLUSDT")
    logger.info("--- SOLUSDT done ---")

    await asyncio.sleep(2)

    # 3. BTC gap backfill (2020-05 → 2021-04, ~12 months missing)
    logger.info("--- BTC gap backfill (retention_years=6) ---")
    btc_engine = CryptoIngestionEngine(BTC_BACKFILL_CFG)
    await btc_engine.sync_raw_data(symbol_override="BTCUSDT")
    logger.info("--- BTCUSDT backfill done ---")

    logger.info("=== ALL SYNCS COMPLETE ===")
    logger.info("Check market_data_parquet/ for downloaded files.")


if __name__ == "__main__":
    asyncio.run(main())
