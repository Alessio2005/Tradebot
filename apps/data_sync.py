# apps/data_sync.py
"""Stage 0 — Sync high-resolution trade-tape data from the Bybit public archive.

Replaces the old ccxt 1h-bar downloader. Uses BybitPublicClient to download
linear-perp trades from public.bybit.com and resample to 5-second OHLCV bars.

Output: market_data_parquet/{symbol}/{year}/{symbol}_YYYY-MM.parquet

Usage:
  python -m apps.data_sync                      # sync entire training_universe
  python -m apps.data_sync sync_symbol=AVAXUSDT # sync single symbol
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="data_sync", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    from tradebot.data.crypto import CryptoIngestionEngine

    _container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(_container, dict):
        raise TypeError(
            f"conf-root moet een mapping zijn, kreeg {type(_container).__name__}"
        )
    cfg_dict = cast("dict[str, Any]", _container)

    # Support single-symbol override: python -m apps.data_sync sync_symbol=AVAXUSDT
    symbol_override: str | None = cfg_dict.get("sync_symbol", None)

    engine = CryptoIngestionEngine(cfg_dict)

    if symbol_override:
        logger.info("Syncing single symbol: %s", symbol_override)
        asyncio.run(engine.sync_raw_data(symbol_override=symbol_override))
    else:
        logger.info("Syncing full universe: %s", engine.symbols)
        asyncio.run(engine.sync_raw_data())


if __name__ == "__main__":
    main()
