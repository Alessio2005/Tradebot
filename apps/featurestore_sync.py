"""apps/featurestore_sync.py — incremental Parquet append for live feed (Stage 1b).

Reads the latest bars from market_data_parquet/, computes features for
bars newer than the FeatureStore's last_timestamp, and appends them.
Designed to run once per bar-close (or be called by live_trader.py).

Usage:
    python apps/featurestore_sync.py [--symbol BTCUSDT] [--interval 1h]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from tradebot.featurestore.store import FeatureStore
from tradebot.utils.time import to_utc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_MARKET_DATA_ROOT = Path("market_data_parquet")
_STORE_ROOT = Path("artefacts/feature_store")


def _load_bars(symbol: str, interval: str) -> pd.DataFrame:
    """Load bars from market_data_parquet/<symbol>/<interval>.parquet."""
    candidates = [
        _MARKET_DATA_ROOT / symbol / f"{interval}.parquet",
        _MARKET_DATA_ROOT / f"{symbol}_{interval}.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            return df
    logger.warning("No bars found for %s %s — checked: %s", symbol, interval, candidates)
    return pd.DataFrame()


def sync(symbol: str, interval: str = "1h") -> int:
    """Append new feature rows for ``symbol`` to the FeatureStore.

    Returns the number of new rows appended.
    """
    store = FeatureStore(_STORE_ROOT)
    latest = store.latest_timestamp(symbol)

    bars = _load_bars(symbol, interval)
    if bars.empty:
        logger.info("No bars for %s — nothing to sync.", symbol)
        return 0

    if latest is not None:
        latest_utc = to_utc(latest)
        bars = bars[bars.index > latest_utc]

    if bars.empty:
        logger.info("%s is up to date (latest=%s).", symbol, latest)
        return 0

    # Simple pass-through: store raw OHLCV as features
    # In production, wire this to features/pipeline.py
    store.append(bars, symbol)
    logger.info("FeatureStore sync: %s appended %d rows.", symbol, len(bars))
    return len(bars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync FeatureStore from market data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    args = parser.parse_args()

    n = sync(args.symbol, args.interval)
    print(f"Synced {n} rows for {args.symbol}.")


if __name__ == "__main__":
    main()
