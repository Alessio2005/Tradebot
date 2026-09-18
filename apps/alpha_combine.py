"""apps/alpha_combine.py — IC-weighted signal combination stage (Stage 6).

Loads alpha signal Parquet files from artefacts/alpha_signals/,
fits the ICWeightedCombiner on historical IC estimates,
and writes the combined signal to artefacts/alpha_signals/combined/.

Usage:
    python apps/alpha_combine.py [--lookback 252] [--symbol BTCUSDT]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from tradebot.alpha.combination import ICWeightedCombiner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_SIGNAL_ROOT = Path("artefacts/alpha_signals")
_OUTPUT_ROOT = _SIGNAL_ROOT / "combined"


def _load_signals(symbol: str) -> pd.DataFrame:
    """Load all individual signal Parquet files for a symbol."""
    sym_dir = _SIGNAL_ROOT / symbol
    if not sym_dir.exists():
        logger.warning("No signal directory for %s: %s", symbol, sym_dir)
        return pd.DataFrame()

    frames = {}
    for path in sorted(sym_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
            signal_col = "signal" if "signal" in df.columns else df.columns[0]
            frames[path.stem] = df[signal_col]
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    return pd.DataFrame(frames) if frames else pd.DataFrame()


def _load_returns(symbol: str) -> pd.Series:
    """Load forward returns for IC estimation."""
    candidates = [
        Path("artefacts") / "features" / symbol / "features.parquet",
        Path("market_data_parquet") / f"{symbol}_1h.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_parquet(path)
            if "close" in df.columns:
                return df["close"].pct_change().shift(-1).dropna()
    return pd.Series(dtype=float)


def combine(symbol: str, lookback: int = 252) -> None:
    """Fit ICWeightedCombiner and save combined signal."""
    signals = _load_signals(symbol)
    if signals.empty:
        logger.info("No signals to combine for %s.", symbol)
        return

    returns = _load_returns(symbol)
    if returns.empty:
        logger.warning("No returns for IC estimation — using equal weights.")
        combined = signals.mean(axis=1)
    else:
        # Align index
        common = signals.index.intersection(returns.index)
        if len(common) < 10:
            logger.warning("Insufficient overlap — using equal weights.")
            combined = signals.mean(axis=1)
        else:
            # `lookback` hoort in de constructor, niet in `fit`. Zoals het hier
            # stond gooide ICWeightedCombiner() `TypeError: missing 1 required
            # positional argument: 'signal_names'` -- deze tak heeft nooit
            # gedraaid en viel altijd door naar de except eronder.
            combiner = ICWeightedCombiner(
                signal_names=list(signals.columns), lookback=lookback,
            )
            combiner.fit(signals.loc[common], returns.loc[common])
            combined = combiner.predict(signals)
            ic_tbl = combiner.ic_table()
            logger.info("IC table:\n%s", ic_tbl.to_string())

    out_dir = _OUTPUT_ROOT / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined.parquet"
    combined.to_frame("combined_signal").to_parquet(out_path)
    logger.info("Combined signal written to %s (%d rows).", out_path, len(combined))


def main() -> None:
    parser = argparse.ArgumentParser(description="IC-weighted alpha signal combination")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--lookback", type=int, default=252)
    args = parser.parse_args()
    combine(args.symbol, args.lookback)


if __name__ == "__main__":
    main()
