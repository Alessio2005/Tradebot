"""regime_features.py — Bear/bull regime features for the feature pipeline.

Added as part of the bear-regime feature expansion (2026-05-22).

All features are strictly causal: bar t uses only data <= t (no lookahead).
All output values are bounded / stationary by construction so that they
pass the ADF stationarity gate without FFD pre-processing:

  feat_ema50_ratio      — close/EMA50 - 1, clipped ±0.5
  feat_ema200_ratio     — close/EMA200 - 1, clipped ±0.5
  feat_ema_bull         — 1 when EMA50 > EMA200 (golden cross), 0 otherwise
  feat_vol_regime       — vol_short/vol_long ratio (stress indicator), clipped [0, 5]
  feat_ret_1d           — 24-bar % return, clipped ±1
  feat_ret_7d           — 168-bar % return, clipped ±1
  feat_ret_30d          — 720-bar % return, clipped ±1
  feat_rsi14            — RSI(14) normalised to [0, 1]

Feature-map bucket: 'micro' (short-horizon momentum + vol features that are
computed on the micro-bar series — same timeframe bucket as feat_autocorr_60,
feat_atr, etc.)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bear/bull regime features for improved model performance in bear markets.

    All features are causal: bar t uses only data <= t.
    No lookahead bias.

    Parameters
    ----------
    df:
        DataFrame with at minimum a ``close`` column.
        Modified in-place AND returned (consistent with the rest of the
        FeatureEngineer pipeline).

    Returns
    -------
    df with new ``feat_*`` columns appended.
    """
    if "close" not in df.columns:
        logger.warning(
            "add_regime_features: 'close' column missing — regime features skipped."
        )
        return df

    close = df["close"]

    # ------------------------------------------------------------------
    # 1. EMA trend features (stationary: ratios centred around 0)
    # ------------------------------------------------------------------
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    df["feat_ema50_ratio"]  = (close / ema50  - 1).clip(-0.5, 0.5)
    df["feat_ema200_ratio"] = (close / ema200 - 1).clip(-0.5, 0.5)
    # Binary golden/death-cross indicator: 1 = bull (EMA50 > EMA200)
    df["feat_ema_bull"]     = (ema50 > ema200).astype(float)

    # ------------------------------------------------------------------
    # 2. Volatility regime (stress indicator)
    #    vol_short/vol_long > 1  → realised vol is spiking (bear/panic)
    #    vol_short/vol_long < 1  → vol is contracting (consolidation/bull)
    # ------------------------------------------------------------------
    ret = close.pct_change()
    vol_short = ret.rolling(24,  min_periods=12).std()
    vol_long  = ret.rolling(168, min_periods=84).std()
    df["feat_vol_regime"] = (
        vol_short / vol_long.replace(0, np.nan)
    ).clip(0, 5).fillna(1.0)

    # ------------------------------------------------------------------
    # 3. Return momentum (directional momentum, stationary by construction)
    #    Periods in bars assume ~24 micro-bars/day:
    #      24 bars  ≈ 1 day
    #      168 bars ≈ 7 days
    #      720 bars ≈ 30 days
    # ------------------------------------------------------------------
    df["feat_ret_1d"]  = close.pct_change(24).clip(-1, 1).fillna(0)
    df["feat_ret_7d"]  = close.pct_change(168).clip(-1, 1).fillna(0)
    df["feat_ret_30d"] = close.pct_change(720).clip(-1, 1).fillna(0)

    # ------------------------------------------------------------------
    # 4. RSI(14) normalised to [0, 1]
    #    Values near 0 → deeply oversold (potential bear exhaustion)
    #    Values near 1 → overbought
    #    Causal: rolling(14).mean() is a one-sided operation.
    # ------------------------------------------------------------------
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["feat_rsi14"] = (1 - 1 / (1 + rs)).fillna(0.5)

    return df


# ---------------------------------------------------------------------------
# Convenience list — used by pipeline.py to extend the feature_map
# ---------------------------------------------------------------------------
REGIME_FEATURE_COLS: list[str] = [
    "feat_ema50_ratio",
    "feat_ema200_ratio",
    "feat_ema_bull",
    "feat_vol_regime",
    "feat_ret_1d",
    "feat_ret_7d",
    "feat_ret_30d",
    "feat_rsi14",
]

__all__ = ["add_regime_features", "REGIME_FEATURE_COLS"]
