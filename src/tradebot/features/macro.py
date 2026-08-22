"""Macro-derived features for bar-level DataFrames.

IMPORTANT: This module transforms already-aligned macro data into
bar-level features. It does NOT load raw macro data (that is data/macro.py).
Alignment with publication-lag has already been applied upstream.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def macro_regime_indicator(
    df_bars: pd.DataFrame,
    macro_df: pd.DataFrame,
    macro_col: str,
    window: int = 10,
) -> pd.Series:
    """Rolling z-score of a macro series, reindexed to bars.

    Args:
        df_bars   : Bar-level DataFrame (index = UTC DatetimeIndex).
        macro_df  : Macro DataFrame with publication-lag-aligned index.
        macro_col : Column name in macro_df to use.
        window    : Rolling window for z-score computation.

    Returns:
        pd.Series aligned to df_bars.index, NaN during warmup.
    """
    if macro_col not in macro_df.columns:
        logger.warning("macro_regime_indicator: column %r not found.", macro_col)
        return pd.Series(np.nan, index=df_bars.index, name=f"feat_macro_{macro_col}")

    series = macro_df[macro_col].reindex(df_bars.index, method="ffill")
    mu  = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std().replace(0, 1e-9)
    zscore = (series - mu) / std
    return zscore.rename(f"feat_macro_{macro_col}")


def macro_momentum(
    df_bars: pd.DataFrame,
    macro_df: pd.DataFrame,
    macro_col: str,
    lag: int = 5,
) -> pd.Series:
    """Signed change in macro level over lag periods (direction indicator)."""
    if macro_col not in macro_df.columns:
        return pd.Series(np.nan, index=df_bars.index, name=f"feat_macro_mom_{macro_col}")

    series = macro_df[macro_col].reindex(df_bars.index, method="ffill")
    change = series.diff(lag)
    return change.rename(f"feat_macro_mom_{macro_col}")


def vix_regime(
    df_bars: pd.DataFrame,
    macro_df: pd.DataFrame,
    vix_col: str = "VIX",
    low_thresh: float = 15.0,
    high_thresh: float = 30.0,
) -> pd.Series:
    """Ternary VIX regime: -1 (calm), 0 (normal), +1 (fear).

    Maps raw VIX level to a signed integer feature the model can directly
    use as a regime multiplier.
    """
    if vix_col not in macro_df.columns:
        return pd.Series(0, index=df_bars.index, name="feat_vix_regime")

    vix = macro_df[vix_col].reindex(df_bars.index, method="ffill").fillna(20.0)
    regime = pd.Series(0, index=df_bars.index, name="feat_vix_regime")
    regime[vix < low_thresh]  = -1
    regime[vix >= high_thresh] = 1
    return regime


def funding_rate_feature(
    df_bars: pd.DataFrame,
    funding_series: pd.Series,
    window: int = 8,
) -> pd.DataFrame:
    """Funding rate level + cumulative 8h signal.

    Args:
        funding_series : Per-bar funding rate series (already aligned, no lag).
        window         : Number of bars for cumulative sum.

    Returns:
        DataFrame with feat_funding_rate and feat_funding_cum columns.
    """
    fr = funding_series.reindex(df_bars.index, method="ffill").fillna(0.0)
    # P1.3-FIX (CHIEF AUDIT 2026-05-23): Bybit funding is booked at bar T
    # but only becomes observable AFTER the settlement tick.  Without shift(1)
    # the rate paid at T leaks directly as a feature on T — the model sees
    # it before it is settled.  The canonical implementation in
    # features/funding_carry.py:69 already applies shift(1); this function
    # must be consistent.
    fr = fr.shift(1).fillna(0.0)
    return pd.DataFrame(
        {
            "feat_funding_rate": fr,
            "feat_funding_cum":  fr.rolling(window, min_periods=1).sum(),
        },
        index=df_bars.index,
    )
