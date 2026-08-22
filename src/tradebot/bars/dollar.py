"""Dollar Bars — close a bar every N USD of notional (AFML §2.3).

Implemented in Wave 13 / AUDIT A-2. Mirrors the structure of
``imbalance.py``: same Numba-kernel pattern, same bid/ask fallback semantics,
same dynamic-threshold option for intraday-seasonality.

Why dollar bars matter for crypto multi-asset:
  Volume Imbalance Bars normalise on signed-volume imbalance. In a steep
  bull-run the *notional* per bar drifts upward proportionally to price.
  Dollar bars hold notional approximately constant, producing a more
  stationary bar-size distribution and inter-asset comparability.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from ._kernels import numba_dollar_bars

logger = logging.getLogger(__name__)


def _calculate_dynamic_dollar_thresholds(
    df: pd.DataFrame,
    target_per_day: int,
    window_days: int = 20,
) -> np.ndarray:
    """Dynamic notional thresholds with per-hour-of-week seasonality.

    Same lookahead-safe shift(1) pattern as imbalance bars: the threshold at
    bar t uses only past observations within the same hour-of-week bucket.
    """
    if len(df) == 0:
        return np.array([], dtype=np.float64)

    notional = df["close"].astype(np.float64) * df.get(
        "tick_volume", df.get("volume", pd.Series(np.zeros(len(df))))
    ).astype(np.float64)

    hourly_not = notional.groupby(pd.Grouper(freq="h")).sum()
    hourly_not = hourly_not.where(hourly_not > 1.0, np.nan)

    hourly_idx = cast(pd.DatetimeIndex, hourly_not.index)
    bucket = (
        hourly_idx.dayofweek.astype(np.int64) * 24
        + hourly_idx.hour.astype(np.int64)
    )
    bucket_series = pd.Series(bucket, index=hourly_idx, name="bucket")

    df_hourly = hourly_not.to_frame("not").assign(bucket=bucket_series)
    shifted = df_hourly.groupby("bucket")["not"].shift(1)

    def _ewm(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if s_clean.empty:
            return s
        return s_clean.ewm(halflife=24 * 7 * 4, adjust=False).mean().reindex(s.index)

    seasonal_expect = (
        shifted.groupby(df_hourly["bucket"], group_keys=False).apply(_ewm)
    )

    daily_not = notional.groupby(pd.Grouper(freq="D")).sum()
    daily_not = daily_not.where(daily_not > 1.0, np.nan)
    daily_fallback_h = (
        daily_not.rolling(window=window_days, min_periods=1).median().shift(1) / 24.0
    )
    daily_fallback_h = daily_fallback_h.reindex(hourly_idx, method="ffill")

    expected_hourly = seasonal_expect.fillna(daily_fallback_h).fillna(1_000_000.0)
    target_per_hour = max(1.0, float(target_per_day) / 24.0)
    hourly_thresholds = expected_hourly / target_per_hour

    return (
        hourly_thresholds.reindex(df.index, method="ffill")
        .fillna(1_000_000.0)
        .values.astype(np.float64)
    )


def generate_dollar_bars(
    df: pd.DataFrame,
    threshold: float = 1_000_000.0,
    daily_bar_target: int | None = None,
    adaptive_window: int = 30,
) -> pd.DataFrame:
    """Transform OHLCV into Dollar Bars.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``open, high, low, close`` and a volume column
        (``tick_volume`` preferred; ``volume`` fallback). Optional
        ``bid_*`` / ``ask_*`` OHLC quartet.
    threshold : float, default 1_000_000.0
        Fixed notional threshold per bar (USD). Ignored if
        ``daily_bar_target`` is provided.
    daily_bar_target : int | None
        If supplied, the threshold is *dynamic* per hour-of-week to target
        roughly this many bars per day on average.
    adaptive_window : int
        Window (days) for the daily-notional fallback.
    """
    if df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index, utc=True)

    vals_date = df.index.astype(np.int64).values
    vals_o = df["open"].values.astype(np.float64)
    vals_h = df["high"].values.astype(np.float64)
    vals_l = df["low"].values.astype(np.float64)
    vals_c = df["close"].values.astype(np.float64)

    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vals_v = df[vol_col].fillna(0).values.astype(np.float64)

    if "bid_open" in df.columns:
        bo = df["bid_open"].values.astype(np.float64)
        bh = df["bid_high"].values.astype(np.float64)
        bl = df["bid_low"].values.astype(np.float64)
        bc = df["bid_close"].values.astype(np.float64)
    else:
        bo = bh = bl = bc = vals_c

    if "ask_open" in df.columns:
        ao = df["ask_open"].values.astype(np.float64)
        ah = df["ask_high"].values.astype(np.float64)
        al = df["ask_low"].values.astype(np.float64)
        ac = df["ask_close"].values.astype(np.float64)
    else:
        ao = ah = al = ac = vals_c

    if daily_bar_target is not None:
        vals_thresholds = _calculate_dynamic_dollar_thresholds(
            df, daily_bar_target, adaptive_window
        )
    else:
        vals_thresholds = np.full(len(df), threshold, dtype=np.float64)

    res = numba_dollar_bars(
        vals_date,
        vals_o, vals_h, vals_l, vals_c,
        vals_v, vals_thresholds,
        bo, bh, bl, bc,
        ao, ah, al, ac,
    )
    ts, o, h, l, c, v, notional, bo_o, bo_h, bo_l, bo_c, ao_o, ao_h, ao_l, ao_c = res

    df_bars = pd.DataFrame(
        {
            "open": o, "high": h, "low": l, "close": c,
            "tick_volume": v, "bar_notional": notional,
            "bid_open": bo_o, "bid_high": bo_h, "bid_low": bo_l, "bid_close": bo_c,
            "ask_open": ao_o, "ask_high": ao_h, "ask_low": ao_l, "ask_close": ao_c,
        },
        index=pd.to_datetime(ts, utc=True),
    )

    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        df_bars = df_bars.drop(
            columns=["bid_open", "bid_high", "bid_low", "bid_close",
                     "ask_open", "ask_high", "ask_low", "ask_close"],
            errors="ignore",
        )
    return df_bars


__all__ = ["generate_dollar_bars", "numba_dollar_bars"]
