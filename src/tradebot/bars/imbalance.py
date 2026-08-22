"""Volume Imbalance Bars (López de Prado AFML §2.2).

Migrated from legacy bars.py. All business logic preserved verbatim;
Numba kernel re-exported from bars._kernels (single source of truth).
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from ._kernels import numba_imbalance_bars

logger = logging.getLogger(__name__)


def _calculate_imbalance_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-tick (tick_vol, signed_imbalance) for the imbalance kernel.

    Scenario A (Bybit): exact taker_buy / taker_sell volumes.
    Scenario B (Forex / generic): tick-rule proxy via bid/ask depth.
    """
    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        vol_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        vol_arr = df[vol_col].fillna(0).values.astype(np.float64)
        imb_arr = (df["taker_buy_volume"] - df["taker_sell_volume"]).fillna(0).values.astype(np.float64)
        return vol_arr, imb_arr

    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    vol_arr = df[vol_col].fillna(0).values.astype(np.float64)
    delta = df["close"].diff()
    _sign = np.sign(delta.fillna(0).to_numpy(dtype=np.float64))
    _sign[_sign == 0] = np.nan
    tick_dir = pd.Series(_sign, index=delta.index).ffill().fillna(1.0).to_numpy(dtype=np.float64)
    imb_arr = (vol_arr * tick_dir).astype(np.float64)
    return vol_arr, imb_arr


def _calculate_dynamic_thresholds(
    df: pd.DataFrame,
    imb_arr: np.ndarray,
    target_per_day: int,
    window_days: int = 20,
) -> np.ndarray:
    """Intraday seasonality-aware dynamic thresholds (per hour-of-week EWMA).

    Lookahead-safe: shift(1) within each bucket so the threshold for a bar
    is based solely on past observations in the same hour-of-week.
    """
    if len(df) == 0:
        return np.array([], dtype=np.float64)

    signed_imbalance = pd.Series(imb_arr, index=df.index)
    hourly_imbalance = (
        signed_imbalance.groupby(pd.Grouper(freq="h")).sum().abs()
    )
    hourly_imbalance = hourly_imbalance.where(hourly_imbalance > 1.0, np.nan)

    hourly_idx = cast(pd.DatetimeIndex, hourly_imbalance.index)
    bucket = (
        hourly_idx.dayofweek.astype(np.int64) * 24
        + hourly_idx.hour.astype(np.int64)
    )
    bucket_series = pd.Series(bucket, index=hourly_idx, name="bucket")

    df_hourly = hourly_imbalance.to_frame("imb").assign(bucket=bucket_series)
    shifted = df_hourly.groupby("bucket")["imb"].shift(1)

    def _time_ewm(s: pd.Series) -> pd.Series:
        s_clean = s.dropna()
        if s_clean.empty:
            return s
        times = cast(pd.DatetimeIndex, s_clean.index)
        halflife = pd.Timedelta(weeks=4)
        ln2 = np.log(2)
        result = pd.Series(index=s_clean.index, dtype=float)
        result.iloc[0] = s_clean.iloc[0]
        for i in range(1, len(s_clean)):
            dt = (times[i] - times[i - 1]) / halflife
            alpha = 1 - np.exp(-ln2 * dt)
            result.iloc[i] = alpha * s_clean.iloc[i] + (1 - alpha) * result.iloc[i - 1]
        return result.reindex(s.index)

    seasonal_expect = (
        shifted.groupby(df_hourly["bucket"], group_keys=False).apply(_time_ewm)
    )

    daily_imb = signed_imbalance.groupby(pd.Grouper(freq="D")).sum().abs()
    daily_imb = daily_imb.where(daily_imb > 100.0, np.nan)
    daily_fallback_h = (
        daily_imb.rolling(window=window_days, min_periods=1).median().shift(1) / 24.0
    )
    daily_fallback_h = daily_fallback_h.reindex(hourly_idx, method="ffill")

    expected_hourly = seasonal_expect.fillna(daily_fallback_h).fillna(10_000.0)
    target_per_hour = max(1.0, float(target_per_day) / 24.0)
    hourly_thresholds = expected_hourly / target_per_hour

    thresholds_aligned = (
        hourly_thresholds.reindex(df.index, method="ffill").fillna(10_000.0)
    )
    return thresholds_aligned.values.astype(np.float64)


def generate_imbalance_bars(
    df: pd.DataFrame,
    imbalance_threshold: float = 50_000,
    daily_bar_target: int | None = None,
    adaptive_window: int = 30,
) -> pd.DataFrame:
    """Transform tick/S5 data into Volume Imbalance Bars."""
    if df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    vals_date = df.index.astype(np.int64).values
    vals_o = df["open"].values.astype(np.float64)
    vals_h = df["high"].values.astype(np.float64)
    vals_l = df["low"].values.astype(np.float64)
    vals_c = df["close"].values.astype(np.float64)

    vals_v, vals_imbalance = _calculate_imbalance_series(df)

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
        vals_thresholds = _calculate_dynamic_thresholds(
            df, vals_imbalance, daily_bar_target, adaptive_window
        )
    else:
        vals_thresholds = np.full(len(df), imbalance_threshold, dtype=np.float64)

    res = numba_imbalance_bars(
        vals_date,
        vals_o, vals_h, vals_l, vals_c,
        vals_v, vals_imbalance, vals_thresholds,
        bo, bh, bl, bc,
        ao, ah, al, ac,
    )
    ts, o, h, l, c, v, bo_o, bo_h, bo_l, bo_c, ao_o, ao_h, ao_l, ao_c = res

    df_bars = pd.DataFrame(
        {
            "open": o, "high": h, "low": l, "close": c, "tick_volume": v,
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
