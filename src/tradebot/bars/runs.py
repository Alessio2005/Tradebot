"""Volume Runs Bars (López de Prado AFML §2.2).

Runs Bars detect VWAP/TWAP algorithm stealth — they close when the dominant
side accumulates volume above threshold even when net imbalance stays low.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd

from ._kernels import numba_runs_bars

logger = logging.getLogger(__name__)


def _calculate_runs_series(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tick (tick_vol, buy_vol, sell_vol) for the runs kernel.

    Preferred path: exact taker_buy / taker_sell volumes (Bybit trades).
    Fallback: bid/ask depth proxy (Forex L1). Last resort: tick-rule.
    WARNING: tick-rule on crypto is <= 70% accurate — logs a warning.
    """
    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        vol_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        tick_vols = df[vol_col].fillna(0.0).values.astype(np.float64)
        buy_vols  = df["taker_buy_volume"].fillna(0.0).values.astype(np.float64)
        sell_vols = df["taker_sell_volume"].fillna(0.0).values.astype(np.float64)
        sell_vols = np.maximum(sell_vols, tick_vols - buy_vols)
        sell_vols = np.maximum(sell_vols, 0.0)
        return tick_vols, buy_vols, sell_vols

    if "taker_buy_volume" in df.columns:
        vol_col = "real_volume" if "real_volume" in df.columns else "tick_volume"
        tick_vols = df[vol_col].fillna(0.0).values.astype(np.float64)
        buy_vols  = df["taker_buy_volume"].fillna(0.0).values.astype(np.float64)
        sell_vols = np.maximum(tick_vols - buy_vols, 0.0)
        return tick_vols, buy_vols, sell_vols

    vol_col   = "tick_volume" if "tick_volume" in df.columns else "volume"
    tick_vols = df[vol_col].fillna(0.0).values.astype(np.float64)

    if "ask_depth" in df.columns and "bid_depth" in df.columns:
        ask_d = df["ask_depth"].fillna(0.0).values.astype(np.float64)
        bid_d = df["bid_depth"].fillna(0.0).values.astype(np.float64)
        depth_diff = ask_d - bid_d
        tick_sign = np.where(depth_diff > 0, 1.0, np.where(depth_diff < 0, -1.0, 0.0))
    else:
        looks_crypto = (
            "real_volume" in df.columns
            or ("tick_volume" in df.columns
                and "ask_depth" not in df.columns
                and "spread" not in df.columns)
        )
        if looks_crypto:
            logger.warning(
                "Runs Bars: tick-rule fallback on crypto data "
                "(no taker_buy/sell columns). Accuracy <= 70%% — "
                "do NOT use in production. Add maker/taker flags."
            )
        else:
            logger.info("Runs Bars: tick-rule fallback (no orderflow signals).")

        delta    = df["close"].diff().fillna(0.0).to_numpy(dtype=np.float64)
        raw_sign = np.sign(delta)
        raw_sign[raw_sign == 0] = np.nan
        tick_sign = (
            pd.Series(raw_sign, index=df.index).ffill().fillna(1.0).to_numpy(dtype=np.float64)
        )

    buy_vols  = tick_vols * np.where(tick_sign > 0, 1.0, 0.0)
    sell_vols = tick_vols * np.where(tick_sign < 0, 1.0, 0.0)
    return tick_vols, buy_vols, sell_vols


def _calculate_runs_thresholds(
    df: pd.DataFrame,
    buy_vols: np.ndarray,
    sell_vols: np.ndarray,
    target_per_day: int,
    window_days: int = 20,
) -> np.ndarray:
    """Dynamic threshold for Runs Bars — lookahead-safe (shift(1) within bucket)."""
    max_run_per_tick = np.maximum(buy_vols, sell_vols)
    max_run_series   = pd.Series(max_run_per_tick, index=df.index)

    hourly_max = max_run_series.groupby(pd.Grouper(freq="h")).sum()
    hourly_max = hourly_max.where(hourly_max > 1.0, np.nan)

    hourly_idx = cast(pd.DatetimeIndex, hourly_max.index)
    bucket = (
        hourly_idx.dayofweek.astype(np.int64) * 24
        + hourly_idx.hour.astype(np.int64)
    )
    bucket_series = pd.Series(bucket, index=hourly_idx, name="bucket")
    df_hourly = hourly_max.to_frame("run").assign(bucket=bucket_series)
    shifted = df_hourly.groupby("bucket")["run"].shift(1)

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
    daily_max = (
        max_run_series.groupby(pd.Grouper(freq="D")).sum()
        .where(lambda s: s > 100.0, np.nan)
        .rolling(window=window_days, min_periods=1)
        .median()
        .shift(1)
        / 24.0
    )
    daily_fallback_h = daily_max.reindex(hourly_idx, method="ffill")
    expected_hourly = seasonal_expect.fillna(daily_fallback_h).fillna(10_000.0)
    target_per_hour = max(1.0, float(target_per_day) / 24.0)
    hourly_thresh = expected_hourly / target_per_hour
    thresholds = hourly_thresh.reindex(df.index, method="ffill").fillna(10_000.0)
    return thresholds.values.astype(np.float64)


def _add_run_aggregate_features(df_bars: pd.DataFrame) -> pd.DataFrame:
    """Voeg stationaire run-aggregaat features toe (v3 T0.1).

    Alle berekeningen zijn CAUSAL (geen lookahead). shift(1) gebruikt waar nodig.
    """
    if "runs_imbalance" not in df_bars.columns:
        return df_bars

    df = df_bars.copy()

    # feat_run_length_avg_20: rolling |imbalance| als proxy voor run-persistentie
    df["feat_run_length_avg_20"] = (
        df["runs_imbalance"].abs().rolling(20, min_periods=5).mean()
    )

    # feat_run_burst_5: hoeveel van de laatste 5 bars had > 2× gemiddeld volume?
    vol_col = "tick_volume" if "tick_volume" in df.columns else "runs_buy_vol"
    if vol_col in df.columns:
        vol_avg_20 = df[vol_col].rolling(20, min_periods=5).mean().shift(1)
        df["feat_run_burst_5"] = (
            (df[vol_col] > 2.0 * vol_avg_20)
            .rolling(5, min_periods=1)
            .sum()
            .fillna(0.0)
        )
    else:
        df["feat_run_burst_5"] = 0.0

    # feat_run_imbalance_zscore_50: stationary Z-score over 50-bar window
    ri = df["runs_imbalance"]
    ri_mean = ri.rolling(50, min_periods=10).mean()
    ri_std  = ri.rolling(50, min_periods=10).std().clip(lower=1e-8)
    df["feat_run_imbalance_zscore_50"] = ((ri - ri_mean) / ri_std).clip(-5.0, 5.0)

    return df


def generate_runs_bars(
    df: pd.DataFrame,
    runs_threshold: float = 50_000.0,
    daily_bar_target: int | None = None,
    adaptive_window: int = 20,
) -> pd.DataFrame:
    """Transform tick/S5 data into Volume Runs Bars.

    API intentionally identical to generate_imbalance_bars so the
    FeaturePipeline can switch via a config flag.
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

    tick_vols, buy_vols, sell_vols = _calculate_runs_series(df)

    if daily_bar_target is not None:
        thresholds = _calculate_runs_thresholds(
            df, buy_vols, sell_vols, daily_bar_target, adaptive_window
        )
    else:
        thresholds = np.full(len(df), runs_threshold, dtype=np.float64)

    vals_o = df["open"].values.astype(np.float64)
    vals_h = df["high"].values.astype(np.float64)
    vals_l = df["low"].values.astype(np.float64)
    vals_c = df["close"].values.astype(np.float64)
    vals_date = df.index.astype(np.int64).values

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

    res = numba_runs_bars(
        vals_date,
        vals_o, vals_h, vals_l, vals_c,
        tick_vols, buy_vols, sell_vols, thresholds,
        bo, bh, bl, bc,
        ao, ah, al, ac,
    )
    (ts, o, h, l, c, v, bv, sv,
     r_bo, r_bh, r_bl, r_bc,
     r_ao, r_ah, r_al, r_ac) = res

    df_bars = pd.DataFrame(
        {
            "open": o, "high": h, "low": l, "close": c,
            "tick_volume": v,
            "runs_buy_vol": bv,
            "runs_sell_vol": sv,
            "bid_open": r_bo, "bid_high": r_bh, "bid_low": r_bl, "bid_close": r_bc,
            "ask_open": r_ao, "ask_high": r_ah, "ask_low": r_al, "ask_close": r_ac,
        },
        index=pd.to_datetime(ts, utc=True),
    )
    total_safe = np.where(v > 0, v, 1.0)
    df_bars["runs_imbalance"] = (bv - sv) / total_safe

    if "taker_buy_volume" in df.columns:
        df_bars = df_bars.drop(
            columns=["bid_open", "bid_high", "bid_low", "bid_close",
                     "ask_open", "ask_high", "ask_low", "ask_close"],
            errors="ignore",
        )
    df_bars = _add_run_aggregate_features(df_bars)
    return df_bars
