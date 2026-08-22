"""Parkinson (1980) high-low range volatility estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _parkinson_kernel(high_arr: np.ndarray, low_arr: np.ndarray, window: int) -> np.ndarray:
    """sigma_P^2 = mean[(ln H/L)^2] / (4 ln 2)  — O(N) sliding window."""
    n = len(high_arr)
    sigma_sq = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return sigma_sq
    inv_4ln2 = 1.0 / (4.0 * 0.6931471805599453)
    hl_sq = np.empty(n, dtype=np.float64)
    for i in range(n):
        lhl = np.log(np.maximum(high_arr[i], 1e-9)) - np.log(np.maximum(low_arr[i], 1e-9))
        hl_sq[i] = lhl * lhl
    rolling_sum = 0.0
    for i in range(window):
        rolling_sum += hl_sq[i]
    sigma_sq[window - 1] = max(0.0, rolling_sum / window * inv_4ln2)
    for i in range(window, n):
        rolling_sum += hl_sq[i] - hl_sq[i - window]
        sigma_sq[i] = max(0.0, rolling_sum / window * inv_4ln2)
    return np.sqrt(sigma_sq)


def get_parkinson_volatility(
    df: pd.DataFrame, window: int = 14, absolute: bool = False
) -> pd.Series:
    """Parkinson (1980) volatility — only high/low required."""
    if df.empty or len(df) < window:
        return pd.Series(0.0, index=df.index, name="feat_vol_park")
    h = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    l = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    vals = _parkinson_kernel(h, l, window)
    c = df["close"].values.astype(np.float64)
    result = pd.Series(vals * c if absolute else vals,
                       index=df.index, name="feat_vol_park")
    return result.ffill().fillna(0.0)
