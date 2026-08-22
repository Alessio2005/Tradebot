"""Garman-Klass (1980) + Barndorff-Nielsen & Shephard jump-variance estimators.

Migrated verbatim from legacy volatility.py. All Numba kernels preserved.
No Yang-Zhang: YZ is designed for DAILY OHLC; overnight component is
undefined for intraday imbalance bars => removed per AFML §2.4 rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _garman_klass_kernel(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    window: int,
) -> np.ndarray:
    """O(N) sliding-window Garman-Klass variance.

    sigma_GK^2 = mean[0.5*(ln H/L)^2] - (2ln2-1)*mean[(ln C/O)^2]
    """
    n = len(close_arr)
    sigma_sq = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return sigma_sq
    log2_const = 2.0 * 0.6931471805599453 - 1.0
    hl = np.empty(n, dtype=np.float64)
    co = np.empty(n, dtype=np.float64)
    for i in range(n):
        lh = np.log(np.maximum(high_arr[i], 1e-9)) - np.log(np.maximum(low_arr[i], 1e-9))
        hl[i] = 0.5 * lh * lh
        lc = np.log(np.maximum(close_arr[i], 1e-9)) - np.log(np.maximum(open_arr[i], 1e-9))
        co[i] = lc * lc
    sum_hl = 0.0; sum_co = 0.0
    for i in range(window):
        sum_hl += hl[i]; sum_co += co[i]
    sigma_sq[window - 1] = max(0.0, sum_hl / window - log2_const * (sum_co / window))
    for i in range(window, n):
        sum_hl += hl[i] - hl[i - window]
        sum_co += co[i] - co[i - window]
        sigma_sq[i] = max(0.0, sum_hl / window - log2_const * (sum_co / window))
    return np.sqrt(sigma_sq)


@njit(cache=True)
def _jump_variance_kernel(close_arr: np.ndarray, window: int) -> np.ndarray:
    """Barndorff-Nielsen & Shephard (2004) jump-variance estimator.

    J_t = max(RV_t - BV_t, 0) where BV uses debiased bipower variation
    (Corsi, Pirino & Reno 2010 sparse-data correction).
    """
    n = len(close_arr)
    jump_sigma = np.full(n, np.nan, dtype=np.float64)
    if n < window + 1:
        return jump_sigma
    pi_2 = 1.5707963267948966
    returns = np.empty(n, dtype=np.float64)
    returns[0] = 0.0
    for i in range(1, n):
        if close_arr[i - 1] > 1e-9 and close_arr[i] > 1e-9:
            returns[i] = np.log(close_arr[i]) - np.log(close_arr[i - 1])
        else:
            returns[i] = 0.0
    for i in range(window, n):
        rv_sum = 0.0; bv_sum = 0.0; nonzero_pairs = 0
        start = i - window + 1
        for j in range(start, i + 1):
            rv_sum += returns[j] * returns[j]
        for j in range(start + 1, i + 1):
            rj = returns[j]; rjm1 = returns[j - 1]
            if rj != 0.0 and rjm1 != 0.0:
                bv_sum += abs(rj) * abs(rjm1); nonzero_pairs += 1
        total_pairs = window - 1
        if window > 1 and total_pairs > 0:
            bv_sum = bv_sum * pi_2 * (window / (window - 1))
            if nonzero_pairs > 0:
                p = float(nonzero_pairs) / float(total_pairs)
                if p >= 0.5:
                    bv_sum = bv_sum / (p * p)
                else:
                    bv_sum = bv_sum * (float(total_pairs) / max(float(nonzero_pairs), 1.0))
        jump_var = rv_sum - bv_sum
        jump_var = max(jump_var, 0.0)
        jump_sigma[i] = np.sqrt(jump_var)
    return jump_sigma


def get_garman_klass_volatility(
    df: pd.DataFrame, window: int = 14, absolute: bool = False
) -> pd.Series:
    """Garman-Klass volatility (or Parkinson fallback when open missing)."""
    from .parkinson import _parkinson_kernel
    if df.empty or len(df) < window:
        return pd.Series(0.0, index=df.index, name="feat_vol_gk")
    h = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    l = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    c = np.ascontiguousarray(df["close"].values, dtype=np.float64)
    if "open" in df.columns:
        o = np.ascontiguousarray(df["open"].values, dtype=np.float64)
        vol_vals = _garman_klass_kernel(o, h, l, c, window)
    else:
        vol_vals = _parkinson_kernel(h, l, window)
    result = pd.Series(vol_vals * c if absolute else vol_vals,
                       index=df.index, name="feat_vol_gk")
    return result.ffill().fillna(0.0)


def get_jump_adjusted_volatility(
    df: pd.DataFrame, window: int = 14, absolute: bool = False
) -> pd.Series:
    """Combined GK + BNS jump-diffusion volatility.

    sigma_total = sqrt(sigma_GK^2 + sigma_jump^2)
    """
    if df.empty or len(df) < window:
        return pd.Series(0.0, index=df.index, name="feat_vol_gk_jump")
    gk = get_garman_klass_volatility(df, window=window, absolute=False)
    gk_vals = np.ascontiguousarray(gk.values, dtype=np.float64)
    c = np.ascontiguousarray(df["close"].values, dtype=np.float64)
    jump_vals = _jump_variance_kernel(c, window)
    jump_vals = np.nan_to_num(jump_vals, nan=0.0, posinf=0.0, neginf=0.0)
    total_sigma = np.sqrt(np.maximum(gk_vals**2 + jump_vals**2, 0.0))
    result = pd.Series(
        total_sigma * c if absolute else total_sigma,
        index=df.index, name="feat_vol_gk_jump"
    )
    return result.ffill().fillna(0.0)


# Alias kept for backward compatibility
gt_garman_klass_volatilility = get_jump_adjusted_volatility
