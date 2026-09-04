# LOC-EXCEPTION: regimedetectie, structurele-breukanalyse en de featurepipeline delen dezelfde
# vensterdefinities; die uit elkaar trekken levert twee bestanden die elkaars constanten
# moeten importeren.
# Cap staat op 1056 regels in scripts/check_file_size.py; groeien is rood.
# src/tradebot/features/regime.py
"""Regime detection, structural break analysis and feature pipeline.

Extracted from ``features.py``.

Contents:
    StationarityComputer      — rolling stationarity-momentum (Numba)
    HurstComputer             — rolling Hurst exponent (Numba, dynamic lags BUG-FIX)
    _cusum_structural_break   — CUSUM structural break test (Numba, M5-FIX bars_since_break)
    _rolling_ols_residuals    — rolling OLS residuals (Numba)
    _ewma_conditional_variance — RiskMetrics EWMA variance (Numba)
    _rolling_wls_residuals    — WLS residuals with EWMA weights (Numba)
    _classify_3_regimes       — 3-regime classifier (Numba, LOOKAHEAD-FIX SK-1)
    RegimeRouter              — 3-regime supervisor (LOOKAHEAD-FIX SK-2 bfill→ffill)
    StructuralBreakRegime     — CUSUM-based structural break detector
    FeaturePipeline           — 3-layer feature pipeline (bar gen → TA → PCA)
"""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
from numba import njit
from omegaconf import DictConfig, OmegaConf

from ..bars import generate_imbalance_bars, generate_runs_bars
from .orthogonalize import FeatureOrthogonalizer
from .regime_features import add_regime_features
from .ta import FeatureEngineer

logger = logging.getLogger("features.regime")


# =============================================================================
# 0. STATIONARITY MOMENTUM COMPUTER (Regime Filter)
# =============================================================================
class StationarityComputer:
    def __init__(self, window: int = 75):
        self.window = window

    def compute(self, prices: np.ndarray) -> np.ndarray:
        if len(prices) == 0:
            return np.array([])
        p = np.log(np.maximum(prices, 1e-9)).astype(np.float64)
        return self._rolling_stat_mom_njit(p, self.window)

    @staticmethod
    @njit(cache=True)
    def _rolling_stat_mom_njit(prices: np.ndarray, window: int) -> np.ndarray:
        n = len(prices)
        out = np.zeros(n, dtype=np.float64)
        ma_price = np.zeros(n, dtype=np.float64)
        z_price = np.zeros(n, dtype=np.float64)

        if n < window * 2:
            return out

        # FIX: Prevent Catastrophic Cancellation (floating-point instability).
        # Use efficient Numba slices instead of Naive Sum of Squares.
        for i in range(window - 1, n):
            window_slice = prices[i - window + 1 : i + 1]
            mu = np.mean(window_slice)
            var = np.var(window_slice)
            sigma = np.sqrt(var)

            ma_price[i] = mu

            if sigma > 1e-9:
                z_raw = (prices[i] - mu) / sigma
                z_price[i] = min(max(z_raw, -4.0), 4.0)
            else:
                z_price[i] = 0.0

        start_idx = (window * 2) - 1

        for i in range(start_idx, n):
            window_slice_ma = ma_price[i - window + 1 : i + 1]
            mu_ma = np.mean(window_slice_ma)
            var_ma = np.var(window_slice_ma)
            sigma_ma = np.sqrt(var_ma)

            if sigma_ma > 1e-9:
                z_raw_ma = (ma_price[i] - mu_ma) / sigma_ma
                z_ma = min(max(z_raw_ma, -4.0), 4.0)
            else:
                z_ma = 0.0

            out[i] = z_price[i] - z_ma

        return out


# =============================================================================
# 1. HURST COMPUTER (Fractal Dimension / Chaos)
# =============================================================================
class HurstComputer:
    def __init__(self, min_window: int, num_lags: int = 20):
        self.min_window = min_window
        self.num_lags = num_lags

    def compute_rolling(self, prices: np.ndarray, window_size: int) -> np.ndarray:
        if len(prices) < 20:
            return np.full(len(prices), 0.5)

        safe_prices = np.maximum(prices, 1e-9)
        log_prices = np.log(safe_prices).astype(np.float64)

        effective_window = window_size
        if len(prices) < window_size:
            effective_window = max(32, len(prices) // 2)

        return self._rolling_hurst_njit(log_prices, effective_window, self.num_lags)

    @staticmethod
    @njit(parallel=False, cache=True)
    def _rolling_hurst_njit(prices: np.ndarray, window: int, lags_count: int) -> np.ndarray:
        n = len(prices)
        output = np.full(n, 0.5)

        if n < window:
            return output

        # BUG-FIX: lags_count was accepted but never used — lags were always
        # hardcoded as [2,4,8,16,32,64,128] (7 values) regardless of the parameter.
        # For window=375 (meso) or window=250 (macro), this means max lag 128 covers
        # only ~34% of the scale range, underestimating H for persistent processes.
        # Fix: generate powers-of-2 lags dynamically up to lags_count entries.
        actual_lags = min(max(lags_count, 3), 20)   # safety: [3, 20]
        lags = np.zeros(actual_lags, dtype=np.int64)
        for _k in range(actual_lags):
            lags[_k] = np.int64(2 ** (_k + 1))     # [2, 4, 8, ..., 2^actual_lags]

        for i in range(window - 1, n):
            chunk = prices[i - window + 1 : i + 1]

            tau = np.zeros(actual_lags, dtype=np.float64)
            valid_lags = 0

            for j in range(actual_lags):
                lag = lags[j]
                if lag >= len(chunk):
                    break

                diffs = chunk[lag:] - chunk[:-lag]
                val = np.sqrt(np.mean(diffs ** 2))

                if val < 1e-12:
                    # Zero-variance at this lag: skip but keep trying larger lags.
                    continue

                tau[valid_lags] = val
                valid_lags += 1

            if valid_lags < 3:
                continue

            x = np.log(lags[:valid_lags].astype(np.float64))
            y = np.log(tau[:valid_lags])

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            num = np.sum((x - x_mean) * (y - y_mean))
            den = np.sum((x - x_mean) ** 2)

            if den > 1e-9:
                hurst = num / den
                output[i] = min(max(hurst, 0.0), 1.0)

        return output


# =============================================================================
# 1.b STRUCTURAL BREAK REGIME DETECTOR (replaces HMM)
# =============================================================================
@njit(cache=True)
def _cusum_structural_break(
    residuals: np.ndarray,
    threshold_sigma: float,
    min_bars_between_breaks: int,
) -> np.ndarray:
    """CUSUM test on residuals of a rolling mean.

    Detects structural breaks when the cumulative deviation from the mean
    exceeds the threshold (threshold_sigma * σ).

    Returns:
        bars_since_break: float32 array — number of bars elapsed since the
        last detected structural break.
        Low (≈ 0–30) = recent regime shift; high = current regime is stable.

    M5-FIX: The previous implementation used a toggle (0.0 ↔ 1.0) that switched
    on every break. This had no absolute semantics: "0.0 after the 2nd break" and
    "0.0 before the first break" represented completely different market conditions.
    CatBoost was learning only "even or odd number of breaks?" rather than
    "how recent was the last regime shift?". bars_since_break provides
    economically meaningful information.
    """
    n = len(residuals)
    flags = np.zeros(n, dtype=np.float32)

    s_pos = 0.0
    s_neg = 0.0

    # LEAKAGE-FIX: Skip the zero-padding at the start for sigma initialisation.
    # The first 'ols_window' residuals are 0.0 (burn-in of _rolling_ols_residuals).
    # Including them in np.std() makes sigma artificially low → CUSUM too
    # sensitive → false structural breaks early in the series.
    first_nonzero = 0
    for _i in range(n):
        if residuals[_i] != 0.0:
            first_nonzero = _i
            break

    init_start = first_nonzero
    init_end = min(first_nonzero + 500, n)
    if init_end > init_start:
        sigma = max(np.std(residuals[init_start:init_end]), 1e-9)
    else:
        sigma = 1e-9
    h = threshold_sigma * sigma

    last_break = -min_bars_between_breaks

    for i in range(1, n):
        # Update rolling sigma every 250 bars (avoids creeping threshold changes)
        if i % 250 == 0:
            start = max(0, i - 500)
            sigma = max(np.std(residuals[start:i]), 1e-9)
            h = threshold_sigma * sigma

        s_pos = max(0.0, s_pos + residuals[i])
        s_neg = min(0.0, s_neg + residuals[i])

        if (s_pos >= h or s_neg <= -h) and (i - last_break) >= min_bars_between_breaks:
            # M5-FIX: Remove toggle; only store break position.
            s_pos = 0.0
            s_neg = 0.0
            last_break = i

        # M5-FIX: bars_since_break as feature value (was: current_regime toggle 0/1)
        flags[i] = float(i - last_break)

    return flags


@njit(cache=True)
def _rolling_ols_residuals(prices: np.ndarray, window: int) -> np.ndarray:
    """Rolling OLS residuals: deviation from local linear trend.

    Used as input for the CUSUM structural break test.
    """
    n = len(prices)
    residuals = np.zeros(n, dtype=np.float64)

    for i in range(window, n):
        y = prices[i - window : i + 1]
        L = len(y)
        x_mean = (L - 1) * 0.5
        y_mean = 0.0
        for k in range(L):
            y_mean += y[k]
        y_mean /= L

        ss_xx = 0.0
        sp_xy = 0.0
        for k in range(L):
            xk = float(k) - x_mean
            ss_xx += xk * xk
            sp_xy += xk * (y[k] - y_mean)

        if ss_xx < 1e-9:
            continue

        slope = sp_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_hat = slope * float(window) + intercept
        residuals[i] = prices[i] - y_hat

    return residuals


@njit(cache=True)
def _ewma_conditional_variance(
    residuals: np.ndarray,
    lam: float = 0.94,
    min_var: float = 1e-12,
) -> np.ndarray:
    """RiskMetrics-style EWMA conditional variance (causal).

        σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_{t-1}

    λ = 0.94 matches the classic JP Morgan RiskMetrics specification for
    daily data. The output is one-bar-lagged relative to the input — no
    lookahead.
    """
    n = len(residuals)
    var = np.empty(n, dtype=np.float64)
    if n == 0:
        return var
    warmup = min(n, 60)
    acc = 0.0
    cnt = 0
    for i in range(warmup):
        r = residuals[i]
        if r != 0.0:
            acc += r * r
            cnt += 1
    init_var = (acc / cnt) if cnt > 0 else min_var
    var[0] = init_var
    for t in range(1, n):
        prev_var = var[t - 1]
        prev_r = residuals[t - 1]
        v = lam * prev_var + (1.0 - lam) * prev_r * prev_r
        v = max(v, min_var)
        var[t] = v
    return var


@njit(cache=True)
def _rolling_wls_residuals(
    prices: np.ndarray,
    window: int,
    lam: float = 0.94,
) -> np.ndarray:
    """Weighted-Least-Squares rolling residuals with EWMA-variance weights.

    Heteroscedasticty-robust variant of ``_rolling_ols_residuals``. Uses
    EWMA conditional variance as weights w_t = 1/σ²_t. The final residual is
    standardised by σ_t → variance-stable → CUSUM H-threshold is no longer
    contaminated by vol-clustering.
    """
    n = len(prices)
    residuals = np.zeros(n, dtype=np.float64)
    if n < window + 2:
        return residuals

    diffs = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diffs[i] = prices[i] - prices[i - 1]
    var_series = _ewma_conditional_variance(diffs, lam=lam)

    for i in range(window, n):
        y = prices[i - window : i + 1]
        L = y.shape[0]
        w_sum = 0.0
        wx_sum = 0.0
        wy_sum = 0.0
        wxx_sum = 0.0
        wxy_sum = 0.0

        base = i - window
        for k in range(L):
            var_k = var_series[base + k]
            var_k = max(var_k, 1e-12)
            w_k = 1.0 / var_k
            xk = float(k)
            yk = y[k]
            w_sum  += w_k
            wx_sum += w_k * xk
            wy_sum += w_k * yk
            wxx_sum += w_k * xk * xk
            wxy_sum += w_k * xk * yk

        if w_sum < 1e-12:
            continue

        xbar = wx_sum / w_sum
        ybar = wy_sum / w_sum
        ss_xx = wxx_sum - w_sum * xbar * xbar
        sp_xy = wxy_sum - w_sum * xbar * ybar

        if ss_xx < 1e-12:
            continue

        slope = sp_xy / ss_xx
        intercept = ybar - slope * xbar
        y_hat = slope * float(window) + intercept
        raw_res = prices[i] - y_hat

        sigma_i = np.sqrt(var_series[i]) if var_series[i] > 1e-12 else 1e-6
        residuals[i] = raw_res / sigma_i

    return residuals


# =============================================================================
# 1.c REGIME ROUTER — 3-regime classifier
# =============================================================================
@njit(cache=True)
def _classify_3_regimes(
    vol_arr: np.ndarray,
    skew_arr: np.ndarray,
    bars_since_break: np.ndarray,
    vol_lo_arr: np.ndarray,   # LOOKAHEAD-FIX (SK-1): per-bar causal expanding quantile
    vol_hi_arr: np.ndarray,   # LOOKAHEAD-FIX (SK-1): per-bar causal expanding quantile
    skew_panic: float,
) -> np.ndarray:
    """Closed-form 3-regime classifier (causal, vector-friendly).

    Regimes:
        0 — Low-Vol Mean-Reversion  : small movements, negative autocorr.
        1 — High-Vol Trending       : volatility above median, persistent trend.
        2 — Liquidation Chaos       : extreme neg-skew + recent structural break.

    LOOKAHEAD-FIX (SK-1): previous signature had scalar vol_lo/vol_hi computed
    via np.quantile(s_valid, q) over the ENTIRE time series — lookahead on bar t.
    Now receives per-bar arrays computed via expanding window with shift(1) in
    RegimeRouter.classify().
    """
    n = vol_arr.size
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        v = vol_arr[i]
        s = skew_arr[i]
        b = bars_since_break[i]
        vol_lo = vol_lo_arr[i]
        vol_hi = vol_hi_arr[i]
        if s <= skew_panic and b < 30.0:
            out[i] = np.int8(2)  # Liquidation chaos
        elif v >= vol_hi:
            out[i] = np.int8(1)  # High-vol trending
        elif v <= vol_lo:
            out[i] = np.int8(0)  # Low-vol mean-reversion
        else:
            # Mid-vol: use bars-since-break as tiebreaker
            out[i] = np.int8(1 if b < 60.0 else 0)
    return out


class RegimeRouter:
    """Three-regime supervisor above the model stack (HMM-equivalent without EM).

    Workflow:
      1. Input features: rolling realized vol (σ), rolling skewness (γ₁) and
         feat_bars_since_break (recency of the last structural break).
      2. Output: per-bar regime-id ∈ {0, 1, 2} + soft probabilities.
      3. Caller (train pipeline) trains a per-regime CatBoost expert; during
         inference the Bandit selects the correct expert via ``predict_regime()``.

    Args:
        vol_window:         bars for σ estimation.
        skew_window:        bars for skew estimation (typically ≥ vol_window × 2).
        vol_lo_quantile:    σ-threshold as quantile of the rolling window.
        vol_hi_quantile:    σ-threshold as quantile of the rolling window.
        skew_panic:         γ₁-threshold for the Liquidation-Chaos regime.
    """

    def __init__(
        self,
        vol_window: int = 60,
        skew_window: int = 240,
        vol_lo_quantile: float = 0.33,
        vol_hi_quantile: float = 0.67,
        skew_panic: float = -1.5,
    ) -> None:
        self.vol_window: int = int(vol_window)
        self.skew_window: int = int(skew_window)
        self.vol_lo_quantile: float = float(vol_lo_quantile)
        self.vol_hi_quantile: float = float(vol_hi_quantile)
        self.skew_panic: float = float(skew_panic)

    def classify(
        self,
        df: pd.DataFrame,
        bars_since_break: np.ndarray,
    ) -> np.ndarray:
        """Return regime-id per bar (int8, length == len(df))."""
        if "close" not in df.columns or len(df) < self.skew_window + 5:
            return np.zeros(len(df), dtype=np.int8)
        log_p = np.log(np.maximum(df["close"].to_numpy(dtype=np.float64), 1e-9))
        ret = np.zeros_like(log_p)
        ret[1:] = log_p[1:] - log_p[:-1]

        ret_s = pd.Series(ret)

        # LOOKAHEAD-FIX (SK-2 — bfill() reverse lookahead):
        # bfill() fills the first (vol_window-1) NaN-bars with the FIRST valid
        # value from the future → direct lookahead.
        # Fix: ffill() propagates forward (causal); remaining NaN (the absolute
        # start of the series) gets 0.0 — safe warmup value.
        s_arr = (
            ret_s
            .rolling(self.vol_window, min_periods=10)
            .std()
            .ffill()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        g_arr = (
            ret_s
            .rolling(self.skew_window, min_periods=30)
            .skew()
            .ffill()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        g_arr = np.nan_to_num(g_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # LOOKAHEAD-FIX (SK-1 — Global Quantile Lookahead):
        # np.quantile(s_valid, q) used the FULL series → thresholds on bar t
        # contained information from bars t+1..T.
        # Fix: expanding window + shift(1) so bar t only sees bars 0..t-1.
        min_calib = max(50, self.vol_window)
        s_valid_series = pd.Series(
            np.where(np.isfinite(s_arr) & (s_arr > 0.0), s_arr, np.nan)
        )
        vol_lo_arr = (
            s_valid_series
            .expanding(min_periods=min_calib)
            .quantile(self.vol_lo_quantile)
            .shift(1)
            .ffill()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        # CHIEF AUDIT 2026-05-23 (P-6): vol_hi warmup must be UNREACHABLE
        # (vol is always ≥ 0) — using fillna(0.0) here makes every warmup bar
        # satisfy ``v >= vol_hi`` and biases the regime label to High-Vol
        # Trending (=1).  Fill with +inf so the high-vol branch fires only
        # AFTER the expanding window has produced a valid quantile.
        # vol_lo keeps fillna(0.0) because a 0.0 lower threshold is also
        # unreachable for non-trivial vol → no bar mistakenly labeled Low-Vol.
        vol_hi_arr = (
            s_valid_series
            .expanding(min_periods=min_calib)
            .quantile(self.vol_hi_quantile)
            .shift(1)
            .ffill()
            .fillna(np.inf)
            .to_numpy(dtype=np.float64)
        )

        bars_arr = np.asarray(bars_since_break, dtype=np.float64)
        if bars_arr.shape[0] != s_arr.shape[0]:
            bars_arr = np.resize(bars_arr, s_arr.shape[0])

        regime_ids = _classify_3_regimes(
            s_arr, g_arr, bars_arr, vol_lo_arr, vol_hi_arr, self.skew_panic,
        )
        # CHIEF AUDIT 2026-05-23 (P-6): force regime label to -1 (burn-in /
        # unknown) for bars where BOTH thresholds are still uncalibrated.
        # ``vol_hi`` is +inf during warmup ⇒ the "high-vol trending" branch
        # never fires; combined with the constant 0.0 vol_lo this would
        # collapse the regime to the mean-revert/tiebreaker path.  We mark
        # those bars explicitly as burn-in so downstream code can ignore them.
        regime_ids = regime_ids.astype(np.int8, copy=True)
        warmup_mask = ~np.isfinite(vol_hi_arr)
        if warmup_mask.any():
            regime_ids[warmup_mask] = np.int8(-1)
        return regime_ids

    def soft_probabilities(
        self,
        regime_ids: np.ndarray,
        smoothing_bars: int = 30,
    ) -> np.ndarray:
        """One-hot regime-ids → smoothed probabilities (T, 3).

        Rolling EMA on the one-hot vector gives "soft" regime probabilities for
        the Contextual Bandit so that routing between experts is gradual rather
        than cliff-edge switching.
        """
        n = regime_ids.shape[0]
        oh = np.zeros((n, 3), dtype=np.float64)
        for k in range(n):
            r = int(regime_ids[k])
            if 0 <= r < 3:
                oh[k, r] = 1.0
        if smoothing_bars <= 1:
            return oh
        alpha = 2.0 / (float(smoothing_bars) + 1.0)
        out = np.zeros_like(oh)
        out[0] = oh[0]
        for k in range(1, n):
            out[k] = alpha * oh[k] + (1.0 - alpha) * out[k - 1]
        rs = out.sum(axis=1, keepdims=True)
        rs = np.where(rs > 1e-9, rs, 1.0)
        return out / rs


class StructuralBreakRegime:
    """Regime detector based on CUSUM structural break tests.

    Replaces HiddenMarkovRegime. Detects when the market equilibrium fundamentally
    changes (macro event, correlation shock, liquidity regime change).

    Output per bar (feat_bars_since_break): number of bars elapsed since the last
    detected structural break. Low = recent regime shift, high = regime is stable.

    M5-FIX: The previous toggle output (0.0 ↔ 1.0) had no absolute semantics —
    "0.0 after the 2nd break" and "0.0 before the first break" represented
    completely different market situations. CatBoost was learning only "even or
    odd number of breaks?" rather than the economically relevant recency of the
    last regime shift. bars_since_break resolves this.

    Advantages over rolling HMM:
      - No Gaussian assumption: works directly on OLS residuals (always stationary)
      - No label switching: output is deterministic and reproducible
      - No EM convergence risk: pure threshold crossing
      - No refit overhead: O(n) Numba kernel
      - Robust to fat-tail crypto data
    """

    def __init__(
        self,
        ols_window: int = 60,
        threshold_sigma: float = 3.0,
        min_bars_between_breaks: int = 30,
        use_wls: bool = True,
        ewma_lambda: float = 0.94,
    ):
        self.ols_window: int = int(ols_window)
        self.threshold_sigma: float = float(threshold_sigma)
        self.min_bars_between_breaks: int = int(min_bars_between_breaks)
        self.use_wls: bool = bool(use_wls)
        self.ewma_lambda: float = float(ewma_lambda)

    def compute_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Compute regime labels for the full DataFrame.

        Args:
            df: DataFrame with at least a 'close' column.

        Returns:
            float32 array: bars_since_break per bar. Low = recent regime shift,
            high = current regime is stable.
        """
        if "close" not in df.columns or len(df) < self.ols_window + 10:
            return np.zeros(len(df), dtype=np.float32)

        log_prices = np.log(
            np.maximum(df["close"].values.astype(np.float64), 1e-9)
        )

        if self.use_wls:
            residuals = _rolling_wls_residuals(
                log_prices,
                self.ols_window,
                lam=self.ewma_lambda,
            )
        else:
            residuals = _rolling_ols_residuals(log_prices, self.ols_window)

        flags = _cusum_structural_break(
            residuals,
            float(self.threshold_sigma),
            int(self.min_bars_between_breaks),
        )

        return flags


# =============================================================================
# 2. FEATURE PIPELINE
# =============================================================================
class FeaturePipeline:
    """Feature pipeline with three layers:

    1. Bar generation (Runs Bars or Imbalance Bars, configurable via config).
    2. TA feature computation per timeframe (micro / meso / macro).
    3. PCA orthogonalisation per timeframe cluster (optional).

    The ``FeatureOrthogonalizer`` instances are stored as attributes so that
    the training pipeline can serialise them (joblib) alongside the model.
    This guarantees the exact same PCA transformation is applied during
    training and live inference.

    Config flags (optional, defaults in code):
        feature_pipeline.use_runs_bars    : bool  (default True)
        feature_pipeline.use_pca_orth     : bool  (default True)
        feature_pipeline.pca_n_components : float (default 0.95)
        feature_pipeline.target_micro_bars: int   (default 30)
    """

    cfg:                  DictConfig
    symbol:               str
    hurst_computer:       HurstComputer
    stat_computer:        StationarityComputer
    ta_engineer:          FeatureEngineer
    orthogonalizer_micro: FeatureOrthogonalizer
    orthogonalizer_meso:  FeatureOrthogonalizer
    orthogonalizer_macro: FeatureOrthogonalizer

    def __init__(self, cfg: DictConfig, symbol: str) -> None:
        self.cfg    = cfg
        self.symbol = symbol

        # HURST-FIX: min_window raised to 700 (meso window). HurstComputer.min_window
        # is the minimum number of bars in the rolling window before the Hurst
        # estimate is activated — must be ≥ window_size used in compute_rolling().
        self.hurst_computer = HurstComputer(min_window=700, num_lags=20)
        self.stat_computer  = StationarityComputer(window=75)
        self.ta_engineer    = FeatureEngineer(dict(cfg))  # type: ignore[arg-type]

        _raw_comps: Any = OmegaConf.select(
            cfg, "feature_pipeline.pca_n_components", default=0.95
        )
        _pca_comps: float = float(_raw_comps) if _raw_comps is not None else 0.95

        self.orthogonalizer_micro  = FeatureOrthogonalizer(n_components=_pca_comps)
        self.orthogonalizer_meso   = FeatureOrthogonalizer(n_components=_pca_comps)
        self.orthogonalizer_macro  = FeatureOrthogonalizer(n_components=_pca_comps)

    def transform(self, df_raw: pd.DataFrame) -> tuple:
        if df_raw is None or df_raw.empty:
            return np.array([]), np.array([]), np.array([]), pd.DataFrame()

        # ── Index normalisation ──────────────────────────────────────────────
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            if "timestamp" in df_raw.columns:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
                df_raw = df_raw.set_index("timestamp")
            else:
                df_raw.index = pd.to_datetime(df_raw.index, utc=True)

        # ── Config (OmegaConf.select for type-safe dotpath access) ────────────
        def _sel(key: str, default: Any) -> Any:
            val = OmegaConf.select(self.cfg, key, default=default)
            return val if val is not None else default

        use_runs     = bool(_sel("feature_pipeline.use_runs_bars",   True))
        use_pca      = bool(_sel("feature_pipeline.use_pca_orth",    True))
        target_micro: int = int(_sel("feature_pipeline.target_micro_bars", 30))

        # ── 1. Bar generation ────────────────────────────────────────────────
        bar_fn = generate_runs_bars if use_runs else generate_imbalance_bars
        bar_type = "Runs" if use_runs else "Imbalance"
        logger.info(
            "[%s] Generating Hierarchical %s Bars (Target: %d micro bars/day)",
            self.symbol, bar_type, target_micro,
        )

        df_micro = bar_fn(df_raw, daily_bar_target=target_micro,                adaptive_window=20)
        df_meso  = bar_fn(df_raw, daily_bar_target=max(1, target_micro // 5),   adaptive_window=20)
        df_macro = bar_fn(df_raw, daily_bar_target=max(1, target_micro // 25),  adaptive_window=20)

        for df_target in [df_micro, df_meso, df_macro]:
            if not df_target.empty and "symbol" in df_raw.columns:
                df_target["symbol"] = df_raw["symbol"].iloc[0]

        logger.info(
            "Bars Generated: Micro=%d, Meso=%d, Macro=%d",
            len(df_micro), len(df_meso), len(df_macro),
        )

        # ── 2. Feature Calculation ───────────────────────────────────────────
        if not df_micro.empty:
            df_micro = self.ta_engineer.add_features(df_micro, "micro")
            df_micro["feat_stat_mom"] = self.stat_computer.compute(
                df_micro["close"].to_numpy(dtype=np.float64)
            )
            # Bear/bull regime features (causal, stationary — no lookahead).
            # Integrated here on the micro-bar series so that the features
            # share the same bar resolution as the other micro TA signals.
            df_micro = add_regime_features(df_micro)

        if not df_meso.empty:
            df_meso = self.ta_engineer.add_features(df_meso, "meso")
            # HURST-FIX: 375 meso bars (6/day) ≈ 62 days → SE ≈ 0.15 (too unreliable).
            # Enlarged to 700 bars (≈ 117 days) for SE ≈ 0.10 — minimum statistically
            # usable. Cost: larger burn-in (700 extra NaN bars at the start).
            df_meso["feat_hurst_meso"] = self.hurst_computer.compute_rolling(
                df_meso["close"].to_numpy(dtype=np.float64), window_size=700
            )
            df_meso["feat_stat_mom_meso"] = self.stat_computer.compute(
                df_meso["close"].to_numpy(dtype=np.float64)
            )

        if not df_macro.empty:
            df_macro = self.ta_engineer.add_features(df_macro, "macro")
            # HURST macro: 250 bars (1.2/day) ≈ 208 days — acceptable. Unchanged.
            df_macro["feat_hurst_macro"] = self.hurst_computer.compute_rolling(
                df_macro["close"].to_numpy(dtype=np.float64), window_size=250
            )
            df_macro["feat_stat_mom_macro"] = self.stat_computer.compute(
                df_macro["close"].to_numpy(dtype=np.float64)
            )

        # StructuralBreakRegime replaces HiddenMarkovRegime.
        # At 30 bars/day: ols_window=60 ≈ 2 days, min_bars=30 ≈ 1 day cooldown.
        sb_computer = StructuralBreakRegime(
            ols_window=60,
            threshold_sigma=3.0,
            min_bars_between_breaks=30,
        )
        # M5-FIX: column renamed from feat_structural_break (toggle) to
        # feat_bars_since_break (recency signal).
        bars_since_break_arr = sb_computer.compute_regimes(df_macro)
        df_macro["feat_bars_since_break"] = bars_since_break_arr

        # AUDIT-FIX (Issue 4 — HMM-style 3-regime supervisor):
        _regime_router = RegimeRouter()
        regime_ids = _regime_router.classify(df_macro, bars_since_break_arr)
        df_macro["feat_regime_id"] = regime_ids.astype(np.int8)
        soft = _regime_router.soft_probabilities(regime_ids, smoothing_bars=30)
        df_macro["feat_regime_p_lowvol"]      = soft[:, 0].astype(np.float32)
        df_macro["feat_regime_p_highvol"]     = soft[:, 1].astype(np.float32)
        df_macro["feat_regime_p_liquidchaos"] = soft[:, 2].astype(np.float32)

        # ── 3. Dynamic Column Detection ──────────────────────────────────────
        cols_micro = [c for c in df_micro.columns if str(c).startswith("feat_")]
        cols_meso  = [c for c in df_meso.columns  if str(c).startswith("feat_")]
        cols_macro = [c for c in df_macro.columns if str(c).startswith("feat_")]

        # ── 4. Merging ───────────────────────────────────────────────────────
        rename_meso  = {c: f"{c}_meso"  if not c.endswith("_meso")  else c for c in cols_meso}
        rename_macro = {c: f"{c}_macro" if not c.endswith("_macro") else c for c in cols_macro}

        df_meso  = df_meso.rename(columns=rename_meso)
        df_macro = df_macro.rename(columns=rename_macro)

        cols_meso_final  = list(rename_meso.values())
        cols_macro_final = list(rename_macro.values())

        if df_micro.empty:
            return np.array([]), np.array([]), np.array([]), pd.DataFrame()

        # FIX (Bar Fragmentation — issue #7): use strictly monotonically
        # increasing index via nanosecond offsets on duplicates.
        df_micro = self._disambiguate_duplicate_index(df_micro)
        df_meso  = self._disambiguate_duplicate_index(df_meso)
        df_macro = self._disambiguate_duplicate_index(df_macro)

        meso_tolerance  = pd.Timedelta(minutes=30)
        macro_tolerance = pd.Timedelta(days=2)

        df_merged = pd.merge_asof(
            df_micro,
            df_meso[cols_meso_final],
            left_index=True, right_index=True,
            direction="backward",
            tolerance=meso_tolerance,
        )
        df_merged = pd.merge_asof(
            df_merged,
            df_macro[cols_macro_final],
            left_index=True, right_index=True,
            direction="backward",
            tolerance=macro_tolerance,
        )

        # ── TRADFI DATA STALENESS FLAG ────────────────────────────────────────
        # AUDIT-FIX (N7 — binary→continuous tradfi staleness):
        # Continuous feat_macro_tradfi_minutes_stale = minutes until the next
        # TradFi liquidity window (13:00 UTC weekdays). Value 0 = market open,
        # >0 = N minutes until open. DST-robust.
        ts_idx = cast(pd.DatetimeIndex, df_merged.index)
        dow_np    = np.asarray(ts_idx.dayofweek, dtype=np.int64)
        hour_np   = np.asarray(ts_idx.hour,      dtype=np.int64)
        minute_np = np.asarray(ts_idx.minute,    dtype=np.int64)

        _OPEN_START_MIN = 13 * 60   # 780 min
        _OPEN_END_MIN   = 21 * 60   # 1260 min

        day_min = hour_np * 60 + minute_np
        is_tradfi_open = (dow_np < 5) & (day_min >= _OPEN_START_MIN) & (day_min < _OPEN_END_MIN)

        mins_until_open = np.where(
            is_tradfi_open,
            0,
            np.where(
                (dow_np < 5) & (day_min < _OPEN_START_MIN),
                _OPEN_START_MIN - day_min,
                np.where(
                    (dow_np < 5) & (day_min >= _OPEN_END_MIN),
                    np.where(
                        dow_np < 4,
                        (1440 - day_min) + _OPEN_START_MIN,
                        (1440 - day_min) + 2 * 1440 + _OPEN_START_MIN,
                    ),
                    (7 - dow_np) * 1440 - day_min + _OPEN_START_MIN,
                ),
            ),
        ).astype(np.float32)

        mins_until_open = np.clip(mins_until_open, 0.0, float(5 * 1440))

        df_merged["feat_macro_tradfi_minutes_stale"] = mins_until_open
        # Backward-compat binary alias
        df_merged["feat_macro_tradfi_stale"] = (mins_until_open > 0).astype(np.int8)

        if "feat_macro_tradfi_minutes_stale" not in cols_macro_final:
            cols_macro_final.append("feat_macro_tradfi_minutes_stale")
        if "feat_macro_tradfi_stale" in cols_macro_final:
            cols_macro_final.remove("feat_macro_tradfi_stale")

        # CHIEF AUDIT 2026-05-23 (P-9): expose a warmup indicator BEFORE the
        # blanket fillna(0.0).  Without this marker the model cannot tell a
        # genuine "zero" feature value from a NaN that was filled with 0.0
        # during warmup — the artificial "neutral" signal contaminates early
        # predictions.  ``feat_is_warmup`` = 1 on bars that had ≥1 NaN feature
        # prior to the fill, 0 otherwise.  Numerical fillna behaviour for
        # downstream CatBoost / PCA is preserved.
        df_merged["feat_is_warmup"] = (
            df_merged.isna().any(axis=1).astype(np.int8)
        )

        df_merged = df_merged.fillna(0.0)

        # Burn-in drop: drop first drop_bars so all rolling features are fully
        # warmed before the model sees data.
        # LEAKAGE-FIX: Hard bound — if the dataset is too small after drop,
        # refuse with a clear error.
        drop_bars = 2000
        # min_usable_bars: minimum runs-bar count accepted after partial burn-in.
        # Set to 100 (from 500) so illiquid symbols (AVAX: ~141 runs bars from
        # 180k 5s input at target_micro_bars=20) can produce live features with
        # a partial burn-in rather than returning empty and silencing all signals.
        # Training is unaffected: historical parquets always produce thousands of
        # runs bars and take the drop_bars=2000 else-branch unconditionally.
        min_usable_bars = 100

        if len(df_merged) <= drop_bars:
            n_remaining = len(df_merged)
            if n_remaining < min_usable_bars:
                logger.error(
                    "Dataset has only %d bars after bar generation — too small to "
                    "drop burn-in (%d) and keep enough data (min %d). Pipeline stopped.",
                    n_remaining, drop_bars, min_usable_bars,
                )
                return np.array([]), np.array([]), np.array([]), pd.DataFrame()
            else:
                actual_drop = n_remaining - min_usable_bars
                if actual_drop > 0:
                    df_merged = df_merged.iloc[actual_drop:].copy()
                logger.warning(
                    "Dataset (%d bars) smaller than burn-in window (%d). "
                    "%d bars dropped — %d usable bars remaining (partial burn-in).",
                    n_remaining, drop_bars, actual_drop, len(df_merged),
                )
        else:
            df_merged = df_merged.iloc[drop_bars:].copy()

        # ── 5. Matrix Extraction ─────────────────────────────────────────────
        X_micro = self._extract_matrix_fast(df_merged, cols_micro)
        X_meso  = self._extract_matrix_fast(df_merged, cols_meso_final)
        X_macro = self._extract_matrix_fast(df_merged, cols_macro_final)

        self._raw_cols_micro = list(cols_micro)
        self._raw_cols_meso  = list(cols_meso_final)
        self._raw_cols_macro = list(cols_macro_final)
        self._X_micro_raw    = X_micro.copy()
        self._X_meso_raw     = X_meso.copy()
        self._X_macro_raw    = X_macro.copy()

        # ── 6. PCA orthogonalisation per cluster (optional) ──────────────────
        # AUDIT C-3: the global ``fit_transform`` here leaks future information
        # because the orthogonalizer's calibration window touches data that
        # ends up in the CPCV test folds. The production training path bypasses
        # this entirely: ``tune.objective.optuna_objective_binary`` runs
        # ``FeatureOrthogonalizer.fit_on_train_indices(...)`` once per fold and
        # uses the raw matrices we exposed above. The pre-CV PCA below is kept
        # only for legacy notebooks / exploratory work and is gated by
        # ``cfg.feature_pipeline.use_pca_pre_cv`` (default False after Wave 13).
        use_pca_pre_cv = bool(
            getattr(self.cfg.feature_pipeline, "use_pca_pre_cv", False)
        )
        if use_pca and use_pca_pre_cv:
            logger.warning(
                "[%s] PCA fit_transform on full df_merged — AUDIT C-3 LEAKAGE "
                "RISK. Set cfg.feature_pipeline.use_pca_pre_cv=False and rely "
                "on the per-fold fit_on_train_indices path in tune/objective.",
                self.symbol,
            )
            if X_micro.shape[1] >= 2:
                X_micro, cols_micro = self.orthogonalizer_micro.fit_transform(
                    X_micro, cols_micro, prefix="micro_pc"
                )
            if X_meso.shape[1] >= 2:
                X_meso, cols_meso_final = self.orthogonalizer_meso.fit_transform(
                    X_meso, cols_meso_final, prefix="meso_pc"
                )
            if X_macro.shape[1] >= 2:
                X_macro, cols_macro_final = self.orthogonalizer_macro.fit_transform(
                    X_macro, cols_macro_final, prefix="macro_pc"
                )
            logger.info(
                "[%s] PCA orth (pre-CV, legacy): micro=%s meso=%s macro=%s",
                self.symbol, X_micro.shape, X_meso.shape, X_macro.shape,
            )
        elif use_pca:
            # Default Wave 13 path: emit raw matrices so the per-fold
            # orthogonalisation in tune/objective can run.
            logger.info(
                "[%s] Skipping pre-CV PCA (AUDIT C-3). Raw matrices passed "
                "to the per-fold orthogonalizers in tune.objective.",
                self.symbol,
            )

        return X_micro, X_meso, X_macro, df_merged

    def _extract_matrix_fast(self, df: pd.DataFrame, cols: list) -> np.ndarray:
        # DTYPE-FIX: return float64 for all financial time series.
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float64)
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return np.zeros((len(df), 0), dtype=np.float64)
        mat = df[valid_cols].values.astype(np.float64)
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _disambiguate_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
        """Add deterministic nanosecond offsets to duplicate timestamps.

        FIX (issue #7, Bar Fragmentation): When multiple VIBs close in the same
        second/microsecond it is unacceptable to use ``~duplicated(keep='last')``
        — that discards valid microstructure information. Instead we preserve all
        bars and offset duplicates with a cumcount number of nanoseconds so the
        index is strictly monotonic (required for ``pd.merge_asof``) without
        physically looking into the future.
        """
        if df.empty:
            return df

        df = df.sort_index(kind="mergesort")

        dup_mask = df.index.duplicated(keep=False)
        if not dup_mask.any():
            return df

        dti = cast(pd.DatetimeIndex, df.index)
        tz = dti.tz

        idx_series = pd.Series(dti, index=np.arange(len(df)))
        offsets_ns = idx_series.groupby(idx_series).cumcount().to_numpy(dtype=np.int64)

        base_ns = dti.astype(np.int64).to_numpy(dtype=np.int64, copy=True)
        new_i8 = base_ns + offsets_ns
        new_dti = cast(
            pd.DatetimeIndex,
            pd.to_datetime(new_i8.astype("datetime64[ns]")),
        )
        if tz is not None:
            new_dti = new_dti.tz_localize("UTC").tz_convert(tz)

        df = df.copy()
        df.index = new_dti
        return df


__all__ = [
    "FeaturePipeline",
    "HurstComputer",
    "RegimeRouter",
    "StationarityComputer",
    "StructuralBreakRegime",
    "_classify_3_regimes",
    "_cusum_structural_break",
    "_ewma_conditional_variance",
    "_rolling_ols_residuals",
    "_rolling_wls_residuals",
]
