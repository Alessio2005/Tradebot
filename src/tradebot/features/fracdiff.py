"""Fractional Differentiation (AUDIT A-1) — AFML chapter 5.

The price level series `P_t` is I(1): naive differencing destroys memory while
keeping the model trainable. Fractional differencing applies the operator
`(1 - L)^d` with `d ∈ (0, 1)` via an FFD (Fixed-Window Fractional Differentiation)
expansion truncated when the weights drop below `threshold` in absolute value.

This module exposes:

- `frac_diff_ffd(series, d, threshold)`   — apply FFD with given d.
- `min_frac_diff(series, ...)`             — binary-search for the smallest d for
                                             which the Augmented Dickey-Fuller
                                             test rejects the unit root.
- `MinFracDiff`                            — sklearn-style transformer that fits
                                             one d per column and produces the
                                             stationary version.

Important — causality (R-1):
  Because FFD is a *one-sided* convolution `y_t = Σ w_k * x_{t-k}` with k ≥ 0,
  every output value uses only past prices. This is the only fractional-diff
  variant that survives the causal gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------
# CHIEF AUDIT-FIX (Sim-to-Reality #3):
#   FFD weight truncation at threshold=1e-4 was producing residual-mass bias.
#   For d=0.4, weights at the truncation point still carry ~0.5-1% of the
#   convolution mass — meaning the output series silently drops a non-trivial
#   chunk of the memory operator's tail.  This biases the price-level features
#   towards a slightly more-differenced version than declared, which:
#     1. Inflates ADF rejection probability (false-positive stationarity)
#     2. Reduces the effective d* found by min_frac_diff() by ~0.05-0.10
#     3. Creates a subtle non-causal-looking artifact at the truncation seam
#   Fix: lower default threshold to 1e-5 (residual mass < 0.05%) and log the
#   cumulative-residual diagnostic so operators can detect truncation bias.
FFD_DEFAULT_THRESHOLD: float = 1e-5
# Maximum acceptable residual-mass before we warn (sum of dropped weight magnitudes).
FFD_RESIDUAL_MASS_WARN: float = 0.01  # 1% of total weight magnitude


def _ffd_weights(
    d: float,
    threshold: float,
    return_residual: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Build the FFD-weight vector (AFML 5.3.2).

    Returns weights in *time-reversed* order: w[0] is the multiplier on
    ``x_t``, w[1] on ``x_{t-1}``, etc. We stop when ``|w[k]| < threshold``.

    Parameters
    ----------
    d : fractional differentiation parameter (0,1).
    threshold : truncation threshold on |w[k]|.
    return_residual : if True, return (weights, residual_mass_estimate)
                      where residual_mass_estimate is an upper bound on
                      the L1-mass of the tail that was dropped relative
                      to the L1-mass of the retained weights.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, float]
        FFD weights (and optional residual-mass estimate).
    """
    if d <= 0.0 or d >= 1.0:
        raise ValueError(f"d must lie in (0, 1); got {d}")
    if threshold <= 0.0:
        raise ValueError("threshold must be > 0")

    weights = [1.0]
    k = 1
    truncation_w: float = 0.0
    # Hard ceiling to avoid runaway expansion for very small thresholds.
    while k < 100_000:
        w = -weights[-1] * (d - (k - 1)) / k
        if abs(w) < threshold:
            truncation_w = abs(w)
            break
        weights.append(w)
        k += 1

    w_arr = np.asarray(weights, dtype=np.float64)
    if not return_residual:
        return w_arr

    # Residual-mass upper bound: FFD weights for d ∈ (0,1) decay as |w_k| ~ k^{-(d+1)}.
    # After truncation at |w_K| ≈ threshold, the tail mass is bounded by
    # threshold × K / d  (geometric majorant). Normalised by retained L1-mass.
    retained_mass = float(np.abs(w_arr).sum())
    tail_upper_bound = truncation_w * max(len(w_arr), 1) / max(d, 1e-3)
    residual_ratio = tail_upper_bound / max(retained_mass, 1e-12)
    if residual_ratio > FFD_RESIDUAL_MASS_WARN:
        logger.debug(
            "FFD weights (d=%.3f, threshold=%.1e): residual-mass ratio "
            "%.3f%% exceeds %.1f%% — truncation may bias the convolution. "
            "Lower threshold or raise weight cap.",
            d, threshold, residual_ratio * 100.0,
            FFD_RESIDUAL_MASS_WARN * 100.0,
        )
    return w_arr, residual_ratio


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = FFD_DEFAULT_THRESHOLD,
) -> pd.Series:
    """Apply FFD operator with parameter ``d`` to ``series``.

    Output index = ``series.index[L-1:]`` where ``L = len(weights)``. The first
    ``L-1`` rows are dropped because they lack enough history for the
    convolution. The function is strictly causal: row ``t`` consumes only
    rows ``[t - L + 1, t]``.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"frac_diff_ffd expects a pd.Series, got {type(series)}")
    # CHIEF AUDIT-FIX (Sim-to-Reality #3): always compute residual-mass diagnostic
    # so truncation bias is surfaced via the module logger.
    w_result = _ffd_weights(d, threshold, return_residual=True)
    if isinstance(w_result, tuple):
        w, _residual = w_result
    else:
        w = w_result
    L = len(w)
    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    if n < L:
        return pd.Series([], dtype=np.float64, name=series.name)

    # Vectorised one-sided convolution. `np.convolve` reverses internally; we
    # invert the weight order so the output is causal.
    out = np.convolve(values, w[::-1], mode="valid")
    idx = series.index[L - 1 :]
    return pd.Series(out, index=idx, name=series.name)


# ---------------------------------------------------------------------------
# Stationarity-driven d* search
# ---------------------------------------------------------------------------
@dataclass
class MinFracDiffResult:
    d: float
    pvalue: float
    n_used_obs: int
    threshold: float
    converged: bool


def min_frac_diff(
    series: pd.Series,
    d_lo: float = 0.05,
    d_hi: float = 0.95,
    threshold: float = FFD_DEFAULT_THRESHOLD,
    p_target: float = 0.05,
    max_iter: int = 24,
) -> MinFracDiffResult:
    """Binary-search the **smallest** ``d`` whose ADF test rejects unit root.

    The smaller ``d``, the more memory preserved in the output series, which
    is the goal of FFD (AFML §5.4). Procedure:

    1. Check ``d_hi`` — if even that does not reject H0, no valid d in
       ``[d_lo, d_hi]``; return ``d_hi`` with the actual p-value.
    2. Check ``d_lo`` — if it rejects already, return ``d_lo``.
    3. Bisect ``[d_lo, d_hi]`` until the gap is below ``2**-max_iter`` or the
       smallest rejecting d is found.

    Requires ``statsmodels``. Gracefully degrades to ``d = 0.5`` if statsmodels
    is unavailable, with a warning.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning(
            "statsmodels not installed; min_frac_diff falling back to d=0.5"
        )
        return MinFracDiffResult(d=0.5, pvalue=float("nan"),
                                 n_used_obs=len(series), threshold=threshold,
                                 converged=False)

    def adf_p(d: float) -> tuple[float, int]:
        y = frac_diff_ffd(series, d=d, threshold=threshold).dropna()
        if len(y) < 50:
            return 1.0, len(y)
        try:
            stat, p, *_ = adfuller(y, maxlag=10, regression="c", autolag=None)
            return float(p), len(y)
        except (ValueError, np.linalg.LinAlgError):
            return 1.0, len(y)

    p_hi, n_hi = adf_p(d_hi)
    if p_hi > p_target:
        # Even maximal differentiation cannot reach stationarity (rare).
        return MinFracDiffResult(d=d_hi, pvalue=p_hi, n_used_obs=n_hi,
                                 threshold=threshold, converged=False)
    p_lo, n_lo = adf_p(d_lo)
    if p_lo <= p_target:
        return MinFracDiffResult(d=d_lo, pvalue=p_lo, n_used_obs=n_lo,
                                 threshold=threshold, converged=True)

    lo, hi = d_lo, d_hi
    best_d, best_p, best_n = d_hi, p_hi, n_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid, n_mid = adf_p(mid)
        if p_mid <= p_target:
            best_d, best_p, best_n = mid, p_mid, n_mid
            hi = mid
        else:
            lo = mid
        if (hi - lo) < 2 ** -max_iter:
            break
    return MinFracDiffResult(d=best_d, pvalue=best_p, n_used_obs=best_n,
                             threshold=threshold, converged=True)


def causal_min_frac_diff(
    series: pd.Series,
    train_end_iloc: int,
    d_lo: float = 0.1,
    d_hi: float = 0.9,
    step: float = 0.05,
    alpha: float = 0.05,
    threshold: float = 1e-5,
) -> float:
    """Bereken minimale d* ALLEEN op train-data — geen lookahead (v3 T0.2).

    In tegenstelling tot min_frac_diff() dat d* over de gehele serie berekent,
    gebruikt deze functie uitsluitend series.iloc[:train_end_iloc].

    Args:
        series:         Tijdserie.
        train_end_iloc: iloc-positie van het laatste train-element (exclusief).
        d_lo, d_hi:     Zoekruimte.
        step:           Stapgrootte (gebruikt als grid als binaire search ontbreekt).
        alpha:          ADF significantieniveau.
        threshold:      FFD convolutie-truncatiedrempel.

    Returns:
        d_star (float). Geeft d_hi terug bij onvoldoende data of falen.
    """
    train_series = series.iloc[:train_end_iloc]
    if len(train_series) < 50:
        return d_hi

    try:
        return min_frac_diff(
            train_series,
            d_lo=d_lo,
            d_hi=d_hi,
            threshold=threshold,
            p_target=alpha,
        ).d
    except Exception:
        return d_hi


# ---------------------------------------------------------------------------
# sklearn-style transformer (per-column d*)
# ---------------------------------------------------------------------------
@dataclass
class MinFracDiff:
    """Per-column fractional-diff transformer with d* learned from ADF gate.

    Fit on a training slice, then transform any (longer) slice with the same
    weight vector. Each column gets its own d*; columns whose price-level
    interpretation does not apply (returns, z-scores, etc.) can be excluded
    via the ``exclude`` argument.
    """
    threshold: float = FFD_DEFAULT_THRESHOLD
    p_target: float = 0.05
    d_lo: float = 0.05
    d_hi: float = 0.95
    exclude: Iterable[str] = field(default_factory=tuple)

    d_per_col_: dict[str, float] = field(default_factory=dict)
    p_per_col_: dict[str, float] = field(default_factory=dict)
    is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "MinFracDiff":
        for col in df.columns:
            if col in self.exclude:
                continue
            series = df[col].astype(np.float64).dropna()
            if len(series) < 200:
                logger.debug("Skipping FFD for %s: only %d obs", col, len(series))
                continue
            result = min_frac_diff(
                series,
                d_lo=self.d_lo,
                d_hi=self.d_hi,
                threshold=self.threshold,
                p_target=self.p_target,
            )
            self.d_per_col_[col] = result.d
            self.p_per_col_[col] = result.pvalue
            logger.info(
                "FFD %s: d*=%.4f p=%.4f obs=%d converged=%s",
                col, result.d, result.pvalue, result.n_used_obs, result.converged,
            )
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("MinFracDiff.transform called before fit().")
        out = {}
        for col, d in self.d_per_col_.items():
            if col not in df.columns:
                continue
            out[f"ffd_{col}_d{d:.3f}"] = frac_diff_ffd(
                df[col].astype(np.float64), d=d, threshold=self.threshold,
            )
        if not out:
            return pd.DataFrame(index=df.index)
        result = pd.concat(out, axis=1)
        result.index = pd.DatetimeIndex(result.index) if not isinstance(
            result.index, pd.DatetimeIndex
        ) else result.index
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


__all__ = [
    "frac_diff_ffd",
    "min_frac_diff",
    "causal_min_frac_diff",
    "MinFracDiff",
    "MinFracDiffResult",
]
