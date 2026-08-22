# src/tradebot/risk/var.py
"""Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall).

All estimators are parametric-free (historical simulation) and causal:
they use only data available up to the current bar.

Functions
---------
historical_var     : VaR from sorted historical returns.
historical_cvar    : CVaR (Expected Shortfall) = mean of tail beyond VaR.
rolling_var        : rolling VaR over a window.
rolling_cvar       : rolling CVaR over a window.
stress_test_var    : VaR scaled up by an extreme-event multiplier.
cornish_fisher_var : Cornish-Fisher adjusted VaR (skew/kurt correction, Wave 16).
evt_gpd_var        : EVT/GPD tail VaR via Peaks-Over-Threshold (Wave 16).
best_var           : Select best VaR estimator by sample size and method (Wave 16).
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.stats as _stats
from scipy.stats import genpareto as _genpareto

logger = logging.getLogger(__name__)

__all__ = [
    "best_var",
    "cornish_fisher_var",
    "evt_gpd_var",
    "historical_cvar",
    "historical_var",
    "rolling_cvar",
    "rolling_var",
    "rolling_var_causal",
    "stress_test_var",
]


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Historical simulation VaR.

    Parameters
    ----------
    returns : array of bar-level returns (negative = loss).
    confidence : e.g. 0.95 for 95 % VaR.

    Returns
    -------
    float : VaR as a positive loss threshold (e.g. 0.02 = 2 %).
        A return worse than -VaR occurs with probability (1 - confidence).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    return float(-np.quantile(r, 1.0 - confidence))


def historical_cvar(
    returns: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Historical simulation CVaR / Expected Shortfall.

    CVaR = mean of returns in the worst (1 - confidence) tail.

    Parameters
    ----------
    returns : bar-level returns.
    confidence : confidence level (e.g. 0.95).

    Returns
    -------
    float : CVaR as a positive loss (e.g. 0.04 = 4 %).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    var = float(-np.quantile(r, 1.0 - confidence))
    tail = r[r <= -var]
    if tail.size == 0:
        return var
    return float(-np.mean(tail))


def rolling_var(
    returns: np.ndarray,
    window: int,
    confidence: float = 0.95,
) -> np.ndarray:
    """Rolling historical VaR.

    Parameters
    ----------
    returns : bar-level returns, shape (T,).
    window : lookback window in bars.
    confidence : VaR confidence level.

    Returns
    -------
    np.ndarray, shape (T,) : VaR at each bar (NaN for the first window-1 bars).

    .. warning::
        CHIEF AUDIT 2026-05-23 (P-12): het window is INCLUSIEF de huidige
        bar (``r[i-window+1 : i+1]``). Voor backward-test scenarios waar
        ``out[i]`` als drempel wordt gebruikt om bar ``i`` te accepteren/
        weigeren ontstaat hierdoor een look-ahead: de return op bar ``i``
        zit in de VaR-schatter die voor die bar wordt gebruikt. Gebruik
        :func:`rolling_var_causal` voor strict bar-i-causale VaR.
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        out[i] = historical_var(r[i - window + 1 : i + 1], confidence=confidence)
    return out


def rolling_var_causal(
    returns: np.ndarray,
    window: int,
    confidence: float = 0.95,
) -> np.ndarray:
    """Strict causal rolling historical VaR.

    CHIEF AUDIT 2026-05-23 (P-12): variant van :func:`rolling_var` waarin
    bar ``i`` ALLEEN gebruik maakt van de voorgaande ``window`` bars
    (``r[i-window : i]``, exclusief bar ``i`` zelf). Geen look-ahead —
    veilig voor live-execution drempels en backtest-signaalgeneratie
    waar de VaR-schatting moet bestaan vóórdat bar ``i`` zich materialiseert.

    Parameters
    ----------
    returns : bar-level returns, shape (T,).
    window : lookback window in bars (exclusief huidige bar).
    confidence : VaR confidence level.

    Returns
    -------
    np.ndarray, shape (T,) : VaR at each bar (NaN voor de eerste ``window``
        bars, want er is dan onvoldoende historie ZONDER de huidige bar).
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window, n):
        out[i] = historical_var(r[i - window : i], confidence=confidence)
    return out


def rolling_cvar(
    returns: np.ndarray,
    window: int,
    confidence: float = 0.95,
) -> np.ndarray:
    """Rolling historical CVaR.

    Parameters
    ----------
    returns : bar-level returns, shape (T,).
    window : lookback window in bars.
    confidence : CVaR confidence level.

    Returns
    -------
    np.ndarray, shape (T,) : CVaR at each bar (NaN for the first window-1 bars).
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(window - 1, n):
        out[i] = historical_cvar(r[i - window + 1 : i + 1], confidence=confidence)
    return out


def cornish_fisher_var(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """Cornish-Fisher adjusted VaR (skew/kurt correction, Wave 16).

    Accounts for non-Gaussian crypto return distributions.
    CF expansion: q_cf = z + (z²-1)/6 * skew + (z³-3z)/24 * excess_kurt
                         - (2z³-5z)/36 * skew²
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return float(np.quantile(arr, 1.0 - confidence)) if len(arr) > 0 else 0.0

    z = float(_stats.norm.ppf(1.0 - confidence))
    skew = float(_stats.skew(arr))
    kurt_excess = float(_stats.kurtosis(arr))  # Fisher definition (excess)

    z_cf = (
        z
        + (z**2 - 1) / 6.0 * skew
        + (z**3 - 3*z) / 24.0 * kurt_excess
        - (2*z**3 - 5*z) / 36.0 * skew**2
    )
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    return float(mean + z_cf * std)


def evt_gpd_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_quantile: float = 0.90,
    seed: int = 42,
) -> float:
    """Extreme Value Theory VaR via Peaks-Over-Threshold + GPD fit (Wave 16).

    Fits Generalized Pareto Distribution to the tail beyond threshold.
    More accurate for extreme quantiles (99%+) on fat-tailed crypto returns.

    CHIEF AUDIT 2026-05-23 (P-11): scipy's ``genpareto.fit`` gebruikt
    L-BFGS-B met een ML-init die intern numpy.random raakt; zonder
    expliciete seed zijn fit-resultaten run-to-run niet bit-identiek.
    De ``seed``-parameter pinpoint de numpy global state direct vóór de
    fit-call, zodat reproduceerbare VaR-getallen ontstaan.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 50:
        return cornish_fisher_var(arr, confidence)

    u = float(np.quantile(arr, threshold_quantile))  # POT threshold
    tail = arr[arr < u] - u  # exceedances (negative side)

    if len(tail) < 10:
        return cornish_fisher_var(arr, confidence)

    try:
        # Fit GPD to losses (flip sign so tail > 0)
        losses = -tail  # now positive
        # CHIEF AUDIT 2026-05-23 (P-11): seed numpy global state vóór fit
        # voor deterministische L-BFGS-B initialisatie.
        np.random.seed(int(seed))
        shape, loc, scale = _genpareto.fit(losses, floc=0)
        # Quantile via GPD CDF inversion
        p_exceed = (1.0 - confidence) / (1.0 - threshold_quantile)
        if p_exceed <= 0 or p_exceed >= 1:
            return cornish_fisher_var(arr, confidence)
        if shape == 0:
            gpd_q = scale * np.log(1.0 / p_exceed)
        else:
            gpd_q = (scale / shape) * ((1.0 / p_exceed) ** shape - 1.0)
        return float(u - gpd_q)
    except Exception:
        return cornish_fisher_var(arr, confidence)


def best_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    method: str = "evt",
) -> float:
    """Select best VaR estimator based on sample size and method (Wave 16).

    method: 'historical' | 'cornish_fisher' | 'evt'
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if method == "evt" and n >= 50:
        return evt_gpd_var(arr, confidence)
    elif method == "cornish_fisher" and n >= 30:
        return cornish_fisher_var(arr, confidence)
    else:
        return float(np.quantile(arr, 1.0 - confidence)) if n > 0 else 0.0


def stress_test_var(
    var: float,
    stress_multiplier: float = 2.0,
    vol_ratio: float = 1.0,
) -> float:
    """Scale VaR by a stress multiplier for scenario analysis.

    Parameters
    ----------
    var : baseline VaR (positive loss fraction).
    stress_multiplier : baseline stress scale (e.g. 2.0 = 2× VaR).
    vol_ratio : current_vol / normal_vol; amplifies scaling in high-vol regimes.

    Returns
    -------
    float : stress-scenario VaR >= ``var``.
    """
    return float(var) * max(float(stress_multiplier) * max(float(vol_ratio), 1.0), 1.0)
