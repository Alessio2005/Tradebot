# src/tradebot/backtest/metrics.py
"""Portfolio performance metrics — Sharpe, Calmar, MaxDD, Deflated Sharpe.

All functions are pure-NumPy; no pandas dependency.  For DataFrame
convenience use the ``from_series`` class methods on the result objects.

================================================================================
CHIEF AUDIT 2026-05-23 (H9): CANONICAL SHARPE CONVENTIONS

The Tradebot codebase historically had FOUR different ``bars_per_year``
defaults scattered around (``365*24``, ``8760``, ``365*24*12``,
``sqrt(365.25)``). To prevent drift, use the named constants below as the
canonical source of truth.

Conventions:
    - Bar-level returns:  bars_per_year = bars_per_day × CALENDAR_DAYS_PER_YEAR
    - Daily returns:      bars_per_year = CALENDAR_DAYS_PER_YEAR
    - Annualization:      multiply mean/std ratio by sqrt(bars_per_year)
    - Zero-padding:       ALWAYS include zero-return bars (calendar-time
                          Sharpe = live equity-curve Sharpe).

Examples:
    1h bars  → BARS_PER_DAY_HOURLY × CALENDAR_DAYS_PER_YEAR ≈ 8766
    5m bars  → BARS_PER_DAY_5M     × CALENDAR_DAYS_PER_YEAR ≈ 105_192
================================================================================
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "annualized_return",
    "annualized_vol",
    "bootstrap_ci",
    "calmar_ratio",
    "deflated_sharpe",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    # CHIEF AUDIT 2026-05-23 (H9): named canonical constants
    "CALENDAR_DAYS_PER_YEAR",
    "BARS_PER_DAY_HOURLY",
    "BARS_PER_DAY_5M",
]

# CHIEF AUDIT 2026-05-23 (H9): canonical Sharpe-annualisation constants.
CALENDAR_DAYS_PER_YEAR: float = 365.25
BARS_PER_DAY_HOURLY: int = 24
BARS_PER_DAY_5M: int = 24 * 12  # 5-minute bars per calendar day

_BARS_PER_YEAR_DEFAULT = 365 * 24  # hourly bars; override via bars_per_year kwarg


def annualized_return(returns: np.ndarray, bars_per_year: int = _BARS_PER_YEAR_DEFAULT) -> float:
    """Compound annualised return.

    Parameters
    ----------
    returns : 1-D array of bar-level returns (decimal, e.g. 0.002 = 0.2 %).
    bars_per_year : number of bars per calendar year.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 0.0
    compound = float(np.prod(1.0 + r))
    years = max(r.size / bars_per_year, 1e-9)
    return float(compound ** (1.0 / years) - 1.0)


def annualized_vol(returns: np.ndarray, bars_per_year: int = _BARS_PER_YEAR_DEFAULT) -> float:
    """Annualised standard deviation of bar-level returns."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * math.sqrt(bars_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    bars_per_year: int = _BARS_PER_YEAR_DEFAULT,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : bar-level returns.
    risk_free : annual risk-free rate (decimal).
    bars_per_year : bars per calendar year.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    rf_bar = (1.0 + risk_free) ** (1.0 / bars_per_year) - 1.0
    excess = r - rf_bar
    sigma = float(np.std(excess, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(excess) / sigma * math.sqrt(bars_per_year))


def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    bars_per_year: int = _BARS_PER_YEAR_DEFAULT,
) -> float:
    """Annualised Sortino ratio (downside deviation denominator)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    rf_bar = (1.0 + risk_free) ** (1.0 / bars_per_year) - 1.0
    excess = r - rf_bar
    downside = excess[excess < 0.0]
    if downside.size < 2:
        return 0.0
    semi_sigma = float(np.std(downside, ddof=1))
    if semi_sigma < 1e-12:
        return 0.0
    return float(np.mean(excess) / semi_sigma * math.sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray) -> tuple[float, int, int]:
    """Maximum drawdown of an equity curve.

    Parameters
    ----------
    equity : 1-D array of equity values (must be > 0).

    Returns
    -------
    (max_dd, peak_idx, trough_idx)
        max_dd      : maximum drawdown as a positive fraction (0.35 = 35 %).
        peak_idx    : index of the equity peak.
        trough_idx  : index of the trough after the peak.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return 0.0, 0, 0
    running_max = np.maximum.accumulate(eq)
    dd = (running_max - eq) / np.maximum(running_max, 1e-12)
    idx = int(np.argmax(dd))
    peak_idx = int(np.argmax(eq[:idx + 1]))
    return float(dd[idx]), peak_idx, idx


def calmar_ratio(
    equity: np.ndarray,
    bars_per_year: int = _BARS_PER_YEAR_DEFAULT,
) -> float:
    """Calmar ratio = annualised return / max drawdown."""
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    ann_ret = annualized_return(returns, bars_per_year=bars_per_year)
    mdd, _, _ = max_drawdown(eq)
    if mdd < 1e-9:
        return 0.0
    return float(ann_ret / mdd)


def deflated_sharpe(
    sr_observed: float,
    n_trials: int,
    n_obs: int,
    sr_benchmark: float = 0.0,
    returns_skew: float = 0.0,
    returns_kurt: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) — Bailey & López de Prado (2014).

    Adjusts the observed Sharpe for selection bias when N trials were run.

    Parameters
    ----------
    sr_observed : best Sharpe ratio found across ``n_trials`` trials.
    n_trials : number of strategy variants evaluated (including the winner).
    n_obs : number of out-of-sample observations.
    sr_benchmark : theoretical mean Sharpe under the null.
    returns_skew : third standardised moment of the returns distribution.
    returns_kurt : fourth standardised moment (3 = Gaussian).

    Returns
    -------
    float : DSR in (0, 1).
    """
    from scipy import stats  # lazy import — scipy is in core deps

    if n_trials < 2 or n_obs < 5:
        return 0.0

    # Expected maximum Sharpe under the null — Bailey & López de Prado (2014) eq. 8.
    # E[max(z₁…zₙ)] (in STANDARD-NORMAL units) =
    #     (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))
    # where γ = Euler-Mascheroni ≈ 0.5772 and e = Euler's number.
    # The previous single-term approximation underestimated this by 0.21-0.35 SR
    # units, making the DSR test anti-conservative (too many strategies passed).
    _gamma_em = 0.5772156649015328   # Euler-Mascheroni constant
    _e        = math.e
    e_max_z   = (
        (1.0 - _gamma_em) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + _gamma_em        * stats.norm.ppf(1.0 - 1.0 / (n_trials * _e))
    )

    # Variance of SR estimator (Mertens 2002).
    var_sr = (
        (1.0 - returns_skew * sr_observed + ((returns_kurt - 1) / 4.0) * sr_observed ** 2)
        / (n_obs - 1)
    )
    sigma_sr = math.sqrt(max(var_sr, 1e-12))

    # DIMENSIONAL-FIX (CHIEF 2026-06-08): ``e_max_z`` is the expected maximum of
    # N standard normals (≈1.67 for N=12) — it is in z/standard-error units, NOT
    # in Sharpe units.  Under H0 the trial Sharpes are ~N(0, σ_SR²), so the
    # expected-max benchmark Sharpe is SR* = σ_SR · e_max_z (Bailey-LdP).  The
    # prior code subtracted ``e_max_z`` directly from a per-observation Sharpe
    # (~0.07), making z ≈ −68 and DSR ≡ 0 for EVERY realistic strategy — a dead
    # gate.  Scaling by σ_SR restores z = SR̂/σ_SR − e_max_z = t_stat − e_max_z.
    sr_benchmark_max = sr_benchmark + e_max_z * sigma_sr
    z = (sr_observed - sr_benchmark_max) / sigma_sr
    return float(stats.norm.cdf(z))


def bootstrap_ci(
    returns: np.ndarray,
    stat_fn: object,
    n_bootstrap: int = 1_000,
    ci: float = 0.95,
    block_size: int = 50,
    seed: int = 42,
    auto_block_size: bool = False,
) -> tuple[float, float]:
    """Stationary block bootstrap confidence interval for a scalar statistic.

    Uses non-overlapping blocks of size ``block_size`` (simple variant of
    Politis & Romano 1994).

    Parameters
    ----------
    returns : bar-level returns array.
    stat_fn : callable(np.ndarray) -> float — the statistic to bootstrap.
    n_bootstrap : number of resamples.
    ci : confidence level (default 0.95).
    block_size : bootstrap block size.
    seed : RNG seed for reproducibility.
    auto_block_size :
        CHIEF AUDIT 2026-05-23 (M12): wanneer True, overschrijdt deze de
        ``block_size`` parameter en kiest een autocorr-gekalibreerde
        block-length via Politis-White heuristic (zelfde routine als
        ``tradebot.backtest.spa._autocorr_block_size``). Voor crypto-returns
        met 20-40 bar vol-clustering levert dit nauwkeuriger CIs op dan
        de hardcoded default van 50. Default False voor backward-compat.

    Returns
    -------
    (lower, upper) confidence interval.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]

    if auto_block_size and r.size >= 30:
        # Lazy-import om circular-deps te vermijden.
        from tradebot.backtest.spa import _autocorr_block_size
        block_size = int(_autocorr_block_size(r))

    if r.size < block_size * 2:
        stat = float(stat_fn(r))  # type: ignore[operator]
        return stat, stat

    n = r.size
    n_blocks = max(n // block_size, 1)

    stats_boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([r[s: s + block_size] for s in starts])[:n]
        stats_boot[i] = float(stat_fn(sample))  # type: ignore[operator]

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats_boot, alpha))
    hi = float(np.quantile(stats_boot, 1.0 - alpha))
    return lo, hi
