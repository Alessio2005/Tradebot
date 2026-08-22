# src/tradebot/alpha/mean_reversion.py
"""Ornstein-Uhlenbeck mean-reversion signal.

Estimates OU parameters (θ, μ, σ) via MLE on log-prices, then
generates a z-score entry signal.

Reference: Uhlenbeck & Ornstein (1930); trading application in
Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market".
"""
from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["OUMeanReversion"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _fit_ou_mle(log_prices: np.ndarray, dt: float = 1.0) -> tuple[float, float, float]:
    """Maximum-likelihood estimation of OU parameters.

    Discrete-time OU: X_{t+dt} = X_t * exp(-θ*dt) + μ*(1-exp(-θ*dt)) + ε

    Returns (θ, μ, σ) where:
        θ = mean-reversion speed (>0 → stationary)
        μ = long-run mean
        σ = diffusion coefficient (annualised)
    """
    x = log_prices
    n = len(x) - 1
    if n < 10:
        return 0.0, float(np.mean(x)), float(np.std(np.diff(x)))

    sx  = float(np.sum(x[:-1]))
    sy  = float(np.sum(x[1:]))
    sxx = float(np.sum(x[:-1] ** 2))
    sxy = float(np.sum(x[:-1] * x[1:]))
    syy = float(np.sum(x[1:] ** 2))

    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return 0.0, float(np.mean(x)), float(np.std(np.diff(x)))

    mu  = (sy * sxx - sx * sxy) / denom
    phi = (sxy - mu * sx - mu * sy + n * mu ** 2) / (sxx - 2 * mu * sx + n * mu ** 2)
    phi = float(np.clip(phi, 1e-6, 1.0 - 1e-9))

    theta = -float(np.log(phi)) / dt

    residuals = x[1:] - phi * x[:-1] - mu * (1 - phi)
    sigma_e = float(np.std(residuals))
    # Convert discrete residual σ to continuous OU σ
    sigma = sigma_e * float(np.sqrt(2 * theta / (1 - phi ** 2))) if phi < 1 - 1e-6 else sigma_e

    return float(max(theta, 0.0)), float(mu), float(max(sigma, 1e-9))


class OUMeanReversion:
    """OU mean-reversion signal via MLE parameter estimation.

    Parameters
    ----------
    symbol :
        Ticker name.
    fit_window :
        Number of bars used to estimate OU parameters.
    entry_zscore :
        z-score threshold beyond which the signal saturates at ±1.
    min_halflife_bars :
        Minimum half-life in bars.  Signals with half-life above this
        threshold are considered trending and are suppressed.
    """

    def __init__(
        self,
        symbol: str,
        fit_window: int = 126,
        entry_zscore: float = 2.0,
        min_halflife_bars: int = 5,
        max_halflife_bars: int = 63,
    ) -> None:
        self.symbol = symbol
        self.fit_window = fit_window
        self.entry_zscore = entry_zscore
        self.min_halflife_bars = min_halflife_bars
        self.max_halflife_bars = max_halflife_bars
        self.signal_id = (
            f"{symbol}_ou_"
            + _params_hash(fw=fit_window, ez=entry_zscore)
        )
        self._theta: float = 0.0
        self._mu: float = 0.0
        self._sigma_eq: float = 1.0

    def fit(self, df: pd.DataFrame) -> None:
        log_p = np.log(df["close"].values[-self.fit_window:])
        self._theta, self._mu, sigma = _fit_ou_mle(log_p)
        # Equilibrium σ of the OU process (std of stationary distribution)
        self._sigma_eq = sigma / (float(np.sqrt(2 * self._theta)) + 1e-9)
        self._sigma_eq = max(self._sigma_eq, 1e-6)

    def predict(self, df: pd.DataFrame) -> SignalResult:
        ts = df.index[-1]
        log_p_now = float(np.log(df["close"].iloc[-1]))

        if self._theta <= 0:
            return SignalResult(self.symbol, ts, 0.0, 0.5, 5, self.signal_id)

        halflife = float(np.log(2) / self._theta)
        if not (self.min_halflife_bars <= halflife <= self.max_halflife_bars):
            return SignalResult(self.symbol, ts, 0.0, 0.5, 5, self.signal_id)

        z = (log_p_now - self._mu) / self._sigma_eq
        # Negative z → below mean → long (expect reversion upward)
        raw = -float(np.tanh(z / self.entry_zscore))
        confidence = float(0.5 + 0.5 * raw)

        return SignalResult(
            symbol=self.symbol,
            timestamp=ts,
            signal=raw,
            confidence=confidence,
            horizon_bars=max(1, int(halflife)),
            signal_id=self.signal_id,
        )

    def feature_names(self) -> List[str]:
        return ["close"]
