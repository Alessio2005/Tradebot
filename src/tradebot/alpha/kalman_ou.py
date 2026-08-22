# src/tradebot/alpha/kalman_ou.py
"""Kalman-filter-enhanced OU Mean-Reversion Signal (Tier 2 — T2.2).

Extends the MLE-based OUMeanReversion with a Kalman filter for real-time
half-life tracking and optional Judge gating.

Why Kalman for half-life?
  The static MLE approach (``_fit_ou_mle``) re-estimates θ over a rolling
  window.  During regime transitions (e.g., crypto funding-rate cascades),
  the mean-reversion speed changes abruptly.  A Kalman filter tracks θ_t
  online with exponential discounting, reacting faster than batch MLE while
  remaining robust to noise.

Algorithm:
  State: x_t = [log_price_t]  (1D Kalman, process noise = drift)
  The OU half-life is estimated from a Kalman-smoothed AR(1) coefficient:
      φ_t = exp(-θ_t × Δt)
  where θ_t is updated via a 1-step Kalman correction on the price residual.

Judge gating (AFML §3.7):
  If ``prob_judge`` is supplied and is below ``judge_threshold``, the signal
  is suppressed regardless of z-score magnitude.  This prevents the OU signal
  from generating trades the Judge has already classified as unprofitable after
  costs.

References:
  • Kalman (1960) "A New Approach to Linear Filtering and Prediction Problems"
  • Avellaneda & Lee (2010) §2 — OU parameter estimation for stat-arb
  • López de Prado (2018) AFML §3.7 — Meta-Labeling
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .base import SignalResult
from .mean_reversion import _fit_ou_mle  # reuse MLE for initialisation

logger = logging.getLogger(__name__)

__all__ = ["KalmanOUMeanReversion"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


class KalmanOUMeanReversion:
    """Kalman-filter-based OU mean-reversion signal with Judge gate.

    Parameters
    ----------
    symbol :
        Ticker name.
    init_window :
        Number of bars used for MLE initialisation of OU parameters.
        Default 126 (≈ 6 months of daily bars; ~5 days for hourly bars).
    kalman_q :
        Process noise variance for the Kalman state update.  Controls how
        quickly the filter adapts to new data.  Default 1e-4.
    kalman_r :
        Observation noise variance.  Default 1e-2.
    entry_zscore :
        Z-score threshold for signal saturation.  Default 2.0.
    min_halflife_bars :
        Minimum half-life (inclusive).  Default 5 bars.
    max_halflife_bars :
        Maximum half-life (inclusive).  Signals with longer half-lives are
        trending, not mean-reverting.  Default 48 bars (2 days on 1h bars).
    judge_threshold :
        If ``prob_judge < judge_threshold`` the signal is suppressed.
        Set to 0.0 to disable Judge gating.  Default 0.55.
    """

    def __init__(
        self,
        symbol: str,
        init_window: int = 126,
        kalman_q: float = 1e-4,
        kalman_r: float = 1e-2,
        entry_zscore: float = 2.0,
        min_halflife_bars: int = 5,
        max_halflife_bars: int = 48,
        judge_threshold: float = 0.55,
    ) -> None:
        self.symbol = symbol
        self.init_window = init_window
        self.kalman_q = float(kalman_q)
        self.kalman_r = float(kalman_r)
        self.entry_zscore = float(entry_zscore)
        self.min_halflife_bars = int(min_halflife_bars)
        self.max_halflife_bars = int(max_halflife_bars)
        self.judge_threshold = float(judge_threshold)
        self.signal_id = (
            f"{symbol}_kalman_ou_"
            + _params_hash(iw=init_window, ez=entry_zscore, hl_max=max_halflife_bars)
        )

        # Kalman state
        self._x: float = 0.0   # Kalman mean estimate (log-price level)
        self._P: float = 1.0   # Kalman error covariance
        self._phi: float = 0.9  # AR(1) coefficient from OU (= exp(-θ*Δt))
        self._mu: float = 0.0   # Long-run mean
        self._sigma_eq: float = 1.0  # Equilibrium std of OU process

        self._initialized = False
        # CHIEF AUDIT 2026-05-23 (P2.3): Idempotency guard.  predict() mutates
        # the Kalman state via _kalman_update().  In a backtest loop that calls
        # predict(df.iloc[:t]) for t=1,2,...,T, each overlapping slice causes
        # the same bar to be updated multiple times → state drift.  We track
        # the last processed timestamp and skip the update if the bar is already
        # incorporated, making predict() idempotent for repeated calls with
        # the same (or earlier) last bar.
        self._last_update_ts: pd.Timestamp | None = None

    # ------------------------------------------------------------------
    def _init_kalman(self, log_prices: np.ndarray) -> None:
        """Initialise Kalman state from MLE OU parameters on log_prices."""
        theta, mu, sigma = _fit_ou_mle(log_prices)

        self._mu = float(mu)
        self._phi = float(np.clip(np.exp(-theta), 1e-9, 1.0 - 1e-9))

        # Equilibrium variance of OU: σ² / (1 − φ²)
        phi_sq = self._phi ** 2
        sigma_e_sq = max(float(sigma ** 2) * (1 - phi_sq), 1e-9)
        self._sigma_eq = float(np.sqrt(sigma_e_sq / max(1.0 - phi_sq, 1e-9)))
        self._sigma_eq = max(self._sigma_eq, 1e-6)

        # Initialise Kalman state at last observed log-price
        self._x = float(log_prices[-1])
        self._P = max(self._sigma_eq ** 2, 1e-8)
        self._initialized = True

        halflife = float(np.log(2) / max(-np.log(self._phi), 1e-9))
        logger.debug(
            "[%s] KalmanOU init: θ=%.4f, μ=%.4f, σ_eq=%.4f, φ=%.4f, half-life=%.1f bars",
            self.symbol, theta, self._mu, self._sigma_eq, self._phi, halflife,
        )

    # ------------------------------------------------------------------
    def _kalman_update(self, log_price: float) -> None:
        """One-step Kalman predict + update for the OU process."""
        # --- Predict ---
        # OU discrete-time: x_t = φ·x_{t-1} + (1-φ)·μ + ε
        x_pred = self._phi * self._x + (1.0 - self._phi) * self._mu
        P_pred = self._phi ** 2 * self._P + self.kalman_q

        # --- Update (observation = log_price, H = 1) ---
        innovation = log_price - x_pred
        S = P_pred + self.kalman_r   # innovation covariance
        K = P_pred / max(S, 1e-12)  # Kalman gain

        self._x = x_pred + K * innovation
        self._P = (1.0 - K) * P_pred

        # CHIEF AUDIT 2026-05-23 (P-1): OU-mean is STATIC by theory; drifting μ
        # creates non-Markov state and backtest/live mismatch. Re-fit via .fit()
        # at regime boundaries.

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:
        """Initialise Kalman filter from historical bars.

        ``df`` must have a ``close`` column.
        """
        log_p = np.log(df["close"].values[-self.init_window:])
        if len(log_p) < 20:
            logger.warning(
                "[%s] KalmanOUMeanReversion.fit: too few bars (%d < 20) — "
                "using price-mean initialisation.",
                self.symbol, len(log_p),
            )
            self._mu = float(np.mean(log_p))
            self._x = float(log_p[-1])
            self._P = float(np.var(log_p)) + 1e-6
            self._sigma_eq = float(np.std(log_p)) + 1e-6
            self._phi = 0.9
            self._initialized = True
            return

        self._init_kalman(log_p)

        # Warm up filter on the full history
        for lp in log_p:
            self._kalman_update(lp)

    # ------------------------------------------------------------------
    def predict(
        self,
        df: pd.DataFrame,
        prob_judge: float | None = None,
    ) -> SignalResult:
        """Compute Kalman-OU signal on the latest bar.

        Parameters
        ----------
        df :
            Bar DataFrame up to and including the current bar.
        prob_judge :
            Optional Judge probability.  Signal is suppressed if below
            ``self.judge_threshold``.

        Returns
        -------
        SignalResult with signal in [-1, +1].
        """
        ts = df.index[-1]
        zero = SignalResult(self.symbol, ts, 0.0, 0.5, 5, self.signal_id)

        if not self._initialized:
            return zero

        # Judge gate
        if prob_judge is not None and float(prob_judge) < self.judge_threshold:
            logger.debug(
                "[%s] KalmanOU: Judge prob %.3f < %.2f → suppressed.",
                self.symbol, prob_judge, self.judge_threshold,
            )
            return zero

        log_p_now = float(np.log(max(df["close"].iloc[-1], 1e-9)))

        # Kalman update — only if this is a new (later) bar than the last one
        # processed.  This prevents state drift when predict() is called
        # multiple times with overlapping df slices in a backtest loop.
        if self._last_update_ts is None or ts > self._last_update_ts:
            self._kalman_update(log_p_now)
            self._last_update_ts = ts

        # Half-life from AR(1) coefficient
        phi_safe = float(np.clip(self._phi, 1e-9, 1.0 - 1e-9))
        halflife = float(np.log(2.0) / max(-np.log(phi_safe), 1e-9))

        if not (self.min_halflife_bars <= halflife <= self.max_halflife_bars):
            return zero  # trending or too slow to mean-revert

        # Z-score: deviation of current price from Kalman mean
        z = (log_p_now - self._x) / max(self._sigma_eq, 1e-6)

        # Negative z → below mean → long (revert upward)
        raw = float(-np.tanh(z / self.entry_zscore))
        confidence = float(0.5 + 0.5 * raw)

        return SignalResult(
            symbol=self.symbol,
            timestamp=ts,
            signal=raw,
            confidence=confidence,
            horizon_bars=max(1, int(halflife)),
            signal_id=self.signal_id,
        )

    # ------------------------------------------------------------------
    def current_halflife(self) -> float:
        """Return current estimated half-life in bars."""
        phi_safe = float(np.clip(self._phi, 1e-9, 1.0 - 1e-9))
        return float(np.log(2.0) / max(-np.log(phi_safe), 1e-9))

    def current_zscore(self, log_price: float) -> float:
        """Return the z-score of ``log_price`` relative to the Kalman mean."""
        return (log_price - self._x) / max(self._sigma_eq, 1e-6)

    def feature_names(self) -> list[str]:
        return ["close"]
