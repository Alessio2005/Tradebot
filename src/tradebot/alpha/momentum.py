# src/tradebot/alpha/momentum.py
"""Time-series and cross-sectional momentum signals (Moskowitz et al. 2012).

TSMomentum  : 12-1 log-return momentum with Jegadeesh-Titman skip-month.
CSMomentum  : Rank-normalised cross-sectional momentum within a universe.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["TSMomentum", "CSMomentum"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


class TSMomentum:
    """Time-series momentum: 12-month minus 1-month log return.

    Implements the Jegadeesh-Titman (1993) skip-month correction:
    signal = log(P[t-1] / P[t-lookback]) — excludes the most-recent month
    to avoid microstructure-driven mean reversion.

    Parameters
    ----------
    symbol :
        Ticker name (embedded in signal_id).
    lookback_bars :
        Total lookback window in bars (default 252 ≈ 12 months daily).
    skip_bars :
        Number of bars to skip at the end (default 21 ≈ 1 month daily).
    vol_scale :
        If True, scale the signal by annualised volatility (TSMOM variant
        from Moskowitz et al. 2012 — uses ex-ante vol to normalise).
    """

    def __init__(
        self,
        symbol: str,
        lookback_bars: int = 252,
        skip_bars: int = 21,
        vol_scale: bool = True,
    ) -> None:
        self.symbol = symbol
        self.lookback_bars = lookback_bars
        self.skip_bars = skip_bars
        self.vol_scale = vol_scale
        self.signal_id = (
            f"{symbol}_tsmom_"
            + _params_hash(lb=lookback_bars, sk=skip_bars, vs=vol_scale)
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self._df = df[["close"]].copy()
        self._fitted = True

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        """Compute time-series momentum signal.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``close`` column.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-3): Number of trailing bars to exclude
            from the realised-vol computation.  Without this exclusion the
            most-recent return ends on bar T while the signal is timestamped
            at T → trade on bar T uses information from bar T (same-bar
            leakage).  Default 1 = causal-safe (signal at T uses returns
            ending at T-1, so trade can enter at T).
        """
        prices = df["close"]
        n = len(prices)
        needed = self.lookback_bars + 1

        if n < needed:
            return SignalResult(
                symbol=self.symbol,
                timestamp=prices.index[-1],
                signal=0.0,
                confidence=0.5,
                horizon_bars=21,
                signal_id=self.signal_id,
            )

        # 12-1 log return: price at (t - skip_bars) / price at (t - lookback_bars)
        p_now  = float(prices.iloc[-(self.skip_bars + 1)])
        p_past = float(prices.iloc[-(self.lookback_bars + 1)])
        if p_past <= 0:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)

        raw_signal = float(np.log(p_now / p_past))

        if self.vol_scale:
            # CHIEF AUDIT 2026-05-23 (P-3): drop the last ``lag_bars`` returns
            # so realised-vol is computed strictly on bars ≤ T - lag_bars.
            lag = max(0, int(lag_bars))
            if lag > 0 and len(prices.values) > lag + 1:
                log_rets = np.log(prices.values[1:-lag] / prices.values[:-1-lag])
            else:
                log_rets = np.log(prices.values[1:] / prices.values[:-1])
            if len(log_rets) == 0:
                ann_vol = 1e-9
            else:
                ann_vol = float(np.std(log_rets[-63:]) * np.sqrt(252)) + 1e-9
            raw_signal /= ann_vol

        # Map to [-1, +1] via tanh with scale 1.0
        signal = float(np.tanh(raw_signal))
        confidence = float(0.5 + 0.5 * signal)

        return SignalResult(
            symbol=self.symbol,
            timestamp=prices.index[-1],
            signal=signal,
            confidence=confidence,
            horizon_bars=21,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> list[str]:
        return ["close"]


class CSMomentum:
    """Cross-sectional rank momentum within a multi-asset universe.

    For use when running across multiple symbols simultaneously.
    Each instance holds one symbol; an external combiner must rank
    signals across the universe.

    Implements the Asness et al. (1997) cross-sectional standardisation:
    z-score of rolling n-bar return within the universe.
    """

    def __init__(
        self,
        symbol: str,
        lookback_bars: int = 126,
    ) -> None:
        self.symbol = symbol
        self.lookback_bars = lookback_bars
        self.signal_id = (
            f"{symbol}_csmom_" + _params_hash(lb=lookback_bars)
        )

    def fit(self, df: pd.DataFrame) -> None:
        self._df = df[["close"]].copy()

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        """Compute cross-sectional momentum signal.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``close`` column.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-3): Number of trailing bars to exclude
            from the return computation so signal_t uses returns ending at
            T - lag_bars (causal-safe; signal at T can be traded at T).
        """
        prices = df["close"]
        lag = max(0, int(lag_bars))
        if len(prices) < self.lookback_bars + 1 + lag:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)

        # CHIEF AUDIT 2026-05-23 (P-3): anchor the latest "now" price at
        # T - lag_bars rather than T to prevent same-bar leakage.
        if lag > 0:
            p_now = float(prices.iloc[-(1 + lag)])
            p_past = float(prices.iloc[-(self.lookback_bars + 1 + lag)])
        else:
            p_now = float(prices.iloc[-1])
            p_past = float(prices.iloc[-(self.lookback_bars + 1)])
        if p_past <= 0:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)
        ret = float(np.log(p_now / p_past))

        # Without a full universe, normalise against own rolling distribution.
        # CHIEF AUDIT 2026-05-23 (P-3): also drop the last ``lag`` bars when
        # building the calibration window so it stays strictly historical.
        if lag > 0 and len(prices.values) > lag:
            prices_for_window = prices.values[:-lag]
        else:
            prices_for_window = prices.values
        if len(prices_for_window) <= self.lookback_bars:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)
        window_rets = np.array([
            np.log(prices_for_window[i] / prices_for_window[i - self.lookback_bars])
            for i in range(self.lookback_bars, len(prices_for_window))
        ])
        if len(window_rets) < 5:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)

        mu, sigma = float(np.mean(window_rets)), float(np.std(window_rets)) + 1e-9
        z = (ret - mu) / sigma
        signal = float(np.tanh(z / 2.0))
        confidence = float(0.5 + 0.5 * signal)

        return SignalResult(
            symbol=self.symbol,
            timestamp=prices.index[-1],
            signal=signal,
            confidence=confidence,
            horizon_bars=21,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> list[str]:
        return ["close"]
