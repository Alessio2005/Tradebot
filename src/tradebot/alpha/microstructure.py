# src/tradebot/alpha/microstructure.py
"""Order-flow imbalance alpha signal (Cont, Kukanov & Stoikov, 2014).

Wraps the ``features.microstructure`` functions to produce a normalised
AlphaSignal-compatible OFI signal.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["OFISignal"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


class OFISignal:
    """Order-Flow Imbalance alpha signal.

    Cumulates taker buy/sell volume over a rolling window, normalises,
    and maps to [-1, +1].  The signal is predictive of short-term
    price impact (Cont et al. 2014).

    Parameters
    ----------
    symbol :
        Ticker name.
    window :
        Rolling window in bars for OFI smoothing.
    scale :
        Tanh steepness parameter.  Higher = more binary signal.
    """

    def __init__(
        self,
        symbol: str,
        window: int = 20,
        scale: float = 3.0,
    ) -> None:
        self.symbol = symbol
        self.window = window
        self.scale = scale
        self.signal_id = (
            f"{symbol}_ofi_" + _params_hash(w=window, sc=scale)
        )

    def fit(self, df: pd.DataFrame) -> None:
        pass  # rolling stateless

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        """Compute OFI signal.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``taker_buy_volume`` and ``taker_sell_volume``.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-5): Number of bars to lag the FEATURE
            value while keeping the SIGNAL timestamp at ``df.index[-1]`` (=
            bar T).  This guarantees the OFI used at bar T is computed only
            from bars ≤ T - lag_bars and is therefore safe to trade on bar T.
        """
        ts = df.index[-1]

        if "taker_buy_volume" not in df.columns or "taker_sell_volume" not in df.columns:
            return SignalResult(self.symbol, ts, 0.0, 0.5, 1, self.signal_id)

        buy  = df["taker_buy_volume"].fillna(0.0)
        sell = df["taker_sell_volume"].fillna(0.0)
        tot  = buy + sell
        raw  = (buy - sell) / tot.replace(0, 1e-9)

        # CHIEF AUDIT 2026-05-23 (P-5): pull the feature value from
        # df.iloc[-(1+lag_bars)] so it is computed strictly on past bars.
        lag = max(0, int(lag_bars))
        smoothed = raw.rolling(self.window, min_periods=1).mean()
        if lag >= len(smoothed):
            return SignalResult(self.symbol, ts, 0.0, 0.5, 1, self.signal_id)
        ofi = float(smoothed.iloc[-(1 + lag)])
        signal = float(np.tanh(ofi * self.scale))
        confidence = float(0.5 + 0.5 * signal)

        return SignalResult(
            symbol=self.symbol,
            timestamp=ts,
            signal=signal,
            confidence=confidence,
            horizon_bars=1,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> list[str]:
        return ["taker_buy_volume", "taker_sell_volume"]
