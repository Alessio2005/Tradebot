# src/tradebot/alpha/carry.py
"""Funding-rate carry signal (crypto-specific).

Annualised funding rate vs volatility-adjusted threshold.

Positive funding → perpetual premium → long basis carry (short perp / long spot).
Negative funding → perpetual discount → short basis carry.

Reference: crypto carry literature; Gorton & Rouwenhorst (2006) commodities carry
adapted for perpetual futures funding rates.
"""
from __future__ import annotations

import hashlib
import logging
from typing import List

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["FundingCarry"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


class FundingCarry:
    """Funding-rate carry signal.

    Parameters
    ----------
    symbol :
        Ticker name.
    funding_col :
        Column name for the funding rate.  Must be in the input DataFrame.
        Expected units: 8-hourly funding rate (e.g. 0.0001 = 0.01%).
    payments_per_year :
        Number of funding payments per year (default 3 × 365 = 1095 for
        8-hourly Bybit schedule).
    vol_window :
        Rolling window for realised volatility (used as denominator to
        compute carry-to-vol ratio).
    min_carry_vol_ratio :
        Minimum |carry / vol| below which signal is suppressed (noise floor).
    """

    def __init__(
        self,
        symbol: str,
        funding_col: str = "funding_rate",
        payments_per_year: int = 1095,
        vol_window: int = 21,
        min_carry_vol_ratio: float = 0.05,
    ) -> None:
        self.symbol = symbol
        self.funding_col = funding_col
        self.payments_per_year = payments_per_year
        self.vol_window = vol_window
        self.min_carry_vol_ratio = min_carry_vol_ratio
        self.signal_id = (
            f"{symbol}_carry_"
            + _params_hash(fc=funding_col, ppy=payments_per_year)
        )

    def fit(self, df: pd.DataFrame) -> None:
        pass  # stateless signal — no fit required

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        """Compute funding-carry signal.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``close`` and ``self.funding_col``.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-5): Number of bars to lag the FEATURE
            value while keeping the SIGNAL timestamp at ``df.index[-1]``.
            Without this lag the funding rate observed AT bar T is used to
            place a trade AT bar T — same-bar leakage.
        """
        ts = df.index[-1]

        if self.funding_col not in df.columns:
            return SignalResult(self.symbol, ts, 0.0, 0.5, 8, self.signal_id)

        # CHIEF AUDIT 2026-05-23 (P-5): pull funding and the realised-vol
        # numerator from df.iloc[-(1+lag_bars)] (and earlier) so the signal
        # at bar T uses only bars ≤ T - lag_bars.
        lag = max(0, int(lag_bars))
        funding_series = df[self.funding_col]
        if lag >= len(funding_series):
            return SignalResult(self.symbol, ts, 0.0, 0.5, 8, self.signal_id)
        funding_8h = float(funding_series.iloc[-(1 + lag)])
        funding_ann = funding_8h * self.payments_per_year

        # Realised vol (annualised) using only bars ≤ T - lag_bars.
        close_arr = df["close"].values
        if lag > 0 and len(close_arr) > lag:
            close_for_vol = close_arr[:-lag]
        else:
            close_for_vol = close_arr
        if len(close_for_vol) < 2:
            return SignalResult(self.symbol, ts, 0.0, 0.5, 8, self.signal_id)
        log_rets = np.log(close_for_vol[1:] / close_for_vol[:-1])
        if len(log_rets) < self.vol_window:
            return SignalResult(self.symbol, ts, 0.0, 0.5, 8, self.signal_id)

        ann_vol = float(np.std(log_rets[-self.vol_window:]) * np.sqrt(252)) + 1e-9
        carry_vol_ratio = funding_ann / ann_vol

        if abs(carry_vol_ratio) < self.min_carry_vol_ratio:
            return SignalResult(self.symbol, ts, 0.0, 0.5, 8, self.signal_id)

        # Positive funding → perp overpriced vs spot → carry benefits from being
        # short the perp (negative signal on perp direction).
        raw = float(-np.tanh(carry_vol_ratio * 3.0))
        confidence = float(0.5 + 0.5 * raw)

        return SignalResult(
            symbol=self.symbol,
            timestamp=ts,
            signal=raw,
            confidence=confidence,
            horizon_bars=8,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> List[str]:
        return ["close", self.funding_col]
