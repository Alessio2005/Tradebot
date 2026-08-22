# src/tradebot/alpha/macro_regime.py
"""Macro-regime overlay — dampens other signals in risk-off environments.

Classifies the current macro regime as risk-on / risk-off based on a
composite of volatility, momentum, and macro indicators.  The overlay
multiplies other signal strengths by a regime multiplier in [0, 1].

Reference: López de Prado (2018) AFML ch. 17 — feature engineering for
macro regimes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["RegimeLabel", "RegimeState", "MacroRegimeOverlay"]


class RegimeLabel(str, Enum):
    RISK_ON  = "risk_on"
    NEUTRAL  = "neutral"
    RISK_OFF = "risk_off"


@dataclass(frozen=True)
class RegimeState:
    label: RegimeLabel
    multiplier: float   # [0, 1] — apply to other signals
    confidence: float   # [0, 1]


class MacroRegimeOverlay:
    """Macro-regime classifier that provides a signal dampening multiplier.

    Uses three risk-off signals:
      1. Realised vol > rolling vol percentile threshold.
      2. Price below its n-bar moving average (trend break).
      3. Optional: macro fear index column (e.g. funding z-score).

    Parameters
    ----------
    symbol :
        Ticker name.
    vol_window :
        Window for current realised volatility.
    trend_window :
        Moving average window for trend-break detection.
    vol_pct_threshold :
        Percentile of rolling vol distribution above which = risk-off.
    fear_col :
        Optional column name for a macro fear indicator (e.g. DXY return,
        crypto fear index).  If absent, only vol + trend are used.
    """

    def __init__(
        self,
        symbol: str,
        vol_window: int = 21,
        trend_window: int = 63,
        vol_pct_threshold: float = 0.80,
        fear_col: Optional[str] = None,
    ) -> None:
        self.symbol = symbol
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_pct_threshold = vol_pct_threshold
        self.fear_col = fear_col
        self.signal_id = f"{symbol}_macro_regime"

    def fit(self, df: pd.DataFrame) -> None:
        pass  # rolling computation — no explicit fit needed

    def regime_state(self, df: pd.DataFrame, lag_bars: int = 1) -> RegimeState:
        """Compute the current regime state from ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``close`` and optional fear column.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-5): Number of trailing bars to exclude
            from all feature computations so the regime label at bar T is
            derived only from bars ≤ T - lag_bars.  The CALLER assigns this
            regime to bar T (= df.index[-1]) for trading on bar T.
        """
        # CHIEF AUDIT 2026-05-23 (P-5): truncate df to bars ≤ T - lag_bars
        # for feature computation; keep ts for caller's reference outside.
        lag = max(0, int(lag_bars))
        if lag > 0 and len(df) > lag:
            df_feat = df.iloc[:-lag]
        else:
            df_feat = df

        close = df_feat["close"]
        if len(close) < 2:
            return RegimeState(RegimeLabel.NEUTRAL, 0.7, 0.5)
        log_rets = np.log(close.values[1:] / close.values[:-1])

        # 1. Volatility regime
        if len(log_rets) < self.vol_window:
            return RegimeState(RegimeLabel.NEUTRAL, 0.7, 0.5)

        current_vol = float(np.std(log_rets[-self.vol_window:]))
        historical_vol = pd.Series(log_rets).rolling(self.vol_window).std().dropna().values
        if len(historical_vol) < 5:
            vol_pct = 0.5
        else:
            vol_pct = float(np.mean(historical_vol <= current_vol))

        vol_score = float(vol_pct >= self.vol_pct_threshold)

        # 2. Trend break: price below MA
        if len(close) >= self.trend_window:
            ma = float(close.iloc[-self.trend_window:].mean())
            trend_score = float(close.iloc[-1] < ma)
        else:
            trend_score = 0.0

        # 3. Optional fear indicator
        fear_score = 0.0
        if self.fear_col is not None and self.fear_col in df_feat.columns:
            fear_val = float(df_feat[self.fear_col].iloc[-1])
            fear_score = float(fear_val > 0)

        n_factors = 2 + int(self.fear_col is not None)
        risk_off_score = (vol_score + trend_score + fear_score) / n_factors

        if risk_off_score >= 0.66:
            label = RegimeLabel.RISK_OFF
            multiplier = max(0.0, 1.0 - risk_off_score)
        elif risk_off_score <= 0.33:
            label = RegimeLabel.RISK_ON
            multiplier = 1.0
        else:
            label = RegimeLabel.NEUTRAL
            multiplier = 0.7

        return RegimeState(label=label, multiplier=multiplier, confidence=1.0 - abs(risk_off_score - 0.5) * 2)

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        state = self.regime_state(df, lag_bars=lag_bars)
        ts = df.index[-1]
        # The overlay itself returns 0 signal — it's applied multiplicatively
        # by the ICWeightedCombiner or research harness.
        return SignalResult(
            symbol=self.symbol,
            timestamp=ts,
            signal=state.multiplier * 2.0 - 1.0,  # mapped from [0,1] to [-1,+1]
            confidence=state.confidence,
            horizon_bars=21,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> List[str]:
        feats = ["close"]
        if self.fear_col is not None:
            feats.append(self.fear_col)
        return feats
