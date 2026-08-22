# src/tradebot/alpha/base.py
"""AlphaSignal protocol + SignalResult dataclass.

Every signal in the alpha library must implement this interface so the
ICWeightedCombiner and research_harness can treat them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = ["SignalResult", "AlphaSignal"]


@dataclass(frozen=True)
class SignalResult:
    """Single-bar prediction from one alpha signal.

    Attributes
    ----------
    symbol :
        Ticker string (e.g. ``"BTCUSDT"``).
    timestamp :
        Bar close time (UTC-aware).
    signal :
        Normalised signal in [-1, +1].  +1 = maximum long conviction.
    confidence :
        Calibrated probability in [0, 1].  0.5 = no view.
    horizon_bars :
        Number of forward bars this signal targets.
    signal_id :
        Reproducible identifier: ``f"{symbol}_{algo}_{params_hash}"``.
    """

    symbol: str
    timestamp: pd.Timestamp
    signal: float
    confidence: float
    horizon_bars: int
    signal_id: str


@runtime_checkable
class AlphaSignal(Protocol):
    """Protocol every alpha signal must satisfy."""

    signal_id: str

    def fit(self, df: pd.DataFrame) -> None:
        """Fit the signal on historical bars ``df``.

        ``df`` must have a UTC-aware DatetimeIndex and contain at minimum
        ``open``, ``high``, ``low``, ``close``, ``volume`` columns.
        All data in ``df`` is treated as in-sample — no lookahead possible
        at fit time (the last row represents bar t-1 relative to live bar t).
        """
        ...

    def predict(self, df: pd.DataFrame) -> SignalResult:
        """Predict on the LATEST bar of ``df``.

        Only the most-recent row is used for the live signal.  Historical
        rows in ``df`` are used solely for rolling/stateful computations.
        Must be called after ``fit``.
        """
        ...

    def feature_names(self) -> list[str]:
        """Return the list of feature names consumed by this signal."""
        ...
