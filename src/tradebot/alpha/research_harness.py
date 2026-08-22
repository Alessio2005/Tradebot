# src/tradebot/alpha/research_harness.py
"""Standardised single-signal backtest loop for alpha research.

Provides a lean walk-forward simulation that:
  1. Rolls a fit window forward in time.
  2. Calls signal.fit(train) then signal.predict(test_window).
  3. Collects SignalResult series alongside realised forward returns.
  4. Computes per-period IC and summary statistics.

This is a RESEARCH tool — it uses simplified assumptions (no transaction
costs, no sizing) to evaluate raw signal quality before integration into
the main CPCV pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .base import AlphaSignal, SignalResult

logger = logging.getLogger(__name__)

__all__ = ["HarnessResult", "run_signal_harness"]


@dataclass
class HarnessResult:
    """Output of a single-signal harness run."""

    signal_id: str
    predictions: List[SignalResult] = field(default_factory=list)
    ic_series: pd.Series = field(default_factory=pd.Series)

    # Summary statistics (populated by run_signal_harness)
    mean_ic: float = 0.0
    icir: float = 0.0    # IC Information Ratio = mean(IC) / std(IC)
    hit_rate: float = 0.0

    def ic_table(self) -> pd.DataFrame:
        return pd.DataFrame({"ic": self.ic_series})


def run_signal_harness(
    signal: AlphaSignal,
    df: pd.DataFrame,
    fit_window: int = 252,
    step_bars: int = 21,
    horizon_bars: int = 21,
    min_fit_bars: int = 63,
) -> HarnessResult:
    """Walk-forward harness for a single AlphaSignal.

    Parameters
    ----------
    signal :
        An AlphaSignal instance (not yet fitted).
    df :
        Full historical OHLCV DataFrame (UTC DatetimeIndex).
    fit_window :
        Number of bars used to fit the signal at each step.
    step_bars :
        Number of bars between each prediction step.
    horizon_bars :
        Forward return horizon for IC computation.
    min_fit_bars :
        Minimum number of bars required before first prediction.

    Returns
    -------
    HarnessResult with per-step IC and summary statistics.
    """
    n = len(df)
    result = HarnessResult(signal_id=signal.signal_id)

    predictions: list[SignalResult] = []
    timestamps: list[pd.Timestamp] = []
    signals_vals: list[float] = []
    fwd_returns: list[float] = []

    first_pred_idx = max(fit_window, min_fit_bars)

    for i in range(first_pred_idx, n - horizon_bars, step_bars):
        train_df = df.iloc[max(0, i - fit_window): i]
        pred_df  = df.iloc[max(0, i - fit_window): i + 1]

        try:
            signal.fit(train_df)
            pred = signal.predict(pred_df)
        except Exception:
            logger.debug("signal.predict failed at bar %d", i, exc_info=True)
            continue

        # Forward return over horizon_bars
        p_now  = float(df["close"].iloc[i])
        p_fwd  = float(df["close"].iloc[i + horizon_bars])
        if p_now <= 0:
            continue
        fwd_ret = float(np.log(p_fwd / p_now))

        predictions.append(pred)
        timestamps.append(pred.timestamp)
        signals_vals.append(pred.signal)
        fwd_returns.append(fwd_ret)

    result.predictions = predictions

    if len(signals_vals) < 5:
        return result

    sigs_arr = np.array(signals_vals)
    rets_arr = np.array(fwd_returns)

    # Rolling IC (window = 21 steps)
    ic_vals: list[float] = []
    ic_ts:   list[pd.Timestamp] = []
    window = 21
    for j in range(window, len(sigs_arr)):
        s_slice = sigs_arr[j - window: j]
        r_slice = rets_arr[j - window: j]
        mask = np.isfinite(s_slice) & np.isfinite(r_slice)
        if mask.sum() < 5:
            continue
        ic, _ = spearmanr(s_slice[mask], r_slice[mask])
        ic_vals.append(float(ic) if np.isfinite(ic) else 0.0)
        ic_ts.append(timestamps[j])

    result.ic_series = pd.Series(ic_vals, index=ic_ts, name="ic")

    # Full-sample IC
    mask_full = np.isfinite(sigs_arr) & np.isfinite(rets_arr)
    if mask_full.sum() >= 5:
        ic_full, _ = spearmanr(sigs_arr[mask_full], rets_arr[mask_full])
        ic_full = float(ic_full) if np.isfinite(ic_full) else 0.0
    else:
        ic_full = 0.0

    result.mean_ic  = float(np.mean(ic_vals)) if ic_vals else ic_full
    result.icir     = float(result.mean_ic / (np.std(ic_vals) + 1e-9)) if ic_vals else 0.0
    result.hit_rate = float(np.mean(np.sign(sigs_arr) == np.sign(rets_arr)))

    logger.info(
        "Harness done: signal_id=%s  IC=%.4f  ICIR=%.4f  hit=%.2f%%  n=%d",
        signal.signal_id,
        result.mean_ic,
        result.icir,
        result.hit_rate * 100,
        len(signals_vals),
    )
    return result
