"""uniqueness.py — AFML ch. 4 sample uniqueness + sample weights.

Extracted from train_regime.py lines 925-1073.
Bit-identical Numba kernel; same public API.

SK-4 fix is already in the source (prange → range); preserved here.

Strangler-fig: once train_regime.py imports from here and the
equivalence test passes, delete lines 925-1073 in train_regime.py.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA KERNEL (SK-4 fix already applied: prange → range)
# =============================================================================

@njit(cache=True)
def _calculate_average_uniqueness(
    t0_indices: np.ndarray,
    t1_indices: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Compute average uniqueness per event (AFML ch. 4).

    RACE-CONDITION-FIX (SK-4):
      prange was misleading — @njit without parallel=True → prange == range.
      If parallel=True were added later, concurrency[start:end] += 1.0 would
      be a non-atomic shared-state slice update → race condition.
      Using range explicitly so intent and behaviour match.

    Args:
        t0_indices : int32 bar-positions of event starts.
        t1_indices : int32 bar-positions of event ends (exclusive).
        n_samples  : total number of bars in the dataset.

    Returns:
        float64 array of uniqueness per event (0..1 range).
    """
    concurrency = np.zeros(n_samples, dtype=np.float64)
    n_obs = len(t0_indices)

    for i in range(n_obs):
        start, end = t0_indices[i], t1_indices[i]
        if start >= n_samples or end > n_samples or start < 0 or end < 0:
            continue
        concurrency[start:end] += 1.0

    for k in range(n_samples):
        if concurrency[k] == 0:
            concurrency[k] = 1.0

    uniqueness = np.zeros(n_obs, dtype=np.float64)
    for i in range(n_obs):
        start, end = t0_indices[i], t1_indices[i]
        if start >= n_samples or end > n_samples or start < 0 or end < 0:
            uniqueness[i] = 0.0
            continue
        duration = end - start
        if duration > 0:
            inv_c = 0.0
            for t in range(start, end):
                inv_c += (1.0 / concurrency[t])
            uniqueness[i] = inv_c / duration
        else:
            uniqueness[i] = 0.0

    return uniqueness


# =============================================================================
# PUBLIC API
# =============================================================================

def get_average_uniqueness(
    df_index: pd.DatetimeIndex,
    t1_indices: pd.Series,
) -> np.ndarray:
    """Global uniqueness over all events (all folds visible).

    WARNING: this leaks OOS test-fold information into train sample weights
    because a train-event overlapping with a test-event gets lower uniqueness,
    even though that test-overlap is unknown in production.
    Use get_average_uniqueness_per_fold() inside CPCV splits.

    Returns float64 array of uniqueness values, same length as t1_indices.
    """
    t0_vals = df_index.get_indexer(t1_indices.index).astype(np.int32)
    t1_vals = np.asarray(t1_indices.values, dtype=np.int32)
    return _calculate_average_uniqueness(t0_vals, t1_vals, len(df_index))


def get_average_uniqueness_per_fold(
    df_index: pd.DatetimeIndex,
    t1_indices: pd.Series,
    train_indices: np.ndarray,
) -> np.ndarray:
    """Per-fold uniqueness — leakage-safe (Item 9 fix).

    Restricts concurrency to train-fold events only.
    OOS events are fully excluded — their overlap does not contaminate
    train sample weights.

    Args:
        df_index      : DatetimeIndex of the full bar series.
        t1_indices    : pd.Series with t1 (exit bar) per event.
        train_indices : integer positions of train events (from CPCV.split()).

    Returns:
        float64 array of uniqueness for each train event (length = len(train_indices)).
    """
    n_bars = len(df_index)
    t0_full = df_index.get_indexer(t1_indices.index).astype(np.int32)
    t1_full = np.asarray(t1_indices.values, dtype=np.int32)

    train_idx_arr = np.asarray(train_indices, dtype=np.int64)
    if train_idx_arr.size == 0:
        return np.zeros(0, dtype=np.float64)

    t0_train = t0_full[train_idx_arr]
    t1_train = t1_full[train_idx_arr]
    return _calculate_average_uniqueness(t0_train, t1_train, n_bars)


def get_sample_weights(
    timestamps: pd.Series,
    uniqueness: np.ndarray,
    returns: np.ndarray | None = None,
    time_decay_span: float = 180.0,
    anchor_time: pd.Timestamp | None = None,
) -> np.ndarray:
    """Sample weights combining uniqueness, return magnitude, and time decay.

    time_decay_span default 180 days (≈ 6 months):
      Crypto regime-shifts every 6-12 months → old data should be downweighted.
      BTC=270, ETH=180, SOL=90 (fast-shifting validator concentration/MEV regime).

    Args:
        timestamps      : pd.Series of event timestamps (DatetimeIndex-like).
        uniqueness      : float64 uniqueness per event (from get_average_uniqueness).
        returns         : optional float64 returns; log(1+|ret|*100)+1 scaling applied.
        time_decay_span : exponential decay half-life in calendar days.
        anchor_time     : reference timestamp for decay (default = max of timestamps).

    Returns:
        float64 sample weights, clipped to [0.05, 10.0].
    """
    n = len(timestamps)
    w = uniqueness.copy()

    if len(w) != n:
        min_len = min(n, len(w))
        timestamps = timestamps.iloc[:min_len]
        w = w[:min_len]
        n = min_len

    if returns is not None:
        if len(returns) != n:
            returns = returns[:n]
        abs_ret = np.abs(returns)
        ret_w = np.log1p(abs_ret * 100.0) + 1.0
        w = w * ret_w

    w = w / (np.mean(w) + 1e-9)

    ts_max = anchor_time if anchor_time is not None else pd.Timestamp(timestamps.max())
    days_old_series = timestamps.rsub(pd.Timestamp(ts_max)).dt.total_seconds() / (24 * 3600)

    if isinstance(days_old_series, pd.Series):
        days_old = days_old_series.to_numpy(dtype=float)
    else:
        days_old = np.asarray(days_old_series, dtype=float)

    days_old = np.maximum(days_old, 0.0)
    decay = np.exp(-days_old / time_decay_span)

    final_w = w * decay
    return np.clip(final_w, 0.05, 10.0)
