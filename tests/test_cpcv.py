"""test_cpcv.py — CPCV invariants: no leakage between train and test folds.

These tests pin the most dangerous "Silent Killer" in the pipeline:
  CPCV with mis-tuned embargo allows the last train-bar to overlap with
  the first test-trade exit window → silent label leakage → inflated
  in-sample Sharpe → catastrophic OOS performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_dataset(n_events: int = 200, max_horizon: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_events, freq="1h", tz="UTC")
    horizons = rng.integers(5, max_horizon + 1, n_events)
    t1_idx   = pd.Series(np.clip(np.arange(n_events) + horizons, 0, n_events - 1),
                          index=idx, dtype=np.int64)
    return idx, t1_idx, max_horizon


def test_embargo_strictly_greater_than_t_max() -> None:
    """Item 6 leakage guard — embargo must exceed t_max bars or raise."""
    from tradebot.cv.cpcv import CombinatorialPurgedCV
    idx, t1, t_max = _make_synthetic_dataset()

    cv = CombinatorialPurgedCV(
        n_groups=4, n_test_groups=2, embargo_bars=t_max,  # ← intentionally too small
        purge_bars=t_max,
    )
    # The first split should raise on the leakage guard.
    with pytest.raises(ValueError, match="embargo"):
        next(iter(cv.split(idx, t1)))


def test_train_test_indices_disjoint() -> None:
    """Train and test indices NEVER overlap inside any fold."""
    from tradebot.cv.cpcv import CombinatorialPurgedCV
    idx, t1, t_max = _make_synthetic_dataset(n_events=300)

    cv = CombinatorialPurgedCV(
        n_groups=5, n_test_groups=2, embargo_bars=t_max + 5,
        purge_bars=t_max + 5,
    )
    for tr, te, _ in cv.split(idx, t1):
        assert len(np.intersect1d(tr, te)) == 0, "train/test overlap!"


def test_embargo_actually_enforced() -> None:
    """No train index lies within ``embargo_bars`` after the latest test exit."""
    from tradebot.cv.cpcv import CombinatorialPurgedCV
    idx, t1, t_max = _make_synthetic_dataset(n_events=300)

    embargo_bars = t_max + 5
    cv = CombinatorialPurgedCV(
        n_groups=5, n_test_groups=2, embargo_bars=embargo_bars,
        purge_bars=embargo_bars,
    )
    for tr, te, _ in cv.split(idx, t1):
        if len(te) == 0 or len(tr) == 0:
            continue
        max_test_t1 = int(np.clip(t1.values[te], 0, len(idx) - 1).max())
        # Any train idx in (max_test_t1, max_test_t1 + embargo_bars] is leakage.
        leakage_zone = (tr > max_test_t1) & (tr <= max_test_t1 + embargo_bars)
        assert not leakage_zone.any(), "embargo violated"


def test_path_reconstruction_yields_distinct_paths() -> None:
    """build_cpcv_return_paths must produce distinct paths (PATH-FIX bug guard)."""
    from tradebot.cv.cpcv import CombinatorialPurgedCV, build_cpcv_return_paths

    idx, t1, t_max = _make_synthetic_dataset(n_events=240)
    cv = CombinatorialPurgedCV(
        n_groups=4, n_test_groups=2, embargo_bars=t_max + 2,
        purge_bars=t_max + 2,
    )
    fold_returns = {}
    rng = np.random.default_rng(1)
    for tr, te, test_groups in cv.split(idx, t1):
        if len(te) == 0:
            continue
        ts_te = idx[te]
        rets = pd.Series(rng.normal(0.0, 0.01, len(te)), index=ts_te)
        fold_returns[test_groups] = rets

    paths = build_cpcv_return_paths(fold_returns, n_groups=4)
    if len(paths) >= 2:
        # Paths must NOT be identical (the PATH-FIX bug was: all paths same).
        identical = (paths[0].equals(paths[1]))
        # Allow rare equality on tiny synthetic data, but not the trivial case
        # where every series length matches and content is identical.
        assert not identical or len(paths[0]) < 5
