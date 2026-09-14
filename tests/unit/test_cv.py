"""tests/unit/test_cv.py — Unit tests for CV tools."""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.cv import WalkForwardCV, get_sequential_bootstrap_indices


def test_sb_output_shape() -> None:
    t0 = np.array([0, 5, 10, 15], dtype=np.int64)
    t1 = np.array([8, 12, 18, 22], dtype=np.int64)
    idx = get_sequential_bootstrap_indices(t0, t1, n_draws=4)
    assert idx.shape == (4,)
    assert idx.dtype == np.int64


def test_sb_indices_in_range() -> None:
    t0 = np.arange(20, dtype=np.int64)
    t1 = t0 + 5
    idx = get_sequential_bootstrap_indices(t0, t1, n_draws=20)
    assert (idx >= 0).all()
    assert (idx < 20).all()


def test_sb_empty_input() -> None:
    t0 = np.array([], dtype=np.int64)
    t1 = np.array([], dtype=np.int64)
    idx = get_sequential_bootstrap_indices(t0, t1, n_draws=5)
    assert idx.shape == (5,)


def test_sb_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError, match="dezelfde lengte"):
        get_sequential_bootstrap_indices(
            np.array([0, 1, 2]), np.array([5, 6]), n_draws=2
        )


def test_sb_raises_on_t1_lt_t0() -> None:
    with pytest.raises(ValueError):
        get_sequential_bootstrap_indices(
            np.array([5, 10]), np.array([3, 15]), n_draws=2
        )


def test_walk_forward_split() -> None:
    cv = WalkForwardCV(train_size=100, test_size=50, step=50, require_embargo=False)
    folds = list(cv.split(300))
    assert len(folds) >= 2
    for fold in folds:
        assert fold.train_end < fold.test_end
        assert len(fold.train_indices) == 100
        assert len(fold.test_indices) == 50


def test_walk_forward_no_overlap() -> None:
    cv = WalkForwardCV(train_size=50, test_size=25, step=25, require_embargo=False)
    folds = list(cv.split(200))
    seen_test = set()
    for fold in folds:
        test_set = set(fold.test_indices.tolist())
        assert not (test_set & seen_test)
        seen_test |= test_set


def test_walk_forward_requires_embargo_by_default() -> None:
    """WalkForwardCV must refuse embargo_bars=0 unless require_embargo=False."""
    with pytest.raises(ValueError, match="embargo_bars=0"):
        WalkForwardCV(train_size=100, test_size=50, step=50)


def test_walk_forward_embargo_drops_boundary_bars() -> None:
    """embargo_bars=H removes the last H train bars so labels can't exit into test."""
    H = 10
    cv = WalkForwardCV(train_size=100, test_size=50, step=50, embargo_bars=H)
    folds = list(cv.split(300))
    assert len(folds) >= 2
    for fold in folds:
        # With H=10 embargo, effective train window is 100-10=90 bars.
        assert len(fold.train_indices) == 100 - H
        # The last train index must be < first test index (no boundary overlap).
        assert fold.train_indices[-1] < fold.test_indices[0]
