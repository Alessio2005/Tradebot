"""Walk-Forward Cross-Validation — sanity check alternative to CPCV.

Two modes:
  - rolling   : fixed-size training window slides forward.
  - anchored  : training window grows from a fixed start date.

Walk-forward is NOT the primary CV for tradebot (CPCV is). It is used as
a sanity check: if CPCV and walk-forward disagree strongly, the CPCV
setup may have a leakage bug.

CHIEF AUDIT-FIX (Sim-to-Reality #12):
  Anchored mode applies a symmetric embargo around every prior test
  boundary.  With many folds and a large embargo this can silently
  decimate the training set — a 30-fold anchored split with embargo=20
  bars can erase >50% of train rows in late folds.  A warning now fires
  when the retained train fraction drops below TRAIN_RETENTION_WARN.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# Warning threshold: when (effective_train / raw_train_size) < this we log a warning.
TRAIN_RETENTION_WARN: float = 0.70


@dataclass
class WalkForwardFold:
    train_idx: np.ndarray
    test_idx:  np.ndarray
    fold_id:   int

    # ── Public aliases used by tests and downstream code ─────────────────────
    @property
    def train_indices(self) -> np.ndarray:
        return self.train_idx

    @property
    def test_indices(self) -> np.ndarray:
        return self.test_idx

    @property
    def train_end(self) -> int:
        """Last exclusive index of the training window."""
        return int(self.train_idx[-1]) + 1 if len(self.train_idx) else 0

    @property
    def test_end(self) -> int:
        """Last exclusive index of the test window."""
        return int(self.test_idx[-1]) + 1 if len(self.test_idx) else 0


class WalkForwardCV:
    """Rolling or anchored walk-forward splitter.

    Args:
        train_size  : Number of bars in each training window (rolling mode).
        test_size   : Number of bars in each test window.
        step        : Number of bars to advance per fold.
        mode        : "rolling" (fixed window) or "anchored" (expanding).
        min_train   : Minimum training bars required (anchored mode only).
        embargo_bars: Bars to drop from the end of each training window before
                      the test boundary. Must be >= your label horizon H so that
                      no train label can exit into the test set. Defaults to 0
                      (backward-compatible) but raises ValueError when > 0 would
                      consume the entire training window.
        require_embargo: When True (default), raises ValueError if embargo_bars
                         is 0 — forces callers to consciously opt out rather than
                         silently running with train/test leakage at the boundary.
                         Set require_embargo=False only for daily-bar studies where
                         horizon H == 1 bar and the boundary leak is negligible.
    """

    def __init__(
        self,
        train_size: int = 1000,
        test_size: int = 250,
        step: int = 125,
        mode: Literal["rolling", "anchored"] = "rolling",
        min_train: int = 500,
        embargo_bars: int = 0,
        require_embargo: bool = True,
    ) -> None:
        if require_embargo and embargo_bars == 0:
            raise ValueError(
                "WalkForwardCV: embargo_bars=0 means the last H train-events can "
                "have labels that exit into the test window — train/test leakage. "
                "Set embargo_bars >= your label horizon H (e.g. embargo_bars=10 "
                "for a 10-bar Triple-Barrier). "
                "Set require_embargo=False only when H==1 and the leak is acceptable."
            )
        self.train_size    = train_size
        self.test_size     = test_size
        self.step          = step
        self.mode          = mode
        self.min_train     = min_train
        self.embargo_bars  = int(embargo_bars)

    def split(
        self, n: int
    ) -> Generator[WalkForwardFold, None, None]:
        """Yield WalkForwardFold objects for n total observations."""
        fold_id = 0
        train_end = self.train_size
        # P0-F FIX: track previous test end so we can embargo it from the
        # CURRENT fold's training window (anchored mode only).
        # In anchored mode previous test bars [prev_test_end-emb, prev_test_end+emb)
        # become training bars in the next fold; any label that entered the prior test
        # fold and exits post-prev_test_end leaks into the new train set.
        prev_test_end: int = 0

        while train_end + self.test_size <= n:
            train_start  = 0 if self.mode == "anchored" else (train_end - self.train_size)
            test_end     = train_end + self.test_size
            # Embargo: drop the last embargo_bars from the training window so no
            # train label with horizon H can exit into [train_end, test_end).
            train_end_embargoed = train_end - self.embargo_bars

            train_idx_raw = np.arange(train_start, max(train_start, train_end_embargoed))

            # P0-F FIX: in anchored mode, also exclude a symmetric embargo window
            # around the previous test fold boundary to prevent labels that span
            # across the fold boundary from contaminating the new training set.
            if self.mode == "anchored" and self.embargo_bars > 0 and prev_test_end > 0:
                emb_lo = max(train_start, prev_test_end - self.embargo_bars)
                emb_hi = min(train_end_embargoed, prev_test_end + self.embargo_bars)
                if emb_lo < emb_hi:
                    train_idx = train_idx_raw[
                        ~((train_idx_raw >= emb_lo) & (train_idx_raw < emb_hi))
                    ]
                else:
                    train_idx = train_idx_raw
            else:
                train_idx = train_idx_raw

            test_idx  = np.arange(train_end, test_end)

            # In rolling mode train_size is constant; min_train only guards
            # anchored mode where the window grows from the start.
            min_ok = (self.mode == "rolling") or (len(train_idx) >= self.min_train)
            if min_ok:
                # CHIEF AUDIT-FIX #12: track embargo erosion in anchored mode.
                raw_train = max(train_end_embargoed - train_start, 1)
                if (
                    self.mode == "anchored"
                    and raw_train > 0
                    and len(train_idx) / raw_train < TRAIN_RETENTION_WARN
                ):
                    logger.warning(
                        "WalkForwardCV fold %d (anchored): symmetric embargo "
                        "eroded train set to %d/%d bars (%.1f%% retained, "
                        "below %.0f%% floor). Reduce embargo_bars or use "
                        "rolling mode.",
                        fold_id, len(train_idx), raw_train,
                        100.0 * len(train_idx) / raw_train,
                        100.0 * TRAIN_RETENTION_WARN,
                    )
                yield WalkForwardFold(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    fold_id=fold_id,
                )
                fold_id += 1

            prev_test_end = int(test_end)  # P0-F: update for next fold
            train_end += self.step

    def n_splits(self, n: int) -> int:
        """Number of splits for n observations."""
        return sum(1 for _ in self.split(n))
