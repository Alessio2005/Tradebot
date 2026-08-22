"""cpcv.py — Combinatorial Purged Cross-Validation + path builder.

Extracted from train_regime.py lines 1076-1384.
Bit-identical to the monolith: same split logic, same path-builder.

Key fixes already in source (preserved here):
  • EMBARGO-FIX (Item 6): embargo expressed in bars, not calendar days.
    embargo_bars > t_max is enforced; raises ValueError if violated.
  • AUDIT-FIX (N16): bar-space purging (purge_bars) replaces calendar-time
    purging in fast regimes (default purge_days kept as backward-compat fallback).
  • AUDIT-FIX (Issue 1): FFD feature_max_lag lifts embargo when FFD tail > t_max.
  • PATH-FIX: path_idx anchoring ensures paths diverge (old bug: all paths identical).

Strangler-fig: once train_regime.py imports from here and the equivalence
test passes, delete lines 1076-1384 in train_regime.py.
"""
from __future__ import annotations

import logging
from collections.abc import Generator
from datetime import timedelta
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CPCV SPLITTER
# =============================================================================

class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV) splitter.

    Partitions the dataset into N groups, generates C(N, k) train/test
    splits with purging and embargo to prevent information leakage.

    EMBARGO-FIX (Item 6):
      Embargo is expressed in BAR-SPACE (not calendar days). Strict
      requirement: embargo_bars > t_max where t_max = max(t1_idx - t0_idx)
      over all events. Embargo shorter than t_max guarantees leakage.

    BAR-PURGE-FIX (N16):
      In fast regimes (300 bars/day), calendar-time purging over-purges 10×.
      Set purge_bars to use bar-space purging (recommended: purge_bars = t_max + 1).
      purge_days is the backward-compat fallback.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_days: int = 2,
        embargo_days: int = 2,
        embargo_bars: int | None = None,
        purge_bars: int | None = None,
        feature_max_lag: int | None = None,
        require_bar_purge: bool = True,
    ) -> None:
        # AUDIT D-2: in intraday regimes (≥ 24 bars/day) the calendar-day
        # purge fallback over-purges by an order of magnitude. Force the
        # caller to supply ``purge_bars`` explicitly unless they opt out via
        # ``require_bar_purge=False`` (allowed only for legacy daily-bar
        # studies). Pre-existing call sites that rely on the calendar
        # fallback now fail loudly so the misconfiguration is impossible to
        # ship to production.
        if require_bar_purge and purge_bars is None:
            raise ValueError(
                "AUDIT D-2: CombinatorialPurgedCV requires purge_bars (not "
                "purge_days). Calendar-time purging over-purges in intraday "
                "regimes — pass purge_bars = t_max + 1 from your label "
                "horizon. Set require_bar_purge=False only when bars are "
                "strictly daily and t_max is interpretable in days."
            )

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.embargo_bars: int | None = (
            int(embargo_bars) if embargo_bars is not None else None
        )
        self.purge_bars: int | None = (
            int(purge_bars) if purge_bars is not None else None
        )
        self.feature_max_lag: int | None = (
            int(feature_max_lag) if feature_max_lag is not None else None
        )

    def split(
        self,
        timestamps: pd.DatetimeIndex,
        t1_indices: pd.Series,
    ) -> Generator[tuple[np.ndarray, np.ndarray, tuple], None, None]:
        """Generate (train_indices, test_indices, test_groups_tuple) triples.

        Args:
            timestamps : DatetimeIndex of all events.
            t1_indices : pd.Series with t1 (exit bar-position, int) per event.

        Yields:
            (train_idx_arr, test_idx_arr, test_groups) where indices are
            integer positions into the timestamps array.
        """
        ts_arr = np.asarray(timestamps)
        t1_arr = np.asarray(t1_indices, dtype=np.int64)

        n_samples  = len(ts_arr)
        group_size = n_samples // self.n_groups

        # ── EMBARGO-FIX (Item 6): derive effective embargo in bars ────────────
        bar_positions: np.ndarray = np.arange(n_samples, dtype=np.int64)
        safe_t1_full: np.ndarray  = np.clip(t1_arr, 0, n_samples - 1)
        horizons_bars: np.ndarray = safe_t1_full - bar_positions
        valid_horiz: np.ndarray   = horizons_bars[horizons_bars > 0]
        t_max_bars: int = int(valid_horiz.max()) if valid_horiz.size > 0 else 1

        if self.embargo_bars is None:
            embargo_bars_eff: int = t_max_bars + 1
        else:
            embargo_bars_eff = self.embargo_bars

        # AUDIT-FIX (Issue 1 — FFD lag lifts embargo):
        if self.feature_max_lag is not None and self.feature_max_lag > 0:
            feature_embargo = int(self.feature_max_lag) + 1
            if feature_embargo > embargo_bars_eff:
                logger.info(
                    "CPCV embargo upgraded by FFD feature_max_lag: "
                    "%d → %d bars (label t_max=%d, ffd_lag=%d).",
                    embargo_bars_eff, feature_embargo,
                    t_max_bars, self.feature_max_lag,
                )
                embargo_bars_eff = feature_embargo

        if embargo_bars_eff <= t_max_bars:
            raise ValueError(
                f"CPCV Item 6 leakage guard: embargo_bars (={embargo_bars_eff}) "
                f"must be STRICTLY greater than t_max (={t_max_bars}) to "
                f"guarantee the last open test-fold trade is closed before "
                f"any train bar can exist."
            )

        # ── Build N groups ────────────────────────────────────────────────────
        group_indices: list[np.ndarray] = []
        for i in range(self.n_groups):
            start_idx = i * group_size
            end_idx   = n_samples if i == self.n_groups - 1 else (i + 1) * group_size
            group_indices.append(np.arange(start_idx, end_idx, dtype=np.int64))

        group_combos = list(combinations(range(self.n_groups), self.n_test_groups))

        _use_bar_purge: bool = self.purge_bars is not None
        _purge_bars_eff: int = int(self.purge_bars) if self.purge_bars is not None else 0
        purge_td = timedelta(days=self.purge_days)

        for test_groups in group_combos:
            train_idx_list: list[np.ndarray] = []
            test_idx_list: list[np.ndarray]  = []

            for tg in test_groups:
                test_idx_list.append(group_indices[tg])
            test_idx_arr: np.ndarray = (
                np.concatenate(test_idx_list)
                if test_idx_list else np.array([], dtype=np.int64)
            )
            if len(test_idx_arr) == 0:
                continue

            # Per test-group: max t1 (bar-index and timestamp)
            test_max_t1_idx_per_group: dict[int, int] = {}
            test_max_t1_ts_per_group: dict[int, Any]  = {}
            for tg in test_groups:
                tg_idx = group_indices[tg]
                if len(tg_idx) > 0:
                    safe_t1 = np.clip(t1_arr[tg_idx], 0, n_samples - 1)
                    test_max_t1_idx_per_group[tg] = int(np.max(safe_t1))
                    test_max_t1_ts_per_group[tg]  = np.max(ts_arr[safe_t1])

            # ── Purge + Embargo per train group ───────────────────────────────
            for trg in range(self.n_groups):
                if trg in test_groups:
                    continue
                tr_group_idx = group_indices[trg]
                if len(tr_group_idx) == 0:
                    continue

                t0_train_arr     = ts_arr[tr_group_idx]
                t1_train_arr     = ts_arr[np.clip(t1_arr[tr_group_idx], 0, n_samples - 1)]
                t0_train_idx_arr = tr_group_idx.astype(np.int64)
                t1_train_idx_arr = np.clip(t1_arr[tr_group_idx], 0, n_samples - 1).astype(np.int64)

                mask = np.ones(len(tr_group_idx), dtype=bool)

                for tg in test_groups:
                    if len(group_indices[tg]) == 0:
                        continue

                    t0_test_min   = ts_arr[group_indices[tg][0]]
                    t1_test_max   = test_max_t1_ts_per_group[tg]
                    t1_test_max_i = test_max_t1_idx_per_group[tg]
                    t0_test_min_i = int(group_indices[tg][0])

                    # PURGE (bar-space or calendar-time)
                    if _use_bar_purge:
                        purge_mask = (
                            (t0_train_idx_arr <= t1_test_max_i + _purge_bars_eff)
                            & (t1_train_idx_arr >= t0_test_min_i - _purge_bars_eff)
                        )
                    else:
                        purge_mask = (
                            (t0_train_arr <= t1_test_max + purge_td)
                            & (t1_train_arr >= t0_test_min - purge_td)
                        )

                    # EMBARGO (always bar-space, Item 6)
                    # RIGHT-side embargo: train events immediately AFTER the test block.
                    embargo_right_mask = (
                        (t0_train_idx_arr > t1_test_max_i)
                        & (t0_train_idx_arr <= t1_test_max_i + embargo_bars_eff)
                    )

                    # P0-E FIX: LEFT-side embargo (AFML §7.4.2).
                    # Train events whose label EXITS within embargo_bars_eff of the
                    # test block's leading edge carry serial-autocorrelation tail
                    # that contaminates the test fold's first bars.
                    # Condition: t1_train falls in [t0_test_min - emb, t0_test_min).
                    embargo_left_mask = (
                        (t1_train_idx_arr >= t0_test_min_i - embargo_bars_eff)
                        & (t1_train_idx_arr < t0_test_min_i)
                    )

                    mask &= ~(purge_mask | embargo_right_mask | embargo_left_mask)

                train_idx_list.append(tr_group_idx[mask])

            train_idx_arr: np.ndarray = (
                np.concatenate(train_idx_list)
                if train_idx_list else np.array([], dtype=np.int64)
            )
            yield train_idx_arr, test_idx_arr, test_groups


# =============================================================================
# CPCV RETURN PATH BUILDER
# =============================================================================

def build_cpcv_return_paths(
    fold_returns_dict: dict[tuple, pd.Series],
    n_groups: int,
) -> list[pd.Series]:
    """Reconstruct continuous backtest paths from CPCV fold returns.

    Correct CPCV evaluation (López de Prado, AFML ch. 12):
      - k test-groups per fold; N groups total; N must be divisible by k.
      - A valid path is a set of N/k folds whose test-groups partition
        {0, 1, ..., N-1} without overlap.
      - Each path covers ALL N groups exactly once (no group appears twice,
        no group is missing).
      - Sharpe distribution over these valid paths is the true OOS metric.

    Algorithm:
      1. Infer k from the first fold tuple.
      2. Enumerate all partitions of {0,...,N-1} into N/k disjoint k-tuples.
      3. Keep only partitions where every k-tuple exists as a fold in
         fold_returns_dict.
      4. Each accepted partition → one path (concat of fold returns, sorted).

    Returns:
        List of pd.Series with trade-returns per complete path, sorted by time.
        Empty list if no valid partitions found (e.g. N not divisible by k).
    """
    if not fold_returns_dict:
        return []

    # Infer k from fold tuple size
    sample_combo = next(iter(fold_returns_dict))
    k = len(sample_combo)
    if k == 0 or n_groups % k != 0:
        raise ValueError(
            f"build_cpcv_return_paths: n_groups={n_groups} must be divisible by k={k}. "
            f"Fix n_groups or n_test_groups. Silent concatenation fallback is FORBIDDEN "
            f"(Wave 14 P0-10) as it destroys train/test separation."
        )

    n_groups // k
    all_groups = set(range(n_groups))

    def _partitions(remaining: frozenset, k: int) -> Generator[tuple[tuple, ...], None, None]:
        """Yield all ordered partitions of ``remaining`` into k-tuples.

        Ordering: always anchor on the smallest element to avoid duplicates.
        """
        if not remaining:
            yield ()
            return
        anchor = min(remaining)
        rest = remaining - {anchor}
        for others in combinations(sorted(rest), k - 1):
            group_tuple = (anchor,) + others
            for tail in _partitions(remaining - set(group_tuple), k):
                yield (group_tuple,) + tail

    paths: list[pd.Series] = []

    for partition in _partitions(frozenset(all_groups), k):
        # Check all folds in this partition exist
        if not all(fold in fold_returns_dict for fold in partition):
            continue
        segments: list[pd.Series] = [
            fold_returns_dict[fold] for fold in partition
        ]
        full_path = pd.concat(segments).sort_index()
        full_path = full_path[~full_path.index.duplicated(keep="first")]
        paths.append(full_path)

    if not paths:
        raise ValueError(
            f"build_cpcv_return_paths: No valid N/k partitions found for "
            f"n_groups={n_groups}, k={k}. All fold combinations must exist in "
            f"fold_returns_dict. Silent concatenation fallback is FORBIDDEN (P0-10)."
        )

    return paths
