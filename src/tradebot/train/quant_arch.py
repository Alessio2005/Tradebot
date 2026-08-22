# src/tradebot/train/quant_arch.py
"""Symmetric quantile scaler, P² online quantile and dynamic embargo.

Extracted from ``quant_architect.py`` (Chief Quantitative Architect — audit items I.1/I.2/II.4).

Contents:
    SymmetricQuantileScalerState  — persistent state dataclass
    SymmetricQuantileScaler       — zero-crossing-aware quantile scaler
    _p2_update_njit               — Numba kernel (Jain & Chlamtac 1985)
    P2OnlineQuantile              — online P²-quantile estimator
    dynamic_embargo_bars          — embargo = max(horizon) * safety_factor
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit

logger = logging.getLogger("train.quant_arch")


# =============================================================================
# I.1 SYMMETRIC QUANTILE SCALER  — zero-crossing-aware
# =============================================================================
@dataclass
class SymmetricQuantileScalerState:
    """Persistent state for :class:`SymmetricQuantileScaler`.

    Separate structure so it can be joblib-serialised cleanly without dragging
    along the online-update machinery.
    """

    n_features: int = 0
    neg_markers: np.ndarray | None = None  # (n_bins, n_feat) — quantiles on (-∞,0]
    pos_markers: np.ndarray | None = None  # (n_bins, n_feat) — quantiles on [0,∞)
    passthrough_mask: np.ndarray | None = None  # (n_feat,) bool — True = binary feature
    n_bins: int = 21
    fitted: bool = False


class SymmetricQuantileScaler:
    """Zero-crossing-aware quantile scaler for oscillator features.

    Problem (audit item I.1): :class:`~execution.market_impact.RankQuantileScaler`
    maps the entire distribution to uniform [0, 1] — the **meaning of the zero
    line** is lost. An RSI of 50 (neutral) gets a different uniform rank in a
    trending regime than in a mean-reverting regime because the whole distribution
    shifts.

    Solution: split input into two buffers ``(-∞, 0]`` and ``[0, ∞)``, scale
    each independently to ``[-1, 0]`` and ``[0, 1]`` respectively. The zero line
    stays — by construction — invariant.

    Safeguard: ``passthrough_mask`` marks binary / discrete features (e.g.
    ``is_weekend``) so the scaler applies **no** rank transformation to them.

    Args:
        n_bins:           number of markers per side (default 21).
        passthrough_mask: bool array of length n_features; True ⇒ feature passes
                          through unchanged.
        clip_eps:         margin against exact ±1.0 (numerical safety for
                          downstream logit/atanh transforms).
    """

    def __init__(
        self,
        n_bins: int = 21,
        passthrough_mask: Sequence[bool] | None = None,
        clip_eps: float = 1e-3,
    ) -> None:
        self.state: SymmetricQuantileScalerState = SymmetricQuantileScalerState(
            n_bins=int(max(n_bins, 5))
        )
        self._init_passthrough: np.ndarray | None = (
            np.asarray(passthrough_mask, dtype=bool) if passthrough_mask is not None else None
        )
        self.clip_eps: float = float(clip_eps)

    def fit(self, X: np.ndarray) -> SymmetricQuantileScaler:
        """Estimate per-column positive and negative quantiles from ``X``.

        Args:
            X: (n_samples, n_features) raw feature matrix.

        Returns:
            self, with ``state.fitted = True``.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, n_feat = X_arr.shape
        st = self.state
        st.n_features = n_feat
        qs = np.linspace(0.0, 1.0, st.n_bins, dtype=np.float64)

        neg = np.zeros((st.n_bins, n_feat), dtype=np.float64)
        pos = np.zeros((st.n_bins, n_feat), dtype=np.float64)

        for j in range(n_feat):
            col = X_arr[:, j]
            col = col[np.isfinite(col)]
            neg_buf = col[col <= 0.0]
            pos_buf = col[col >= 0.0]

            if neg_buf.size >= 2:
                neg[:, j] = np.quantile(neg_buf, qs)
            else:
                neg[:, j] = np.linspace(-1.0, 0.0, st.n_bins)
            if pos_buf.size >= 2:
                pos[:, j] = np.quantile(pos_buf, qs)
            else:
                pos[:, j] = np.linspace(0.0, 1.0, st.n_bins)

        # Force 0.0 on the boundary marker to prevent rounding jitter around
        # the zero line.
        neg[-1, :] = 0.0
        pos[0, :] = 0.0

        st.neg_markers = neg
        st.pos_markers = pos
        if self._init_passthrough is not None and self._init_passthrough.size == n_feat:
            st.passthrough_mask = self._init_passthrough.astype(bool)
        else:
            st.passthrough_mask = np.zeros(n_feat, dtype=bool)

        st.fitted = True
        if n_samples == 0:
            logger.warning("SymmetricQuantileScaler.fit on empty matrix — passthrough fallback.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project each column to [-1, 0] (negative half) or [0, 1].

        Returns:
            float64 array, same shape as ``X`` after ``reshape(-1, 1)`` if 1D.
            Passthrough columns are copied verbatim.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        st = self.state
        if not st.fitted or st.neg_markers is None or st.pos_markers is None:
            logger.warning("SymmetricQuantileScaler.transform before fit() — passthrough.")
            return X_arr.copy()

        _, n_feat = X_arr.shape
        if n_feat != st.n_features:
            raise ValueError(
                f"SymmetricQuantileScaler dim-mismatch: fitted on {st.n_features} "
                f"features, got {n_feat}."
            )

        out = np.empty_like(X_arr, dtype=np.float64)
        qs = np.linspace(0.0, 1.0, st.n_bins, dtype=np.float64)
        passthrough: np.ndarray = (
            st.passthrough_mask
            if st.passthrough_mask is not None
            else np.zeros(n_feat, dtype=bool)
        )
        eps = self.clip_eps

        for j in range(n_feat):
            if bool(passthrough[j]):
                out[:, j] = X_arr[:, j]
                continue

            col = X_arr[:, j]
            neg_col = st.neg_markers[:, j]
            pos_col = st.pos_markers[:, j]
            scaled = np.zeros_like(col)

            mask_neg = col < 0.0
            if mask_neg.any():
                ranks_neg = np.interp(col[mask_neg], neg_col, qs, left=0.0, right=1.0)
                scaled[mask_neg] = ranks_neg - 1.0  # [0,1] → [-1,0]

            mask_pos = col > 0.0
            if mask_pos.any():
                ranks_pos = np.interp(col[mask_pos], pos_col, qs, left=0.0, right=1.0)
                scaled[mask_pos] = ranks_pos

            out[:, j] = np.clip(scaled, -1.0 + eps, 1.0 - eps)

        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# =============================================================================
# I.2 P² ONLINE QUANTILE  — Numba acceleration for HFT latency
# =============================================================================
@njit(cache=True, fastmath=False)
def _p2_update_njit(
    x: float,
    q_h: np.ndarray,   # (5,) heights
    q_n: np.ndarray,   # (5,) marker positions
    quantile: float,
    count: int,
) -> int:
    """Numba implementation of one P²-update (Jain & Chlamtac 1985).

    Releases the GIL — can be called from a prange-loop for parallel online
    quantile tracking across feature columns. Returns the new count (caller
    must assign it).
    """
    if x < q_h[0]:
        q_h[0] = x
        k = 0
    elif x >= q_h[4]:
        q_h[4] = x
        k = 3
    elif x < q_h[1]:
        k = 0
    elif x < q_h[2]:
        k = 1
    elif x < q_h[3]:
        k = 2
    else:
        k = 3

    for i in range(k + 1, 5):
        q_n[i] += 1.0

    new_count = count + 1
    n_prime = np.empty(5, dtype=np.float64)
    n_prime[0] = 1.0
    n_prime[1] = 1.0 + 2.0 * quantile * new_count
    n_prime[2] = 1.0 + 4.0 * quantile * new_count
    n_prime[3] = 3.0 + 2.0 * quantile * new_count
    n_prime[4] = float(new_count)

    for i in range(1, 4):
        d = n_prime[i] - q_n[i]
        if (d >= 1.0 and q_n[i + 1] - q_n[i] > 1.0) or (
            d <= -1.0 and q_n[i - 1] - q_n[i] < -1.0
        ):
            dd = 1.0 if d >= 0.0 else -1.0
            qi = q_h[i]
            qim1 = q_h[i - 1]
            qip1 = q_h[i + 1]
            ni = q_n[i]
            nim1 = q_n[i - 1]
            nip1 = q_n[i + 1]
            denom_outer = nip1 - nim1
            denom_up = nip1 - ni
            denom_dn = ni - nim1
            if denom_outer == 0.0 or denom_up == 0.0 or denom_dn == 0.0:
                if dd > 0.0 and denom_up > 0.0:
                    q_h[i] = qi + (qip1 - qi) / denom_up
                elif denom_dn > 0.0:
                    q_h[i] = qi - (qi - qim1) / denom_dn
            else:
                parabolic = qi + (dd / denom_outer) * (
                    (ni - nim1 + dd) * (qip1 - qi) / denom_up
                    + (nip1 - ni - dd) * (qi - qim1) / denom_dn
                )
                if qim1 < parabolic < qip1:
                    q_h[i] = parabolic
                elif dd > 0.0:
                    q_h[i] = qi + (qip1 - qi) / denom_up
                else:
                    q_h[i] = qi - (qi - qim1) / denom_dn
            q_n[i] = ni + dd

    return new_count


class P2OnlineQuantile:
    """Online P²-quantile estimator with Numba kernel.

    Drop-in replacement for situations where
    :class:`~execution.market_impact.RankQuantileScaler` has too much Python
    overhead (HFT inner-loop). Returns a single-quantile estimate; instantiate
    N instances for multiple quantiles.
    """

    def __init__(self, quantile: float = 0.5, n_features: int = 1) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1).")
        self.quantile: float = float(quantile)
        self.n_features: int = int(max(n_features, 1))
        self._heights: np.ndarray | None = None
        self._positions: np.ndarray | None = None
        self._count: int = 0
        self._warmup_buf: list[np.ndarray] = []

    def _initialize(self, first5: np.ndarray) -> None:
        sorted_block = np.sort(first5, axis=0)
        self._heights = sorted_block.T.astype(np.float64).copy()  # (n_feat, 5)
        q = self.quantile
        positions_template = np.array(
            [1.0, 1.0 + 2.0 * q, 1.0 + 4.0 * q, 3.0 + 2.0 * q, 5.0],
            dtype=np.float64,
        )
        self._positions = np.tile(positions_template, (self.n_features, 1))
        self._count = 5

    def update(self, x: np.ndarray) -> None:
        """Add one observation (vector of length n_features).

        The count is incremented ONCE PER SAMPLE — not per feature. For each
        feature the same pre-update count is passed to the Numba kernel, keeping
        marker progression consistent across all features.
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        if x_arr.size != self.n_features:
            raise ValueError(
                f"P2OnlineQuantile.update expects {self.n_features} features, "
                f"got {x_arr.size}."
            )

        if self._heights is None:
            self._warmup_buf.append(x_arr.copy())
            if len(self._warmup_buf) >= 5:
                first5 = np.vstack(self._warmup_buf[:5])
                self._initialize(first5)
                overflow = list(self._warmup_buf[5:])
                self._warmup_buf = []
                for extra in overflow:
                    self.update(extra)
            return

        heights = self._heights
        positions = self._positions
        assert positions is not None

        pre_count = self._count
        post_count = pre_count
        for j in range(self.n_features):
            post_count = _p2_update_njit(
                float(x_arr[j]),
                heights[j],
                positions[j],
                self.quantile,
                pre_count,
            )
        self._count = post_count

    def value(self) -> np.ndarray | None:
        """Current quantile estimate per feature (None during warmup)."""
        if self._heights is None:
            return None
        return self._heights[:, 2].copy()


# =============================================================================
# II.4 DYNAMIC EMBARGO  — Embargo = max(horizon) · safety_factor
# =============================================================================
def dynamic_embargo_bars(
    horizons_bars: Sequence[int],
    safety_factor: float = 1.2,
    min_embargo: int = 1,
) -> int:
    """Compute the dynamic CPCV embargo (audit item II.4).

    Standard purging covers overlap within one label horizon, but ignores that
    multi-period returns give information overlap that still correlates *after*
    the horizon (autocorrelation, vol-clustering). The recommended heuristic::

        embargo = ceil(max(horizon_bars) · safety_factor)

    With ``safety_factor = 1.2`` we add 20% extra buffer — wide enough for the
    typical 5-bar autocorrelation tail of crypto returns.

    Args:
        horizons_bars: list of label horizons (in bars).
        safety_factor: multiplier on max-horizon (≥ 1.0).
        min_embargo:   lower bound (e.g. 1 bar for very short horizons).

    Returns:
        Embargo in bars (integer).
    """
    if not horizons_bars:
        return int(min_embargo)
    h_max = int(max(int(h) for h in horizons_bars))
    embargo = math.ceil(h_max * float(max(safety_factor, 1.0)))
    return max(int(min_embargo), embargo)


__all__ = [
    "P2OnlineQuantile",
    "SymmetricQuantileScaler",
    "SymmetricQuantileScalerState",
    "_p2_update_njit",
    "dynamic_embargo_bars",
]
