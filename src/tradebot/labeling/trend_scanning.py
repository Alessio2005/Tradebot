"""trend_scanning.py — TrendScanningLabeler + Numba OLS kernel.

Extracted from train_regime.py lines 420-718.
Bit-identical to the monolith: same Numba kernel, same class API.

Changes vs. monolith:
  • Import path only (strangler-fig step 2).
  • _triple_barrier_per_event imported from labeling.triple_barrier
    instead of defined inline — breaks the circular dependency once
    that module is extracted.
  • .values.astype() → numba_array() at Numba call-sites AFTER schema
    enforcement is added in Wave 2 Step 9.  Not yet changed here to
    preserve bit-identical output during transition.

Strangler-fig: once train_regime.py imports TrendScanningLabeler from
here and passes the equivalence test, delete lines 515-718 in train_regime.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numba import njit

from tradebot.labeling.triple_barrier import _triple_barrier_per_event

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA KERNEL (bit-identical to train_regime.py lines 423-512)
# =============================================================================

@njit(cache=True)
def _trend_scan_core(
    close: np.ndarray,
    t_min: int,
    t_max: int,
    min_tstat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba kernel — Trend Scanning Labels (AFML ch. 17).

    For each bar i: scan forward windows L ∈ [t_min, t_max], fit OLS on
    close[i:i+L] vs time index, compute t-stat of the slope.
    Choose L* = argmax |t-stat|.  Label = sign(slope) if |t-stat| >= min_tstat.

    Returns:
        labels    : int8  (-1, 0, +1)
        t_stats   : float64 (t-stat at optimal L)
        best_spans: int32  (chosen horizon length L*)
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    t_stats = np.full(n, np.nan)
    best_spans = np.zeros(n, dtype=np.int32)

    for i in range(n - t_min):
        best_abs_t = 0.0
        best_t = np.nan
        best_L = 0

        for L in range(t_min, t_max + 1):
            end = i + L
            if end > n:
                break

            x_mean = (L - 1) * 0.5
            ss_xx = 0.0
            for k in range(L):
                d = float(k) - x_mean
                ss_xx += d * d

            if ss_xx < 1e-9:
                continue

            y_mean = 0.0
            for k in range(L):
                y_mean += close[i + k]
            y_mean /= L

            sp_xy = 0.0
            for k in range(L):
                sp_xy += (float(k) - x_mean) * (close[i + k] - y_mean)

            slope = sp_xy / ss_xx

            mse = 0.0
            for k in range(L):
                y_hat = slope * (float(k) - x_mean) + y_mean
                res = close[i + k] - y_hat
                mse += res * res
            if L > 2:
                mse /= (L - 2)
            else:
                continue

            se_slope = (mse / ss_xx) ** 0.5
            if se_slope < 1e-12:
                continue

            tstat = slope / se_slope
            abs_t = tstat if tstat > 0.0 else -tstat

            if abs_t > best_abs_t:
                best_abs_t = abs_t
                best_t = tstat
                best_L = L

        t_stats[i] = best_t
        best_spans[i] = best_L

        if not (best_t != best_t):  # not NaN  # noqa: PLR0124
            if best_abs_t >= min_tstat:
                if best_t > 0.0:
                    labels[i] = 1
                elif best_t < 0.0:
                    labels[i] = -1

    return labels, t_stats, best_spans


# =============================================================================
# PUBLIC CLASS
# =============================================================================

class TrendScanningLabeler:
    """Hybrid labeler: Trend Scanning (direction) + Triple Barrier (returns).

    Workflow of label_data():
      1. Trend Scanner (AFML ch. 17): OLS slope over adaptive window
         [t_min, t_max] bars. Label = 1 if |t-stat| ≥ min_tstat and
         direction matches side. best_span = optimal horizon length.
      2. Triple Barrier (dynamic horizon = best_span):
         PT = pt_width × ATR, SL = sl_width × ATR.
         Entry = open of bar t+1 (no lookahead bias).
         Realistic P&L via bid/ask OHLC.

    label() returns a raw direction DataFrame (useful for analysis).
    label_data() returns (labels, t1, returns) Series.
    """

    def __init__(
        self,
        t_min: int = 5,
        t_max: int = 30,
        min_tstat: float = 2.0,
        sl_jump_max_mult: float = 0.0,
    ) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.min_tstat = min_tstat
        self.sl_jump_max_mult: float = float(sl_jump_max_mult)

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "TrendScanningLabeler.label() is REMOVED (Wave 14 P0-8). "
            "Use label_exploration_only() for analysis-only or "
            "label_data(..., require_events=True) for production."
        )

    def label_exploration_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Trend Scanning labels for the full DataFrame.

        WARNING: This method produces LOOKAHEAD BIAS by rolling OLS over all
        bars. Use only for exploratory analysis, never for Judge training.

        Returns a DataFrame with columns: label (int8), tstat (float64),
        best_span (int32).
        """
        # CHIEF AUDIT 2026-05-23 (P_LAAG): Upgraded from DeprecationWarning to
        # RuntimeError so that `warnings.simplefilter('ignore')` in notebooks
        # cannot accidentally allow this lookahead path to reach production code.
        # If you genuinely need offline exploration, call _label_exploration_only_unsafe()
        # explicitly after acknowledging the bias.
        raise RuntimeError(
            "label_exploration_only() produces LOOKAHEAD BIAS (forward OLS). "
            "It is FORBIDDEN in any training or backtest loop. "
            "For production labeling use label_data(..., require_events=True, "
            "use_fixed_horizon=True). "
            "For offline research only, call _label_exploration_only_unsafe() "
            "to acknowledge the bias explicitly."
        )
        if df.empty or "close" not in df.columns:
            return pd.DataFrame(
                {"label": [], "tstat": [], "best_span": []},
                index=df.index,
            )
        close_arr = df["close"].values.astype(np.float64)
        labels_arr, tstat_arr, span_arr = _trend_scan_core(
            close_arr, int(self.t_min), int(self.t_max), float(self.min_tstat)
        )
        return pd.DataFrame(
            {"label": labels_arr, "tstat": tstat_arr, "best_span": span_arr},
            index=df.index,
        )

    def _label_exploration_only_unsafe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Research-only escape hatch for label_exploration_only().

        Calling this method signals explicit acknowledgement of the lookahead
        bias.  NEVER call this from any training loop, backtest engine, or
        pipeline that feeds a live system.
        """
        if df.empty or "close" not in df.columns:
            return pd.DataFrame(
                {"label": [], "tstat": [], "best_span": []},
                index=df.index,
            )
        close_arr = df["close"].values.astype(np.float64)
        labels_arr, tstat_arr, span_arr = _trend_scan_core(
            close_arr, int(self.t_min), int(self.t_max), float(self.min_tstat)
        )
        return pd.DataFrame(
            {"label": labels_arr, "tstat": tstat_arr, "best_span": span_arr},
            index=df.index,
        )

    def label_data(
        self,
        df: pd.DataFrame,
        event_timestamps: pd.DatetimeIndex | None = None,
        side: str = "LONG",
        pt_width: float = 2.0,
        sl_width: float = 1.0,
        spread: float = 0.0,
        require_events: bool = True,
        # CHIEF AUDIT 2026-05-23 (P1.1): use_fixed_horizon eliminates the
        # oracle-span lookahead.  When False (legacy), the Triple Barrier
        # horizon = TrendScan best_span (future OLS knows the optimal hold
        # duration).  When True, horizon = self.t_max for all events — the
        # PT/SL barriers remain causal; only the timeout is fixed.
        # bidirectional_backtest passes True; legacy callers keep False.
        use_fixed_horizon: bool = False,
        # SHORT-model verbetering (Agent C — 2026-05-22):
        # Optionele side-specifieke barrier-overrides. Wanneer opgegeven voor
        # side="SHORT", vervangen ze pt_width/sl_width voor SHORT-labeling.
        # Activeer via conf/symbols/{symbol}.yaml:
        #   label_pt_width_short: 1.5
        #   label_sl_width_short: 0.8
        # Reden: SHORT trades in bear-trends bereiken PT sneller (trend-
        # versterking). Een kleiner PT geeft meer label=1 samples →
        # betere class balance → SHORT-model leert winst herkennen.
        # Tight SL filtert onterechte SHORT-entries scherper → schonere
        # negatieve voorbeelden.
        pt_width_short: float | None = None,
        sl_width_short: float | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Hybrid labeler: Trend Scanning direction + Triple Barrier returns.

        Step 1 — Trend Scanner: determines per bar whether a statistically
                  significant trend exists (|t-stat| ≥ min_tstat) and its
                  duration (best_span).
        Step 2 — Triple Barrier (dynamic horizon = best_span):
                  Determines whether the move actually reached the PT.
                  barrier_lbl=1 confirms; 0 (SL/timeout) is a negative label.

        Args:
            df              : DataFrame with at minimum open/high/low/close.
            event_timestamps: Subset of timestamps (CUSUM events) or None for all bars.
            side            : 'LONG' or 'SHORT'.
            pt_width        : Profit-taking multiple of ATR (default 2.0).
            sl_width        : Stop-loss multiple of ATR (default 1.0).
            spread          : Half bid-ask spread for entry price adjustment.
            require_events  : AUDIT B-2 gate. When True (default for production),
                              ``event_timestamps`` MUST be supplied so labels
                              are scoped to CUSUM events; passing ``None``
                              raises ``ValueError``. Set False only for
                              sanity-check exploration and back-compat tests.
            pt_width_short  : Optional override for PT barrier when side='SHORT'.
                              When None, falls back to pt_width. Load from
                              conf/symbols/{symbol}.yaml: label_pt_width_short.
            sl_width_short  : Optional override for SL barrier when side='SHORT'.
                              When None, falls back to sl_width.

        Returns:
            labels  : pd.Series[int]   — 0/1, indexed on event_timestamps or df.index
            t1      : pd.Series[int]   — bar-index of trade exit (barrier or horizon)
            returns : pd.Series[float] — realised net return
        """
        if require_events and event_timestamps is None:
            raise ValueError(
                "AUDIT B-2: TrendScanningLabeler.label_data requires "
                "event_timestamps (CUSUM events). Labelling every bar produces "
                "extreme overlap and inflated sample counts. Pass an "
                "event_timestamps index from `cusum_filter(...)`, or set "
                "require_events=False to opt out (sanity-checks only)."
            )
        side_upper = side.upper()
        side_int   = 1 if side_upper == "LONG" else -1

        # SHORT-model verbetering (Agent C — 2026-05-22):
        # Pas barrier-parameters aan voor SHORT wanneer side-specifieke overrides
        # zijn opgegeven. Dit corrigeert de class-balance asymmetrie: SHORT trades
        # in bear-trends bereiken een kleiner PT vaker (label=1 rate omhoog) en
        # een tight SL filtert fout-SHORT-entries scherper (label=0 kwaliteit omhoog).
        # Backward-compat: wanneer pt_width_short/sl_width_short=None, valt de
        # code terug op de oorspronkelijke pt_width/sl_width (geen gedragsverandering).
        if side_upper == "SHORT":
            eff_pt_width = pt_width_short if pt_width_short is not None else pt_width
            eff_sl_width = sl_width_short if sl_width_short is not None else sl_width
            if pt_width_short is not None or sl_width_short is not None:
                logger.info(
                    "[label_data][SHORT] Side-specifieke barriers actief: "
                    "pt_width=%.2f (was %.2f), sl_width=%.2f (was %.2f). "
                    "Verwacht hogere label=1 rate door kleinere PT in bear-trend.",
                    eff_pt_width, pt_width, eff_sl_width, sl_width,
                )
        else:
            eff_pt_width = pt_width
            eff_sl_width = sl_width

        close_arr  = df["close"].values.astype(np.float64)
        n          = len(close_arr)

        # ── 1. Trend Scanner ──────────────────────────────────────────────────
        _, ts_tstats, ts_spans = _trend_scan_core(
            close_arr, int(self.t_min), int(self.t_max), float(self.min_tstat)
        )

        if side_upper == "LONG":
            binary = (ts_tstats >= self.min_tstat).astype(np.int8)
        else:
            binary = (ts_tstats <= -self.min_tstat).astype(np.int8)

        # ── 2. Triple Barrier (dynamic horizon = best_span) ───────────────────
        open_arr = (
            df["open"].values.astype(np.float64)
            if "open" in df.columns
            else close_arr
        )
        bid_high = (
            df["bid_high"].values.astype(np.float64)
            if "bid_high" in df.columns
            else df["high"].values.astype(np.float64)
        )
        bid_low = (
            df["bid_low"].values.astype(np.float64)
            if "bid_low" in df.columns
            else df["low"].values.astype(np.float64)
        )
        ask_high = (
            df["ask_high"].values.astype(np.float64)
            if "ask_high" in df.columns
            else df["high"].values.astype(np.float64)
        )
        ask_low = (
            df["ask_low"].values.astype(np.float64)
            if "ask_low" in df.columns
            else df["low"].values.astype(np.float64)
        )

        # ── SPREAD-FIX: apply half_spread to synthetic bid/ask proxies ──────────
        # When only mid-price OHLC is available, we approximate bid/ask by
        # widening the high/low by half_spread.  Without this, the barrier
        # monitor uses unwidened prices while the entry price IS widened —
        # an inconsistency that over-counts profitable PT hits by ~spread/ATR.
        _hs: float = float(spread) / 2.0  # half-spread already halved again below
        # Note: spread param = full spread; _triple_barrier_per_event gets half_spread
        # = spread/2, so the proxy widening should equal that same half_spread.
        # Reuse the bid_high/ask_high already built above — but widen them.
        if "bid_high" not in df.columns:
            _widen = np.full(len(close_arr), _hs, dtype=np.float64)
            bid_high = bid_high - _widen   # bid = mid - hs
            bid_low  = bid_low  - _widen
            ask_high = ask_high + _widen   # ask = mid + hs
            ask_low  = ask_low  + _widen

        if "feat_vol_gk" in df.columns:
            atr = df["feat_vol_gk"].ffill().fillna(1e-5).values.astype(np.float64)
        else:
            _h = df["high"].values.astype(np.float64)
            _l = df["low"].values.astype(np.float64)
            _prev_c = np.roll(close_arr, 1)
            _tr = np.maximum(_h - _l, np.abs(_h - _prev_c))
            atr = (
                pd.Series(_tr)
                .rolling(14)
                .mean()
                .ffill()
                .fillna(1e-5)
                .to_numpy(dtype=np.float64)
            )

        _high_arr = (
            df["high"].values.astype(np.float64)
            if "high" in df.columns
            else close_arr
        )
        _low_arr = (
            df["low"].values.astype(np.float64)
            if "low" in df.columns
            else close_arr
        )
        _wick_arr = (_high_arr - _low_arr) / (atr + 1e-9)

        if use_fixed_horizon:
            # CHIEF AUDIT 2026-05-23 (P1.1): Fixed horizon removes oracle bias.
            # TrendScan best_span is forward-looking (optimal exit in hindsight);
            # using it as the Triple Barrier timeout leaks future information.
            # With use_fixed_horizon=True, all events timeout at t_max bars —
            # the PT/SL barriers still trigger causally from actual price data.
            horizons = np.full(n, int(self.t_max), dtype=np.int32)
        else:
            horizons = np.clip(ts_spans.astype(np.int32), self.t_min, self.t_max)
        all_indices = np.arange(n, dtype=np.int32)

        _barrier_lbl, t1_barrier, realized_ret = _triple_barrier_per_event(
            open_arr, bid_high, bid_low, ask_high, ask_low,
            close_arr, atr,
            all_indices, horizons,
            float(eff_pt_width), float(eff_sl_width), side_int,
            float(spread) / 2.0,
            1,          # execution_delay_bars
            _wick_arr,
            float(self.sl_jump_max_mult),
        )

        combined = np.where(
            (binary == 1) & (_barrier_lbl == 1), 1, 0
        ).astype(int)

        t1_abs = np.clip(t1_barrier, 0, n - 1)

        logger.info(
            "[label_data][%s] TrendScan+Barrier: %d positive labels / %d bars "
            "(PT=%.1f×ATR, SL=%.1f×ATR).",
            side, int(combined.sum()), n, eff_pt_width, eff_sl_width,
        )

        labels_s = pd.Series(combined,     index=df.index, dtype=int)
        t1_s     = pd.Series(t1_abs,       index=df.index, dtype=int)
        rets_s   = pd.Series(realized_ret, index=df.index, dtype=float)

        if event_timestamps is not None:
            labels_s = labels_s.reindex(event_timestamps, fill_value=0)
            t1_s     = t1_s.reindex(event_timestamps,     fill_value=-1)
            rets_s   = rets_s.reindex(event_timestamps,   fill_value=0.0)

        return labels_s, t1_s, rets_s
