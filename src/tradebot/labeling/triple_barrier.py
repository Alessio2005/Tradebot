"""triple_barrier.py — Triple Barrier Numba kernel.

Extracted from train_regime.py lines 720-925.
Bit-identical to the monolith: same Numba kernel, same signature.

Public name: triple_barrier_per_event (drop the leading underscore — it is
now a proper module-level API, not a monolith-private function).

The underscore variant _triple_barrier_per_event is re-exported for
backward-compat during the strangler-fig transition period; import it as:
    from tradebot.labeling.triple_barrier import _triple_barrier_per_event

Strangler-fig: once trend_scanning.py imports from here and the equivalence
test passes, remove lines 720-925 from train_regime.py.
"""
from __future__ import annotations

import numpy as np
from numba import njit

# =============================================================================
# NUMBA KERNEL — bit-identical to train_regime.py lines 723-925
# =============================================================================

@njit(cache=True)
def _triple_barrier_per_event(
    open_arr: np.ndarray,
    bid_high: np.ndarray,
    bid_low: np.ndarray,
    ask_high: np.ndarray,
    ask_low: np.ndarray,
    close_arr: np.ndarray,
    atr: np.ndarray,
    event_indices: np.ndarray,
    horizons: np.ndarray,
    pt_width: float,
    sl_width: float,
    side: int,
    half_spread: float,
    execution_delay_bars: int,
    jump_ratio_arr: np.ndarray,
    sl_jump_max_mult: float,
    min_future_bars: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triple Barrier kernel — dynamic horizon per event.

    Horizon comes from the Trend Scanner (best_span per bar) so barrier
    width aligns with the detected trend duration.

    EXEC-LATENCY-FIX (Item 3):
      Signal computed at bar i; entry on open[i+1+execution_delay_bars].
      Default execution_delay_bars=1 (API roundtrip + slippage).

    SPREAD-FIX:
      For mid-price data (Bybit): entry = mid ± half_spread.
      For bid/ask data (Dukascopy): half_spread=0.0.

    AUDIT-FIX (Round 3 — SOL adaptive SL):
      wick_ratio = (high-low)/ATR. High wick → wider SL by sl_jump_max_mult.
      sl_jump_max_mult=0.0 → disabled (BTC/ETH default).
      sl_jump_max_mult=0.5 → SL up to 50% wider at extreme wicks (SOL).

    Returns:
        barrier_labels : int8  (1 = PT hit, 0 = SL or horizon timeout)
        t1_indices     : int32 (bar-index of trade close)
        realized_ret   : float64 (net directional return)
    """
    n_events = len(event_indices)
    n_total  = len(close_arr)

    barrier_labels = np.zeros(n_events, dtype=np.int8)
    t1_indices     = np.full(n_events, -1, dtype=np.int32)
    realized_ret   = np.zeros(n_events, dtype=np.float64)

    entry_offset: int = 1 + int(execution_delay_bars)
    entry_offset = max(entry_offset, 1)

    # CHIEF AUDIT 2026-05-23 (P-9): min_future_bars=1 is te krap als
    # execution_delay_bars>=1 (entry_offset wordt dan 2). De PT/SL-loop
    # heeft op zijn minst (execution_delay_bars+1) bars na het signaal
    # nodig om de eerste echte exit-evaluatie te kunnen doen. Lift het
    # bodem naar (execution_delay_bars + 1) zodat de check op de timeout-
    # index (regel ~85) altijd voldoende ruimte garandeert.
    min_future_bars = max(min_future_bars, int(execution_delay_bars) + 1)

    for j in range(n_events):
        i       = event_indices[j]
        horizon = int(horizons[j])

        timeout_idx_raw = i + entry_offset + horizon - 1

        # P0-15: discard events without sufficient future bars (no clipping)
        if timeout_idx_raw >= n_total - min_future_bars:
            t1_indices[j] = -1
            barrier_labels[j] = -9
            continue

        if i < 0 or i + entry_offset >= n_total:
            t1_indices[j] = -1
            barrier_labels[j] = -9
            continue

        curr_atr = atr[i]
        if np.isnan(curr_atr) or curr_atr <= 0:
            t1_indices[j] = -1
            barrier_labels[j] = -9
            continue

        raw_open  = open_arr[i + entry_offset]
        curr_price = raw_open + half_spread if side == 1 else raw_open - half_spread
        if curr_price <= 0.0:
            t1_indices[j] = -1
            barrier_labels[j] = -9
            continue

        dist_pt = curr_atr * pt_width

        # ── SOL adaptive SL ───────────────────────────────────────────────────
        # CHIEF AUDIT 2026-05-23 (P-4): lookahead-fix. jump_ratio[i] op event-
        # bar i = (high[i]-low[i])/atr[i] — beide bar-i grootheden, wick is
        # pas bekend AAN HET EINDE van bar i. Gebruik bar i-1 (consistent met
        # CUSUM ATR-shift). Voor i=0 fall back op 0.0 (max(i-1, 0)).
        _eff_sl_mult: float = sl_width
        if sl_jump_max_mult > 0.0:
            _jr_idx: int = max(i - 1, 0)
            _jr_i: float = float(jump_ratio_arr[_jr_idx]) if _jr_idx < len(jump_ratio_arr) else 0.0
            _wick_score: float = (_jr_i - 1.0) / 2.0
            _wick_score = max(_wick_score, 0.0)
            _wick_score = min(_wick_score, 1.0)
            _eff_sl_mult = sl_width * (1.0 + sl_jump_max_mult * _wick_score)
        dist_sl = curr_atr * _eff_sl_mult

        if side == 1:   # LONG
            target_price = curr_price + dist_pt
            stop_price   = curr_price - dist_sl
        else:           # SHORT
            target_price = curr_price - dist_pt
            stop_price   = curr_price + dist_sl

        exit_bar = min(timeout_idx_raw, n_total - 1)

        for bar in range(i + entry_offset, exit_bar + 1):
            if side == 1:   # LONG: monitor via ask-high (worst-case PT) and bid-low (SL)
                if ask_high[bar] >= target_price:
                    barrier_labels[j] = 1
                    exit_bar = bar
                    break
                if bid_low[bar] <= stop_price:
                    barrier_labels[j] = 0
                    exit_bar = bar
                    break
            else:           # SHORT: monitor via bid-low (PT) and ask-high (SL)
                if bid_low[bar] <= target_price:
                    barrier_labels[j] = 1
                    exit_bar = bar
                    break
                if ask_high[bar] >= stop_price:
                    barrier_labels[j] = 0
                    exit_bar = bar
                    break

        t1_indices[j] = exit_bar

        # ── Realised return ───────────────────────────────────────────────────
        if exit_bar < n_total:
            exit_price = close_arr[exit_bar]
            if side == 1:
                realized_ret[j] = (exit_price - curr_price) / curr_price
            else:
                realized_ret[j] = (curr_price - exit_price) / curr_price
        else:
            realized_ret[j] = 0.0

    return barrier_labels, t1_indices, realized_ret


# ── Public alias (drop leading underscore for the module-level API) ───────────
triple_barrier_per_event = _triple_barrier_per_event


# =============================================================================
# TripleBarrierLabeler — scikit-learn-compatible class wrapper
# =============================================================================

import logging as _logging

import pandas as pd

_logger = _logging.getLogger(__name__)


class TripleBarrierLabeler:
    """Triple Barrier Method labeler (López de Prado AFML ch. 3).

    Wraps the Numba kernel ``_triple_barrier_per_event`` with a DataFrame-
    compatible interface.  Designed for use downstream of CUSUM events and
    TrendScanning horizons.

    Parameters
    ----------
    pt_width:
        Profit-taking barrier as ATR multiple (default 2.0).
    sl_width:
        Stop-loss barrier as ATR multiple (default 1.0).
    half_spread:
        Simulated half-spread in price units (0.0 = mid-price only).
    execution_delay_bars:
        Bars between signal generation and entry (default 1).
    sl_jump_max_mult:
        Adaptive SL multiplier for jump risk (0.0 = disabled; 0.5 for SOL).
    """

    def __init__(
        self,
        pt_width: float = 2.0,
        sl_width: float = 1.0,
        half_spread: float = 0.0,
        execution_delay_bars: int = 1,
        sl_jump_max_mult: float = 0.0,
    ) -> None:
        self.pt_width = float(pt_width)
        self.sl_width = float(sl_width)
        self.half_spread = float(half_spread)
        self.execution_delay_bars = int(execution_delay_bars)
        self.sl_jump_max_mult = float(sl_jump_max_mult)

    def label(
        self,
        df: pd.DataFrame,
        event_indices: np.ndarray,
        horizons: np.ndarray,
        atr: np.ndarray,
        side: int = 1,
        half_spread_arr: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Apply triple barrier labels to a set of events.

        Parameters
        ----------
        df:
            OHLCV bar DataFrame (must contain 'open', 'close', and either
            bid_high/bid_low/ask_high/ask_low or 'high'/'low' columns).
        event_indices:
            Bar indices of CUSUM events (int32 array).
        horizons:
            Per-event barrier horizon in bars (int32 array, same length).
        atr:
            ATR array aligned to df (float64, same length as df).
        side:
            +1 = LONG events, -1 = SHORT events.
        half_spread_arr:
            Optional per-bar half-spread array (price units). Used to widen
            bid/ask quartet when only mid-price OHLC is available
            (AUDIT E-2). If ``None``, falls back to the scalar
            ``self.half_spread``.

        Returns
        -------
        pd.DataFrame with columns:
            barrier_label (int8), t1_idx (int32), realized_return (float64)
        Index aligned to ``df.index[event_indices]``.
        """
        close = df["close"].to_numpy(dtype=np.float64)
        open_ = df["open"].to_numpy(dtype=np.float64) if "open" in df.columns else close.copy()

        if "bid_high" in df.columns:
            bid_high = df["bid_high"].to_numpy(dtype=np.float64)
            bid_low  = df["bid_low"].to_numpy(dtype=np.float64)
            ask_high = df["ask_high"].to_numpy(dtype=np.float64)
            ask_low  = df["ask_low"].to_numpy(dtype=np.float64)
        else:
            high = df["high"].to_numpy(dtype=np.float64) if "high" in df.columns else close.copy()
            low  = df["low"].to_numpy(dtype=np.float64)  if "low"  in df.columns else close.copy()
            # AUDIT E-2: widen the ask_high / bid_low proxies by the half-spread
            # so the barrier monitor does not over-count PT triggers.
            if half_spread_arr is None:
                hs = np.full(len(close), float(self.half_spread), dtype=np.float64)
            else:
                hs = np.asarray(half_spread_arr, dtype=np.float64)
                if hs.shape[0] != len(close):
                    raise ValueError(
                        f"half_spread_arr length {hs.shape[0]} != n_bars {len(close)}"
                    )
            # CHIEF AUDIT-FIX (Sim-to-Reality #20):
            #   Stale or spike half_spread values (e.g. 2× the rolling median
            #   during data gaps) inflate the synthetic ask_high beyond the
            #   true PT level, producing spurious barrier_label=1 hits.  We
            #   clamp ``hs`` to ``[0, max(3 × rolling_median, ATR-scaled cap)]``
            #   so a single corrupted bar cannot poison the barrier monitor.
            #
            #   Cap = max(3 × median(hs), 0.5 × ATR) — accommodates the rare
            #   but legitimate wide-spread regime while filtering print noise.
            hs_median = float(np.nanmedian(hs[hs > 0.0])) if np.any(hs > 0.0) else 0.0
            atr_cap = 0.5 * np.nanmedian(np.asarray(atr, dtype=np.float64))
            hs_cap = max(3.0 * hs_median, float(atr_cap), float(self.half_spread))
            n_spiked = int(np.sum(hs > hs_cap))
            if n_spiked > 0:
                _logger.warning(
                    "TripleBarrier (Sim-to-Reality #20): clamped %d half_spread "
                    "spike bar(s) (>%.6f, median=%.6f). Synthetic bid/ask widened "
                    "beyond rolling envelope — check upstream spread feed for gaps.",
                    n_spiked, hs_cap, hs_median,
                )
            hs = np.minimum(np.maximum(hs, 0.0), hs_cap)
            ask_high = high + hs
            ask_low  = low  + hs
            bid_high = high - hs
            bid_low  = low  - hs

        # ── AUDIT B-3: jump-ratio = wick / ATR. Replaces the zero placeholder so
        #               sl_jump_max_mult actually has an effect for SOL events.
        if "high" in df.columns and "low" in df.columns:
            high_arr = df["high"].to_numpy(dtype=np.float64)
            low_arr  = df["low"].to_numpy(dtype=np.float64)
            jump = (high_arr - low_arr) / (np.asarray(atr, dtype=np.float64) + 1e-9)
            jump = np.where(np.isfinite(jump), jump, 0.0).astype(np.float64)
        else:
            jump = np.zeros(len(close), dtype=np.float64)

        ev_idx = np.asarray(event_indices, dtype=np.int32)
        hz     = np.asarray(horizons, dtype=np.int32)
        atr_   = np.asarray(atr, dtype=np.float64)

        barrier_labels, t1_indices, realized_ret = _triple_barrier_per_event(
            open_, bid_high, bid_low, ask_high, ask_low, close,
            atr_, ev_idx, hz,
            self.pt_width, self.sl_width, int(side),
            self.half_spread, self.execution_delay_bars,
            jump, self.sl_jump_max_mult,
        )

        # P0-15: filter out rejected events (insufficient future bars)
        valid_mask = t1_indices >= 0
        if not valid_mask.all():
            n_rejected = int((~valid_mask).sum())
            _logger.info(
                "TripleBarrierLabeler: %d events rejected (insufficient future bars)", n_rejected
            )
        barrier_labels = barrier_labels[valid_mask]
        t1_indices = t1_indices[valid_mask]
        realized_ret = realized_ret[valid_mask]
        ev_idx = ev_idx[valid_mask]

        event_ts = df.index[ev_idx]
        result = pd.DataFrame(
            {
                "barrier_label":    barrier_labels,
                "t1_idx":           t1_indices,
                "realized_return":  realized_ret,
            },
            index=event_ts,
        )
        _logger.info(
            "TripleBarrierLabeler: %d events → %d PT / %d SL (side=%+d)",
            len(ev_idx),
            int((barrier_labels == 1).sum()),
            int((barrier_labels == 0).sum()),
            side,
        )
        return result
