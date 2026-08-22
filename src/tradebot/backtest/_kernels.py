"""_kernels.py — Numba-compiled core for non-overlapping trade simulation.

Extracted from train_regime.py lines 285-350.

IMPORTANT — Side encoding (L4-NOTE, preserved from source):
  This kernel uses a NON-STANDARD internal encoding:
    Long  = 2,  Short = 0,  Flat = 1
  The rest of the codebase uses +1 / -1 / 0.
  Reason: using 0 for Flat clashed with Short in early array pre-allocation.
  Conversion back: int(side == 2) - int(side == 0)
  Callers (internal_backtest, bidirectional_backtest) handle this conversion.

Numba cache is enabled: first-run compilation is cached in .numba_cache/.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def calc_non_overlapping_stats(
    probs_short: np.ndarray,     # float64, C-contiguous
    probs_long: np.ndarray,      # float64, C-contiguous
    ret_short: np.ndarray,       # float64, C-contiguous — net TBM returns
    ret_long: np.ndarray,        # float64, C-contiguous — net TBM returns
    t1_short: np.ndarray,        # int32,   C-contiguous — exit bar index
    t1_long: np.ndarray,         # int32,   C-contiguous — exit bar index
    spreads: np.ndarray,         # float64, C-contiguous — per-bar cost
    min_conf_long: np.ndarray,   # float64, C-contiguous — per-bar threshold
    min_conf_short: np.ndarray,  # float64, C-contiguous — per-bar threshold
):
    """Simulate a non-overlapping trade sequence bar-by-bar.

    On each bar the kernel decides:
      - Signal? (prob >= threshold)
      - Conflict? (both Long and Short) → ratio tie-breaker
      - Jump: skip to bar AFTER the exit bar of the current trade

    Returns
    -------
    active_rets   : float64[n_trades]  — net returns per trade
    active_sides  : int32[n_trades]    — 2=Long, 0=Short (L4-NOTE encoding)
    active_indices: int32[n_trades]    — entry bar position
    """
    n = len(probs_long)
    active_rets    = np.zeros(n, dtype=np.float64)
    active_sides   = np.zeros(n, dtype=np.int32)
    active_indices = np.zeros(n, dtype=np.int32)
    count = 0

    i = 0
    while i < n:
        p_long  = probs_long[i]
        p_short = probs_short[i]
        side = 1  # Flat

        thresh_l = min_conf_long[i]
        thresh_s = min_conf_short[i]

        is_long_signal  = p_long  >= thresh_l
        is_short_signal = p_short >= thresh_s

        if is_long_signal and not is_short_signal:
            side = 2
        elif is_short_signal and not is_long_signal:
            side = 0
        elif is_long_signal and is_short_signal:
            ratio_l = p_long  / thresh_l
            ratio_s = p_short / thresh_s
            side = 2 if ratio_l > ratio_s else 0

        if side == 1:
            i += 1
            continue

        raw_ret = ret_long[i]  if side == 2 else ret_short[i]
        t1_idx  = t1_long[i]  if side == 2 else t1_short[i]
        # CHIEF AUDIT 2026-05-23 (H2): spread-aftrek-audit — gehandhaafd.
        # TBM (_triple_barrier_per_event) trekt entry half_spread af bij
        # entry-prijs en exit half_spread af bij timeout/PT/SL. Op de Bybit
        # mid-price (bid/ask niet beschikbaar) is dat een MODEL, geen
        # mechanische bid/ask-fill. De extra ``spreads[i] * 0.5`` hier modelleert
        # de "verborgen" maker→taker slippage die niet door de TBM mid-price
        # geometrie wordt gevangen — bv. queue-jump kosten en adverse-selection
        # premie bij snelle exits. Voor bid/ask data (spread_val = 0) is dit
        # automatisch 0, dus geen dubbel-charge. Behouden om backtest
        # conservatief te houden; de bandit-reward in bidirectional.py / per_side.py
        # gebruikt dezelfde half-spread aftrek (CHIEF AUDIT K2) zodat learner
        # en evaluator dezelfde net-PnL zien.
        net_ret = raw_ret - spreads[i] * 0.5  # half-spread per leg

        active_rets[count]    = net_ret
        active_sides[count]   = side
        active_indices[count] = i
        count += 1

        i = t1_idx + 1 if t1_idx > i else i + 1

    return active_rets[:count], active_sides[:count], active_indices[:count]
