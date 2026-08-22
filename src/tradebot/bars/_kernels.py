"""Numba kernels for imbalance bars and runs bars.

Single source of truth — imported by imbalance.py and runs.py.
Never import from bars.py (legacy) in new code.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def numba_imbalance_bars(
    dates, opens, highs, lows, closes,
    tick_volumes, imbalances, thresholds,
    bid_opens, bid_highs, bid_lows, bid_closes,
    ask_opens, ask_highs, ask_lows, ask_closes,
):
    """Volume Imbalance Bar kernel.

    Audit-fix K4 (Phantom Bars): cap to one bar emission per input tick.
    Residual imbalance rolls to the next real tick (information preserved).
    """
    n = len(closes)
    out_dates  = np.zeros(n, dtype=np.int64)
    out_open   = np.zeros(n, dtype=np.float64)
    out_high   = np.zeros(n, dtype=np.float64)
    out_low    = np.zeros(n, dtype=np.float64)
    out_close  = np.zeros(n, dtype=np.float64)
    out_vol    = np.zeros(n, dtype=np.float64)
    out_bid_open  = np.zeros(n, dtype=np.float64)
    out_bid_high  = np.zeros(n, dtype=np.float64)
    out_bid_low   = np.zeros(n, dtype=np.float64)
    out_bid_close = np.zeros(n, dtype=np.float64)
    out_ask_open  = np.zeros(n, dtype=np.float64)
    out_ask_high  = np.zeros(n, dtype=np.float64)
    out_ask_low   = np.zeros(n, dtype=np.float64)
    out_ask_close = np.zeros(n, dtype=np.float64)

    current_imbalance = 0.0
    current_tick_vol  = 0.0
    idx_start = 0
    bar_idx   = 0
    cur_h = -1.0; cur_l = 1e15
    cur_bid_h = -1.0; cur_bid_l = 1e15
    cur_ask_h = -1.0; cur_ask_l = 1e15
    next_open     = opens[0]
    next_bid_open = bid_opens[0]
    next_ask_open = ask_opens[0]

    for i in range(n):
        current_imbalance += imbalances[i]
        current_tick_vol  += tick_volumes[i]
        cur_threshold = thresholds[i]
        if cur_threshold <= 1e-9:
            cur_threshold = 10_000.0

        if i == idx_start:
            next_open = opens[i]; cur_h = highs[i]; cur_l = lows[i]
            next_bid_open = bid_opens[i]; cur_bid_h = bid_highs[i]; cur_bid_l = bid_lows[i]
            next_ask_open = ask_opens[i]; cur_ask_h = ask_highs[i]; cur_ask_l = ask_lows[i]
        else:
            cur_h = max(cur_h, highs[i])
            cur_l = min(cur_l, lows[i])
            cur_bid_h = max(cur_bid_h, bid_highs[i])
            cur_bid_l = min(cur_bid_l, bid_lows[i])
            cur_ask_h = max(cur_ask_h, ask_highs[i])
            cur_ask_l = min(cur_ask_l, ask_lows[i])

        if abs(current_imbalance) >= cur_threshold:
            out_dates[bar_idx] = dates[i]
            out_open[bar_idx]  = next_open
            out_high[bar_idx]  = cur_h
            out_low[bar_idx]   = cur_l
            out_close[bar_idx] = closes[i]
            ratio = cur_threshold / abs(current_imbalance)
            allocated_vol = current_tick_vol * ratio
            out_vol[bar_idx] = allocated_vol
            out_bid_open[bar_idx]  = next_bid_open
            out_bid_high[bar_idx]  = cur_bid_h
            out_bid_low[bar_idx]   = cur_bid_l
            out_bid_close[bar_idx] = bid_closes[i]
            out_ask_open[bar_idx]  = next_ask_open
            out_ask_high[bar_idx]  = cur_ask_h
            out_ask_low[bar_idx]   = cur_ask_l
            out_ask_close[bar_idx] = ask_closes[i]
            bar_idx += 1
            if current_imbalance > 0:
                current_imbalance -= cur_threshold
            else:
                current_imbalance += cur_threshold
            current_tick_vol -= allocated_vol
            cur_h = highs[i]; cur_l = lows[i]; next_open = closes[i]
            cur_bid_h = bid_highs[i]; cur_bid_l = bid_lows[i]; next_bid_open = bid_closes[i]
            cur_ask_h = ask_highs[i]; cur_ask_l = ask_lows[i]; next_ask_open = ask_closes[i]
            idx_start = i + 1

    return (
        out_dates[:bar_idx],
        out_open[:bar_idx],  out_high[:bar_idx],  out_low[:bar_idx],  out_close[:bar_idx],
        out_vol[:bar_idx],
        out_bid_open[:bar_idx], out_bid_high[:bar_idx], out_bid_low[:bar_idx], out_bid_close[:bar_idx],
        out_ask_open[:bar_idx], out_ask_high[:bar_idx], out_ask_low[:bar_idx], out_ask_close[:bar_idx],
    )


@njit(cache=True)
def numba_runs_bars(
    dates, opens, highs, lows, closes,
    tick_volumes, buy_vols, sell_vols, thresholds,
    bid_opens, bid_highs, bid_lows, bid_closes,
    ask_opens, ask_highs, ask_lows, ask_closes,
):
    """Volume Runs Bar kernel.

    Closes a bar when max(cum_buy_vol, cum_sell_vol) >= threshold.
    Returns per-bar buy/sell volumes as microstructure features.
    """
    n = len(closes)
    out_dates    = np.zeros(n, dtype=np.int64)
    out_open     = np.zeros(n, dtype=np.float64)
    out_high     = np.zeros(n, dtype=np.float64)
    out_low      = np.zeros(n, dtype=np.float64)
    out_close    = np.zeros(n, dtype=np.float64)
    out_vol      = np.zeros(n, dtype=np.float64)
    out_buy_vol  = np.zeros(n, dtype=np.float64)
    out_sell_vol = np.zeros(n, dtype=np.float64)
    out_bid_open  = np.zeros(n, dtype=np.float64)
    out_bid_high  = np.zeros(n, dtype=np.float64)
    out_bid_low   = np.zeros(n, dtype=np.float64)
    out_bid_close = np.zeros(n, dtype=np.float64)
    out_ask_open  = np.zeros(n, dtype=np.float64)
    out_ask_high  = np.zeros(n, dtype=np.float64)
    out_ask_low   = np.zeros(n, dtype=np.float64)
    out_ask_close = np.zeros(n, dtype=np.float64)

    cum_buy_vol  = 0.0
    cum_sell_vol = 0.0
    cum_tick_vol = 0.0
    bar_idx   = 0
    idx_start = 0
    cur_h = -1.0; cur_l = 1e15
    cur_bid_h = -1.0; cur_bid_l = 1e15
    cur_ask_h = -1.0; cur_ask_l = 1e15
    next_open     = opens[0]
    next_bid_open = bid_opens[0]
    next_ask_open = ask_opens[0]

    for i in range(n):
        cum_buy_vol  += buy_vols[i]
        cum_sell_vol += sell_vols[i]
        cum_tick_vol += tick_volumes[i]
        cur_threshold = thresholds[i]
        if cur_threshold <= 1e-9:
            cur_threshold = 10_000.0

        if i == idx_start:
            next_open = opens[i]; cur_h = highs[i]; cur_l = lows[i]
            next_bid_open = bid_opens[i]; cur_bid_h = bid_highs[i]; cur_bid_l = bid_lows[i]
            next_ask_open = ask_opens[i]; cur_ask_h = ask_highs[i]; cur_ask_l = ask_lows[i]
        else:
            cur_h = max(cur_h, highs[i])
            cur_l = min(cur_l, lows[i])
            cur_bid_h = max(cur_bid_h, bid_highs[i])
            cur_bid_l = min(cur_bid_l, bid_lows[i])
            cur_ask_h = max(cur_ask_h, ask_highs[i])
            cur_ask_l = min(cur_ask_l, ask_lows[i])

        if max(cum_buy_vol, cum_sell_vol) >= cur_threshold:
            out_dates[bar_idx]    = dates[i]
            out_open[bar_idx]     = next_open
            out_high[bar_idx]     = cur_h
            out_low[bar_idx]      = cur_l
            out_close[bar_idx]    = closes[i]
            out_vol[bar_idx]      = cum_tick_vol
            out_buy_vol[bar_idx]  = cum_buy_vol
            out_sell_vol[bar_idx] = cum_sell_vol
            out_bid_open[bar_idx]  = next_bid_open
            out_bid_high[bar_idx]  = cur_bid_h
            out_bid_low[bar_idx]   = cur_bid_l
            out_bid_close[bar_idx] = bid_closes[i]
            out_ask_open[bar_idx]  = next_ask_open
            out_ask_high[bar_idx]  = cur_ask_h
            out_ask_low[bar_idx]   = cur_ask_l
            out_ask_close[bar_idx] = ask_closes[i]
            bar_idx += 1
            # CHIEF AUDIT 2026-05-23 (FIX 5 / runs-bar residual): only reset
            # the side that actually triggered the threshold; the unconsumed
            # opposite-side flow must roll into the next bar to avoid silently
            # dropping information.  Mirrors the imbalance-bar residual logic
            # (lines 88-92) and the dollar-bar overflow (lines 282-294).
            # Tick volume is also rolled by the residual ratio of the triggering
            # side so the per-bar volume tally stays consistent.
            if cum_buy_vol >= cur_threshold and cum_buy_vol >= cum_sell_vol:
                buy_residual = cum_buy_vol - cur_threshold
                if cum_buy_vol > 1e-12:
                    tick_residual = cum_tick_vol * (buy_residual / cum_buy_vol)
                else:
                    tick_residual = 0.0
                cum_buy_vol  = buy_residual if buy_residual > 0.0 else 0.0
                cum_tick_vol = tick_residual if tick_residual > 0.0 else 0.0
                # Sell-side flow keeps accumulating (it never hit the threshold).
            else:
                sell_residual = cum_sell_vol - cur_threshold
                if cum_sell_vol > 1e-12:
                    tick_residual = cum_tick_vol * (sell_residual / cum_sell_vol)
                else:
                    tick_residual = 0.0
                cum_sell_vol = sell_residual if sell_residual > 0.0 else 0.0
                cum_tick_vol = tick_residual if tick_residual > 0.0 else 0.0
                # Buy-side flow keeps accumulating.
            cur_h = highs[i]; cur_l = lows[i]; next_open = closes[i]
            cur_bid_h = bid_highs[i]; cur_bid_l = bid_lows[i]; next_bid_open = bid_closes[i]
            cur_ask_h = ask_highs[i]; cur_ask_l = ask_lows[i]; next_ask_open = ask_closes[i]
            idx_start = i + 1

    return (
        out_dates[:bar_idx],
        out_open[:bar_idx],  out_high[:bar_idx],  out_low[:bar_idx],  out_close[:bar_idx],
        out_vol[:bar_idx],   out_buy_vol[:bar_idx], out_sell_vol[:bar_idx],
        out_bid_open[:bar_idx], out_bid_high[:bar_idx], out_bid_low[:bar_idx], out_bid_close[:bar_idx],
        out_ask_open[:bar_idx], out_ask_high[:bar_idx], out_ask_low[:bar_idx], out_ask_close[:bar_idx],
    )


@njit(cache=True)
def numba_dollar_bars(
    dates, opens, highs, lows, closes,
    tick_volumes, thresholds,
    bid_opens, bid_highs, bid_lows, bid_closes,
    ask_opens, ask_highs, ask_lows, ask_closes,
):
    """Dollar Bar kernel (AFML §2.3, AUDIT A-2).

    Closes a bar when the cumulative *notional* (price × volume) crosses
    ``threshold``. The notional uses the executed price of each tick (close).
    Residual notional rolls into the next bar (volume is allocated by ratio
    on overflow ticks, mirroring the imbalance-bars pattern, to keep notional
    per bar near-constant).
    """
    n = len(closes)
    out_dates    = np.zeros(n, dtype=np.int64)
    out_open     = np.zeros(n, dtype=np.float64)
    out_high     = np.zeros(n, dtype=np.float64)
    out_low      = np.zeros(n, dtype=np.float64)
    out_close    = np.zeros(n, dtype=np.float64)
    out_vol      = np.zeros(n, dtype=np.float64)
    out_notional = np.zeros(n, dtype=np.float64)
    out_bid_open  = np.zeros(n, dtype=np.float64)
    out_bid_high  = np.zeros(n, dtype=np.float64)
    out_bid_low   = np.zeros(n, dtype=np.float64)
    out_bid_close = np.zeros(n, dtype=np.float64)
    out_ask_open  = np.zeros(n, dtype=np.float64)
    out_ask_high  = np.zeros(n, dtype=np.float64)
    out_ask_low   = np.zeros(n, dtype=np.float64)
    out_ask_close = np.zeros(n, dtype=np.float64)

    cum_notional = 0.0
    cum_volume   = 0.0
    bar_idx      = 0
    idx_start    = 0
    cur_h = -1.0; cur_l = 1e15
    cur_bid_h = -1.0; cur_bid_l = 1e15
    cur_ask_h = -1.0; cur_ask_l = 1e15
    next_open     = opens[0]
    next_bid_open = bid_opens[0]
    next_ask_open = ask_opens[0]

    for i in range(n):
        tick_notional = closes[i] * tick_volumes[i]
        cum_notional += tick_notional
        cum_volume   += tick_volumes[i]
        cur_threshold = thresholds[i]
        if cur_threshold <= 1e-9:
            cur_threshold = 1_000_000.0

        if i == idx_start:
            next_open = opens[i]; cur_h = highs[i]; cur_l = lows[i]
            next_bid_open = bid_opens[i]; cur_bid_h = bid_highs[i]; cur_bid_l = bid_lows[i]
            next_ask_open = ask_opens[i]; cur_ask_h = ask_highs[i]; cur_ask_l = ask_lows[i]
        else:
            cur_h = max(cur_h, highs[i])
            cur_l = min(cur_l, lows[i])
            cur_bid_h = max(cur_bid_h, bid_highs[i])
            cur_bid_l = min(cur_bid_l, bid_lows[i])
            cur_ask_h = max(cur_ask_h, ask_highs[i])
            cur_ask_l = min(cur_ask_l, ask_lows[i])

        if cum_notional >= cur_threshold:
            out_dates[bar_idx]     = dates[i]
            out_open[bar_idx]      = next_open
            out_high[bar_idx]      = cur_h
            out_low[bar_idx]       = cur_l
            out_close[bar_idx]     = closes[i]
            out_vol[bar_idx]       = cum_volume
            out_notional[bar_idx]  = cum_notional
            out_bid_open[bar_idx]  = next_bid_open
            out_bid_high[bar_idx]  = cur_bid_h
            out_bid_low[bar_idx]   = cur_bid_l
            out_bid_close[bar_idx] = bid_closes[i]
            out_ask_open[bar_idx]  = next_ask_open
            out_ask_high[bar_idx]  = cur_ask_h
            out_ask_low[bar_idx]   = cur_ask_l
            out_ask_close[bar_idx] = ask_closes[i]
            bar_idx += 1
            # CHIEF AUDIT 2026-05-23 (P3.2): Carry the excess notional from
            # the triggering tick into the next bar.  The old code reset to 0.0,
            # discarding the fraction of the tick that overflowed the threshold.
            # At large ticks (illiquid alts, news spikes) this caused 5-15%
            # bar-size inconsistency.  The proportional volume residual is
            # allocated using the same fraction as the notional residual.
            residual_notional = cum_notional - cur_threshold
            if tick_notional > 1e-12:
                residual_volume = tick_volumes[i] * (residual_notional / tick_notional)
            else:
                residual_volume = 0.0
            cum_notional = residual_notional if residual_notional > 0.0 else 0.0
            cum_volume   = residual_volume   if residual_volume   > 0.0 else 0.0
            cur_h = highs[i]; cur_l = lows[i]; next_open = closes[i]
            cur_bid_h = bid_highs[i]; cur_bid_l = bid_lows[i]; next_bid_open = bid_closes[i]
            cur_ask_h = ask_highs[i]; cur_ask_l = ask_lows[i]; next_ask_open = ask_closes[i]
            idx_start = i + 1

    return (
        out_dates[:bar_idx],
        out_open[:bar_idx],  out_high[:bar_idx],  out_low[:bar_idx],  out_close[:bar_idx],
        out_vol[:bar_idx],   out_notional[:bar_idx],
        out_bid_open[:bar_idx], out_bid_high[:bar_idx], out_bid_low[:bar_idx], out_bid_close[:bar_idx],
        out_ask_open[:bar_idx], out_ask_high[:bar_idx], out_ask_low[:bar_idx], out_ask_close[:bar_idx],
    )
