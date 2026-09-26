"""cusum.py — Symmetric CUSUM filter + get_cusum_events (fused).

Extracted from train_regime.py lines 209-250 (_symmetric_cusum_filter)
and lines 1390-1458 (get_cusum_events).

Design decisions:
  • _symmetric_cusum_filter (Numba @njit): unchanged kernel — bit-identical
    to train_regime.py version. Only the import path changes.
  • get_cusum_events: fused function that handles ATR computation, the
    CAUSAL-FIX (one-bar shift so threshold at bar t uses ATR[t-1]),
    and degenerate-bar filtering.
  • Public API: symmetric_cusum_filter() (thin wrapper, typed) +
    get_cusum_events() (full pipeline function).

Strangler-fig: train_regime.py line 209 should import from here and
delete its own definition after equivalence test passes.
"""
from __future__ import annotations

import logging
from typing import cast

import numpy as np
import pandas as pd
from numba import njit

logger = logging.getLogger(__name__)


# =============================================================================
# NUMBA KERNEL — unchanged from train_regime.py for bit-identical output
# =============================================================================

@njit(cache=True)
def _symmetric_cusum_filter_kernel(
    prices: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Numba-optimised Symmetric CUSUM filter.

    Returns integer positions (bar indices) where accumulated price action
    breaches the dynamic threshold in either direction.

    Args:
        prices     : float64 array of close prices.
        thresholds : float64 array of per-bar dynamic thresholds (e.g. ATR-scaled).

    Returns:
        int32 array of event bar indices (pre-allocated, trimmed).
    """
    n = len(prices)
    s_pos = 0.0
    s_neg = 0.0

    event_indices = np.zeros(n, dtype=np.int32)
    count = 0

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        h = thresholds[i]

        if s_pos >= h:
            event_indices[count] = i
            count += 1
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg <= -h:
            event_indices[count] = i
            count += 1
            s_pos = 0.0
            s_neg = 0.0

    return event_indices[:count]


# =============================================================================
# PUBLIC API
# =============================================================================

def symmetric_cusum_filter(
    prices: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Thin typed wrapper around the Numba kernel.

    Parameters
    ----------
    prices     : float64 C-contiguous close-price array.
    thresholds : float64 C-contiguous per-bar threshold array (same length).

    Returns
    -------
    np.ndarray[int32]
        Sorted bar indices of detected events.
    """
    if len(prices) != len(thresholds):
        raise ValueError(
            f"prices ({len(prices)}) and thresholds ({len(thresholds)}) "
            "must have equal length."
        )
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
    return _symmetric_cusum_filter_kernel(prices, thresholds)


def directional_cusum_filter(
    prices: np.ndarray,
    thresholds_up: np.ndarray,
    thresholds_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Publieke, getypeerde ingang op de directionele kernel.

    Returns
    -------
    (up_indices, down_indices) : int32-arrays met de bars waarop de cumulatieve
        beweging de drempel omhoog resp. omlaag passeerde. Samen zijn zij
        exact `symmetric_cusum_filter` bij gelijke drempels.
    """
    if not (len(prices) == len(thresholds_up) == len(thresholds_down)):
        raise ValueError("prices en beide drempelreeksen moeten even lang zijn.")
    return _directional_cusum_filter_kernel(
        np.ascontiguousarray(prices, dtype=np.float64),
        np.ascontiguousarray(thresholds_up, dtype=np.float64),
        np.ascontiguousarray(thresholds_down, dtype=np.float64),
    )


def get_cusum_events(
    df: pd.DataFrame,
    threshold_multiplier: float = 3.5,
) -> pd.DatetimeIndex:
    """Compute CUSUM event timestamps from a bar DataFrame.

    MFT calibration: threshold_multiplier = 3.5 gives ~5-10 events/day at
    30 bars/day.  CatBoost filters down to 1-2 best trades.

    Multiplier guidelines:
      1.0 → ~30-50 events/day  (HFT/scalp)
      2.0 → ~15-25 events/day
      3.5 → ~5-10 events/day   (MFT target ✓)
      5.0 → ~2-5 events/day    (conservative swing)

    CAUSAL-FIX (Item 9):
      ATR at bar t is shifted one bar (atr_causal[t] = atr[t-1]) so the
      CUSUM threshold at bar t only uses information from bar t-1.
      Without this, ATR at bar t uses high/low of bar t to decide if bar t
      is an event — a within-bar lookahead bias.

    Parameters
    ----------
    df                   : Bar DataFrame with at minimum 'close' column
                           and a DatetimeIndex.  'feat_vol_gk' is used as
                           ATR if present; otherwise a 14-bar TR ATR is computed.
    threshold_multiplier : ATR multiplier (default 3.5 for MFT).

    Returns
    -------
    pd.DatetimeIndex
        Timestamps of detected CUSUM events, de-duplicated and sorted.
    """
    close = df["close"].to_numpy(dtype=np.float64)

    # ── ATR computation (causal: feat_vol_gk preferred) ──────────────────────
    if "feat_vol_gk" in df.columns:
        atr = df["feat_vol_gk"].ffill().fillna(1e-5).to_numpy(dtype=np.float64)
    else:
        high = df["high"].to_numpy(dtype=np.float64)
        low  = df["low"].to_numpy(dtype=np.float64)
        prev_close = np.roll(close, 1)
        tr  = np.maximum(high - low, np.abs(high - prev_close))
        atr = (
            pd.Series(tr, index=df.index)
            .rolling(14, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    # ── CAUSAL-FIX: shift ATR one bar so threshold[t] = ATR[t-1] ─────────────
    # Bar 0 has no prior bar → use 1e-5 (neutral) instead of atr[0].
    # atr[0] is computed from bar-0's own high/low — strict within-bar leak.
    atr_causal = np.empty_like(atr)
    atr_causal[1:] = atr[:-1]
    atr_causal[0]  = 1e-5

    # ── Threshold vector ──────────────────────────────────────────────────────
    thresholds = np.clip(atr_causal * threshold_multiplier, 1e-5, np.inf)

    # ── Run CUSUM kernel ──────────────────────────────────────────────────────
    event_idx = _symmetric_cusum_filter_kernel(close, thresholds)

    if event_idx.size == 0:
        logger.warning(
            "get_cusum_events: zero events detected "
            "(multiplier=%.1f, n_bars=%d). Check ATR quality.",
            threshold_multiplier,
            len(close),
        )
        return cast(pd.DatetimeIndex, df.index[:0])

    # ── FIX (Item 55): filter degenerate bars (O=H=L=C) ─────────────────────
    if "feat_bar_is_degenerate" in df.columns:
        deg_flags = df["feat_bar_is_degenerate"].to_numpy(dtype=np.int8)
        keep_mask = deg_flags[event_idx] == 0
        event_idx = event_idx[keep_mask]
    elif {"high", "low", "open", "close"}.issubset(df.columns):
        high_np  = df["high"].to_numpy(dtype=np.float64)
        low_np   = df["low"].to_numpy(dtype=np.float64)
        open_np  = df["open"].to_numpy(dtype=np.float64)
        deg_mask = (
            ((high_np - low_np) < 1e-9)
            & (np.abs(close - open_np) < 1e-9)
        )
        if event_idx.size > 0:
            event_idx = event_idx[~deg_mask[event_idx]]

    n_events = len(event_idx)
    logger.info(
        "get_cusum_events: %d events / %d bars (multiplier=%.1f)",
        n_events, len(close), threshold_multiplier,
    )

    return cast(pd.DatetimeIndex, df.index[event_idx])


# =============================================================================
# SHORT-MODEL VERBETERING (Agent C — 2026-05-22)
# =============================================================================
# get_cusum_events() retourneert SYMMETRISCHE events (UP én DOWN gecombineerd).
# De richtingsbepaling voor SHORT training gebeurt via TrendScanning (OLS-helling
# < -min_tstat) of EMA-crossover — niet via de CUSUM filter zelf.
#
# Root cause van het SHORT-probleem:
#   cusum_threshold_multiplier is per asset geconfigureerd op LONG-performance
#   (AVAX: 2.2, DOT: 2.0). Deze hoge drempel filtert ook DOWN-impulsen weg die
#   structurele SHORT-entries zijn in een bear-markt. Resultaat: te weinig SHORT
#   training events → slechte class balance → model leert SHORT winst niet.
#
# Fix A: get_cusum_events_for_short() met optioneel lagere threshold_multiplier.
# In een bear-markt (DOWN-trending): DOWN-impulsen zijn het echte signaal,
# niet de ruis. Een lagere SHORT-drempel geeft meer potentiële SHORT-events
# terwijl hoge UP-drempel (voor LONG) de LONG-kwaliteit bewaakt.
#
# Gebruik:
#   from tradebot.labeling.cusum import get_cusum_events_for_short
#   short_events = get_cusum_events_for_short(df, sym_cfg)
#   # Geeft DatetimeIndex van events met lagere drempel voor SHORT training.
#
# Let op: de pipeline in tune_hparams.py en train_cpcv.py gebruikt op dit
# moment het gecombineerde events_path artefact (Stage 1 → Stage 2/3).
# Voor volledige integratie: zie conf_config.yaml aanbevelingen hieronder.
# =============================================================================

@njit(cache=True)
def _directional_cusum_filter_kernel(
    prices: np.ndarray,
    thresholds_up: np.ndarray,
    thresholds_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Asymmetrische CUSUM filter met aparte drempels voor UP en DOWN events.

    SHORT-model verbetering (Agent C — 2026-05-22):
    Geeft UP-events en DOWN-events apart terug zodat de caller de SHORT-events
    kan selecteren met een lagere threshold_multiplier dan LONG-events.

    Args:
        prices          : float64 array van close prices.
        thresholds_up   : float64 array van per-bar drempels voor UP events.
        thresholds_down : float64 array van per-bar drempels voor DOWN events.
                          Stel lager in dan thresholds_up voor meer SHORT-events.

    Returns:
        up_indices   : int32 array van bar-indices van UP events.
        down_indices : int32 array van bar-indices van DOWN events.
    """
    n = len(prices)
    s_pos = 0.0
    s_neg = 0.0

    up_idx   = np.zeros(n, dtype=np.int32)
    down_idx = np.zeros(n, dtype=np.int32)
    n_up   = 0
    n_down = 0

    for i in range(1, n):
        diff = prices[i] - prices[i - 1]
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        h_up   = thresholds_up[i]
        h_down = thresholds_down[i]

        if s_pos >= h_up:
            up_idx[n_up] = i
            n_up += 1
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg <= -h_down:
            down_idx[n_down] = i
            n_down += 1
            s_pos = 0.0
            s_neg = 0.0

    return up_idx[:n_up], down_idx[:n_down]


def get_cusum_events_for_short(
    df: pd.DataFrame,
    threshold_multiplier: float = 3.5,
    short_multiplier_override: float | None = None,
) -> pd.DatetimeIndex:
    """Bereken CUSUM event timestamps specifiek voor SHORT training.

    SHORT-model verbetering (Agent C — 2026-05-22):

    Rationale:
        De symmetrische CUSUM filter (get_cusum_events) gebruikt één drempel
        voor zowel UP als DOWN events. In een bear-markt zijn DOWN-impulsen
        de kern van het SHORT-signaal, maar een hoge threshold_multiplier
        (bv. 2.2 voor AVAX) filtert ook structurele bear-moves weg.

        Door een lagere drempel te gebruiken voor DOWN events (SHORT training),
        krijgt het SHORT-model meer training events → betere class balance →
        het model leert SHORT-winstgevendheid herkennen.

    Args:
        df                      : Bar DataFrame met 'close' kolom en DatetimeIndex.
                                  'feat_vol_gk' optioneel voor ATR-berekening.
        threshold_multiplier    : ATR multiplier voor LONG events (default drempel).
                                  Dit is de cusum_threshold_multiplier uit de
                                  per-symbol YAML config.
        short_multiplier_override: Lagere ATR multiplier specifiek voor DOWN/SHORT
                                  events. Wanneer None: gebruikt threshold_multiplier
                                  (identiek aan get_cusum_events — geen verandering).
                                  Laad uit conf/symbols/{symbol}.yaml:
                                  cusum_threshold_multiplier_short.

    Returns:
        pd.DatetimeIndex van DOWN-CUSUM event timestamps.
        Gebruik get_cusum_events() voor gecombineerde UP+DOWN events.

    Voorbeeld (tune_hparams / Stage 1):
        short_mult = sym_cfg.get("cusum_threshold_multiplier_short",
                                  threshold_multiplier)
        short_events = get_cusum_events_for_short(
            df, threshold_multiplier=sym_mult, short_multiplier_override=short_mult
        )
    """
    close = df["close"].to_numpy(dtype=np.float64)

    # ── ATR computation (causal: feat_vol_gk preferred) ──────────────────────
    if "feat_vol_gk" in df.columns:
        atr = df["feat_vol_gk"].ffill().fillna(1e-5).to_numpy(dtype=np.float64)
    else:
        high = df["high"].to_numpy(dtype=np.float64)
        low  = df["low"].to_numpy(dtype=np.float64)
        prev_close = np.roll(close, 1)
        tr  = np.maximum(high - low, np.abs(high - prev_close))
        atr = (
            pd.Series(tr, index=df.index)
            .rolling(14, min_periods=1)
            .mean()
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

    # ── CAUSAL-FIX: shift ATR one bar ────────────────────────────────────────
    atr_causal = np.empty_like(atr)
    atr_causal[1:] = atr[:-1]
    atr_causal[0]  = 1e-5

    # ── Threshold vectoren: asymmetrisch wanneer override aanwezig ────────────
    short_mult = short_multiplier_override if short_multiplier_override is not None else threshold_multiplier
    thresholds_up   = np.clip(atr_causal * threshold_multiplier, 1e-5, np.inf)
    thresholds_down = np.clip(atr_causal * short_mult, 1e-5, np.inf)

    # ── Run asymmetrische CUSUM kernel ────────────────────────────────────────
    _up_idx, down_idx = _directional_cusum_filter_kernel(close, thresholds_up, thresholds_down)

    if down_idx.size == 0:
        logger.warning(
            "get_cusum_events_for_short: zero DOWN events detected "
            "(long_mult=%.1f, short_mult=%.1f, n_bars=%d). "
            "Overweeg cusum_threshold_multiplier_short te verlagen in de symbol YAML.",
            threshold_multiplier, short_mult, len(close),
        )
        return cast(pd.DatetimeIndex, df.index[:0])

    # ── FIX (Item 55): filter degenerate bars ─────────────────────────────────
    if "feat_bar_is_degenerate" in df.columns:
        deg_flags = df["feat_bar_is_degenerate"].to_numpy(dtype=np.int8)
        down_idx = down_idx[deg_flags[down_idx] == 0]
    elif {"high", "low", "open", "close"}.issubset(df.columns):
        high_np = df["high"].to_numpy(dtype=np.float64)
        low_np  = df["low"].to_numpy(dtype=np.float64)
        open_np = df["open"].to_numpy(dtype=np.float64)
        deg_mask = (
            ((high_np - low_np) < 1e-9)
            & (np.abs(close - open_np) < 1e-9)
        )
        if down_idx.size > 0:
            down_idx = down_idx[~deg_mask[down_idx]]

    n_down_events = len(down_idx)
    logger.info(
        "get_cusum_events_for_short: %d DOWN events / %d bars "
        "(long_mult=%.1f, short_mult=%.1f, ratio=%.1f× meer vs symmetrisch)",
        n_down_events, len(close), threshold_multiplier, short_mult,
        float(n_down_events) / max(
            # Benadering van symmetrische count voor logging: verhouding drempels
            float(n_down_events) * (short_mult / max(threshold_multiplier, 1e-9)),
            1.0,
        ),
    )

    return cast(pd.DatetimeIndex, df.index[down_idx])


# ---------------------------------------------------------------------------
# Backward-compat aliases (labeling/__init__.py public API)
# ---------------------------------------------------------------------------
# cusum_filter: identical to symmetric_cusum_filter (thin Numba wrapper).
cusum_filter = symmetric_cusum_filter

# dual_cusum_events: alias for get_cusum_events.
# Note: returns combined DatetimeIndex (both up/down events merged), not two
# separate arrays — callers that need direction must use get_cusum_events
# directly and inspect the price direction at each event.
dual_cusum_events = get_cusum_events
