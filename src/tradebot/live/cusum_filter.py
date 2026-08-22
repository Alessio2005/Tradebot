# src/tradebot/live/cusum_filter.py
"""Standalone CUSUM filter — AFML Chapter 2 correct implementation.

AFML design principle:
    CUSUM is the SAMPLING MECHANISM that determines WHEN a new observation
    (runs bar) is created.  It is NOT a trading signal by itself.

Correct live flow:
    1. Step CUSUM on every raw 5s bar  (O(1), just a diff + compare)
    2. When CUSUM fires  →  FeaturePipeline runs NOW on buffered bars
    3. ModelSignal.predict_on_event() on the FRESH features
    4. JudgeGate  →  rebalancing

Original (broken) flow:
    FeaturePipeline on fixed 538-bar schedule  →  CUSUM steps on raw bars
    →  CUSUM fires  →  predict on STALE features (up to 44.8 min old)
    Gap: model was trained on (CUSUM event, fresh features) pairs.
         Live was predicting on (CUSUM event, stale features) pairs.

One filter per symbol — direction (LONG/SHORT) is determined by the ML model,
not by CUSUM.  This matches AFML Chapter 2 where CUSUM is bidirectional:
    s_pos ≥ h  OR  s_neg ≤ -h  →  event (runs bar formed)

ATR threshold:
    threshold = ATR_last_event × cusum_multiplier
    ATR is updated from fresh features after every event.
    Between events the previous ATR is used (causal, no lookahead).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["CUSUMFilter", "build_cusum_filters"]

_WARMUP_BARS = 200   # rows to read from feature parquet for ATR seed


class CUSUMFilter:
    """Bidirectional CUSUM filter per symbol.

    Parameters
    ----------
    multiplier :
        ATR multiplier for the event threshold (2.0–2.5 depending on symbol).
    symbol :
        For logging only.
    """

    def __init__(self, multiplier: float, symbol: str = "") -> None:
        self._mult = multiplier
        self._sym = symbol
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._prev_close: Optional[float] = None
        self._last_atr: Optional[float] = None   # seeded at warm-start

    # ------------------------------------------------------------------
    # Warm-start
    # ------------------------------------------------------------------

    def warmup_from_parquet(self, feat_path: Path) -> None:
        """Seed last_atr and prev_close from the feature parquet.

        Reads the last _WARMUP_BARS rows and extracts feat_vol_gk (GK-vol
        proxy used as the CUSUM ATR threshold) and close.  Resets s_pos/s_neg
        to zero — a conservative start that avoids a spurious first-bar event.
        """
        if not feat_path.exists():
            logger.warning("CUSUMFilter [%s]: parquet not found at %s — cold start.", self._sym, feat_path)
            return
        df = pd.read_parquet(feat_path).tail(_WARMUP_BARS)
        if "feat_vol_gk" in df.columns:
            self._last_atr = float(df["feat_vol_gk"].iloc[-1])
        elif "feat_atr" in df.columns:
            self._last_atr = float(df["feat_atr"].iloc[-1])
        if "close" in df.columns:
            self._prev_close = float(df["close"].iloc[-1])
        logger.info(
            "CUSUMFilter [%s]: warm-started — last_atr=%.6f prev_close=%.4f",
            self._sym, self._last_atr or 0.0, self._prev_close or 0.0,
        )

    # ------------------------------------------------------------------
    # Per-bar stepping
    # ------------------------------------------------------------------

    def step(self, close: float) -> bool:
        """Step CUSUM with one raw 5s bar close.  Returns True when an event fires.

        O(1) — no feature computation.  Called on every incoming bar.
        """
        if self._prev_close is None or self._last_atr is None:
            self._prev_close = close
            return False

        threshold = max(self._last_atr * self._mult, 1e-8)
        diff = close - self._prev_close
        self._s_pos = max(0.0, self._s_pos + diff)
        self._s_neg = min(0.0, self._s_neg + diff)
        self._prev_close = close

        if self._s_pos >= threshold or self._s_neg <= -threshold:
            self._s_pos = 0.0
            self._s_neg = 0.0
            return True
        return False

    # ------------------------------------------------------------------
    # ATR update (called after each event with fresh features)
    # ------------------------------------------------------------------

    def update_atr(self, features: pd.DataFrame) -> None:
        """Update ATR threshold from freshly computed event features.

        Must be called immediately after FeaturePipeline runs at event time
        so the next CUSUM interval uses the volatility at THIS event.
        """
        if "feat_vol_gk" in features.columns:
            atr = float(features["feat_vol_gk"].iloc[-1])
        elif "feat_atr" in features.columns:
            atr = float(features["feat_atr"].iloc[-1])
        else:
            return
        if atr > 1e-8:
            self._last_atr = atr


def build_cusum_filters(
    symbols: list[str],
    multipliers: Dict[str, float],
    artefacts_dir: Path,
) -> Dict[str, CUSUMFilter]:
    """Create and warm-start one CUSUMFilter per symbol.

    Parameters
    ----------
    symbols :
        Trading pairs.
    multipliers :
        symbol → cusum_threshold_multiplier (from conf_config.yaml).
    artefacts_dir :
        Root artefacts directory containing features/{SYM}.parquet.
    """
    filters: Dict[str, CUSUMFilter] = {}
    for sym in symbols:
        mult = multipliers.get(sym, 2.0)
        cf = CUSUMFilter(multiplier=mult, symbol=sym)
        feat_path = artefacts_dir / "features" / f"{sym}.parquet"
        cf.warmup_from_parquet(feat_path)
        filters[sym] = cf
        logger.info("CUSUMFilter [%s]: ready, multiplier=%.1f", sym, mult)
    return filters
