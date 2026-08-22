"""data/funding.py — Per-bar funding-rate loader.

Extracted from train_regime.py lines 342-417 (_load_per_bar_funding_rate).
Bit-identical to the monolith.  Public name: load_per_bar_funding_rate
(drop leading underscore — it is now a proper module-level API).

FUNDING-FIX (Item 2):
  Bybit linear-perp funding is posted every 8 hours (majors).  This helper reads
  ``macro_crypto_{symbol}.parquet`` (raw ``fundingRate`` column, written
  by ``CryptoMacroFetcher.calculate_macro_features``) and places the rate
  on the first bar that starts ON OR AFTER the funding timestamp.
  All other bars get 0.0, so ``sum(signed_lev * funding_rate)`` computes
  correctly in the backtest loop.

Strangler-fig: once train_regime.py imports from here and the equivalence
test passes, delete lines 342-417 in train_regime.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_per_bar_funding_rate(
    symbol: str,
    bar_index: pd.DatetimeIndex,
    macro_dir: Path,
) -> np.ndarray:
    """Load per-bar funding rate for ``symbol`` on the ``bar_index`` grid.

    Parameters
    ----------
    symbol    : Ticker symbol (e.g. 'BTCUSDT').
    bar_index : DatetimeIndex of the bar series (all bars in the backtest).
    macro_dir : Directory containing ``macro_crypto_{symbol}.parquet``.

    Returns
    -------
    np.ndarray[float64]
        Array of length ``len(bar_index)``.  Each element is the decimal
        funding rate (NOT annualised) at that bar, 0.0 elsewhere.
        On missing file or parse error: array of zeros + log warning
        (never raises — backward-compatible).
    """
    n = len(bar_index)
    out = np.zeros(n, dtype=np.float64)

    fpath = Path(macro_dir) / f"macro_crypto_{symbol}.parquet"
    if not fpath.exists():
        logger.warning(
            "[%s] Funding-rate parquet not found (%s) — backtest without "
            "funding costs.",
            symbol, fpath.name,
        )
        return out

    try:
        df = pd.read_parquet(fpath)

        if "fundingRate" not in df.columns:
            logger.warning(
                "[%s] No 'fundingRate' column in %s — re-run "
                "CryptoMacroFetcher to persist raw rate. Funding set to 0.",
                symbol, fpath.name,
            )
            return out

        # ── Index normalisation ───────────────────────────────────────────────
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        # ── UTC-aware bar index ───────────────────────────────────────────────
        bar_idx_dt = cast(pd.DatetimeIndex, pd.DatetimeIndex(bar_index))
        if bar_idx_dt.tz is None:
            bar_idx_dt = cast(
                pd.DatetimeIndex, bar_idx_dt.tz_localize("UTC")
            )

        # ── Funding-tick → bar mapping ────────────────────────────────────────
        # For each funding event: find the first bar whose timestamp is
        # GREATER THAN OR EQUAL to the funding timestamp (forward search).
        #
        # CHIEF AUDIT 2026-05-23 (FIX 1 / off-by-one): bars are right-labelled
        # i.e. timestamp T = bar (T-Δ, T].  A funding tick at 08:00:00 belongs
        # to the bar with timestamp 08:00:00 (its close moment), NOT to the
        # bar at 08:00:05 (the next bar that starts strictly after 08:00:00).
        # ``side="right"`` returns the first index with bar_ts >  funding_ts,
        # which incorrectly shifts every funding event to the NEXT bar.
        # ``side="left"`` returns the first index with bar_ts >= funding_ts,
        # which correctly lands funding on the same-close bar when present
        # and on the next bar only when no exact match exists.
        funding_series = df["fundingRate"].dropna()
        funding_idx    = cast(pd.DatetimeIndex, funding_series.index)

        if funding_idx.tz is None:
            funding_idx = cast(
                pd.DatetimeIndex, funding_idx.tz_localize("UTC")
            )

        funding_vals = np.asarray(
            funding_series.to_numpy(), dtype=np.float64
        )
        positions = np.asarray(
            bar_idx_dt.searchsorted(funding_idx, side="left"),
            dtype=np.int64,
        )

        for k in range(funding_vals.size):
            rate_f = float(funding_vals[k])
            if not np.isfinite(rate_f) or rate_f == 0.0:
                continue
            pos = int(positions[k])
            if 0 <= pos < n:
                out[pos] += rate_f

        return out

    except Exception as exc:
        logger.warning(
            "[%s] Funding-rate load failed: %s", symbol, exc
        )
        return out


# ── Backward-compat alias (leading-underscore monolith name) ──────────────────
_load_per_bar_funding_rate = load_per_bar_funding_rate
