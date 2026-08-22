"""Open-interest feature block (CHIEF data review 2026-06-14).

Turns the per-bar open-interest LEVEL (see
``data.open_interest.load_per_bar_open_interest``) into causal features:

- ``feat_meso_oi_zscore_30d``       — z-score of the OI level over 30 days.
- ``feat_meso_oi_chg_24h``          — 24h log-change in OI (build-up vs unwind).
- ``feat_meso_oi_price_divergence`` — sign(ΔOI_24h) × sign(price_ret_24h):
      +1 → OI and price move together (trend / fresh positioning),
      -1 → OI rises while price falls, or vice-versa (covering / hedging).

All series use ``shift(1)`` to guarantee causality (R-1): the row at bar ``t``
cannot read the OI sample or close that lands inside bar ``t``.  This mirrors
``features.funding_carry.compute_funding_features`` exactly so there is no
research/live divergence.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_oi_features(
    bar_index: pd.DatetimeIndex,
    oi_per_bar: np.ndarray,
    close: np.ndarray | None = None,
    bars_per_day: int = 24,
    zscore_window_days: int = 30,
) -> pd.DataFrame:
    """Build a causal open-interest feature block.

    Parameters
    ----------
    bar_index :
        DatetimeIndex of the feature matrix.
    oi_per_bar :
        Per-bar OI level array (length must equal ``len(bar_index)``).  Use
        :func:`tradebot.data.open_interest.load_per_bar_open_interest`.
    close :
        Optional per-bar close array (same length) for the divergence feature.
        If None, ``feat_meso_oi_price_divergence`` is set to 0.
    bars_per_day :
        Bars per calendar day. Default 24 (1h bars on 24/7 crypto).
    zscore_window_days :
        Window for the rolling z-score.
    """
    if len(oi_per_bar) != len(bar_index):
        raise ValueError(
            f"oi_per_bar length {len(oi_per_bar)} != "
            f"bar_index length {len(bar_index)}"
        )

    oi = pd.Series(np.asarray(oi_per_bar, dtype=np.float64),
                   index=bar_index, name="open_interest")

    # Causality: shift by one bar so the model only sees OI samples that have
    # already been published by the start of the current bar.
    oi_lag = oi.shift(1)

    # ── 24h log-change (guard against zeros) ──────────────────────────────
    oi_prev = oi_lag.shift(bars_per_day)
    safe = (oi_lag > 0) & (oi_prev > 0)
    oi_chg = pd.Series(0.0, index=bar_index)
    oi_chg[safe] = np.log(oi_lag[safe] / oi_prev[safe])

    # ── rolling z-score of the level ──────────────────────────────────────
    zw = max(bars_per_day, bars_per_day * zscore_window_days)
    mean_z = oi_lag.rolling(window=zw, min_periods=bars_per_day).mean()
    std_z = oi_lag.rolling(window=zw, min_periods=bars_per_day).std(ddof=0)
    z_score = ((oi_lag - mean_z) / (std_z + 1e-12)).clip(-6.0, 6.0)

    # ── OI/price divergence ───────────────────────────────────────────────
    if close is not None:
        if len(close) != len(bar_index):
            raise ValueError("close length must equal bar_index length")
        px = pd.Series(np.asarray(close, dtype=np.float64), index=bar_index)
        px_lag = px.shift(1)
        px_ret = px_lag.pct_change(bars_per_day)
        divergence = (np.sign(oi_chg) * np.sign(px_ret)).astype(np.float64)
    else:
        divergence = pd.Series(0.0, index=bar_index)

    out = pd.DataFrame(
        {
            "feat_meso_oi_zscore_30d":       z_score.astype(np.float32),
            "feat_meso_oi_chg_24h":          oi_chg.astype(np.float32),
            "feat_meso_oi_price_divergence": divergence.astype(np.float32),
        },
        index=bar_index,
    )
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logger.info(
        "OI features built: z_range=[%.2f, %.2f] chg_range=[%.4f, %.4f]",
        float(out["feat_meso_oi_zscore_30d"].min()),
        float(out["feat_meso_oi_zscore_30d"].max()),
        float(out["feat_meso_oi_chg_24h"].min()),
        float(out["feat_meso_oi_chg_24h"].max()),
    )
    return out


__all__ = ["compute_oi_features"]
