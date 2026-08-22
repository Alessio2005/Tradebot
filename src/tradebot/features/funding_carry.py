"""Funding-rate feature block (AUDIT E-4).

Bybit perp funding rates are posted every 8h and are autocorrelated.
Positive funding → longs pay shorts → market is long-skewed → short-term
mean-reversion / squeeze risk if price falls. Negative funding → the
opposite. The backtest already debits funding from PnL via
``data.funding.load_per_bar_funding_rate``, but the model never *sees* the
rate as a predictor.

This module turns the same per-bar funding array into causal features:

- ``feat_micro_funding_rate``     — current funding rate (per 8h period).
- ``feat_micro_funding_anno``      — annualised: rate × 3 × 365.
- ``feat_meso_funding_cum_24h``    — rolling sum over the previous 24h.
- ``feat_meso_funding_zscore_30d`` — z-score over the trailing 30 days.
- ``feat_meso_funding_sign``       — sign of the trailing 24h funding.

All series use ``shift(1)`` to guarantee causality (R-1): the row at bar
``t`` cannot read the funding tick that lands inside bar ``t``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_funding_features(
    bar_index: pd.DatetimeIndex,
    funding_per_bar: np.ndarray,
    bars_per_day: int = 24,
    zscore_window_days: int = 30,
) -> pd.DataFrame:
    """Build a causal funding-rate feature block.

    Parameters
    ----------
    bar_index :
        DatetimeIndex of the feature matrix.
    funding_per_bar :
        Per-bar funding rate array (length must equal ``len(bar_index)``).
        Use :func:`tradebot.data.funding.load_per_bar_funding_rate` to
        produce this.
    bars_per_day :
        Bars per calendar day for the meso aggregations. Default 24 (1 h
        bars on 24/7 crypto).
    zscore_window_days :
        Window for the rolling z-score.

    Returns
    -------
    pd.DataFrame indexed identically to ``bar_index`` with feature columns
    starting with ``feat_micro_funding_*`` or ``feat_meso_funding_*``.
    """
    if len(funding_per_bar) != len(bar_index):
        raise ValueError(
            f"funding_per_bar length {len(funding_per_bar)} != "
            f"bar_index length {len(bar_index)}"
        )

    rate = pd.Series(np.asarray(funding_per_bar, dtype=np.float64),
                     index=bar_index, name="funding_rate")

    # Causality: shift by one bar so the model only sees funding ticks that
    # have *already* been posted by the start of the current bar.
    rate_lagged = rate.shift(1).fillna(0.0)

    # ── Micro features ────────────────────────────────────────────────────
    feat_funding_rate = rate_lagged
    feat_funding_anno = rate_lagged * 3.0 * 365.0   # 3 fundings/day * 365

    # ── Meso aggregations ────────────────────────────────────────────────
    cum_24h = rate_lagged.rolling(window=bars_per_day, min_periods=1).sum()

    zw = max(bars_per_day, bars_per_day * zscore_window_days)
    mean_z = rate_lagged.rolling(window=zw, min_periods=bars_per_day).mean()
    std_z  = rate_lagged.rolling(window=zw, min_periods=bars_per_day).std(ddof=0)
    z_score = ((rate_lagged - mean_z) / (std_z + 1e-12)).clip(-6.0, 6.0)

    sign_24h = np.sign(cum_24h).astype(np.float64)

    out = pd.DataFrame(
        {
            "feat_micro_funding_rate":     feat_funding_rate.astype(np.float32),
            "feat_micro_funding_anno":     feat_funding_anno.astype(np.float32),
            "feat_meso_funding_cum_24h":   cum_24h.astype(np.float32),
            "feat_meso_funding_zscore_30d": z_score.astype(np.float32),
            "feat_meso_funding_sign":      sign_24h.astype(np.float32),
        },
        index=bar_index,
    )
    out = out.fillna(0.0)
    logger.info(
        "Funding features built: rate_range=[%.6f, %.6f] z_range=[%.2f, %.2f]",
        float(out["feat_micro_funding_rate"].min()),
        float(out["feat_micro_funding_rate"].max()),
        float(out["feat_meso_funding_zscore_30d"].min()),
        float(out["feat_meso_funding_zscore_30d"].max()),
    )
    return out


__all__ = ["compute_funding_features"]
