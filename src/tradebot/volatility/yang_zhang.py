"""Yang-Zhang (2000) overnight-gap volatility estimator.

WARNING: YZ is designed for DAILY OHLC bars where the overnight component
log(open_t / close_{t-1}) measures the closed-market gap.  For intraday
volume-imbalance bars this component is theoretically undefined and leads
to mis-calibrated Triple-Barrier PT/SL.

This module exists for completeness and equity/daily-bar use cases only.
Do NOT use on intraday crypto bars — use garman_klass.py instead.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_yang_zhang_volatility(
    df: pd.DataFrame,
    window: int = 20,
    k: float = 0.34,
    absolute: bool = False,
) -> pd.Series:
    """Yang-Zhang (2000) combined open-close + Rogers-Satchell estimator.

    Only valid for daily OHLC. Raises ValueError on intraday data
    (bar duration < 1 hour inferred from index frequency).
    """
    if df.empty or len(df) < window:
        return pd.Series(0.0, index=df.index, name="feat_vol_yz")

    # CHIEF AUDIT 2026-05-23 (P-13): runtime guard against intraday data.
    # The Yang-Zhang formula models an overnight gap component
    # (log(open_t / close_{t-1})) that is only meaningful for closed-market
    # daily bars.  For intraday bars the gap is mechanically zero/noise and
    # the resulting σ mis-calibrates Triple-Barrier PT/SL.
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            freq = getattr(df.index, "freq", None)
            if freq is not None:
                # pandas freq → bar duration in seconds
                bar_seconds = pd.tseries.frequencies.to_offset(freq).nanos / 1e9
                if bar_seconds < 86400:  # < 1 day
                    raise ValueError(
                        "Yang-Zhang estimator requires daily (or longer) OHLC "
                        f"bars; got freq={freq} (~{bar_seconds:.0f}s). "
                        "Use garman_klass.py for intraday bars."
                    )
            elif len(df.index) >= 2:
                # Fallback: infer cadence from the median index spacing.
                diffs = pd.Series(df.index).diff().dropna()
                if not diffs.empty:
                    median_seconds = float(diffs.median().total_seconds())
                    if 0 < median_seconds < 86400:
                        logger.warning(
                            "Yang-Zhang received bars with median spacing "
                            "%.0fs (< 1 day) — overnight-gap term is invalid "
                            "for intraday data.  Use garman_klass.py instead.",
                            median_seconds,
                        )
    except ValueError:
        raise
    # Phase 0: aangescherpt van `except Exception`. Deze guard leidt alleen de
    # bar-cadans af uit de index; ontbreekt het freq-attribuut of is de index
    # geen tijdreeks, dan is de inferentie onbeslist (AttributeError/TypeError).
    # De ValueError-tak hierboven blijft doorwerpen. Elke andere fout is een bug.
    except (AttributeError, TypeError) as exc:  # pragma: no cover
        logger.debug("Yang-Zhang intraday guard inconclusive: %s", exc)

    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    c_prev = np.roll(c, 1); c_prev[0] = c[0]

    log_oc = np.log(o / c_prev)    # overnight return
    log_co = np.log(c / o)         # intraday open-to-close
    log_hc = np.log(h / c)
    log_lc = np.log(l / c)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)

    rs = log_hc * log_ho + log_lc * log_lo  # Rogers-Satchell

    n = window
    oc_var = pd.Series(log_oc**2).rolling(n).mean().values
    co_var = pd.Series(log_co**2).rolling(n).mean().values
    rs_var = pd.Series(rs).rolling(n).mean().values

    yz_var = oc_var + k * co_var + (1 - k) * rs_var
    vol = np.sqrt(np.maximum(yz_var, 0.0))
    result = pd.Series(vol * c if absolute else vol,
                       index=df.index, name="feat_vol_yz")
    return result.ffill().fillna(0.0)
