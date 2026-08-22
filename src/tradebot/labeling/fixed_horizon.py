"""Fixed-Horizon Labeling — baseline alternative to Triple Barrier.

Simple forward-return sign over a fixed window. Useful as a sanity check:
if CPCV performance with triple-barrier labels is not better than fixed-horizon,
the triple-barrier setup is misconfigured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_horizon_labels(
    close: pd.Series,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary label: 1 if forward return > threshold, else -1.

    Args:
        close     : Close-price series with UTC DatetimeIndex.
        horizon   : Look-forward bars.
        threshold : Minimum absolute return to generate a signal.

    Returns:
        pd.Series with values in {-1, 0, 1}, NaN for the last `horizon` bars.
        0 = below threshold (no position).
    """
    fwd_ret = close.shift(-horizon) / close - 1.0
    labels = pd.Series(np.where(fwd_ret > threshold, 1,
                                np.where(fwd_ret < -threshold, -1, 0)),
                       index=close.index, dtype=np.int8)
    labels.iloc[-horizon:] = np.nan
    return labels.rename("label")
