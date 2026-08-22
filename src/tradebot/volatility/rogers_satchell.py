"""Rogers-Satchell (1991) intraday volatility — no overnight gap required."""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_rogers_satchell_volatility(
    df: pd.DataFrame, window: int = 14, absolute: bool = False
) -> pd.Series:
    """Rogers-Satchell variance: sum of intraday log-price products."""
    if df.empty or len(df) < window:
        return pd.Series(0.0, index=df.index, name="feat_vol_rs")
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    rs = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
    vol = np.sqrt(np.maximum(pd.Series(rs).rolling(window).mean().values, 0.0))
    result = pd.Series(vol * c if absolute else vol,
                       index=df.index, name="feat_vol_rs")
    return result.ffill().fillna(0.0)
