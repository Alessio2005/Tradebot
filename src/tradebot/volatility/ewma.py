"""Exponentially weighted moving average (EWMA) volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def get_ewma_volatility(
    df: pd.DataFrame,
    halflife: int = 20,
    min_periods: int = 5,
    absolute: bool = False,
) -> pd.Series:
    """EWMA close-to-close log-return volatility (annualised)."""
    if df.empty:
        return pd.Series(0.0, index=df.index, name="feat_vol_ewma")
    log_ret = np.log(df["close"] / df["close"].shift(1))
    ewma_var = log_ret.ewm(halflife=halflife, min_periods=min_periods).var()
    vol = np.sqrt(ewma_var.clip(lower=0.0))
    if absolute:
        vol = vol * df["close"]
    vol.name = "feat_vol_ewma"
    return vol.ffill().fillna(0.0)
