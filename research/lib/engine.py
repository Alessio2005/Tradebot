"""Vectorised daily book. Decide on close t, hold close t -> close t+lag (lag=1 is the base case).

Cost: COST per unit |dw| (taker 5.5 + half-spread 1.0 bps, conf/execution/fees.yaml).
Funding: a long pays positive daily funding, a short receives it.
Weights are fractions of equity per coin (signed); gross leverage is capped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel

COST = 6.5e-4
GROSS_CAP = 4.0  # mandate max_leverage


def cap_gross(w: pd.DataFrame, cap: float = GROSS_CAP) -> pd.DataFrame:
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    return w.mul(np.minimum(1.0, cap / gross).fillna(1.0), axis=0)


def run_book(w: pd.DataFrame, p: Panel, lag: int = 1, cost: float = COST, funding: bool = True) -> pd.DataFrame:
    w = w.reindex(p.index).fillna(0.0)
    held = w.shift(lag).fillna(0.0)
    gross_ret = (held * p.ret.fillna(0.0)).sum(axis=1)
    fund = (held * p.funding).sum(axis=1) if funding else 0.0 * gross_ret
    turnover = w.diff().abs().sum(axis=1).shift(lag - 1).fillna(0.0)
    net = gross_ret - fund - cost * turnover
    return pd.DataFrame({"net": net, "gross": gross_ret, "funding": -fund, "costs": -cost * turnover,
                         "turnover": turnover, "gross_lev": held.abs().sum(axis=1)})


def vol_target(r: pd.Series, target: float = 0.20, span: int = 60, floor: float = 0.02, max_scale: float = 3.0) -> pd.Series:
    """Scale a daily return series so its EWMA vol (known at t) targets `target`; applied with lag 1."""
    vol = r.ewm(span=span, min_periods=20).std() * np.sqrt(365.0)
    scale = (target / vol.clip(lower=floor)).clip(upper=max_scale).shift(1)
    return r * scale.fillna(0.0)
