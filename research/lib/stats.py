"""Performance statistics. Sharpe SE / bootstrap come from the repo's own measurement kernel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from tradebot.validation.inference import block_bootstrap_ci, sharpe_with_se

BPY = 365.0


def perf(r: pd.Series, start=None, end=None, boot: bool = True) -> dict:
    r = r.loc[start:end].dropna()
    se = sharpe_with_se(r, bars_per_year=BPY)
    eq = (1 + r).cumprod()
    out = {
        "sharpe": float(se.sharpe), "se": float(se.se),
        "ann_ret": float(r.mean() * BPY), "ann_vol": float(r.std() * np.sqrt(BPY)),
        "max_dd": float((eq / eq.cummax() - 1).min()), "years": len(r) / BPY,
        "skew": float(sps.skew(r)), "kurt": float(sps.kurtosis(r, fisher=False)),
    }
    out["calmar"] = out["ann_ret"] / abs(out["max_dd"]) if out["max_dd"] < 0 else float("nan")
    if boot:
        ci = block_bootstrap_ci(r, bars_per_year=BPY, seed=7)
        out["ci_lo"], out["ci_hi"] = float(ci.low), float(ci.high)
    return out


def sr(r: pd.Series, start=None, end=None) -> float:
    r = r.loc[start:end].dropna()
    s = r.std()
    return float(r.mean() / s * np.sqrt(BPY)) if s > 0 else 0.0


def by_year(r: pd.Series) -> dict:
    out = {}
    for y, g in r.groupby(r.index.year):
        out[int(y)] = {"ret": float(g.sum()), "sr": sr(g), "n": len(g)}
    return out


def dsr(r: pd.Series, n_trials: int, start=None, end=None) -> float:
    """Deflated Sharpe, Bailey & Lopez de Prado (2014), normal-approximation sr_variance = 1/T.

    Uses the repo implementation; sr_hat per bar. M is the HONEST number of variants tried.
    """
    from tradebot.backtest.metrics import deflated_sharpe

    r = r.loc[start:end].dropna()
    T = len(r)
    res = deflated_sharpe(
        float(r.mean() / r.std()), n_obs=T, n_trials=int(n_trials), sr_variance=1.0 / T,
        skew=float(sps.skew(r)), kurtosis=float(sps.kurtosis(r, fisher=False)),
        bars_per_year=BPY, approximation="normal",
    )
    return float(getattr(res, "dsr", getattr(res, "value", np.nan)))
