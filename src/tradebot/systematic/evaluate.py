"""Metriek, inferentie, poorten en de robuustheidsscore van het boek.

Eén implementatie per grootheid, en het is die van de repo: Sharpe en zijn Lo-SE uit
`validation/inference.py::sharpe_with_se`, het bootstrapinterval uit
`block_bootstrap_ci`, de DSR uit `backtest/metrics.py::deflated_sharpe`, de PBO uit
`backtest/pbo.py::compute_pbo`. Deze module rekent niets opnieuw uit dat daar al staat.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..backtest.metrics import annualized_return, deflated_sharpe, max_drawdown, sortino_ratio
from ..backtest.pbo import compute_pbo
from ..validation.inference import block_bootstrap_ci, sharpe_with_se
from .book import BookResult
from .market import BARS_PER_YEAR, BookMarket

__all__ = [
    "BOOTSTRAP_SEED",
    "dsr_record",
    "gate_z",
    "pbo_record",
    "regime_breakdown",
    "robustness_score",
    "summarize",
    "yearly",
]

BOOTSTRAP_SEED = 20261007
COMPONENTS = ("gross", "funding", "fees", "spread", "slippage", "impact", "net")


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd * math.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0


def summarize(res: BookResult, *, bootstrap: bool = True) -> dict[str, Any]:
    """Alle metriek van één venster van één run, netto tenzij anders vermeld."""
    f = res.frame
    r = f["net"]
    n = int(r.size)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + r.to_numpy())))
    mdd, _, _ = max_drawdown(equity)
    cagr = annualized_return(r.to_numpy(), bars_per_year=BARS_PER_YEAR)
    est = sharpe_with_se(r, bars_per_year=BARS_PER_YEAR)
    out: dict[str, Any] = {
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "n_obs": n, "t_years": n / BARS_PER_YEAR,
        "cagr": cagr,
        "ann_vol": float(r.std(ddof=1) * math.sqrt(BARS_PER_YEAR)),
        "sharpe": est.sharpe, "sharpe_se": est.se, "t_stat": est.t_stat,
        "sharpe_per_bar": est.sharpe_per_bar, "skew": est.skew, "kurtosis": est.kurtosis,
        "p_sharpe_gt_0": float(stats.norm.cdf(est.sharpe / est.se)) if est.se > 0 else float("nan"),
        "sortino": sortino_ratio(r.to_numpy(), bars_per_year=BARS_PER_YEAR),
        "max_drawdown": mdd,
        "calmar": float(cagr / mdd) if mdd > 1e-9 else float("nan"),
        "gross_sharpe": _sharpe(f["gross"]),
        "gross_plus_funding_sharpe": _sharpe(f["gross"] + f["funding"]),
        "ann_turnover": float(f["turnover"].mean() * BARS_PER_YEAR),
        "n_trades": int(f["n_trades"].sum()),
        "avg_gross_leverage": float(f["gross_leverage"].mean()),
        "max_gross_leverage": float(f["gross_leverage"].max()),
        "avg_net_leverage": float(f["net_leverage"].mean()),
        "pct_bars_invested": float((f["gross_leverage"] > 1e-9).mean()),
        "hit_rate_days": float((r > 0).sum() / max((r != 0).sum(), 1)),
        "worst_day": float(r.min()), "best_day": float(r.max()),
    }
    for c in COMPONENTS:
        out[f"ann_{c}"] = float(f[c].mean() * BARS_PER_YEAR)
    if bootstrap:
        ci = block_bootstrap_ci(r, statistic="sharpe", bars_per_year=BARS_PER_YEAR,
                                seed=BOOTSTRAP_SEED)
        out["sharpe_ci_low"] = ci.low
        out["sharpe_ci_high"] = ci.high
        out["bootstrap_block_length"] = ci.block_length
        out["bootstrap_n_boot"] = ci.n_boot
    return out


def yearly(res: BookResult) -> dict[str, dict[str, float]]:
    """Per kalenderjaar: netto rendement, Sharpe, max drawdown."""
    out: dict[str, dict[str, float]] = {}
    r = res.frame["net"]
    for year, chunk in r.groupby(r.index.year):
        eq = np.concatenate(([1.0], np.cumprod(1.0 + chunk.to_numpy())))
        out[str(year)] = {"return": float(eq[-1] - 1.0), "sharpe": _sharpe(chunk),
                          "max_drawdown": max_drawdown(eq)[0], "n_obs": int(chunk.size)}
    return out


def regime_masks(market: BookMarket, index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Causale regimes op de besluitbar *t*, toegepast op het rendement van *t+1*.

    bull/bear: BTC boven/onder zijn 200-daags gemiddelde; hoge/lage vol: BTC-30d-vol
    boven/onder zijn expanderende mediaan; hoge/lage correlatie: gemiddelde paarsgewijze
    60d-correlatie boven/onder haar expanderende mediaan.
    """
    btc = market.close["BTCUSDT"]
    bull = (btc > btc.rolling(200, min_periods=200).mean()).shift(1)
    vol30 = market.ret["BTCUSDT"].rolling(30, min_periods=30).std()
    hi_vol = (vol30 > vol30.expanding(min_periods=200).median()).shift(1)
    corr = market.ret.rolling(60, min_periods=60).corr().to_numpy()
    n = market.ret.shape[1]
    off = ~np.eye(n, dtype=bool)
    blocks = corr.reshape(len(market.index), n, n)
    avg_vals = np.array([
        float(np.nanmean(b[off])) if np.isfinite(b[off]).sum() >= 2 else np.nan
        for b in blocks])
    avg = pd.Series(avg_vals, index=market.index)
    hi_corr = (avg > avg.expanding(min_periods=200).median()).shift(1)
    masks = {
        "bull_btc_above_ma200": bull == True,  # noqa: E712 -- NaN telt als niet-bull
        "bear_btc_below_ma200": bull == False,  # noqa: E712
        "high_vol": hi_vol == True,  # noqa: E712
        "low_vol": hi_vol == False,  # noqa: E712
        "high_correlation": hi_corr == True,  # noqa: E712
        "low_correlation": hi_corr == False,  # noqa: E712
    }
    return {k: v.reindex(index).fillna(False).astype(bool) for k, v in masks.items()}


def regime_breakdown(res: BookResult, market: BookMarket) -> dict[str, dict[str, float]]:
    r = res.frame["net"]
    out = {}
    for name, mask in regime_masks(market, pd.DatetimeIndex(r.index)).items():
        chunk = r[mask]
        out[name] = {"share_of_bars": float(mask.mean()), "sharpe": _sharpe(chunk),
                     "ann_return_arith": float(chunk.mean() * BARS_PER_YEAR)}
    return out


def dsr_record(r: pd.Series, *, n_trials: int, trial_sharpes_per_bar: Sequence[float]) -> dict[str, Any]:
    """DSR bij `n_trials`, met V[SR] = max(empirische trialvariantie, 1/T) (conservatief)."""
    est = sharpe_with_se(r, bars_per_year=BARS_PER_YEAR)
    n = int(r.dropna().size)
    emp = float(np.var(np.asarray(trial_sharpes_per_bar, dtype=float), ddof=1)) \
        if len(trial_sharpes_per_bar) >= 2 else 0.0
    var = max(emp, 1.0 / n)
    res = deflated_sharpe(est.sharpe_per_bar, n_obs=n, n_trials=int(n_trials), sr_variance=var,
                          skew=est.skew, kurtosis=est.kurtosis, bars_per_year=BARS_PER_YEAR)
    rec = res.to_dict()
    rec["sr_variance_empirical"] = emp
    rec["required_annual_sharpe"] = float(res.sr_zero * math.sqrt(BARS_PER_YEAR))
    return rec


def pbo_record(variants: Mapping[str, pd.Series], *, n_subsets: int = 16) -> dict[str, Any]:
    """PBO (CSCV) over een familie varianten op hetzelfde venster."""
    frame = pd.DataFrame(dict(variants)).dropna()
    res = compute_pbo(frame.to_numpy(), n_subsets=n_subsets)
    return {"pbo": float(res["pbo"]), "logit_pbo": float(res["logit_pbo"]),
            "n_combinations": int(res["n_combinations"]), "n_variants": int(frame.shape[1]),
            "n_subsets": n_subsets}


def gate_z(sr_gate: float, se_gate: float, sr_dev: float) -> float:
    """`(SR_gate − SR_dev) / SE_gate`: hoeveel standaardfouten het poortsample onder W_DEV valt."""
    return float((sr_gate - sr_dev) / se_gate) if se_gate > 0 else float("nan")


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0)) if math.isfinite(x) else 0.0


def robustness_score(m: Mapping[str, float]) -> dict[str, float]:
    """De vooraf vastgelegde score (spec §8), 0..100, uitsluitend op W_DEV-grootheden."""
    sr = float(m["sharpe"])
    pos = sr > 0.0
    parts = {
        "level": 25 * _clip01(sr / 1.0),
        "consistency": 15 * (_clip01(min(m["sharpe_train"], m["sharpe_validate"]) / sr) if pos else 0.0),
        "cost": 10 * (_clip01(m["sharpe_2x_cost"] / sr) if pos else 0.0),
        "delay": 10 * (_clip01(m["sharpe_lag2"] / sr) if pos else 0.0),
        "plateau": 10 * _clip01(m["plateau_fraction"]),
        "confidence": 10 * _clip01((m["p_sharpe_gt_0"] - 0.5) / 0.475),
        "drawdown": 10 * _clip01(1.0 - m["max_drawdown"] / 0.40),
        "overfit": 5 * _clip01(1.0 - m["pbo"]),
        "universe": 5 * _clip01(m["leave_one_out_positive_fraction"]),
    }
    parts["total"] = float(sum(parts.values()))
    return parts
