"""Sleeve builders: each returns a target-weight frame (fractions of equity per coin, signed).

No parameter here is fitted. Every default is the literature/pre-registered value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel
from .engine import cap_gross, run_book

LOOKS = (7, 14, 28, 56, 112)


def tsmom_score(p: Panel, looks=LOOKS, long_flat: bool = False) -> pd.DataFrame:
    logp = np.log(p.close)
    s = sum(np.sign(logp - logp.shift(L)) for L in looks) / len(looks)
    s = s.where(logp.shift(max(looks)).notna())
    return s.clip(lower=0) if long_flat else s


def ts_weights(p: Panel, score: pd.DataFrame, per_coin_vol: float = 0.40) -> pd.DataFrame:
    """Equal-risk TS book: w = score * (vol_target_per_coin / sigma_i) / N_live (as in the uploaded T1/T2)."""
    live = score.notna() & p.sigma_ann.notna()
    n = live.sum(axis=1).replace(0, np.nan)
    return cap_gross((score * per_coin_vol / p.sigma_ann).where(live).div(n, axis=0).fillna(0.0))


def xs_weights(p: Panel, score: pd.DataFrame, min_live: int = 4, rebalance_every: int = 7) -> pd.DataFrame:
    """Dollar-neutral, inverse-vol XS book with unit gross exposure; rebalanced every `rebalance_every` days."""
    live = score.notna() & p.sigma_ann.notna()
    ok = live.sum(axis=1) >= min_live
    z = score.where(live)
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1).replace(0, np.nan), axis=0)
    raw = (z / p.sigma_ann).where(live)
    w = raw.div(raw.abs().sum(axis=1).replace(0, np.nan), axis=0).where(ok).fillna(0.0)
    if rebalance_every > 1:
        # rebalance on a fixed calendar phase (day-of-grid modulo), hold in between
        phase = np.arange(len(w)) % rebalance_every == 0
        w = w.where(pd.Series(phase, index=w.index), np.nan).ffill().fillna(0.0)
    return w


def pv_target(w: pd.DataFrame, p: Panel, target: float = 0.20, span: int = 60, floor: float = 0.03,
              max_scale: float = 6.0, lag: int = 1) -> pd.DataFrame:
    """Scale WEIGHTS so the book's EWMA vol (known at close t) targets `target`; costs of re-scaling are paid."""
    r0 = run_book(w, p, lag=lag)["net"]
    vol = r0.ewm(span=span, min_periods=30).std() * np.sqrt(365.0)
    scale = (target / vol.clip(lower=floor)).clip(upper=max_scale).fillna(0.0)
    return cap_gross(w.mul(scale, axis=0))


def mom_z(p: Panel, k: int) -> pd.DataFrame:
    logp = np.log(p.close)
    sd = p.ret.ewm(span=30, min_periods=20).std()
    return (logp - logp.shift(k)) / (sd * np.sqrt(k))


def xs_mom_score(p: Panel, ks=(14, 28, 56, 112)) -> pd.DataFrame:
    parts = []
    for k in ks:
        m = mom_z(p, k)
        parts.append(m.sub(m.mean(axis=1), axis=0).div(m.std(axis=1).replace(0, np.nan), axis=0))
    return sum(parts) / len(parts)


def fund_z(p: Panel) -> pd.DataFrame:
    f3 = p.funding.rolling(3).sum()
    return (f3 - f3.rolling(90).mean()) / f3.rolling(90).std()


def donchian_score(p: Panel, windows=(20, 40, 80, 160), long_flat: bool = False) -> pd.DataFrame:
    """Hold +1 after a close above the prior N-day high, -1 (or 0) after a close below the prior N-day low,
    until the opposite break. Ensemble = mean over windows."""
    outs = []
    for N in windows:
        hi = p.close.rolling(N).max().shift(1); lo = p.close.rolling(N).min().shift(1)
        sig = pd.DataFrame(np.nan, index=p.index, columns=p.close.columns)
        sig[p.close > hi] = 1.0
        sig[p.close < lo] = 0.0 if long_flat else -1.0
        sig = sig.ffill().where(hi.notna())
        outs.append(sig)
    return sum(outs) / len(outs)


def smooth_score(score: pd.DataFrame, span: int = 5, band: float = 0.15) -> pd.DataFrame:
    """EMA-smoothed signal with a rest band: only move the held signal when it is > band away from target."""
    ema = score.ewm(span=span, min_periods=1).mean().where(score.notna())
    out = ema.copy()
    prev = pd.Series(0.0, index=score.columns)
    for i, t in enumerate(ema.index):
        tgt = ema.iloc[i]
        move = (tgt - prev).abs() > band
        cur = prev.where(~move, tgt)
        cur = cur.where(tgt.notna())
        out.iloc[i] = cur
        prev = cur.fillna(0.0)
    return out
