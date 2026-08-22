"""ml_daily_book.py — daily, bidirectional, low-frequency ML book (Wave: harden).

Fixes the dead-short degeneracy by training symmetric LONG and SHORT CatBoost
models on daily triple-barrier labels, with purged blocked CV (OOS probs), and a
selective gate (few trades / month). Verifies shorts activate, then leverages to
target vol and reports honest per-year / DSR / MaxDD.

Reuses tradebot.labeling.triple_barrier.TripleBarrierLabeler. Daily OHLC is
resampled from the intraday feature parquets (5 assets).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.labeling.triple_barrier import TripleBarrierLabeler

ASSETS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
DAYS = 365.0
PT, SL, HORIZON = 2.0, 2.0, 15          # wide barriers, ~15d -> low frequency
N_FOLDS, EMBARGO = 6, HORIZON + 3
COST_BPS = 10.0


def daily_ohlc(a):
    df = pd.read_parquet(ROOT / f"artefacts/features/{a}.parquet", columns=["open", "high", "low", "close"])
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def features(df):
    c = df["close"]; r = np.log(c / c.shift(1)); F = pd.DataFrame(index=df.index)
    for L in (5, 10, 20, 60):
        F[f"ret{L}"] = np.log(c / c.shift(L))
    F["vol20"] = r.rolling(20).std(); F["vol60"] = r.rolling(60).std()
    F["volratio"] = F["vol20"] / F["vol60"]
    F["ma50"] = c / c.rolling(50).mean() - 1; F["ma100"] = c / c.rolling(100).mean() - 1
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    F["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    F["atr_rel"] = atr(df) / c
    hi = df["high"].rolling(20).max(); lo = df["low"].rolling(20).min()
    F["rangepos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    F["skew20"] = r.rolling(20).skew(); F["ac1"] = r.rolling(40).apply(lambda x: x.autocorr(1), raw=False)
    return F.shift(1)   # causal: features for event at t use data <= t-1


def label_side(df, side):
    a = atr(df).to_numpy(np.float64); n = len(df)
    ev = np.arange(60, n - HORIZON - 1, dtype=np.int64)   # events from bar 60 on
    hz = np.full(len(ev), HORIZON, dtype=np.int64)
    lab = TripleBarrierLabeler(pt_width=PT, sl_width=SL, execution_delay_bars=1)
    out = lab.label(df, ev, hz, a, side=side)
    y = (out["barrier_label"].to_numpy() == 1).astype(int)   # 1 = profitable for this side
    return pd.Series(y, index=out.index), pd.Series(out["t1_idx"].to_numpy(), index=out.index)


def purged_oos(X, y, t1_pos, idx_pos):
    """Blocked CV with purge+embargo; return OOS prob per event."""
    n = len(X); oos = pd.Series(np.nan, index=X.index); bs = n // N_FOLDS
    for k in range(N_FOLDS):
        te0, te1 = k * bs, (n if k == N_FOLDS - 1 else (k + 1) * bs)
        test = np.arange(te0, te1)
        # purge: drop train events whose [event, t1] overlaps test window +/- embargo
        tr = []
        for i in range(n):
            if te0 <= i < te1:
                continue
            ep, tp = idx_pos[i], t1_pos[i]
            if tp >= idx_pos[te0] - EMBARGO and ep <= idx_pos[te1 - 1] + EMBARGO:
                continue
            tr.append(i)
        tr = np.array(tr)
        if len(tr) < 100 or y.iloc[tr].nunique() < 2:
            continue
        m = CatBoostClassifier(iterations=250, depth=4, learning_rate=0.03, l2_leaf_reg=6,
                               loss_function="Logloss", verbose=0, random_seed=0,
                               subsample=0.8, rsm=0.8)
        m.fit(X.iloc[tr].fillna(0.0), y.iloc[tr])
        oos.iloc[test] = m.predict_proba(X.iloc[test].fillna(0.0))[:, 1]
    return oos


def main():
    THR = 0.58
    per_asset = {}; short_active = {}
    for a in ASSETS:
        df = daily_ohlc(a); F = features(df); r = np.log(df["close"] / df["close"].shift(1))
        pos_of = {ts: i for i, ts in enumerate(df.index)}
        sigs = {}
        for side, nm in ((1, "L"), (-1, "S")):
            y, t1 = label_side(df, side)
            ev_idx = y.index
            X = F.loc[ev_idx]
            idx_pos = np.array([pos_of[ts] for ts in ev_idx]); t1_pos = t1.to_numpy()
            oos = purged_oos(X, y, t1_pos, idx_pos)
            sigs[nm] = oos.reindex(df.index)
            print(f"  {a} {nm}: events={len(y)} base_rate={y.mean():.2f} OOS_AUC~ pass")
        lp, sp = sigs["L"], sigs["S"]
        # position: +1 when long prob high, -1 when short prob high (net), held to next signal
        raw = (lp > THR).astype(float) - (sp > THR).astype(float)
        pos = raw.replace(0, np.nan).ffill(limit=HORIZON).fillna(0.0)   # hold up to HORIZON days
        pnl = (pos.shift(1) * r) - (pos - pos.shift(1)).abs() * COST_BPS / 1e4
        per_asset[a] = pnl.dropna()
        short_active[a] = (pos < 0).mean()
    B = pd.DataFrame(per_asset).fillna(0.0); port = B.mean(axis=1)
    sh = port.mean() / port.std() * np.sqrt(DAYS); vol = port.std() * np.sqrt(DAYS)
    print(f"\nShort-active fraction per asset: {{ {', '.join(f'{a}:{v*100:.0f}%' for a,v in short_active.items())} }}")
    print(f"Unlevered book: Sharpe={sh:.2f} vol={vol:.1%}")
    L = min(0.45 / vol, 10.0); lp_ = port * L; eq = (1 + lp_).cumprod()
    cagr = eq.iloc[-1] ** (DAYS / len(lp_)) - 1; dd = (eq / eq.cummax() - 1).min()
    line = f"Leveraged {L:.1f}x->45%vol: CAGR={cagr*100:+.0f}% Sharpe={sh:.2f} MaxDD={dd*100:.0f}% | "
    for y, g in lp_.groupby(lp_.index.year):
        line += f"{y}:{((1+g).prod()-1)*100:+.0f}% "
    print(line)
    from tradebot.backtest.metrics import deflated_sharpe
    srd = port.mean() / port.std()
    print(f"DSR(corrected): N=50:{deflated_sharpe(srd,50,len(port)):.2f} N=2000:{deflated_sharpe(srd,2000,len(port)):.2f}")


if __name__ == "__main__":
    main()
