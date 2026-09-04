"""walkforward_forward.py — adaptive, recency-weighted forward test (Wave 18).

User point (valid): one static strategy across 5 regimes is the wrong test; what
matters is being robust FORWARD for the coming year, trained on data up to now and
refit as the market moves. Backtest = validation, weight the CURRENT regime.

Method (strict, no future leak):
  - rolling WALK-FORWARD: at each quarter t0, train ONLY on rows dated < t0 - embargo,
    test on [t0, t0+91d]. Pure past->future. Refit every quarter.
  - RECENCY sample-weights: exp decay by age (half-life configurable) so the model
    tracks the current regime, not the 2021/2022 world.
  - REGIME GUARD: de-gross the (short-biased) book when the market is in a strong
    bull mania (its known failure mode), sized off trailing market trend.
  - judge on the LAST 12 MONTHS forward (proxy for "coming year"), report all quarters.

Run:  python research/walkforward_forward.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
FUNDING = ROOT / "artefacts" / "funding_universe.parquet"
DAYS = 365.0; H = 10; COST_BPS = 10.0; MIN_HIST = 900; TARGET_VOL = 0.40
EMBARGO = pd.Timedelta(days=H + 3)
HALFLIFE_D = 365.0           # recency half-life
WF_START = "2023-01-01"      # first forward test quarter


def base_feats(df):
    c = df["close"]; r = np.log(c / c.shift(1)); F = pd.DataFrame(index=df.index)
    for L in (5, 10, 20, 60):
        F[f"ret{L}"] = np.log(c / c.shift(L))
    F["vol20"] = r.rolling(20).std(); F["vol60"] = r.rolling(60).std()
    F["volratio"] = F["vol20"] / F["vol60"]
    F["ma50"] = c / c.rolling(50).mean() - 1; F["ma100"] = c / c.rolling(100).mean() - 1
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    F["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hi = c.rolling(20).max(); lo = c.rolling(20).min()
    F["rangepos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    F["skew20"] = r.rolling(20).skew(); F["ac1"] = r.rolling(40).apply(lambda x: x.autocorr(1), raw=False)
    o, h, l = df["open"], df["high"], df["low"]
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    F["gk20"] = gk.rolling(20).mean()
    return F


def build_pool(panel, syms, close, rets, mkt, btc_ret, fund_z):
    rows = []
    for s in syms:
        df = panel[panel.symbol == s].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        F = base_feats(df).reindex(close.index)
        b = (rets[s].rolling(60).cov(btc_ret) / btc_ret.rolling(60).var()).clip(-3, 3)
        F["resid_mom20"] = (rets[s] - b * btc_ret).rolling(20).sum()
        F["rel_ret20"] = rets[s].rolling(20).sum() - mkt.rolling(20).sum()
        F["fund_z"] = fund_z[s]
        F = F.shift(1); F["__sym"] = s; F["__date"] = close.index
        F["__fwd"] = np.log(close[s].shift(-H) / close[s]).values
        rows.append(F)
    P = pd.concat(rows, ignore_index=True)
    basef = [c for c in P.columns if not c.startswith("__")]
    for col in basef:
        P[f"xr_{col}"] = P.groupby("__date")[col].rank(pct=True)
    med = P.groupby("__date")["__fwd"].transform("median")
    P["__win"] = (P["__fwd"] > med).astype(int)
    feats = [c for c in P.columns if not c.startswith("__")]
    return P.dropna(subset=["__win"]).reset_index(drop=True), feats


def main():
    panel = pd.read_parquet(PANEL)
    syms = [s for s, g in panel.groupby("symbol") if len(g) >= MIN_HIST]
    close = panel.pivot_table(index="date", columns="symbol", values="close")[syms].sort_index()
    rets = np.log(close / close.shift(1)); mkt = rets.mean(axis=1); btc_ret = rets["BTCUSDT"]
    try:
        fund = pd.read_parquet(FUNDING).reindex(close.index).reindex(columns=syms)
        fund_z = ((fund - fund.rolling(30).mean()) / fund.rolling(30).std()).shift(1)
    except Exception:
        fund_z = pd.DataFrame(0.0, index=close.index, columns=syms)

    P, FEATS = build_pool(panel, syms, close, rets, mkt, btc_ret, fund_z)
    P["__date"] = pd.to_datetime(P["__date"]); P = P.sort_values("__date").reset_index(drop=True)
    print(f"pool rows={len(P)}  feats={len(FEATS)}  walk-forward from {WF_START}, halflife={HALFLIFE_D:.0f}d", flush=True)

    qs = pd.date_range(pd.Timestamp(WF_START, tz="UTC"), P["__date"].max(), freq="91D")
    oos = pd.Series(np.nan, index=P.index)
    for t0 in qs:
        t1 = t0 + pd.Timedelta(days=91)
        tr = P["__date"] < (t0 - EMBARGO)
        te = (P["__date"] >= t0) & (P["__date"] < t1)
        if tr.sum() < 5000 or te.sum() == 0:
            continue
        age = (t0 - P.loc[tr, "__date"]).dt.days.values
        w = np.exp(-np.log(2) * age / HALFLIFE_D)        # recency weight
        m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=8,
                               loss_function="Logloss", verbose=0, random_seed=0, subsample=0.8, rsm=0.8)
        m.fit(P.loc[tr, FEATS].fillna(0.0), P.loc[tr, "__win"], sample_weight=w)
        oos[te.nonzero()[0] if hasattr(te, 'nonzero') else te] = m.predict_proba(P.loc[te, FEATS].fillna(0.0))[:, 1]
    P["__p"] = oos.values
    mask = P["__p"].notna()
    auc = roc_auc_score(P.loc[mask, "__win"], P.loc[mask, "__p"])
    ic = P.loc[mask, "__p"].corr(P.loc[mask, "__fwd"], method="spearman")
    print(f"WALK-FORWARD forward OOS: AUC={auc:.4f}  IC={ic:+.4f}", flush=True)

    # harvest: neutralise vol+beta, smooth, band  + REGIME GUARD (de-gross in bull mania)
    score = P[mask].pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last").reindex(close.index).reindex(columns=syms)
    vol = rets.rolling(30).std().shift(1); logvol = np.log(vol.clip(lower=1e-4))
    betaf = rets.rolling(60).cov(btc_ret).div(btc_ret.rolling(60).var(), axis=0).shift(1)
    rk = score.rank(axis=1, pct=True); sig = rk.sub(rk.mean(axis=1), axis=0); sign = sig * np.nan
    for dtm in sig.index:
        y = sig.loc[dtm].dropna()
        if len(y) < 15:
            continue
        X = pd.DataFrame({"lv": logvol.loc[dtm], "b": betaf.loc[dtm]}).reindex(y.index).fillna(0.0)
        X.insert(0, "c", 1.0); A = X.values
        if np.isfinite(A).all():
            try:
                coef, *_ = np.linalg.lstsq(A, y.values, rcond=None); sign.loc[dtm, y.index] = y.values - A @ coef
            except np.linalg.LinAlgError:
                sign.loc[dtm, y.index] = y.values
        else:
            sign.loc[dtm, y.index] = y.values
    w = sign.ewm(span=5).mean(); g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
    # regime guard: market 60d trend strength -> cut gross when strongly bull (mania)
    mkt_trend = mkt.rolling(60).mean() / mkt.rolling(60).std()
    gross = (1.0 - (mkt_trend.shift(1).clip(lower=0) / mkt_trend.clip(lower=0).quantile(0.9)).clip(0, 1) * 0.7).fillna(1.0)
    held = wn * 0.0; prev = pd.Series(0.0, index=wn.columns)
    for dtm in wn.index:
        tgt = wn.loc[dtm] * gross.get(dtm, 1.0); mv = (tgt - prev).abs() > 0.004
        prev = prev.where(~mv, tgt); held.loc[dtm] = prev
    pnl = ((held.shift(1) * rets).sum(axis=1) - (held - held.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4).dropna()
    bv = pnl.std() * np.sqrt(DAYS); pnl = pnl * (TARGET_VOL / bv)

    sh = pnl.mean() / pnl.std() * np.sqrt(DAYS)
    yrs = {y: (1 + g_).prod() - 1 for y, g_ in pnl.groupby(pnl.index.year)}
    last12 = pnl[pnl.index >= (pnl.index.max() - pd.Timedelta(days=365))]
    sh12 = last12.mean() / last12.std() * np.sqrt(DAYS); ret12 = (1 + last12).prod() - 1
    dd = ((1 + pnl).cumprod() / (1 + pnl).cumprod().cummax() - 1).min()
    print("\n=== WALK-FORWARD FORWARD BOOK (recency-weighted, regime-guarded) ===")
    print(f"  forward Sharpe={sh:.2f}  MaxDD={dd*100:.0f}%  DSR(N2000)={deflated_sharpe(pnl.mean()/pnl.std(),2000,len(pnl)):.3f}")
    print("  forward per-year: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in yrs.items()))
    print(f"  LAST 12 MONTHS (proxy for coming year): return={ret12*100:+.0f}%  Sharpe={sh12:.2f}")
    # quarter-by-quarter recent
    q = pnl.resample("91D").apply(lambda x: (1 + x).prod() - 1)
    print("  recent quarters: " + " ".join(f"{d.date()}:{v*100:+.0f}%" for d, v in q.tail(8).items()))


if __name__ == "__main__":
    main()
