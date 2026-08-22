"""xs_alpha_rework.py — cross-sectional ML rework, models PERSISTED (Wave 15).

Addresses the real gap in Wave 14: every prior probe used PER-ASSET features only,
so the models learned volatility/path artifacts (AUC up, P&L <=0) instead of
relative-value. This rework introduces genuinely new inputs and a proper target:

  NEW FEATURES (per asset, causal, data <= t-1):
    - base TA (returns, vol, RSI, MA, range, skew, autocorr, GK vol)
    - CROSS-SECTIONAL RANK of each base feature across the universe each day
    - BTC-residual momentum (rolling-beta residual returns -> idiosyncratic trend)
    - market-relative return (asset ret minus cross-sectional mean)
    - FUNDING carry z-score (lagged; cross-sectional carry signal)
  NEW TARGET: relative winner -> 1 if asset fwd H-day return > cross-sectional
    median that day (predict relative out/under-performance, where MN alpha lives).

Trains ONE pooled CatBoost per CV fold with global time-purge+embargo, collects OOS
relative-winner probs, PERSISTS the per-fold models to artefacts/tracks_breadth/,
then builds the dollar-neutral decile book and measures honestly.

Run:  python scripts/xs_alpha_rework.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
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
OUTDIR = ROOT / "artefacts" / "tracks_breadth"
DAYS = 365.0
H = 10
N_FOLDS, EMBARGO = 6, H + 3
COST_BPS = 10.0
MIN_HIST = 900
TARGET_VOL = 0.40
DECILE = 0.20


def base_feats(df):
    c = df["close"]; r = np.log(c / c.shift(1)); F = pd.DataFrame(index=df.index)
    for L in (5, 10, 20, 60):
        F[f"ret{L}"] = np.log(c / c.shift(L))
    F["vol20"] = r.rolling(20).std(); F["vol60"] = r.rolling(60).std()
    F["volratio"] = F["vol20"] / F["vol60"]
    F["ma50"] = c / c.rolling(50).mean() - 1; F["ma100"] = c / c.rolling(100).mean() - 1
    d = c.diff(); up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    F["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hi = df["high"].rolling(20).max(); lo = df["low"].rolling(20).min()
    F["rangepos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    F["skew20"] = r.rolling(20).skew(); F["ac1"] = r.rolling(40).apply(lambda x: x.autocorr(1), raw=False)
    o, h, l = df["open"], df["high"], df["low"]
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    F["gk20"] = gk.rolling(20).mean()
    return F


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    syms = [s for s, g in panel.groupby("symbol") if len(g) >= MIN_HIST]
    import os
    if os.environ.get("MAX_ASSETS"):
        syms = syms[: int(os.environ["MAX_ASSETS"])]
    print(f"XS rework: {len(syms)} assets", flush=True)

    # wide close + return panels
    close = panel.pivot_table(index="date", columns="symbol", values="close")[syms].sort_index()
    rets = np.log(close / close.shift(1))
    btc = np.log(panel.pivot_table(index="date", columns="symbol", values="close")["BTCUSDT"]
                 ).pipe(lambda s: s).reindex(close.index)
    btc_ret = btc.diff()
    mkt_ret = rets.mean(axis=1)                       # equal-weight market

    # funding panel (lagged 1d for causality)
    try:
        fund = pd.read_parquet(FUNDING).reindex(close.index).reindex(columns=syms)
        fund_z = (fund - fund.rolling(30).mean()) / fund.rolling(30).std()
        fund_z = fund_z.shift(1)
    except Exception:
        fund_z = pd.DataFrame(0.0, index=close.index, columns=syms)

    # per-asset base features -> dict of frames
    base = {}
    for sym in syms:
        df = panel[panel.symbol == sym].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        base[sym] = base_feats(df).reindex(close.index)
    base_cols = list(next(iter(base.values())).columns)

    # assemble long pooled frame with cross-sectional + residual + funding features
    print("Assembling cross-sectional feature panel...", flush=True)
    # BTC-residual momentum: rolling beta of asset ret on btc ret, residual cum over 20d
    beta = {}
    for sym in syms:
        cov = rets[sym].rolling(60).cov(btc_ret); var = btc_ret.rolling(60).var()
        b = (cov / var).clip(-3, 3)
        resid = rets[sym] - b * btc_ret
        beta[sym] = resid.rolling(20).sum()           # idiosyncratic 20d momentum

    rows = []
    for sym in syms:
        f = base[sym].copy()
        f.columns = [f"b_{c}" for c in base_cols]
        f["resid_mom20"] = beta[sym]
        f["rel_ret20"] = rets[sym].rolling(20).sum() - mkt_ret.rolling(20).sum()
        f["fund_z"] = fund_z[sym]
        f = f.shift(1)                                # causal: event at t uses data <= t-1
        f["__sym"] = sym; f["__date"] = close.index
        # target: relative winner over next H days
        fwd = np.log(close[sym].shift(-H) / close[sym])
        f["__fwd"] = fwd.values
        rows.append(f)
    P = pd.concat(rows, ignore_index=True)

    # cross-sectional rank features (per date, across assets) on the base TA cols
    print("Computing cross-sectional ranks...", flush=True)
    feat_for_rank = [f"b_{c}" for c in base_cols] + ["resid_mom20", "rel_ret20", "fund_z"]
    for col in feat_for_rank:
        P[f"xr_{col}"] = P.groupby("__date")[col].rank(pct=True)
    # cross-sectional median forward return -> relative-winner label
    med = P.groupby("__date")["__fwd"].transform("median")
    P["__win"] = (P["__fwd"] > med).astype(int)
    P = P.dropna(subset=["__win", "__fwd"])

    FEATS = [c for c in P.columns if c.startswith("b_") or c.startswith("xr_")
             or c in ("resid_mom20", "rel_ret20", "fund_z")]
    print(f"Pooled rows={len(P)}  features={len(FEATS)}  base win-rate={P['__win'].mean():.3f}", flush=True)

    # global time-purged CV -> OOS prob, persist per-fold models
    dt = pd.to_datetime(P["__date"]).values
    P = P.sort_values("__date").reset_index(drop=True); dt = pd.to_datetime(P["__date"]).values
    edges = pd.to_datetime(np.quantile(dt.astype("int64"), np.linspace(0, 1, N_FOLDS + 1)))
    emb = np.timedelta64(EMBARGO, "D")
    oos = np.full(len(P), np.nan)
    saved = []
    for k in range(N_FOLDS):
        lo, hi = edges[k].to_datetime64(), edges[k + 1].to_datetime64()
        test = (dt >= lo) & (dt < hi) if k < N_FOLDS - 1 else (dt >= lo) & (dt <= hi)
        overlap = (dt >= (lo - emb)) & (dt <= (hi + emb))
        train = (~test) & (~overlap)
        if train.sum() < 2000:
            continue
        m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=6,
                               loss_function="Logloss", verbose=0, random_seed=0,
                               subsample=0.8, rsm=0.8)
        m.fit(P.loc[train, FEATS].fillna(0.0), P.loc[train, "__win"])
        oos[test.nonzero()[0]] = m.predict_proba(P.loc[test, FEATS].fillna(0.0))[:, 1]
        fp = OUTDIR / f"xs_fold{k}.joblib"
        joblib.dump({"model": m, "feats": FEATS, "fold": k}, fp)
        saved.append(fp.name)
    P["__p"] = oos
    # persist OOS score panel + returns so book construction can iterate without retraining
    P.loc[P["__p"].notna(), ["__date", "__sym", "__p", "__fwd"]].to_parquet(
        ROOT / "artefacts" / "xs_oos_scores.parquet")
    close.to_parquet(ROOT / "artefacts" / "xs_close_panel.parquet")
    mask = P["__p"].notna()
    auc = roc_auc_score(P.loc[mask, "__win"], P.loc[mask, "__p"])
    ic = P.loc[mask, "__p"].corr(P.loc[mask, "__fwd"], method="spearman")
    print(f"\nOOS relative-winner AUC = {auc:.4f}   rank-IC(p vs fwd ret) = {ic:.4f}", flush=True)
    Pm = P[mask].copy(); Pm["__yr"] = pd.to_datetime(Pm["__date"]).dt.year
    icy = Pm.groupby("__yr").apply(lambda g: g["__p"].corr(g["__fwd"], method="spearman"))
    print("  per-year rank-IC: " + " ".join(f"{y}:{v:+.3f}" for y, v in icy.items()))
    print(f"PERSISTED models: {saved}", flush=True)

    # dollar-neutral decile book
    score = P.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last").reindex(columns=syms)
    score = score.reindex(close.index)
    invvol = 1.0 / rets.rolling(30).std().shift(1).clip(lower=1e-4)
    pos = pd.DataFrame(0.0, index=close.index, columns=syms)
    for dt_ in score.index:
        row = score.loc[dt_].dropna()
        if len(row) < 10:
            continue
        klo, khi = row.quantile(DECILE), row.quantile(1 - DECILE)
        w = pd.Series(0.0, index=syms)
        longs, shorts = row[row >= khi].index, row[row <= klo].index
        w[longs] = invvol.loc[dt_, longs].fillna(0.0)
        w[shorts] = -invvol.loc[dt_, shorts].fillna(0.0)
        g = w.abs().sum()
        if g > 0:
            pos.loc[dt_] = w / g
    pnl_gross = (pos.shift(1) * rets).sum(axis=1)
    turn = (pos - pos.shift(1)).abs().sum(axis=1)
    pnl = (pnl_gross - turn * COST_BPS / 1e4).dropna()
    bv = pnl.std() * np.sqrt(DAYS)
    pnl = pnl * (TARGET_VOL / bv) if bv > 0 else pnl
    sh = pnl.mean() / pnl.std() * np.sqrt(DAYS); vol = pnl.std() * np.sqrt(DAYS)
    eq = (1 + pnl).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(pnl)) - 1
    dd = (eq / eq.cummax() - 1).min(); srd = pnl.mean() / pnl.std()
    beta_mkt = pnl.reindex(mkt_ret.index).cov(mkt_ret) / mkt_ret.var()
    yr = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in pnl.groupby(pnl.index.year))
    print("\n===== CROSS-SECTIONAL RELATIVE-VALUE BOOK — DECILE (dollar-neutral, 10bps, 40% vol) =====")
    print(f"  Sharpe={sh:.2f}  CAGR={cagr*100:+.0f}%  vol={vol:.0%}  MaxDD={dd*100:.0f}%  "
          f"beta={beta_mkt:.2f}  DSR(N2000)={deflated_sharpe(srd,2000,len(pnl)):.3f}")
    print(f"  per-year: {yr}")

    # continuous score-weighted book (smoother harvest of the IC)
    sdm = score.sub(score.mean(axis=1), axis=0)       # cross-sectionally demean
    cw = (sdm * invvol).fillna(0.0)
    cg = cw.abs().sum(axis=1).replace(0, np.nan)
    cwn = cw.div(cg, axis=0).fillna(0.0)
    cpnl = ((cwn.shift(1) * rets).sum(axis=1)
            - (cwn - cwn.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4).dropna()
    cbv = cpnl.std() * np.sqrt(DAYS); cpnl = cpnl * (TARGET_VOL / cbv) if cbv > 0 else cpnl
    csh = cpnl.mean() / cpnl.std() * np.sqrt(DAYS)
    ceq = (1 + cpnl).cumprod(); ccagr = ceq.iloc[-1] ** (DAYS / len(cpnl)) - 1
    cdd = (ceq / ceq.cummax() - 1).min(); csrd = cpnl.mean() / cpnl.std()
    cyr = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in cpnl.groupby(cpnl.index.year))
    print("\n===== CONTINUOUS SCORE-WEIGHTED BOOK =====")
    print(f"  Sharpe={csh:.2f}  CAGR={ccagr*100:+.0f}%  MaxDD={cdd*100:.0f}%  "
          f"DSR(N2000)={deflated_sharpe(csrd,2000,len(cpnl)):.3f}")
    print(f"  per-year: {cyr}")

    json.dump({"auc": auc, "ic": ic, "sharpe": sh, "cagr": cagr, "maxdd": dd,
               "beta": beta_mkt, "n_models": len(saved), "feats": len(FEATS)},
              open(ROOT / "reports" / "xs_rework_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
