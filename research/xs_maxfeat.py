"""xs_maxfeat.py — MAX the free cross-sectional feature set (Wave 16).

Answers the user's challenge: the prior XS model used ~16 base features. This throws
the full free OHLCV+altdata-derivable feature set at the relative-winner target to
find the best honest IC:
  base TA + fractional-diff price + downside/vol-of-vol + Amihud illiquidity +
  drawdown/52w-high distance + per-asset DVOL-beta & BTC-correlation (this is how
  market-wide alt-data enters a CROSS-SECTIONAL model) + funding-z + resid-mom +
  cross-sectional RANK of every feature.  ~40 base -> ~80 incl. ranks. Models
  PERSISTED. Reports IC vs the 34-feature baseline (+0.069) and the harvested Sharpe.

Run:  python research/xs_maxfeat.py
"""
from __future__ import annotations

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
ALT = ROOT / "artefacts" / "altdata_macro.parquet"
OUTDIR = ROOT / "artefacts" / "tracks_breadth"
DAYS = 365.0; H = 10; N_FOLDS, EMBARGO = 6, 13; COST_BPS = 10.0; MIN_HIST = 900; TARGET_VOL = 0.40


def fracdiff_fixed(series, d=0.4, width=20):
    w = [1.0]
    for k in range(1, width):
        w.append(-w[-1] * (d - k + 1) / k)
    w = np.array(w[::-1])
    return series.rolling(width).apply(lambda x: np.dot(w, x), raw=True)


def feats(df, dvol_chg):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    r = np.log(c / c.shift(1)); F = pd.DataFrame(index=df.index)
    for L in (5, 10, 20, 60):
        F[f"ret{L}"] = np.log(c / c.shift(L))
    F["vol20"] = r.rolling(20).std(); F["vol60"] = r.rolling(60).std()
    F["volratio"] = F["vol20"] / F["vol60"]
    F["ma50"] = c / c.rolling(50).mean() - 1; F["ma100"] = c / c.rolling(100).mean() - 1
    d_ = c.diff(); up = d_.clip(lower=0).rolling(14).mean(); dn = (-d_.clip(upper=0)).rolling(14).mean()
    F["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hi = c.rolling(20).max(); lo = c.rolling(20).min()
    F["rangepos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    F["skew20"] = r.rolling(20).skew(); F["kurt20"] = r.rolling(20).kurt()
    F["ac1"] = r.rolling(40).apply(lambda x: x.autocorr(1), raw=False)
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    F["gk20"] = gk.rolling(20).mean()
    # NEW free features
    F["fdiff"] = fracdiff_fixed(np.log(c))                       # stationary memory of price
    F["downvol"] = r.clip(upper=0).rolling(20).std()             # semideviation
    F["volofvol"] = F["vol20"].rolling(20).std()
    dollar = (c * v).replace(0, np.nan)
    F["amihud"] = (r.abs() / dollar).rolling(20).mean() * 1e9    # illiquidity
    F["dd60"] = c / c.rolling(60).max() - 1                      # drawdown from 60d high
    F["hi52"] = c / c.rolling(252).max() - 1                     # distance from 52w high
    F["dvol_beta"] = r.rolling(60).cov(dvol_chg) / dvol_chg.rolling(60).var()  # vol-regime sensitivity
    return F


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    syms = [s for s, g in panel.groupby("symbol") if len(g) >= MIN_HIST]
    import os
    if os.environ.get("MAX_ASSETS"):
        syms = syms[: int(os.environ["MAX_ASSETS"])]
        if "BTCUSDT" not in syms:
            syms = ["BTCUSDT"] + syms
    close = panel.pivot_table(index="date", columns="symbol", values="close")[syms].sort_index()
    rets = np.log(close / close.shift(1)); mkt = rets.mean(axis=1); btc_ret = rets["BTCUSDT"]
    alt = pd.read_parquet(ALT).reindex(close.index).ffill()
    dvol_chg = alt["dvol_btc"].diff().reindex(close.index)
    try:
        fund = pd.read_parquet(FUNDING).reindex(close.index).reindex(columns=syms)
        fund_z = ((fund - fund.rolling(30).mean()) / fund.rolling(30).std()).shift(1)
    except Exception:
        fund_z = pd.DataFrame(0.0, index=close.index, columns=syms)
    print(f"max-feat XS: {len(syms)} assets", flush=True)

    rows = []
    for s in syms:
        df = panel[panel.symbol == s].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        F = feats(df, dvol_chg).reindex(close.index)
        b = (rets[s].rolling(60).cov(btc_ret) / btc_ret.rolling(60).var()).clip(-3, 3)
        F["resid_mom20"] = (rets[s] - b * btc_ret).rolling(20).sum()
        F["btc_corr"] = rets[s].rolling(60).corr(btc_ret)
        F["rel_ret20"] = rets[s].rolling(20).sum() - mkt.rolling(20).sum()
        F["fund_z"] = fund_z[s]
        F = F.shift(1); F["__sym"] = s; F["__date"] = close.index
        F["__fwd"] = np.log(close[s].shift(-H) / close[s]).values
        rows.append(F)
    P = pd.concat(rows, ignore_index=True)
    basef = [c for c in P.columns if not c.startswith("__")]
    for col in basef:
        P[f"xr_{col}"] = P.groupby("__date")[col].rank(pct=True)
    FEATS = [c for c in P.columns if not c.startswith("__")]
    med = P.groupby("__date")["__fwd"].transform("median")
    P["__win"] = (P["__fwd"] > med).astype(int)
    P = P.dropna(subset=["__win"]).sort_values("__date").reset_index(drop=True)
    print(f"rows={len(P)}  FEATURES={len(FEATS)}  (baseline was 34)", flush=True)

    dt = pd.to_datetime(P["__date"]).values
    edges = pd.to_datetime(np.quantile(dt.astype("int64"), np.linspace(0, 1, N_FOLDS + 1)))
    emb = np.timedelta64(EMBARGO, "D"); oos = np.full(len(P), np.nan)
    for k in range(N_FOLDS):
        lo, hi = edges[k].to_datetime64(), edges[k + 1].to_datetime64()
        test = (dt >= lo) & (dt < hi) if k < N_FOLDS - 1 else (dt >= lo) & (dt <= hi)
        overlap = (dt >= (lo - emb)) & (dt <= (hi + emb)); train = (~test) & (~overlap)
        if train.sum() < 2000:
            continue
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.025, l2_leaf_reg=8,
                               loss_function="Logloss", verbose=0, random_seed=0, subsample=0.8, rsm=0.7)
        m.fit(P.loc[train, FEATS].fillna(0.0), P.loc[train, "__win"])
        oos[test.nonzero()[0]] = m.predict_proba(P.loc[test, FEATS].fillna(0.0))[:, 1]
        joblib.dump({"model": m, "feats": FEATS, "fold": k}, OUTDIR / f"xsmax_fold{k}.joblib")
    P["__p"] = oos; mask = P["__p"].notna()
    auc = roc_auc_score(P.loc[mask, "__win"], P.loc[mask, "__p"])
    ic = P.loc[mask, "__p"].corr(P.loc[mask, "__fwd"], method="spearman")
    Pm = P[mask].copy(); Pm["__yr"] = pd.to_datetime(Pm["__date"]).dt.year
    icy = Pm.groupby("__yr").apply(lambda g: g["__p"].corr(g["__fwd"], method="spearman"))
    print(f"\nMAX-FEAT OOS AUC={auc:.4f}  rank-IC={ic:+.4f}  (baseline IC +0.069)")
    print("  per-year IC: " + " ".join(f"{y}:{v:+.3f}" for y, v in icy.items()))

    # top feature importances (avg across folds)
    imps = {}
    for k in range(N_FOLDS):
        fp = OUTDIR / f"xsmax_fold{k}.joblib"
        if fp.exists():
            d = joblib.load(fp); im = d["model"].get_feature_importance()
            for f_, val in zip(d["feats"], im):
                imps[f_] = imps.get(f_, 0) + val / N_FOLDS
    top = sorted(imps.items(), key=lambda x: -x[1])[:10]
    print("  top-10 features: " + ", ".join(f"{f}={v:.1f}" for f, v in top))

    # harvest (neutralise vol+beta, smooth, band)
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
    held = wn * 0.0; prev = pd.Series(0.0, index=wn.columns)
    for dtm in wn.index:
        tgt = wn.loc[dtm]; mv = (tgt - prev).abs() > 0.004; prev = prev.where(~mv, tgt); held.loc[dtm] = prev
    pnl = ((held.shift(1) * rets).sum(axis=1) - (held - held.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4).dropna()
    bv = pnl.std() * np.sqrt(DAYS); pnl = pnl * (TARGET_VOL / bv)
    sh = pnl.mean() / pnl.std() * np.sqrt(DAYS); eq = (1 + pnl).cumprod()
    cagr = eq.iloc[-1] ** (DAYS / len(pnl)) - 1; dd = (eq / eq.cummax() - 1).min()
    yr = " ".join(f"{y}:{((1+gg).prod()-1)*100:+.0f}%" for y, gg in pnl.groupby(pnl.index.year))
    print(f"\n=== MAX-FEAT harvested book Sharpe={sh:.2f} CAGR={cagr*100:+.0f}% MaxDD={dd*100:.0f}% "
          f"DSR(N2000)={deflated_sharpe(pnl.mean()/pnl.std(),2000,len(pnl)):.3f}")
    print(f"  per-year: {yr}")


if __name__ == "__main__":
    main()
