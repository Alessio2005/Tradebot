"""xs_multihorizon.py — multi-horizon ensemble to lift the XS-ML IC (Wave 15).

The single-horizon (H=10) XS model gives rank-IC +0.069 / harvested Sharpe ~0.97.
Ensembling relative-winner models across horizons H in {5,10,20} is the standard way
to raise IC and smooth the signal. Trains all three (purged CV, models PERSISTED to
artefacts/tracks_breadth/), averages the OOS scores, and harvests with the best
construction (rank-demean + vol/beta-neutralise + EWMA5 + no-trade band).

Run:  python scripts/xs_multihorizon.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
FUNDING = ROOT / "artefacts" / "funding_universe.parquet"
OUTDIR = ROOT / "artefacts" / "tracks_breadth"
DAYS = 365.0; N_FOLDS = 6; COST_BPS = 10.0; MIN_HIST = 900; TARGET_VOL = 0.40
HORIZONS = [5, 10, 20]


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
    close = panel.pivot_table(index="date", columns="symbol", values="close")[syms].sort_index()
    rets = np.log(close / close.shift(1)); mkt = rets.mean(axis=1)
    btc_ret = rets["BTCUSDT"]
    try:
        fund = pd.read_parquet(FUNDING).reindex(close.index).reindex(columns=syms)
        fund_z = ((fund - fund.rolling(30).mean()) / fund.rolling(30).std()).shift(1)
    except Exception:
        fund_z = pd.DataFrame(0.0, index=close.index, columns=syms)

    base = {s: base_feats(panel[panel.symbol == s].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()).reindex(close.index) for s in syms}
    base_cols = list(next(iter(base.values())).columns)
    beta = {}
    for s in syms:
        b = (rets[s].rolling(60).cov(btc_ret) / btc_ret.rolling(60).var()).clip(-3, 3)
        beta[s] = (rets[s] - b * btc_ret).rolling(20).sum()

    # static feature panel (horizon-independent)
    rows = []
    for s in syms:
        f = base[s].copy(); f.columns = [f"b_{c}" for c in base_cols]
        f["resid_mom20"] = beta[s]
        f["rel_ret20"] = rets[s].rolling(20).sum() - mkt.rolling(20).sum()
        f["fund_z"] = fund_z[s]
        f = f.shift(1); f["__sym"] = s; f["__date"] = close.index
        rows.append(f)
    P0 = pd.concat(rows, ignore_index=True)
    rank_cols = [f"b_{c}" for c in base_cols] + ["resid_mom20", "rel_ret20", "fund_z"]
    for col in rank_cols:
        P0[f"xr_{col}"] = P0.groupby("__date")[col].rank(pct=True)
    FEATS = [c for c in P0.columns if c.startswith("b_") or c.startswith("xr_")
             or c in ("resid_mom20", "rel_ret20", "fund_z")]

    score_sum = None
    for H in HORIZONS:
        P = P0.copy()
        fwd_map = {s: np.log(close[s].shift(-H) / close[s]) for s in syms}
        P["__fwd"] = P.apply(lambda r: fwd_map[r["__sym"]].get(r["__date"], np.nan), axis=1) \
            if False else np.concatenate([fwd_map[s].values for s in syms])  # fast path below
        # fast forward assembly (order of P0 is per-sym blocks)
        P["__fwd"] = np.concatenate([fwd_map[s].reindex(close.index).values for s in syms])
        med = P.groupby("__date")["__fwd"].transform("median")
        P["__win"] = (P["__fwd"] > med).astype(int)
        P = P.dropna(subset=["__win"]).sort_values("__date").reset_index(drop=True)
        dt = pd.to_datetime(P["__date"]).values
        edges = pd.to_datetime(np.quantile(dt.astype("int64"), np.linspace(0, 1, N_FOLDS + 1)))
        emb = np.timedelta64(H + 3, "D"); oos = np.full(len(P), np.nan)
        for k in range(N_FOLDS):
            lo, hi = edges[k].to_datetime64(), edges[k + 1].to_datetime64()
            test = (dt >= lo) & (dt < hi) if k < N_FOLDS - 1 else (dt >= lo) & (dt <= hi)
            overlap = (dt >= (lo - emb)) & (dt <= (hi + emb)); train = (~test) & (~overlap)
            if train.sum() < 2000:
                continue
            m = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=6,
                                   loss_function="Logloss", verbose=0, random_seed=0, subsample=0.8, rsm=0.8)
            m.fit(P.loc[train, FEATS].fillna(0.0), P.loc[train, "__win"])
            oos[test.nonzero()[0]] = m.predict_proba(P.loc[test, FEATS].fillna(0.0))[:, 1]
            joblib.dump({"model": m, "feats": FEATS, "H": H, "fold": k}, OUTDIR / f"xs_H{H}_fold{k}.joblib")
        P["__p"] = oos; m = P["__p"].notna()
        ic = P.loc[m, "__p"].corr(P.loc[m, "__fwd"], method="spearman")
        print(f"  H={H}: OOS AUC={roc_auc_score(P.loc[m,'__win'],P.loc[m,'__p']):.4f}  rank-IC={ic:+.4f}", flush=True)
        sc = P[m].pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last").reindex(close.index).reindex(columns=syms)
        sc = sc.rank(axis=1, pct=True)            # rank per horizon before averaging
        score_sum = sc if score_sum is None else score_sum.add(sc, fill_value=np.nan)
    score = score_sum / len(HORIZONS)

    # harvest: neutralise vs vol+beta, smooth, band
    vol = rets.rolling(30).std().shift(1); logvol = np.log(vol.clip(lower=1e-4))
    betaf = rets.rolling(60).cov(btc_ret).div(btc_ret.rolling(60).var(), axis=0).shift(1)
    sig = score.sub(score.mean(axis=1), axis=0); sign = sig * np.nan
    for dtm in sig.index:
        y = sig.loc[dtm].dropna()
        if len(y) < 15:
            continue
        X = pd.DataFrame({"lv": logvol.loc[dtm], "b": betaf.loc[dtm]}).reindex(y.index).fillna(0.0)
        X.insert(0, "c", 1.0); A = X.values
        if not np.isfinite(A).all():
            sign.loc[dtm, y.index] = y.values; continue
        try:
            coef, *_ = np.linalg.lstsq(A, y.values, rcond=None); sign.loc[dtm, y.index] = y.values - A @ coef
        except np.linalg.LinAlgError:
            sign.loc[dtm, y.index] = y.values
    w = sign.ewm(span=5).mean(); g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
    held = wn * 0.0; prev = pd.Series(0.0, index=wn.columns)
    for dtm in wn.index:
        tgt = wn.loc[dtm]; mv = (tgt - prev).abs() > 0.004; prev = prev.where(~mv, tgt); held.loc[dtm] = prev
    pnl = ((held.shift(1) * rets).sum(axis=1) - (held - held.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4).dropna()
    bv = pnl.std() * np.sqrt(DAYS); pnl = pnl * (TARGET_VOL / bv)
    sh = pnl.mean() / pnl.std() * np.sqrt(DAYS); eq = (1 + pnl).cumprod()
    cagr = eq.iloc[-1] ** (DAYS / len(pnl)) - 1; dd = (eq / eq.cummax() - 1).min()
    b = pnl.cov(mkt.reindex(pnl.index)) / mkt.reindex(pnl.index).var()
    yr = " ".join(f"{y}:{((1+gg).prod()-1)*100:+.0f}%" for y, gg in pnl.groupby(pnl.index.year))
    print(f"\n=== MULTI-HORIZON ENSEMBLE book (neutralised+smooth+band, 10bps, 40% vol) ===")
    print(f"  Sharpe={sh:.2f} CAGR={cagr*100:+.0f}% MaxDD={dd*100:.0f}% beta={b:+.2f} "
          f"DSR(N2000)={deflated_sharpe(pnl.mean()/pnl.std(),2000,len(pnl)):.3f}")
    print(f"  per-year: {yr}")


if __name__ == "__main__":
    main()
