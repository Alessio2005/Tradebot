"""meta_label_book.py — meta-labeling done right (Wave 14, LdP central technique).

The breadth direct-classification rework failed (AUC up, P&L <=0). This tests the
canonical Lopez de Prado structure: a PRIMARY signal sets the side (the only signal
with faint positive edge = cross-sectional 20d reversal), a SECONDARY CatBoost
meta-model decides bet/size by predicting p(this bet wins). If the meta-model has
real skill it filters losers and lifts Sharpe; if not, the ML rework is exhausted.

Run:  python scripts/meta_label_book.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
DAYS = 365.0
H = 15                       # bet horizon (days)
N_FOLDS, EMBARGO = 6, H + 3
COST_BPS = 10.0
MIN_HIST = 900
TARGET_VOL = 0.40
FEATCOLS = ["ret5", "ret10", "ret20", "ret60", "vol20", "vol60", "volratio", "ma50",
            "ma100", "rsi", "atr_rel", "rangepos", "skew20", "ac1", "gk20", "dvol_z"]


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
    o, h, l = df["open"], df["high"], df["low"]
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    F["gk20"] = gk.rolling(20).mean()
    dv = (c * df["volume"]); F["dvol_z"] = (dv - dv.rolling(60).mean()) / dv.rolling(60).std()
    return F.shift(1)


def main():
    panel = pd.read_parquet(PANEL)
    syms = [s for s, g in panel.groupby("symbol") if len(g) >= MIN_HIST]
    import os
    if os.environ.get("MAX_ASSETS"):
        syms = syms[: int(os.environ["MAX_ASSETS"])]
    print(f"Meta-labeling rework: {len(syms)} assets", flush=True)

    blocks, closes = [], {}
    for sym in syms:
        df = panel[panel.symbol == sym].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        closes[sym] = df["close"]
        F = features(df)
        c = df["close"]
        ret20 = np.log(c / c.shift(20))                 # through t-? -> shift below for causality
        side = -np.sign(ret20.shift(1))                 # PRIMARY: reversal, causal (info <= t-1)
        fwd = np.log(c.shift(-H) / c)                    # bet P&L direction over next H days
        win = ((side * fwd) > 0).astype(int)            # meta target: did the bet win?
        blk = F[FEATCOLS].copy()
        blk["__win"] = win.values; blk["__side"] = side.values
        blk["__fwd"] = fwd.values; blk["__sym"] = sym
        blk["__date"] = df.index
        blk = blk[(side != 0)].dropna(subset=["__win"])
        blocks.append(blk)
    P = pd.concat(blocks, ignore_index=True).sort_values("__date").reset_index(drop=True)
    print(f"Meta events: {len(P)}  base win-rate={P['__win'].mean():.3f}", flush=True)

    # global time-purged CV -> OOS p(win)
    dt = pd.to_datetime(P["__date"]).values
    edges = pd.to_datetime(np.quantile(dt.astype("int64"), np.linspace(0, 1, N_FOLDS + 1)))
    emb = pd.Timedelta(days=EMBARGO)
    oos = np.full(len(P), np.nan)
    for k in range(N_FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test = (dt >= lo) & (dt < hi) if k < N_FOLDS - 1 else (dt >= lo) & (dt <= hi)
        # bet opened at date d closes ~d+H; purge train rows within [lo-emb, hi+emb]
        overlap = (dt >= (lo - emb).to_datetime64()) & (dt <= (hi + emb).to_datetime64())
        train = (~test) & (~overlap)
        if train.sum() < 1000:
            continue
        m = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, l2_leaf_reg=6,
                               loss_function="Logloss", verbose=0, random_seed=0,
                               subsample=0.8, rsm=0.8)
        m.fit(P.loc[train, FEATCOLS].fillna(0.0), P.loc[train, "__win"])
        oos[test.nonzero()[0]] = m.predict_proba(P.loc[test, FEATCOLS].fillna(0.0))[:, 1]
    P["__pwin"] = oos
    m = P["__pwin"].notna()
    auc = roc_auc_score(P.loc[m, "__win"], P.loc[m, "__pwin"])
    print(f"META-MODEL OOS AUC (predict bet wins) = {auc:.4f}  (n={int(m.sum())})", flush=True)

    # build books: primary-only vs meta-sized
    closes_df = pd.DataFrame(closes)
    rets = np.log(closes_df / closes_df.shift(1))
    invvol = 1.0 / rets.rolling(30).std().shift(1).clip(lower=1e-4)
    all_dates = sorted(pd.to_datetime(P["__date"]).unique())

    def book(weight_col):
        pos = pd.DataFrame(0.0, index=pd.DatetimeIndex(all_dates), columns=syms)
        for _, row in P[m].iterrows():
            d = pd.Timestamp(row["__date"]); s = syms.index(row["__sym"])
            w = row["__side"] * (row[weight_col] if weight_col else 1.0)
            pos.iat[pos.index.get_loc(d), s] = w
        # forward-fill each bet for H days (hold), inv-vol scale, gross-normalise daily
        pos = pos.replace(0.0, np.nan).ffill(limit=H)
        pos = pos.reindex(rets.index).fillna(0.0) * invvol.reindex(rets.index).fillna(0.0)
        g = pos.abs().sum(axis=1).replace(0, np.nan)
        posn = pos.div(g, axis=0).fillna(0.0)
        pnl_gross = (posn.shift(1) * rets).sum(axis=1)
        turn = (posn - posn.shift(1)).abs().sum(axis=1)
        pnl = (pnl_gross - turn * COST_BPS / 1e4).dropna()
        bv = pnl.std() * np.sqrt(DAYS)
        return pnl * (TARGET_VOL / bv) if bv > 0 else pnl

    for nm, wc, gate in (("primary reversal only", None, None),
                          ("meta-sized (w=p_win)", "__pwin", None)):
        # optional gate p_win>0.5
        Psave = P.copy()
        if nm.startswith("meta"):
            P.loc[P["__pwin"] < 0.50, "__pwin"] = 0.0
        pnl = book(wc)
        P[:] = Psave
        sh = pnl.mean() / pnl.std() * np.sqrt(DAYS); vol = pnl.std() * np.sqrt(DAYS)
        eq = (1 + pnl).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(pnl)) - 1
        dd = (eq / eq.cummax() - 1).min(); srd = pnl.mean() / pnl.std()
        yr = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in pnl.groupby(pnl.index.year))
        print(f"\n[{nm}] Sharpe={sh:.2f} CAGR={cagr*100:+.0f}% vol={vol:.0%} MaxDD={dd*100:.0f}% "
              f"DSR(N2000)={deflated_sharpe(srd,2000,len(pnl)):.3f}\n   {yr}")


if __name__ == "__main__":
    main()
