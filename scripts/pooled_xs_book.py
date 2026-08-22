"""pooled_xs_book.py — pooled cross-sectional ML book (Wave 14 rework).

The AFML-correct breadth rework: instead of 99 thin per-asset models, train ONE
LONG and ONE SHORT CatBoost on ALL assets' events pooled (~150k events), with a
GLOBAL time-purged + embargoed CV so no test-period information leaks. Then form
the canonical dollar-neutral cross-sectional decile book (long top / short bottom
by score) — the natural home for Fundamental-Law breadth — and test BOTH the score
direction and its reverse (the 5-asset smoke showed directional IC < 0 = reversal).

Run:  python scripts/pooled_xs_book.py
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
from tradebot.labeling.triple_barrier import TripleBarrierLabeler

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
DAYS = 365.0
PT, SL, HORIZON = 2.0, 2.0, 15
N_FOLDS, EMBARGO = 6, HORIZON + 3
COST_BPS = 10.0
MIN_HIST = 900
TARGET_VOL = 0.40
DECILE = 0.20      # long top 20% / short bottom 20% each day


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


FEATCOLS = ["ret5", "ret10", "ret20", "ret60", "vol20", "vol60", "volratio", "ma50",
            "ma100", "rsi", "atr_rel", "rangepos", "skew20", "ac1", "gk20", "dvol_z"]


def build_pool(syms, panel):
    """Stack per-asset events into one pooled labeled frame for each side."""
    rows = {1: [], -1: []}
    daily_close, daily_score_index = {}, {}
    for sym in syms:
        df = panel[panel.symbol == sym].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        daily_close[sym] = df["close"]
        F = features(df)
        a = atr(df).to_numpy(np.float64); n = len(df)
        ev = np.arange(60, n - HORIZON - 1, dtype=np.int64)
        hz = np.full(len(ev), HORIZON, dtype=np.int64)
        for side in (1, -1):
            lab = TripleBarrierLabeler(pt_width=PT, sl_width=SL, execution_delay_bars=1)
            out = lab.label(df, ev, hz, a, side=side)
            y = (out["barrier_label"].to_numpy() == 1).astype(int)
            ev_ts = out.index
            t1_ts = df.index[out["t1_idx"].to_numpy()]
            X = F.loc[ev_ts, FEATCOLS]
            blk = X.copy()
            blk["__y"] = y; blk["__sym"] = sym
            blk["__ev"] = ev_ts; blk["__t1"] = t1_ts
            rows[side].append(blk)
    pool = {s: pd.concat(rows[s], ignore_index=True) for s in (1, -1)}
    return pool, daily_close


def pooled_oos(pool_side):
    """Global time-purged CV over the pooled events -> OOS prob per row."""
    P = pool_side.sort_values("__ev").reset_index(drop=True)
    ev = P["__ev"].values; t1 = P["__t1"].values
    edges = pd.to_datetime(np.quantile(ev.astype("int64"), np.linspace(0, 1, N_FOLDS + 1)))
    emb = pd.Timedelta(days=EMBARGO)
    oos = np.full(len(P), np.nan)
    for k in range(N_FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test = (ev >= lo) & (ev < hi) if k < N_FOLDS - 1 else (ev >= lo) & (ev <= hi)
        # purge: drop train rows whose [ev,t1] overlaps [lo,hi] +/- embargo
        overlap = (t1 >= (lo - emb)) & (ev <= (hi + emb))
        train = (~test) & (~overlap)
        if train.sum() < 500 or P.loc[test, "__y"].nunique() < 1:
            continue
        m = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, l2_leaf_reg=6,
                               loss_function="Logloss", verbose=0, random_seed=0,
                               subsample=0.8, rsm=0.8)
        m.fit(P.loc[train, FEATCOLS].fillna(0.0), P.loc[train, "__y"])
        oos[test.nonzero()[0]] = m.predict_proba(P.loc[test, FEATCOLS].fillna(0.0))[:, 1]
    P["__oos"] = oos
    return P


def xs_book(score_panel, daily_close, syms, reverse=False):
    """Dollar-neutral cross-sectional decile book from a daily score panel."""
    sc = score_panel.copy()
    if reverse:
        sc = -sc
    rets = pd.DataFrame({s: np.log(daily_close[s] / daily_close[s].shift(1)) for s in syms})
    rets = rets.reindex(sc.index)
    invvol = 1.0 / rets.rolling(30).std().shift(1).clip(lower=1e-4)
    pos = pd.DataFrame(0.0, index=sc.index, columns=sc.columns)
    for dt in sc.index:
        row = sc.loc[dt].dropna()
        if len(row) < 10:
            continue
        klo, khi = row.quantile(DECILE), row.quantile(1 - DECILE)
        longs = row[row >= khi].index; shorts = row[row <= klo].index
        w = pd.Series(0.0, index=sc.columns)
        if len(longs):
            w[longs] = invvol.loc[dt, longs].fillna(0.0)
        if len(shorts):
            w[shorts] = -invvol.loc[dt, shorts].fillna(0.0)
        g = w.abs().sum()
        if g > 0:
            pos.loc[dt] = w / g            # gross-normalised, ~dollar-neutral
    pnl_gross = (pos.shift(1) * rets).sum(axis=1)
    turn = (pos - pos.shift(1)).abs().sum(axis=1)
    pnl = (pnl_gross - turn * COST_BPS / 1e4).dropna()
    bv = pnl.std() * np.sqrt(DAYS)
    pnl = pnl * (TARGET_VOL / bv) if bv > 0 else pnl
    return pnl


def report(name, pnl):
    sh = pnl.mean() / pnl.std() * np.sqrt(DAYS); vol = pnl.std() * np.sqrt(DAYS)
    eq = (1 + pnl).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(pnl)) - 1
    dd = (eq / eq.cummax() - 1).min(); srd = pnl.mean() / pnl.std()
    line = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in pnl.groupby(pnl.index.year))
    print(f"\n[{name}] Sharpe={sh:.2f} CAGR={cagr*100:+.0f}% vol={vol:.0%} MaxDD={dd*100:.0f}% "
          f"DSR(N2000)={deflated_sharpe(srd,2000,len(pnl)):.3f}")
    print(f"   per-year: {line}")
    return sh


def main():
    panel = pd.read_parquet(PANEL)
    syms = [s for s, g in panel.groupby("symbol") if len(g) >= MIN_HIST]
    import os
    if os.environ.get("MAX_ASSETS"):
        syms = syms[: int(os.environ["MAX_ASSETS"])]
    print(f"Pooled cross-sectional rework: {len(syms)} assets", flush=True)

    pool, daily_close = build_pool(syms, panel)
    print(f"Pooled events: L={len(pool[1])}  S={len(pool[-1])}", flush=True)

    scores = {}
    for side in (1, -1):
        P = pooled_oos(pool[side])
        m = P["__oos"].notna() & P["__y"].notna()
        auc = roc_auc_score(P.loc[m, "__y"], P.loc[m, "__oos"])
        print(f"  pooled {'LONG' if side==1 else 'SHORT'} OOS AUC = {auc:.4f}  (n={int(m.sum())})", flush=True)
        # daily wide score panel per side (ffill within horizon)
        wide = P.pivot_table(index="__ev", columns="__sym", values="__oos", aggfunc="last")
        wide = wide.reindex(sorted(set().union(*[daily_close[s].index for s in syms]))).ffill(limit=HORIZON)
        scores[side] = wide.reindex(columns=syms)

    score = (scores[1] - scores[-1])     # directional cross-sectional score
    print("\n===== CROSS-SECTIONAL DECILE BOOKS (dollar-neutral, inv-vol, 10bps, 40% vol) =====")
    report("momentum (long top score)", xs_book(score, daily_close, syms, reverse=False))
    report("reversal (long bottom score)", xs_book(score, daily_close, syms, reverse=True))
    # also a pure realised-return reversal benchmark (no ML) for context
    rets = pd.DataFrame({s: np.log(daily_close[s] / daily_close[s].shift(1)) for s in syms})
    mom20 = rets.rolling(20).sum()
    report("raw 20d return reversal (no ML)", xs_book(-mom20, daily_close, syms, reverse=False))


if __name__ == "__main__":
    main()
