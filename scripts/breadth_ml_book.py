"""breadth_ml_book.py — 70x2 (99x2) directional ML breadth book (Wave 14).

Mandate: scale the directional ML book from 5 assets to ~70+ assets x 2 sides
(LONG/SHORT models), rework per Lopez de Prado / Bailey, and measure whether
breadth can lift net Sharpe toward ~3 (the level 100% net CAGR requires).

Method (AFML, leakage-free):
  - Per asset: causal features (data <= t-1), triple-barrier L/S labels,
    purged + embargoed blocked CV -> OOS probabilities. 99 x 2 = 198 models.
  - Decisive measurement: per-asset & POOLED out-of-sample AUC; directional
    rank-IC of (p_long - p_short) vs the realised forward H-day return.
  - Portfolio: inverse-vol risk-parity over assets, vol-targeted, honest 10bps,
    held to the barrier horizon. PnL = pos.shift(1)*ret - turnover*cost (no
    signed-return construction artifact).
  - Fundamental Law of Active Management (Grinold): IR = IC * sqrt(breadth).
    Project the achievable Sharpe from the measured IC and effective breadth.

Run:  python scripts/breadth_ml_book.py
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
from tradebot.labeling.triple_barrier import TripleBarrierLabeler
from tradebot.backtest.metrics import deflated_sharpe

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
DAYS = 365.0
PT, SL, HORIZON = 2.0, 2.0, 15
N_FOLDS, EMBARGO = 6, HORIZON + 3
COST_BPS = 10.0
THR = 0.55
MIN_HIST = 900           # require >=~2.5yr per asset
TARGET_VOL = 0.40


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
    # Garman-Klass vol (uses OHLC) and dollar-volume z
    o, h, l = df["open"], df["high"], df["low"]
    gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
    F["gk20"] = gk.rolling(20).mean()
    dv = (c * df["volume"]); F["dvol_z"] = (dv - dv.rolling(60).mean()) / dv.rolling(60).std()
    return F.shift(1)   # causal: event at t uses data <= t-1


def label_side(df, side):
    a = atr(df).to_numpy(np.float64); n = len(df)
    ev = np.arange(60, n - HORIZON - 1, dtype=np.int64)
    hz = np.full(len(ev), HORIZON, dtype=np.int64)
    lab = TripleBarrierLabeler(pt_width=PT, sl_width=SL, execution_delay_bars=1)
    out = lab.label(df, ev, hz, a, side=side)
    y = (out["barrier_label"].to_numpy() == 1).astype(int)
    return pd.Series(y, index=out.index), pd.Series(out["t1_idx"].to_numpy(), index=out.index)


def purged_oos(X, y, t1_pos, idx_pos):
    n = len(X); oos = pd.Series(np.nan, index=X.index); bs = n // N_FOLDS
    for k in range(N_FOLDS):
        te0, te1 = k * bs, (n if k == N_FOLDS - 1 else (k + 1) * bs)
        test = np.arange(te0, te1)
        tr = []
        lo_pos, hi_pos = idx_pos[te0], idx_pos[te1 - 1]
        for i in range(n):
            if te0 <= i < te1:
                continue
            if t1_pos[i] >= lo_pos - EMBARGO and idx_pos[i] <= hi_pos + EMBARGO:
                continue
            tr.append(i)
        tr = np.array(tr)
        if len(tr) < 150 or y.iloc[tr].nunique() < 2:
            continue
        m = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03, l2_leaf_reg=6,
                               loss_function="Logloss", verbose=0, random_seed=0,
                               subsample=0.8, rsm=0.8)
        m.fit(X.iloc[tr].fillna(0.0), y.iloc[tr])
        oos.iloc[test] = m.predict_proba(X.iloc[test].fillna(0.0))[:, 1]
    return oos


def main():
    panel = pd.read_parquet(PANEL)
    syms = [s for s, g in panel.groupby("symbol") if len(g) >= MIN_HIST]
    import os
    if os.environ.get("MAX_ASSETS"):
        syms = syms[: int(os.environ["MAX_ASSETS"])]
    print(f"Assets with >={MIN_HIST}d history: {len(syms)}  (models = {len(syms)*2})", flush=True)

    auc_rows = []           # (sym, side, auc, n)
    pnl_cols = {}           # sym -> daily pnl series (vol-normed, pre-book-target)
    pos_cols = {}           # sym -> daily net position (for turnover/IC)
    ret_cols = {}           # sym -> daily log return
    score_cols = {}         # sym -> daily directional score (p_long - p_short)
    short_active = {}

    for si, sym in enumerate(syms):
        df = panel[panel.symbol == sym].set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        F = features(df); r = np.log(df["close"] / df["close"].shift(1))
        pos_of = {ts: i for i, ts in enumerate(df.index)}
        sigs = {}
        for side, nm in ((1, "L"), (-1, "S")):
            y, t1 = label_side(df, side)
            ev_idx = y.index
            X = F.loc[ev_idx]
            idx_pos = np.array([pos_of[ts] for ts in ev_idx]); t1_pos = t1.to_numpy()
            oos = purged_oos(X, y, t1_pos, idx_pos)
            m = oos.notna() & y.notna()
            if m.sum() > 50 and y[m].nunique() == 2:
                auc_rows.append((sym, nm, roc_auc_score(y[m], oos[m]), int(m.sum())))
            sigs[nm] = oos.reindex(df.index)
        lp, sp = sigs["L"].fillna(0.0), sigs["S"].fillna(0.0)
        score = (sigs["L"] - sigs["S"])               # directional score on event days
        raw = (lp > THR).astype(float) - (sp > THR).astype(float)
        pos = raw.replace(0, np.nan).ffill(limit=HORIZON).fillna(0.0)
        # inverse-vol risk scaling so high-vol alts don't dominate the book
        rv = r.rolling(30).std().shift(1).clip(lower=1e-4)
        vpos = (pos / rv) * (TARGET_VOL / np.sqrt(DAYS))   # ~per-asset vol-normalised unit
        pnl = (vpos.shift(1) * r) - (vpos - vpos.shift(1)).abs() * COST_BPS / 1e4
        pnl_cols[sym] = pnl.dropna(); pos_cols[sym] = pos; ret_cols[sym] = r
        score_cols[sym] = score
        short_active[sym] = float((pos < 0).mean())
        if si % 15 == 0:
            print(f"  {si}/{len(syms)} done ({sym})", flush=True)

    # ---- OOS AUC summary (the decisive base-rate) ----
    auc = pd.DataFrame(auc_rows, columns=["sym", "side", "auc", "n"])
    pooled_auc = (auc["auc"] * auc["n"]).sum() / auc["n"].sum()
    print("\n===== OUT-OF-SAMPLE DISCRIMINATIVE SKILL (per-asset x side) =====")
    print(f"  models scored: {len(auc)}   median AUC={auc.auc.median():.4f}  "
          f"mean={auc.auc.mean():.4f}  n-weighted pooled={pooled_auc:.4f}")
    print(f"  AUC>0.55: {(auc.auc>0.55).mean()*100:.0f}%   AUC>0.52: {(auc.auc>0.52).mean()*100:.0f}%   "
          f"AUC<0.50: {(auc.auc<0.50).mean()*100:.0f}%")
    print(f"  L median={auc[auc.side=='L'].auc.median():.4f}  S median={auc[auc.side=='S'].auc.median():.4f}")

    # ---- directional rank-IC (pooled, forward H-day return) ----
    ic_vals = []
    for sym in syms:
        c = panel[panel.symbol == sym].set_index("date")["close"].sort_index()
        fwd = np.log(c.shift(-HORIZON) / c)
        s = score_cols[sym].reindex(c.index)
        d = pd.concat([s, fwd], axis=1).dropna()
        if len(d) > 100:
            ic_vals.append(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))
    mean_ic = float(np.nanmean(ic_vals))
    print(f"\n  directional rank-IC (score vs fwd {HORIZON}d ret): mean={mean_ic:.4f}  "
          f"median={np.nanmedian(ic_vals):.4f}  (n_assets={len(ic_vals)})")

    # ---- breadth portfolio ----
    B = pd.DataFrame(pnl_cols).sort_index()
    port = B.sum(axis=1) / np.sqrt((B != 0).sum(axis=1).clip(lower=1))  # risk-parity aggregate
    port = port.dropna()
    # re-scale whole book to TARGET_VOL
    bv = port.std() * np.sqrt(DAYS)
    port = port * (TARGET_VOL / bv) if bv > 0 else port
    sh = port.mean() / port.std() * np.sqrt(DAYS); vol = port.std() * np.sqrt(DAYS)
    eq = (1 + port).cumprod()
    cagr = eq.iloc[-1] ** (DAYS / len(port)) - 1
    dd = (eq / eq.cummax() - 1).min()
    print("\n===== BREADTH BOOK (inverse-vol risk-parity, vol-targeted, 10bps) =====")
    print(f"  Sharpe={sh:.2f}  CAGR={cagr*100:+.0f}%  vol={vol:.0%}  MaxDD={dd*100:.0f}%")
    line = "  per-year: "
    for yv, g in port.groupby(port.index.year):
        line += f"{yv}:{((1+g).prod()-1)*100:+.0f}% "
    print(line)
    avg_short = np.mean(list(short_active.values()))
    print(f"  avg short-active fraction: {avg_short*100:.0f}%")
    srd = port.mean() / port.std()
    print(f"  DSR(corrected): N=200:{deflated_sharpe(srd,200,len(port)):.3f}  "
          f"N=2000:{deflated_sharpe(srd,2000,len(port)):.3f}")

    # ---- Fundamental Law projection ----
    print("\n===== FUNDAMENTAL LAW OF ACTIVE MANAGEMENT (Grinold) =====")
    print("  IR = IC * sqrt(breadth).  Crypto cross-section is highly correlated, so")
    print("  effective independent breadth << nominal (assets x rebalances).")
    n_assets = len(syms)
    rebals_yr = DAYS / HORIZON       # independent time-bets per asset per year
    for eff_assets in (n_assets, 8, 4):
        breadth = eff_assets * rebals_yr
        ir = abs(mean_ic) * np.sqrt(breadth)
        tag = "nominal" if eff_assets == n_assets else f"eff~{eff_assets} indep"
        print(f"  breadth={breadth:6.0f} ({tag:14s}) -> projected IR={ir:.2f}")
    need_ic_3 = 3.0 / np.sqrt(8 * rebals_yr)
    print(f"  -> to reach IR=3 at eff-breadth(8 assets): need IC={need_ic_3:.3f} "
          f"(measured |IC|={abs(mean_ic):.3f})")


if __name__ == "__main__":
    main()
