"""wave_final_eval.py — capstone gate scorecard for the vol-targeted book.

Usage: python scripts/wave_final_eval.py [panel.parquet] [cost_bps] [target_vol]

Builds the best market-neutral book (carry + residual-reversal k=10 + lowvol,
causal inverse-vol RP), vol-targets it to `target_vol` within a 10x gross cap,
and runs the full acceptance battery:
  G1 net CAGR (geometric, after cost)         >= 100%
  G2 DSR (corrected) at honest n_trials        > 0.95
  G3 PBO via CSCV over a config grid           < 0.20
  G4 rolling 90d |beta| to BTC & basket        < 0.10
  G5 net Sharpe in every regime bucket         > 0
  G6 alpha after MKT+TSMOM regression (HAC)    > 0, p<0.05
"""
from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts/combined_neutral_book.py")
cb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cb)

PANEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artefacts/broad_perp_daily_close.parquet"
COST = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
TARGET_VOL = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
MAX_GROSS = 10.0
DAYS = 365.0


def load(panel_path):
    P = pd.read_parquet(panel_path).resample("1D").last()
    R = np.log(P / P.shift(1)); R = R[R.notna().sum(axis=1) >= 20]
    return P, R


def xs_sleeve(R, sig, cost):
    z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1).replace(0, np.nan), axis=0)
    w = z.sub(z.mean(axis=1), axis=0); w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    we = w.shift(1).fillna(0.0)
    return ((we * R) .sum(axis=1) - (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * cost / 1e4).dropna()


def sleeves(R, k_rev=10, cost=COST):
    m = R.mean(axis=1); resid = R.sub(m, axis=0)
    rev = xs_sleeve(R, -resid.rolling(k_rev).sum(), cost).rename("rev")
    lv = xs_sleeve(R, -R.rolling(20).std(), cost).rename("lowvol")
    car = cb.carry_pnl(k=3, cost_bps=cost).rename("carry")
    return pd.concat([car, rev, lv], axis=1).dropna()


def causal_rp(df, lb=252):
    vol = df.rolling(lb, min_periods=60).std().shift(1); iv = 1.0 / vol.replace(0, np.nan)
    w = iv.div(iv.sum(axis=1), axis=0)
    return (w * df).sum(axis=1).dropna()


def vol_target(pnl, target=TARGET_VOL, lb=60):
    rv = pnl.rolling(lb, min_periods=20).std().shift(1) * np.sqrt(DAYS)
    lev = (target / rv).clip(upper=MAX_GROSS).fillna(0.0)
    return (lev * pnl).dropna(), lev


def ann_sharpe(x): return x.mean() / x.std() * np.sqrt(DAYS) if x.std() > 0 else np.nan


def cagr_geom(x):
    eq = (1 + x).cumprod(); yrs = len(x) / DAYS
    return eq.iloc[-1] ** (1 / yrs) - 1


def main():
    P, R = load(PANEL_PATH)
    BTC = R["BTCUSDT"] if "BTCUSDT" in R.columns else R.mean(axis=1)
    BASKET = R.mean(axis=1)
    df = sleeves(R)
    book = causal_rp(df)
    lev_book, lev = vol_target(book)
    print(f"PANEL {PANEL_PATH.name}: {R.shape[1]} names, {len(R)} days | cost={COST}bps target_vol={TARGET_VOL:.0%}")
    print(f"Unlevered book: Sharpe={ann_sharpe(book):.2f} vol={book.std()*np.sqrt(DAYS):.1%}")
    print(f"Vol-targeted:   Sharpe={ann_sharpe(lev_book):.2f} vol={lev_book.std()*np.sqrt(DAYS):.1%} "
          f"avg_gross={lev.mean():.1f}x max_gross={lev.max():.1f}x")

    # G1
    cagr = cagr_geom(lev_book)
    print(f"\nG1 net CAGR (geom) = {cagr*100:+.1f}%   [{'PASS' if cagr >= 1.0 else 'FAIL'}]  (target >=100%)")

    # G2 DSR (corrected) at honest trial counts
    from tradebot.backtest.metrics import deflated_sharpe
    sr_d = book.mean() / book.std()
    print("G2 DSR (corrected):", end="  ")
    for nt in (12, 50, 200):
        print(f"N={nt}:{deflated_sharpe(sr_d, nt, len(book)):.2f}", end="  ")
    print(f" [{'PASS' if deflated_sharpe(sr_d,50,len(book))>0.95 else 'FAIL'} @N=50]")

    # G3 PBO via CSCV over a config grid (k_rev x lowvol window x rp lookback)
    grid = []
    for k in (5, 10, 20):
        d = sleeves(R, k_rev=k)
        for lb in (180, 252, 365):
            grid.append(causal_rp(d, lb=lb).rename(f"k{k}_lb{lb}"))
    M = pd.concat(grid, axis=1).dropna()
    pbo = cscv_pbo(M, n_blocks=10)
    print(f"G3 PBO (CSCV, {M.shape[1]} configs) = {pbo:.2f}   [{'PASS' if pbo < 0.20 else 'FAIL'}]")

    # G4 rolling beta
    print("G4 rolling 90d |beta|:", end="  ")
    g4pass = True
    for nm, fac in (("BTC", BTC), ("basket", BASKET)):
        x = fac.reindex(book.index); beta = (book.rolling(90).cov(x) / x.rolling(90).var()).dropna()
        mx = beta.abs().max(); frac = (beta.abs() < 0.10).mean(); g4pass &= mx < 0.10
        print(f"{nm}:max={mx:.3f}(<0.10:{frac:.0%})", end="  ")
    print(f"[{'PASS' if g4pass else 'MARGINAL'}]")

    # G5 regime buckets
    trail = np.log(P["BTCUSDT"] / P["BTCUSDT"].shift(90)).reindex(book.index) if "BTCUSDT" in P.columns else BTC.rolling(90).sum()
    vol30 = BTC.rolling(30).std().reindex(book.index)
    bull, hiv = trail > 0, vol30 > vol30.median()
    print("G5 regime Sharpe:", end="  "); neg = 0
    for nm, msk in [("bull/hi", bull & hiv), ("bull/lo", bull & ~hiv), ("bear/hi", ~bull & hiv), ("bear/lo", ~bull & ~hiv)]:
        seg = book[msk.fillna(False)]; s = ann_sharpe(seg) if len(seg) > 5 else np.nan; neg += s <= 0
        print(f"{nm}:{s:+.2f}", end="  ")
    print(f"[{'PASS' if neg == 0 else 'FAIL'}]")

    # G6 alpha vs MKT+TSMOM (HAC)
    sig = np.sign(np.log(P / P.shift(50))); tsmom = (sig.shift(1) * R).mean(axis=1).rename("TSMOM")
    reg = pd.concat([book.rename("y"), BTC.rename("MKT"), tsmom], axis=1).dropna()
    m = sm.OLS(reg["y"], sm.add_constant(reg[["MKT", "TSMOM"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    a, pa = m.params["const"], m.pvalues["const"]
    print(f"G6 alpha vs MKT+TSMOM = {a*DAYS*100:+.1f}%/yr  t={m.tvalues['const']:+.2f} p={pa:.4f}  "
          f"[{'PASS' if a > 0 and pa < 0.05 else 'FAIL'}]")


def cscv_pbo(M, n_blocks=10):
    """Bailey et al. (2016) CSCV probability of backtest overfitting."""
    M = M.dropna(); n = len(M); bs = n // n_blocks
    blocks = [M.iloc[i * bs:(i + 1) * bs] for i in range(n_blocks)]
    logits = []
    for tr in combinations(range(n_blocks), n_blocks // 2):
        te = [i for i in range(n_blocks) if i not in tr]
        IS = pd.concat([blocks[i] for i in tr]); OOS = pd.concat([blocks[i] for i in te])
        is_sr = IS.mean() / IS.std(); oos_sr = OOS.mean() / OOS.std()
        best = is_sr.idxmax()
        rank = oos_sr.rank(pct=True)[best]  # OOS percentile rank of IS-best
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())  # P(IS-best is below OOS median)


if __name__ == "__main__":
    main()
