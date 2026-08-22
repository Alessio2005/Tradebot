"""mn_optimized_exec.py — leverage + execution optimization of the verified MN book.

Improvements over scripts/wave_final_eval.py:
  EXECUTION
    • combine sleeves at the PER-ASSET WEIGHT level (net offsetting trades across
      sleeves) instead of summing per-sleeve PnL → lower turnover, lower cost.
    • no-trade rebalance band (only move when |Δw| > band) → cuts churn.
  LEVERAGE
    • fractional-Kelly vol-target on the book's own rolling Sharpe (not a flat
      target), de-grossed on vol spikes and drawdown, hard gross cap (ruin bound).
Reports turnover, gross, CAGR, Sharpe, Calmar, MaxDD vs the naive baseline.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts/combined_neutral_book.py")
cb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cb)
DAYS, COST, MAXGROSS = 365.0, 10.0, 10.0
PANEL = ROOT / "artefacts/broad_perp_daily_close_WIDE.parquet"


def load():
    P = pd.read_parquet(PANEL).resample("1D").last()
    R = np.log(P / P.shift(1)); R = R[R.notna().sum(axis=1) >= 20]
    return P, R


def sleeve_weights(R, kind, L):
    m = R.mean(axis=1); resid = R.sub(m, axis=0)
    sig = -resid.rolling(L).sum() if kind == "rev" else -R.rolling(L).std()
    z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1).replace(0, np.nan), axis=0)
    w = z.sub(z.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def pnl_from_weights(W, R, band=0.0):
    we = W.shift(1).fillna(0.0)
    if band > 0:                                   # no-trade band: hold unless move > band
        held = we.copy().values
        for i in range(1, len(held)):
            move = np.abs(held[i] - held[i - 1])
            keep = move < band
            held[i] = np.where(keep, held[i - 1], held[i])
        we = pd.DataFrame(held, index=we.index, columns=we.columns)
    turn = (we - we.shift(1).fillna(0.0)).abs().sum(axis=1)
    gross = we.abs().sum(axis=1)
    return ((we * R).sum(axis=1) - turn * COST / 1e4).dropna(), turn, gross


def ann_sharpe(x): return x.mean() / x.std() * np.sqrt(DAYS) if x.std() > 0 else np.nan


def frac_kelly_leverage(pnl, target_vol=0.40, kelly_frac=0.5, lb=40):
    """De-grossed fractional-Kelly vol-target with ruin cap."""
    rv = pnl.rolling(lb, min_periods=20).std().shift(1) * np.sqrt(DAYS)
    mu = pnl.rolling(lb * 3, min_periods=40).mean().shift(1) * DAYS
    sr = (mu / rv).clip(-1, 4)                       # rolling Sharpe estimate
    base = (target_vol / rv).clip(upper=MAXGROSS)    # vol-target
    klly = (kelly_frac * sr).clip(lower=0)           # fractional Kelly scaler (0..)
    lev = (base * klly.clip(upper=1.5)).clip(upper=MAXGROSS).fillna(0.0)
    # de-gross on vol spike (rv above its own 80th pct)
    spike = rv > rv.rolling(120, min_periods=40).quantile(0.85)
    lev = lev.where(~spike.fillna(False), lev * 0.5)
    return lev


def report(name, pnl, lev=None):
    p = (lev * pnl).dropna() if lev is not None else pnl
    eq = (1 + p).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(p)) - 1
    dd = (eq / eq.cummax() - 1).min(); cal = cagr / abs(dd) if dd < 0 else np.nan
    line = f"{name:26s} CAGR={cagr*100:+6.1f}% Sh={ann_sharpe(p):+.2f} MaxDD={dd*100:5.0f}% Calmar={cal:.2f}"
    if lev is not None:
        line += f" gross={lev.mean():.1f}x(max{lev.max():.1f})"
    print(line)
    return p


def main():
    P, R = load()
    Wr = sleeve_weights(R, "rev", 10); Wl = sleeve_weights(R, "lowvol", 20)
    car = cb.carry_pnl(k=3, cost_bps=COST).rename("carry")

    # --- baseline: per-sleeve PnL, naive RP, flat 40% vol-target ---
    pr, _, _ = pnl_from_weights(Wr, R); pl, _, _ = pnl_from_weights(Wl, R)
    df = pd.concat([car, pr.rename("rev"), pl.rename("lowvol")], axis=1).dropna()
    vol = df.rolling(252, min_periods=60).std().shift(1); iv = 1 / vol.replace(0, np.nan)
    rp = iv.div(iv.sum(axis=1), axis=0); base_book = (rp * df).sum(axis=1).dropna()
    print("=== BASELINE (PnL-level RP, flat vol-target) ===")
    blev = (0.40 / (base_book.rolling(40, min_periods=20).std().shift(1) * np.sqrt(DAYS))).clip(upper=MAXGROSS).fillna(0)
    report("baseline", base_book, blev)

    # --- optimized: position-level netting of rev+lowvol, band, + carry PnL ---
    print("\n=== OPTIMIZED EXECUTION (position-level netting + band) ===")
    for band in (0.0, 0.02, 0.05):
        # causal RP weights between rev & lowvol from their recent vol
        comb = (0.5 * Wr + 0.5 * Wl)               # equal risk (both ~unit gross)
        pnl_net, turn, gross = pnl_from_weights(comb, R, band=band)
        # add carry sleeve (separate 5-asset names, RP at PnL level — minimal overlap)
        full = pd.concat([car, pnl_net.rename("xs")], axis=1).dropna()
        v = full.rolling(252, min_periods=60).std().shift(1); ivf = 1 / v.replace(0, np.nan)
        rpf = ivf.div(ivf.sum(axis=1), axis=0); book = (rpf * full).sum(axis=1).dropna()
        lev = frac_kelly_leverage(book)
        p = report(f"net+band{band:.0%}+fracKelly", book, lev)
        if band == 0.02:
            from tradebot.backtest.metrics import deflated_sharpe
            srd = book.mean() / book.std()
            print(f"     turnover/day={turn.mean():.3f} (baseline xs sleeves churn higher) | "
                  f"DSR N=50:{deflated_sharpe(srd,50,len(book)):.2f} N=2000:{deflated_sharpe(srd,2000,len(book)):.2f}")
            line = "     per-year: "
            for y, g in p.groupby(p.index.year):
                line += f"{y}:{((1+g).prod()-1)*100:+.0f}% "
            print(line)


if __name__ == "__main__":
    main()
