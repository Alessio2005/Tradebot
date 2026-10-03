"""Stage A: where does the existing TSMOM P&L come from? (diagnostics only, no new trials)"""
import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.stats import perf, sr, by_year

p = load_panel(); logp = np.log(p.close)
def sig(Ls, lf=False):
    s = sum(np.sign(logp - logp.shift(L)) for L in Ls) / len(Ls)
    s = s.where(logp.shift(max(Ls)).notna()); return s.clip(lower=0) if lf else s
def wts(s, tgt=0.40):
    live = s.notna() & p.sigma_ann.notna(); n = live.sum(axis=1).replace(0, np.nan)
    return cap_gross((s * tgt / p.sigma_ann).where(live).div(n, axis=0).fillna(0.0))
A, B = "2022-01-01", "2026-06-23"
w1 = wts(sig([7,14,28,56,112])); w2 = wts(sig([7,14,28,56,112], True))
for nm, w in [("T1", w1), ("T2", w2)]:
    held = w.shift(1).fillna(0)
    contrib = (held * p.ret.fillna(0)).loc[A:B]
    print(nm, "gross contribution per coin, ann:", (contrib.mean() * 365).round(3).to_dict())
    print(nm, "mean weight per coin:", w.loc[A:B].mean().round(3).to_dict())
    print(nm, "share of days with a position:", float((w.abs().sum(axis=1) > 0).loc[A:B].mean()))
# is the trend book just long beta? correlation with equal-weight index
ew = p.ret.mean(axis=1)
for nm, w in [("T1", w1), ("T2", w2)]:
    r = run_book(w, p)["net"].loc[A:B]
    print(nm, "corr with EW index", round(float(np.corrcoef(r, ew.loc[A:B])[0,1]), 2),
          "| net SR", round(sr(r), 2), "| lag2 SR", round(sr(run_book(w, p, lag=2)["net"], A, B), 2),
          "| lag0 (look-ahead, upper bound) SR", round(sr(run_book(w, p, lag=0)["net"], A, B), 2))
# single-lookback decomposition at lag 1 and lag 2: which lookbacks die with a one-day delay?
print("\nlookback  SR(lag1)  SR(lag2)  turnover/day")
for L in [7, 14, 28, 56, 112]:
    w = wts(sig([L]))
    b1, b2 = run_book(w, p), run_book(w, p, lag=2)
    print(f"{L:>6}   {sr(b1['net'], A, B):+.2f}     {sr(b2['net'], A, B):+.2f}      {b1['turnover'].loc[A:B].mean():.3f}")
# year x coin gross P&L for T1
held = w1.shift(1).fillna(0); c = (held * p.ret.fillna(0))
print("\nT1 gross P&L by year x coin"); print(c.groupby(c.index.year).sum().round(3))
# equal weight index trend: simple 'market' timing check
mkt = p.close.pct_change().mean(axis=1); mi = (1 + mkt.fillna(0)).cumprod()
print("\nEW index buy&hold SR", round(sr(mkt, A, B), 2))
