"""Stage C screen (DISCOVERY ONLY, <= 2023-12-31): which simple predictors carry forward-return information?

Panel Fama-MacBeth style XS rank-IC and pooled time-series IC, vol-normalised forward returns,
Newey-West t (lags = h+4). Strict bar: |t| >= 3 (Harvey-Liu-Zhu multiple-testing hurdle).
Each (predictor) is booked as ONE trial (horizons are read together; the screen reports both).
"""
import sys, json; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from scipy import stats
from lib.data import load_panel
from lib.trials import log_trial

p = load_panel()
A, B = pd.Timestamp("2021-10-20", tz="UTC"), pd.Timestamp("2023-12-31", tz="UTC")
logp = np.log(p.close); r = p.ret
sd = r.ewm(span=30, min_periods=20).std().shift(0)           # causal vol known at t
ew = r.mean(axis=1)

def mom(k): return (logp - logp.shift(k)) / (sd * np.sqrt(k))
preds = {f"mom_{k}": mom(k) for k in [1, 3, 7, 14, 28, 56, 112]}
f3 = p.funding.rolling(3).sum(); preds["fund_z"] = (f3 - f3.rolling(90).mean()) / f3.rolling(90).std()
preds["fund_lvl7"] = p.funding.rolling(7).mean()
oi = np.log(p.oi.where(p.oi > 0))
preds["oi_chg1"] = (oi - oi.shift(1)).shift(1); preds["oi_chg5"] = (oi - oi.shift(5)).shift(1)   # +1 bar lag as in spec
tv = p.turnover; preds["vol_surge"] = np.log(tv / tv.rolling(30).mean())
preds["rv_ratio"] = np.log(r.rolling(20).std() / r.rolling(120).std())
beta = r.rolling(60).cov(ew).div(ew.rolling(60).var(), axis=0)
res = r - beta.mul(ew, axis=0); preds["resid_z20"] = res.rolling(20).sum() / (res.rolling(60).std() * np.sqrt(20))
preds["btc_lead"] = pd.DataFrame({s: (r["BTCUSDT"] / sd["BTCUSDT"]) for s in r.columns}).where(
    pd.DataFrame({s: s != "BTCUSDT" for s in r.columns}, index=r.index))
hi = p.high.rolling(20).max().shift(1); lo = p.low.rolling(20).min().shift(1)
preds["dist_hi20"] = np.log(p.close / hi) / sd; preds["dist_lo20"] = np.log(p.close / lo) / sd

def fwd(h): return (logp.shift(-h) - logp) / (sd * np.sqrt(h))   # vol-normalised forward return, h days ahead
def nw_t(x, lags):
    x = pd.Series(x).dropna().to_numpy(); n = len(x); m = x.mean(); e = x - m
    s = (e @ e) / n
    for l in range(1, lags + 1): s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return m / np.sqrt(s / n), m, n

rows = []
for name, X in preds.items():
    for h in [1, 5]:
        Y = fwd(h); sl = (X.index >= A) & (X.index <= B)
        Xs, Ys = X[sl], Y[sl]
        # cross-sectional rank IC per date
        ics = []
        for t in Xs.index:
            x, y = Xs.loc[t], Ys.loc[t]; ok = x.notna() & y.notna()
            if ok.sum() >= 4: ics.append((t, stats.spearmanr(x[ok], y[ok]).statistic))
        ics = pd.Series(dict(ics)); t_xs, m_xs, _ = nw_t(ics, h + 4)
        # pooled time-series IC: per-date mean over coins of standardized product (sign-bet)
        z = Xs.sub(Xs.mean(), axis=1).div(Xs.std(), axis=1)         # per-coin standardise over window
        ts = (np.sign(z) * Ys).mean(axis=1)                          # mean signed forward vol-norm return
        t_ts, m_ts, n = nw_t(ts, h + 4)
        rows.append({"pred": name, "h": h, "XS_IC": m_xs, "XS_t": t_xs, "TS_meanSignedFwd": m_ts, "TS_t": t_ts, "days": n})
tab = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(tab.round(3).to_string(index=False))
tab.to_csv("research/results/stage_c_screen_discovery.csv", index=False)
print("\n|t|>=3:\n", tab[(tab.XS_t.abs() >= 3) | (tab.TS_t.abs() >= 3)].round(3).to_string(index=False))
for name in preds:
    log_trial(f"SCREEN_{name}", "screen", "discovery-only predictor screen, h in {1,5}")
