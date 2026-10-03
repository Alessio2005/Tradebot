"""Stage H: kitchen-sink linear model (1 trial, fixed hyperparameters, no search).
Pooled panel ridge (alpha=1000) of the 5-day vol-normalised forward return on the 15 screen predictors, refit every
quarter on data older than (t - 6 days); OOS predictions from 2022-01-01. Traded as a TS book (pred * 0.40/sigma/N) and as a
dollar-neutral XS book. pv20, 6.5 bps, funding. Both windows reported."""
import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.sleeves import ts_weights, xs_weights, pv_target, mom_z, fund_z
from lib.stats import sr, perf, by_year
from lib.trials import log_trial, M

p = load_panel(); r = p.ret; logp = np.log(p.close)
sd = r.ewm(span=30, min_periods=20).std(); ew = r.mean(axis=1)
F = {f"mom_{k}": mom_z(p, k) for k in [1, 3, 7, 14, 28, 56, 112]}
F["fund_z"] = fund_z(p); F["fund_lvl7"] = p.funding.rolling(7).mean()
oi = np.log(p.oi.where(p.oi > 0)); F["oi_chg1"] = (oi - oi.shift(1)).shift(1); F["oi_chg5"] = (oi - oi.shift(5)).shift(1)
F["vol_surge"] = np.log(p.turnover / p.turnover.rolling(30).mean()); F["rv_ratio"] = np.log(r.rolling(20).std() / r.rolling(120).std())
beta = r.rolling(60).cov(ew).div(ew.rolling(60).var(), axis=0); res = r - beta.mul(ew, axis=0)
F["resid_z20"] = res.rolling(20).sum() / (res.rolling(60).std() * np.sqrt(20))
hi = p.high.rolling(20).max().shift(1); lo = p.low.rolling(20).min().shift(1)
F["dist_hi20"] = np.log(p.close / hi) / sd; F["dist_lo20"] = np.log(p.close / lo) / sd
names = list(F)
H = 5
Y = ((logp.shift(-H) - logp) / (sd * np.sqrt(H))).clip(-4, 4)
D = pd.concat({k: v.stack(future_stack=True) for k, v in F.items()}, axis=1)
D["y"] = Y.stack(future_stack=True); D.index.names = ["t", "coin"]
D = D.dropna(subset=names)
pred = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
q_starts = pd.date_range("2022-01-01", "2026-06-23", freq="QS", tz="UTC")
for i, q0 in enumerate(q_starts):
    q1 = q_starts[i + 1] if i + 1 < len(q_starts) else pd.Timestamp("2026-06-24", tz="UTC")
    tr = D[(D.index.get_level_values("t") < q0 - pd.Timedelta(days=H + 1))].dropna(subset=["y"])
    mu, sg = tr[names].mean(), tr[names].std().replace(0, 1)
    m = Ridge(alpha=1000.0).fit(((tr[names] - mu) / sg).clip(-4, 4), tr["y"])
    te = D[(D.index.get_level_values("t") >= q0) & (D.index.get_level_values("t") < q1)]
    pr = pd.Series(m.predict(((te[names] - mu) / sg).clip(-4, 4)), index=te.index)
    pred.update(pr.unstack("coin"))
pred.loc[pred.index < pd.Timestamp("2022-01-01", tz="UTC")] = np.nan
DEV = ("2022-01-01", "2026-06-23"); VAL = ("2024-01-01", "2026-06-23")
# panel IC of the OOS prediction
ics = [pred.loc[t].corr(Y.loc[t], method="spearman") for t in pred.dropna(how="all").index if pred.loc[t].notna().sum() >= 4 and Y.loc[t].notna().sum() >= 4]
print("OOS mean XS rank-IC (5d fwd):", round(float(np.nanmean(ics)), 4), " n days", len(ics))
tsw = pv_target(ts_weights(p, pred.clip(-1, 1)), p)
xsw = pv_target(xs_weights(p, pred, rebalance_every=1), p)
for nm, w in [("ML_TS", tsw), ("ML_XS", xsw)]:
    b = run_book(w, p); x = b["net"]
    print(f"{nm}: dev SR {sr(x,*DEV):5.2f} | val SR {sr(x,*VAL):5.2f} | gross dev {sr(b['gross'],*DEV):5.2f} | turn {b['turnover'].loc[DEV[0]:DEV[1]].mean():.3f} | by year", {y: round(v['sr'], 2) for y, v in by_year(x.loc[DEV[0]:DEV[1]]).items()})
log_trial("ML_RIDGE", "ml", "pooled ridge a=1000, quarterly refit, TS and XS book (one trial)")
print("M =", M())
