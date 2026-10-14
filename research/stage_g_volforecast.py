"""Stage G: can a better vol forecast than the repo's EWMA(0.94) improve risk scaling?  Second moments are far more
predictable than first moments, so this is where a robust (if modest) Sharpe gain can live.

Target: log realised variance over the NEXT 5 days per coin. Candidates (all causal at close t):
  EWMA94 (baseline), EWMA97, Garman-Klass EWMA94, HAR blend (log-RV 1d/7d/30d, expanding-window OLS refit yearly).
Metric: QLIKE loss (robust to noisy proxies, Patton 2011), out-of-sample 2022+ (HAR fitted on data < refit date only).
"""
import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.trials import log_trial
p = load_panel(); r = p.ret
H = 5
rv_next = (r ** 2).rolling(H).sum().shift(-H)                         # realised variance next H days (sum of squares)
def ew(x, lam): return x.ewm(alpha=1 - lam, min_periods=20).mean()
v94 = ew(r ** 2, 0.94) * H; v97 = ew(r ** 2, 0.97) * H
gk = 0.5 * np.log(p.high / p.low) ** 2 - (2 * np.log(2) - 1) * np.log(p.close / p.open) ** 2
vgk = ew(gk, 0.94) * H
# HAR in log space: log RV_next ~ a + b1 log rv1 + b2 log rv7 + b3 log rv30
rv1 = (r ** 2).clip(lower=1e-8); rv7 = (r ** 2).rolling(7).mean(); rv30 = (r ** 2).rolling(30).mean()
X = pd.concat({"l1": np.log(rv1), "l7": np.log(rv7), "l30": np.log(rv30)}, axis=1)
Y = np.log(rv_next.clip(lower=1e-8))
def stack(d):  # long format
    return pd.DataFrame({k: d[k].stack() if isinstance(d, dict) else d[k].stack() for k in d}) if False else None
cols = r.columns
har = pd.DataFrame(np.nan, index=r.index, columns=cols)
years = range(2022, 2027)
for yr in years:
    t0 = pd.Timestamp(f"{yr}-01-01", tz="UTC"); t1 = pd.Timestamp(f"{yr + 1}-01-01", tz="UTC")
    train_end = t0 - pd.Timedelta(days=H + 1)
    xs, ys = [], []
    for c in cols:
        d = pd.DataFrame({"y": Y[c], "l1": X["l1"][c], "l7": X["l7"][c], "l30": X["l30"][c]}).loc[:train_end].dropna()
        xs.append(d[["l1", "l7", "l30"]]); ys.append(d["y"])
    xt, yt = pd.concat(xs), pd.concat(ys)
    A = np.column_stack([np.ones(len(xt)), xt.to_numpy()]); beta = np.linalg.lstsq(A, yt.to_numpy(), rcond=None)[0]
    # log-normal bias correction: exp(pred + 0.5 resid var)
    resid_var = float(np.var(yt.to_numpy() - A @ beta))
    for c in cols:
        m = (r.index >= t0) & (r.index < t1)
        xx = np.column_stack([np.ones(m.sum()), X["l1"][c][m], X["l7"][c][m], X["l30"][c][m]])
        har.loc[m, c] = np.exp(xx @ beta + 0.5 * resid_var)
cands = {"EWMA94": v94, "EWMA97": v97, "GK_EWMA94": vgk, "HAR": har}
def qlike(f, y):  # mean over valid, f,y variances
    ok = f.notna() & y.notna() & (f > 0) & (y > 0)
    ratio = (y / f)[ok]; return float((ratio - np.log(ratio) - 1).stack().mean() if hasattr(ratio, "stack") else (ratio - np.log(ratio) - 1).mean())
print("QLIKE of next-5d variance forecast (lower is better), by year (HAR is out-of-sample by construction):")
res = {}
for nm, f in cands.items():
    res[nm] = {}
    for yr in years:
        m = (r.index >= pd.Timestamp(f"{yr}-01-01", tz="UTC")) & (r.index < pd.Timestamp(f"{yr + 1}-01-01", tz="UTC")) & (r.index <= pd.Timestamp("2026-06-17", tz="UTC"))
        ff, yy = f[m], rv_next[m]
        ok = ff.notna() & yy.notna() & (ff > 0) & (yy > 0); ratio = (yy / ff).where(ok)
        res[nm][yr] = float(((ratio - np.log(ratio) - 1).stack()).mean())
T = pd.DataFrame(res); T.loc["all"] = T.mean(); print(T.round(3))
print("\nrelative to EWMA94 (negative = better):"); print((T.div(T["EWMA94"], axis=0) - 1).round(3))
log_trial("VOLFC_family", "risk", "vol-forecast QLIKE comparison, EWMA94/97/GK/HAR")
