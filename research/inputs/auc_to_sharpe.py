import numpy as np
from scipy import stats
rng = np.random.default_rng(0)
eps = 0.0025/ (np.sqrt(5)*0.042)   # ex-ante cost ~25bps / barrier ~9.4%
print("eps", round(eps,3))
print("AUC  hit(top15%)  meanR   SR/yr@78 trades  SR/yr@400 trades(breadth x5)")
for a in [0.51, 0.52, 0.53, 0.55, 0.58, 0.60]:
    mu = np.sqrt(2)*stats.norm.ppf(a)
    n = 2_000_000; y = rng.random(n) < 0.5
    s = rng.normal(mu*y, 1.0)
    thr = np.quantile(s, 0.85); hit = y[s >= thr].mean()
    m = 2*hit - 1 - eps; sd = np.sqrt(1 - (2*hit-1)**2)
    print(f"{a:.2f}  {hit:.3f}         {m:+.3f}  {m/sd*np.sqrt(78):+.2f}            {m/sd*np.sqrt(400):+.2f}")
# drift needed: P(up first) = 1/(1+exp(-2 mu b / s^2)), b = sqrt5 * s  -> 2*sqrt5*(mu/s)
print("\nconditional daily Sharpe (mu/sigma) -> P(win) for 1:1 barrier at sqrt5 sigma")
for sr_d in [0.01, 0.02, 0.03, 0.05, 0.08]:
    p = 1/(1+np.exp(-2*np.sqrt(5)*sr_d))
    print(f"daily SR {sr_d:.2f} (annual {sr_d*np.sqrt(365):.2f}) -> p_win {p:.3f}")
