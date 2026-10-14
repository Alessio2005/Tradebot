import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.sleeves import *
from lib.stats import sr, by_year, perf
from lib.trials import log_trial, M
DISC = ("2020-09-01", "2023-12-31"); D6 = ("2021-10-20", "2023-12-31")
p = load_panel()
s_mix = 0.5 * tsmom_score(p) + 0.5 * tsmom_score(p, long_flat=True)
w_trend = pv_target(ts_weights(p, s_mix), p)
w_xs = pv_target(xs_weights(p, xs_mom_score(p)), p)
w_combo = pv_target(cap_gross(0.5 * w_trend + 0.5 * w_xs), p)
rows = {}
for nm, w in [("TREND_MIX", w_trend), ("XS_MOM", w_xs), ("COMBO_TX", w_combo)]:
    b = run_book(w, p); r = b["net"]; rows[nm] = r
    print(f"{nm:10} disc SR {sr(r,*DISC):5.2f} | d6 SR {sr(r,*D6):5.2f} | gross d6 {sr(b['gross'],*D6):5.2f} | ann d6 {r.loc[D6[0]:D6[1]].mean()*365:6.3f} | vol d6 {r.loc[D6[0]:D6[1]].std()*np.sqrt(365):5.3f} | mdd d6 {perf(r,*D6,boot=False)['max_dd']:6.3f}")
    print("           by year:", {y: round(v['sr'], 2) for y, v in by_year(r.loc[:DISC[1]]).items()},
          "| lag2 d6", round(sr(run_book(w, p, lag=2)["net"], *D6), 2),
          "| 2x cost d6", round(sr(run_book(w, p, cost=13e-4)["net"], *D6), 2))
    if nm != "XS_MOM": log_trial(nm, "sleeve", "discovery combo", disc_sharpe=sr(r, *DISC))
R = pd.DataFrame(rows).loc[D6[0]:D6[1]]; print(R.corr().round(2))
print("M =", M())
