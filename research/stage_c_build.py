"""Stage C build: sleeve books on DISCOVERY only (<= 2023-12-31). The validation window is sealed."""
import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book
from lib.sleeves import *
from lib.stats import perf, sr, by_year
from lib.trials import log_trial

DISC = ("2020-09-01", "2023-12-31"); DISC6 = ("2021-10-20", "2023-12-31")
p = load_panel()
books = {}
books["TR_LS"] = pv_target(ts_weights(p, tsmom_score(p)), p)
books["TR_LF"] = pv_target(ts_weights(p, tsmom_score(p, long_flat=True)), p)
books["XS_MOM"] = pv_target(xs_weights(p, xs_mom_score(p)), p)
books["FUND_XS"] = pv_target(xs_weights(p, -fund_z(p)), p)
rets = {}
print(f"{'sleeve':8} {'win':8} {'SR':>6} {'ann':>7} {'vol':>6} {'mdd':>7} | gross SR | turn/day")
for k, w in books.items():
    b = run_book(w, p); rets[k] = b["net"]
    for wn, (a, e) in {"disc": DISC, "disc6": DISC6}.items():
        x = perf(b["net"], a, e, boot=False)
        print(f"{k:8} {wn:8} {x['sharpe']:6.2f} {x['ann_ret']:7.3f} {x['ann_vol']:6.3f} {x['max_dd']:7.3f} | {sr(b['gross'], a, e):6.2f}   | {b['turnover'].loc[a:e].mean():.3f}")
    print("   by year:", {y: round(v['sr'], 2) for y, v in by_year(b["net"].loc[:DISC[1]]).items()})
R = pd.DataFrame(rets).loc[DISC6[0]:DISC6[1]]
print("\ncorrelation (disc6):"); print(R.corr().round(2))
for k in books:
    log_trial(k, "sleeve", "pv20 weekly/daily as pre-registered", disc_sharpe=sr(rets[k], *DISC))
import pickle; pickle.dump(books, open("research/results/stage_c_books.pkl", "wb"))
