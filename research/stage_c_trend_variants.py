import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book
from lib.sleeves import *
from lib.stats import sr, by_year
from lib.trials import log_trial
DISC = ("2020-09-01", "2023-12-31"); D6 = ("2021-10-20", "2023-12-31")
p = load_panel()
def show(name, w, log=True):
    b = run_book(w, p); r = b["net"]
    print(f"{name:26} disc {sr(r,*DISC):5.2f} | d6 {sr(r,*D6):5.2f} | gross d6 {sr(b['gross'],*D6):5.2f} | turn {b['turnover'].loc[D6[0]:D6[1]].mean():.3f} | lag2 d6 {sr(run_book(w,p,lag=2)['net'],*D6):5.2f}")
    if log: log_trial(name, "sleeve", "discovery variant", disc_sharpe=sr(r, *DISC))
    return r
base_ls = tsmom_score(p); base_lf = tsmom_score(p, long_flat=True)
print("--- structure (booked trials)")
show("TR_BRK_LS", pv_target(ts_weights(p, donchian_score(p)), p))
show("TR_BRK_LF", pv_target(ts_weights(p, donchian_score(p, long_flat=True)), p))
show("TR_SM_LS", pv_target(ts_weights(p, smooth_score(base_ls)), p))
print("--- sensitivity (NOT trials): TR_LS / TR_LF without portfolio vol target and with other spans")
show("T1 raw (per-coin 40%/N)", ts_weights(p, base_ls), log=False)
show("T2 raw", ts_weights(p, base_lf), log=False)
for span in (30, 60, 120):
    show(f"TR_LS pv20 span{span}", pv_target(ts_weights(p, base_ls), p, span=span), log=False)
for tgt in (0.10, 0.20, 0.30):
    show(f"TR_LS pv{int(tgt*100)}", pv_target(ts_weights(p, base_ls), p, target=tgt), log=False)
print("--- lookback neighbourhood (NOT trials): pv20 LS")
for Ls in [(7,14,28),(14,28,56),(28,56,112),(7,14,28,56,112),(5,10,20,40,80,160)]:
    show(f"TR_LS {Ls}", pv_target(ts_weights(p, tsmom_score(p, Ls)), p), log=False)
