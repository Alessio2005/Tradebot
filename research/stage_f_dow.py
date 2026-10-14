"""Stage F: day-of-week screen on DISCOVERY ONLY (<= 2023-12-31). Is there a weekday structure that could explain why
7-day-multiple lookbacks beat off-by-one ladders? 7 weekdays booked as 1 trial (single family)."""
import sys; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.sleeves import tsmom_score, ts_weights, pv_target
from lib.engine import run_book
from lib.stats import sr
from lib.trials import log_trial
p = load_panel()
A, B = "2021-10-20", "2023-12-31"
ew = p.ret.mean(axis=1).loc[A:B]; vol = p.ret.abs().mean(axis=1).loc[A:B]
df = pd.DataFrame({"r": ew, "absr": vol}); df["dow"] = df.index.dayofweek
g = df.groupby("dow")
tab = pd.DataFrame({"mean_ret_bp": g.r.mean() * 1e4, "t": g.r.mean() / (g.r.std() / np.sqrt(g.r.count())), "mean_absret_bp": g.absr.mean() * 1e4})
tab.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print(tab.round(2))
# ladders shifted by +-1 day off the 7-multiples: is the 7-multiple advantage structural or luck? (diagnostic, not a trial)
print("\nladder alignment diagnostic (MIX, pv20): disc6 Sharpe 2021-10-20..2023-12-31 and 2024+")
for Ls in [(7,14,28,56,112), (6,13,27,55,111), (8,15,29,57,113), (7,14,28,56,112), (9,16,30,58,114), (5,12,26,54,110)]:
    s = 0.5 * tsmom_score(p, Ls) + 0.5 * tsmom_score(p, Ls, long_flat=True)
    r = run_book(pv_target(ts_weights(p, s), p), p)["net"]
    print(Ls, "d6 %.2f | val %.2f" % (sr(r, A, B), sr(r, "2024-01-01", "2026-06-23")))
log_trial("SCREEN_dow", "screen", "weekday effect, EW index, discovery")
