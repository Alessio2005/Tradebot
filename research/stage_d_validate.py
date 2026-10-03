"""Stage D: ONE read of the validation window (2024-01-01 -> 2026-06-23) for the frozen COMBO_TX and its sleeves,
plus the preregistered acceptance gates on the 2022-01-01 -> 2026-06-23 development OOS."""
import sys, json; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.sleeves import *
from lib.stats import sr, by_year, perf, dsr
from lib.trials import M

VAL = ("2024-01-01", "2026-06-23"); DEV = ("2022-01-01", "2026-06-23"); PRE = ("2022-01-01", "2023-12-31")
p = load_panel()
s_mix = 0.5 * tsmom_score(p) + 0.5 * tsmom_score(p, long_flat=True)
w_trend = pv_target(ts_weights(p, s_mix), p)
w_xs = pv_target(xs_weights(p, xs_mom_score(p)), p)
w_combo = pv_target(cap_gross(0.5 * w_trend + 0.5 * w_xs), p)
out = {}
print(f"M = {M()}")
for nm, w in [("TREND_MIX", w_trend), ("XS_MOM", w_xs), ("COMBO_TX", w_combo)]:
    b = run_book(w, p); r = b["net"]
    d = {"val": perf(r, *VAL), "dev": perf(r, *DEV), "pre": perf(r, *PRE, boot=False)}
    d["lag2_dev"] = sr(run_book(w, p, lag=2)["net"], *DEV); d["cost2_dev"] = sr(run_book(w, p, cost=13e-4)["net"], *DEV)
    d["lag2_val"] = sr(run_book(w, p, lag=2)["net"], *VAL); d["cost2_val"] = sr(run_book(w, p, cost=13e-4)["net"], *VAL)
    d["by_year"] = by_year(r.loc[DEV[0]:DEV[1]])
    d["dsr_dev_M"] = dsr(r, M(), *DEV)
    out[nm] = d
    v, dv = d["val"], d["dev"]
    print(f"\n{nm}: VALIDATION SR {v['sharpe']:.2f} (SE {v['se']:.2f}, CI [{v['ci_lo']:.2f},{v['ci_hi']:.2f}]) ann {v['ann_ret']:.3f} vol {v['ann_vol']:.3f} mdd {v['max_dd']:.3f}")
    print(f"   DEV 2022+ SR {dv['sharpe']:.2f} (SE {dv['se']:.2f}, CI [{dv['ci_lo']:.2f},{dv['ci_hi']:.2f}]) ann {dv['ann_ret']:.3f} vol {dv['ann_vol']:.3f} mdd {dv['max_dd']:.3f} | 2022-23 SR {d['pre']['sharpe']:.2f}")
    print(f"   lag2 dev {d['lag2_dev']:.2f} val {d['lag2_val']:.2f} | 2xcost dev {d['cost2_dev']:.2f} val {d['cost2_val']:.2f} | DSR(M={M()}) dev {d['dsr_dev_M']:.3f}")
    print("   by year:", {y: (round(x['ret'], 3), round(x['sr'], 2)) for y, x in d["by_year"].items()})
json.dump(out, open("research/results/stage_d_validation.json", "w"), indent=1, default=float)
