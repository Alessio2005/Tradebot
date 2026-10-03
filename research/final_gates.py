"""Apply the preregistered acceptance gates A1-A6 (A7, the 60-bar holdout, is deliberately NOT read: no candidate earns it)."""
import sys, json; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.sleeves import *
from lib.stats import sr, by_year, perf, dsr
from lib.trials import M, _read

DEV = ("2022-01-01", "2026-06-23"); PRE = ("2022-01-01", "2023-12-31"); VAL = ("2024-01-01", "2026-06-23")
p = load_panel()
ladders = [tuple(int(L * 2 ** k) for k in range(n)) for L in (5, 7, 10, 14, 20, 28) for n in (3, 4, 5)]
ws = []
for Ls in ladders:
    ls, lf = tsmom_score(p, Ls), tsmom_score(p, Ls, long_flat=True)
    for s in (ls, lf, 0.5 * ls + 0.5 * lf): ws.append(pv_target(ts_weights(p, s), p))
w_avg = pv_target(cap_gross(sum(ws) / len(ws)), p)
s_mix = 0.5 * tsmom_score(p) + 0.5 * tsmom_score(p, long_flat=True)
w_trend = pv_target(ts_weights(p, s_mix), p); w_xs = pv_target(xs_weights(p, xs_mom_score(p)), p)
cands = {"TREND_MIX (frozen, 7-112)": w_trend, "COMBO_TX (frozen)": pv_target(cap_gross(0.5 * w_trend + 0.5 * w_xs), p), "TREND_MULTIVERSE_AVG (54 specs)": w_avg}
out = {}
for nm, w in cands.items():
    r = run_book(w, p)["net"]; d = perf(r, *DEV); v = perf(r, *VAL, boot=False); yr = by_year(r.loc[DEV[0]:DEV[1]])
    lag2 = sr(run_book(w, p, lag=2)["net"], *DEV); c2 = sr(run_book(w, p, cost=13e-4)["net"], *DEV)
    g = {"A1 net SR>=1.0": (d["sharpe"], d["sharpe"] >= 1.0),
         "A2 SR>=0.7 in 22-23 AND 24+": ((sr(r, *PRE), v["sharpe"]), sr(r, *PRE) >= 0.7 and v["sharpe"] >= 0.7),
         "A3 >=4/5 yrs positive, none < -10%": ({k: round(x['ret'], 3) for k, x in yr.items()}, sum(x['ret'] > 0 for x in yr.values()) >= 4 and min(x['ret'] for x in yr.values()) > -0.10),
         "A4 maxDD <= 25%": (d["max_dd"], d["max_dd"] >= -0.25),
         "A5 2xcost>=0.8 AND lag2>=0.6": ((c2, lag2), c2 >= 0.8 and lag2 >= 0.6),
         "A6 CI95 lower bound > 0": (d["ci_lo"], d["ci_lo"] > 0)}
    out[nm] = {"gates": {k: {"value": x, "pass": bool(ok)} for k, (x, ok) in g.items()}, "dev": d, "val_sharpe": v["sharpe"],
               "dsr_M": dsr(r, M(), *DEV), "M": M(), "all_pass": all(ok for _, ok in g.values())}
    print(f"\n{nm}  (M={M()}, DSR={out[nm]['dsr_M']:.3f}, dev SR {d['sharpe']:.2f}, CI [{d['ci_lo']:.2f},{d['ci_hi']:.2f}])")
    for k, (x, ok) in g.items(): print(f"  {'PASS' if ok else 'FAIL'}  {k}: {x}")
json.dump(out, open("research/results/final_gates.json", "w"), indent=1, default=float)
# power arithmetic
for sr_true in (0.5, 1.0, 1.5):
    se1 = np.sqrt((1 + 0.5 * sr_true ** 2) / 1.0)
    print(f"true SR {sr_true}: years for E[lower 95% bound]>0 : {(1.645 * se1 / sr_true) ** 2:.1f} ; for 80% power: {((1.645 + 0.84) * se1 / sr_true) ** 2:.1f}")
tr = _read(); fam = pd.Series([t["family"] for t in tr]).value_counts(); print("\ntrials by family:", fam.to_dict(), "| new:", len(tr), "| M:", M())
