"""Stage E: multiverse of the TS-trend family. Report ALL specs, select nothing. 54 specs = 18 geometric lookback
ladders x {LS, LF, MIX}. The deliverable candidate is the equal-weight AVERAGE of all specs (no selection)."""
import sys, json; sys.path.insert(0, "research")
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.sleeves import *
from lib.stats import sr, by_year, perf, dsr
from lib.trials import log_trial, M

DEV = ("2022-01-01", "2026-06-23"); PRE = ("2022-01-01", "2023-12-31"); VAL = ("2024-01-01", "2026-06-23")
p = load_panel()
ladders = [tuple(int(Lmin * 2 ** k) for k in range(n)) for Lmin in (5, 7, 10, 14, 20, 28) for n in (3, 4, 5)]
specs, rows, ws = [], [], []
for Ls in ladders:
    ls, lf = tsmom_score(p, Ls), tsmom_score(p, Ls, long_flat=True)
    for kind, s in (("LS", ls), ("LF", lf), ("MIX", 0.5 * ls + 0.5 * lf)):
        w = pv_target(ts_weights(p, s), p); b = run_book(w, p)["net"]
        rows.append({"ladder": str(Ls), "kind": kind, "dev": sr(b, *DEV), "pre": sr(b, *PRE), "val": sr(b, *VAL), "full6": sr(b, "2021-10-20", DEV[1])})
        ws.append(w)
T = pd.DataFrame(rows); pd.set_option("display.width", 200)
print(T.round(2).to_string(index=False))
print("\nDISTRIBUTION over 54 specs (Sharpe):"); print(T[["dev", "pre", "val", "full6"]].describe().round(2))
print("\nby kind (median dev/val):"); print(T.groupby("kind")[["dev", "pre", "val"]].median().round(2))
w_avg = pv_target(cap_gross(sum(ws) / len(ws)), p); bk = run_book(w_avg, p); r = bk["net"]
res = {"dev": perf(r, *DEV), "val": perf(r, *VAL), "pre": perf(r, *PRE, boot=False), "by_year": by_year(r.loc[DEV[0]:DEV[1]]),
       "lag2_dev": sr(run_book(w_avg, p, lag=2)["net"], *DEV), "cost2_dev": sr(run_book(w_avg, p, cost=13e-4)["net"], *DEV),
       "lag2_val": sr(run_book(w_avg, p, lag=2)["net"], *VAL), "cost2_val": sr(run_book(w_avg, p, cost=13e-4)["net"], *VAL),
       "turnover": float(bk["turnover"].loc[DEV[0]:DEV[1]].mean())}
log_trial("TREND_MULTIVERSE_AVG", "sleeve", "equal-weight average of 54 trend specs; no selection")
res["dsr_dev_M"] = dsr(r, M(), *DEV); res["M"] = M()
print("\nTREND_MULTIVERSE_AVG:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k not in ("dev", "val", "pre", "by_year")}))
for k in ("pre", "dev", "val"):
    v = res[k]; print(f"  {k}: SR {v['sharpe']:.2f} (SE {v['se']:.2f}" + (f", CI [{v['ci_lo']:.2f},{v['ci_hi']:.2f}]" if "ci_lo" in v else "") + f") ann {v['ann_ret']:.3f} vol {v['ann_vol']:.3f} mdd {v['max_dd']:.3f}")
print("  by year:", {y: (round(x['ret'], 3), round(x['sr'], 2)) for y, x in res["by_year"].items()})
T.to_csv("research/results/stage_e_multiverse.csv", index=False)
json.dump(res, open("research/results/stage_e_multiverse_avg.json", "w"), indent=1, default=float)
r.to_frame("net").to_parquet("research/results/multiverse_avg_returns.parquet")
