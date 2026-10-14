"""Sensitivity (reported in full, not selected): single lookbacks and doubled costs for the T1/T2 family."""
import os, json, numpy as np, pandas as pd
from pathlib import Path
os.chdir("/home/claude/alessio2005/tradebot")
src = open("/home/claude/research/trend_carry.py").read().split("out = {}")[0]
exec(src)
def signal_L(Ls, long_flat=False):
    s = sum(np.sign(logp - logp.shift(L)) for L in Ls) / len(Ls)
    s = s.where(logp.shift(max(LOOKS)).notna())
    return s.clip(lower=0) if long_flat else s
rows = []
for Ls in [[7],[14],[28],[56],[112],LOOKS]:
    for lf in [False, True]:
        for cost_mult in [1, 2]:
            globals()["COST"] = 6.5e-4 * cost_mult
            r = book(signal_L(Ls, lf))["net"]
            for pname,(a,b) in {"full": ("2020-09-01","2026-06-23"), "2022+": ("2022-01-01","2026-06-23")}.items():
                x = r.loc[a:b]; rows.append({"lookbacks": str(Ls), "long_flat": lf, "cost_x": cost_mult, "period": pname,
                                             "sharpe": round(x.mean()/x.std()*np.sqrt(365), 2)})
t = pd.DataFrame(rows).pivot_table(index=["lookbacks","long_flat","cost_x"], columns="period", values="sharpe")
print(t.to_string()); t.to_csv("/home/claude/research/trend_sensitivity.csv")
