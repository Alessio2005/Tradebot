"""Step 0: reproduce the T1/T2/B0 numbers of the uploaded trend_carry.py with the research lib."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.stats import perf

p = load_panel()
logp = np.log(p.close)
LOOKS = [7, 14, 28, 56, 112]

def signal(long_flat=False):
    s = sum(np.sign(logp - logp.shift(L)) for L in LOOKS) / len(LOOKS)
    s = s.where(logp.shift(max(LOOKS)).notna())
    return s.clip(lower=0) if long_flat else s

def weights(s):
    live = s.notna() & p.sigma_ann.notna()
    n_live = live.sum(axis=1).replace(0, np.nan)
    w = (s * 0.40 / p.sigma_ann).where(live).div(n_live, axis=0).fillna(0.0)
    return cap_gross(w)

res = {}
for name, s in {"T1": signal(), "T2": signal(True)}.items():
    r = run_book(weights(s), p)["net"]
    res[name] = {per: perf(r, a, b, boot=False) for per, (a, b) in
                 {"full": ("2020-09-01", "2026-06-23"), "2022+": ("2022-01-01", "2026-06-23")}.items()}
    print(name, {k: round(v["sharpe"], 2) for k, v in res[name].items()})
print("zip: T1 full 0.99 / 2022+ 0.64 ; T2 full 1.20 / 2022+ 0.74")
