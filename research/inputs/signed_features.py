"""Trial S1: same pipeline, but directional features signed by trade side (structural fix)."""
import os, dataclasses, json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/claude/alessio2005/tradebot"); os.chdir(ROOT)
exec(open('/home/claude/research/_setup.py').read())
DIRECTIONAL = ["ffd_logp","ret1_n","ret5_n","ret20_n","dist_hi20","dist_lo20","kalman_z",
               "funding_z","oi_chg1","oi_chg5","mkt_ret5","resid_z20"]
feat = ds.features.copy()
for c in DIRECTIONAL: feat[c] = feat[c] * feat["side"]
# dist_hi/lo swap meaning under signing: for shorts, distance to low is the 'breakout' distance
ds2 = dataclasses.replace(ds, features=feat)
res = {}
for kind in MODEL_KINDS:
    f = walk_forward_fit_predict(ds2, cv, lambda rows, y: fit_light_model(ds2, rows, y, kind, cfg),
                                 n_bars=n_bars, embargo_bars=emb)
    res[kind] = auc(f)
    print(kind, res[kind])
json.dump(res, open('/home/claude/research/signed_features.json','w'))
