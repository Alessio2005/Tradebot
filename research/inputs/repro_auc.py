"""Reproduce the OOS AUC of the weekly meta-label pipeline on the development sample.

Does NOT touch the ledger or the holdout lock; data strictly before 2026-06-24 (the spec's holdout split).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/claude/alessio2005/tradebot")
import os; os.chdir(ROOT)
from tradebot.data.weekly_market import load_weekly_market
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.schemas.config import ExecutionConfig, load_config
from tradebot.labeling.barrier_fills import round_trip_cost
from tradebot.features.weekly_set import fit_common_d_star
from tradebot.labeling.breakout import calibrate_k
from tradebot.train.weekly_dataset import build_weekly_dataset
from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.train.light_models import MODEL_KINDS, fit_light_model
from tradebot.train.meta_label import walk_forward_fit_predict, shuffled_targets

t0 = time.time()
cfg = weekly_meta_config()
full = load_weekly_market(ROOT, cfg.symbols)
split = pd.Timestamp(cfg.holdout_split_utc)
market = full.truncate(split)
print("grid", market.grid[0], "->", market.grid[-1], len(market.grid))
exec_cfg = load_config(ROOT / "conf/execution/fees.yaml", ExecutionConfig)
cost_rt = round_trip_cost(exec_cfg)
first_test = pd.Timestamp(cfg.first_test_start_utc)
closes = {s: market.ohlcv[s]["close"] for s in market.symbols}
d_star = fit_common_d_star({s: np.log(c) for s, c in closes.items()}, until=first_test)
start = market.sigma_daily.dropna(how="all").index[0]
k, rates = calibrate_k(closes, {s: market.sigma_daily[s] for s in market.symbols},
                       k_grid=cfg.k_grid, target_per_week=cfg.events_per_week_target,
                       start=start, end=first_test)
print("cost_rt", cost_rt, "d*", d_star, "k", k, rates)
wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt)
ds = wd.dataset
print("events", len(ds), "effective_n", ds.effective_n, "dropped", wd.n_dropped_nan)
print("positive rate", float(np.mean(ds.target)))
first_pos = int(market.grid.searchsorted(first_test))
emb = cfg.horizon_bars + 1
cv = WalkForwardCV(train_size=first_pos, test_size=cfg.test_bars, step=cfg.test_bars,
                   mode="anchored", min_train=first_pos, embargo_bars=emb)
n_bars = len(market.grid)
def run(kind, target=None):
    return walk_forward_fit_predict(ds, cv, lambda rows, y: fit_light_model(ds, rows, y, kind, cfg),
                                    n_bars=n_bars, embargo_bars=emb, target=target)
def auc(folds, weighted=True):
    y = np.concatenate([f.target for f in folds]); p = np.concatenate([f.probability for f in folds])
    w = np.concatenate([f.uniqueness for f in folds])
    return float(roc_auc_score(y, p, sample_weight=w if weighted else None)), len(y)
out = {"k": k, "d_star": d_star, "n_events": int(len(ds)), "effective_n": float(ds.effective_n),
       "pos_rate": float(np.mean(ds.target))}
folds_all = {}
for kind in MODEL_KINDS:
    f = run(kind); folds_all[kind] = f
    a, n = auc(f); au, _ = auc(f, False)
    out[f"auc_{kind}"] = a; out[f"auc_unw_{kind}"] = au; out["n_oos"] = n
    print(kind, "OOS AUC w=%.4f unw=%.4f n=%d" % (a, au, n), "t=%.0fs" % (time.time()-t0))
sh = [auc(run("ensemble", perm))[0] for perm in shuffled_targets(ds, cfg.seed, 3)]
out["shuffle_aucs"] = sh
print("shuffled", sh)
# save OOS predictions for diagnosis
f = folds_all["ensemble"]
pred = pd.DataFrame({"row": np.concatenate([x.row_index for x in f]),
                     "p": np.concatenate([x.probability for x in f]),
                     "y": np.concatenate([x.target for x in f]),
                     "w": np.concatenate([x.uniqueness for x in f]),
                     "fold": np.concatenate([[x.fold_id]*len(x.row_index) for x in f])})
pred.to_parquet("/home/claude/research/oos_ensemble.parquet")
ev = wd.events.copy(); ev.to_parquet("/home/claude/research/events.parquet")
ds.features.to_parquet("/home/claude/research/features.parquet")
json.dump(out, open("/home/claude/research/repro_auc.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
