import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
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
cfg = weekly_meta_config()
full = load_weekly_market(ROOT, cfg.symbols)
market = full.truncate(pd.Timestamp(cfg.holdout_split_utc))
cost_rt = round_trip_cost(load_config(ROOT / "conf/execution/fees.yaml", ExecutionConfig))
first_test = pd.Timestamp(cfg.first_test_start_utc)
closes = {s: market.ohlcv[s]["close"] for s in market.symbols}
d_star = fit_common_d_star({s: np.log(c) for s, c in closes.items()}, until=first_test)
start = market.sigma_daily.dropna(how="all").index[0]
k, rates = calibrate_k(closes, {s: market.sigma_daily[s] for s in market.symbols},
                       k_grid=cfg.k_grid, target_per_week=cfg.events_per_week_target, start=start, end=first_test)
wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt)
ds = wd.dataset
first_pos = int(market.grid.searchsorted(first_test)); emb = cfg.horizon_bars + 1
cv = WalkForwardCV(train_size=first_pos, test_size=cfg.test_bars, step=cfg.test_bars,
                   mode="anchored", min_train=first_pos, embargo_bars=emb)
n_bars = len(market.grid)
def auc(folds):
    y = np.concatenate([f.target for f in folds]); p = np.concatenate([f.probability for f in folds])
    w = np.concatenate([f.uniqueness for f in folds])
    return round(float(roc_auc_score(y, p, sample_weight=w)),4), round(float(roc_auc_score(y, p)),4)
