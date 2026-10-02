"""Van gecertificeerde markt naar de gepoolde meta-label-dataset (spec §5-§7).

De volgorde van rijen is die van `train.meta_label.build_dataset`: symbolen
alfabetisch, binnen een symbool op eventbar. `events` volgt precies die
volgorde, zodat het tradeboek per rij de fill en de sigma terugvindt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.weekly_market import WeeklyMarket
from ..features.weekly_set import build_feature_panel
from ..labeling.barrier_fills import barrier_fill_returns, select_events, with_cost_aware_target
from ..labeling.breakout import breakout_side
from ..labeling.vol_barriers import BarrierLabels, label_triple_barrier
from ..schemas.config import LabelingConfig
from ..schemas.weekly_meta import WeeklyMetaConfig
from .meta_label import MetaLabelDataset, build_dataset

__all__ = ["WeeklyDataset", "build_weekly_dataset", "labeling_config"]


@dataclass(frozen=True)
class WeeklyDataset:
    dataset: MetaLabelDataset
    events: pd.DataFrame
    labels: dict[str, BarrierLabels]
    grid: pd.DatetimeIndex
    k: float
    d_star: float
    n_dropped_nan: dict[str, int]


def labeling_config(cfg: WeeklyMetaConfig) -> LabelingConfig:
    return LabelingConfig(profit_target_sigma=cfg.barrier_sigma,
                          stop_loss_sigma=cfg.barrier_sigma,
                          horizon_bars=cfg.horizon_bars, entry_lag_bars=1,
                          min_sigma_obs=60)


def build_weekly_dataset(
    market: WeeklyMarket,
    cfg: WeeklyMetaConfig,
    *,
    k: float,
    d_star: float,
    cost_rt: float,
    side_sign: float = 1.0,
) -> WeeklyDataset:
    """Events, 1:1-labels met barrièrefills, en features; rijen met een NaN-feature vallen af."""
    lab_cfg = labeling_config(cfg)
    sides = {s: side_sign * breakout_side(market.ohlcv[s]["close"], market.sigma_daily[s], k)
             for s in market.symbols}
    features = build_feature_panel(market, sides, d_star=d_star, corr_window=cfg.corr_window)
    labels: dict[str, BarrierLabels] = {}
    dropped: dict[str, int] = {}
    rows: list[pd.DataFrame] = []
    for s in sorted(market.symbols):
        o = market.ohlcv[s]
        raw = label_triple_barrier(
            o["high"].to_numpy(), o["low"].to_numpy(), o["close"].to_numpy(),
            market.sigma_daily[s].to_numpy(), sides[s].to_numpy(), lab_cfg)
        fills = barrier_fill_returns(raw, o["open"].to_numpy(), o["close"].to_numpy(),
                                     lab_cfg, stop_slippage_bps=cfg.stop_slippage_bps)
        lab = with_cost_aware_target(raw, fills, round_trip_cost=cost_rt)
        finite = np.isfinite(features[s].iloc[lab.event_idx].to_numpy(dtype=float)).all(axis=1)
        dropped[s] = int((~finite).sum())
        lab = select_events(lab, finite)
        labels[s] = lab
        rows.append(pd.DataFrame({
            "symbol": s, "event_bar": lab.event_idx, "exit_bar": lab.exit_idx,
            "side": lab.side, "fill_return": lab.realized_return, "sigma": lab.sigma,
            "target": lab.meta_label.astype(np.int64),
        }))
    dataset = build_dataset(features, labels)
    return WeeklyDataset(dataset=dataset, events=pd.concat(rows, ignore_index=True),
                         labels=labels, grid=market.grid, k=float(k),
                         d_star=float(d_star), n_dropped_nan=dropped)
