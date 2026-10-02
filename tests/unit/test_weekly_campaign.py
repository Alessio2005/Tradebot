"""De campagne van begin tot eind op een synthetische markt: elke poortmeting bestaat en is eindig."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.registry.preregistration import StopCriterion
from tradebot.schemas.config import ExecutionConfig, RiskConfig, load_config
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.validation.weekly_campaign import run_campaign_on_market, trade_candidates
from tradebot.validation.weekly_verdict import DEVELOPMENT_METRICS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def result():
    market = synthetic_market(n=1100, seed=21)
    cfg = weekly_meta_config().model_copy(update={
        "first_test_start_utc": str(market.grid[500]),
        "holdout_split_utc": str(market.grid[-1] + pd.Timedelta(days=1)),
        "forest_n_estimators": 40, "n_shuffle_replicates": 1, "mc_paths": 1000,
        "k_grid": (1.0, 1.5, 2.0),
        # De synthetische markt is klein en kent geen echte doorbraken: met het productiedoel van
        # 2,5 per week kiest hij de grootste k en blijft het ensemble-boek vlak (nul variantie).
        "events_per_week_target": 5.5,
    })
    criteria = [StopCriterion(f"c_{m}", m, "<", -1e9, "archive", "synthetic")
                for m in DEVELOPMENT_METRICS]
    return run_campaign_on_market(
        market, cfg, criteria=criteria,
        exec_cfg=load_config(ROOT / "conf/execution/fees.yaml", ExecutionConfig),
        risk_cfg=load_config(ROOT / "conf/risk/default.yaml", RiskConfig),
        impact=None)


def test_every_gate_metric_is_measured_and_finite(result) -> None:
    for metric in DEVELOPMENT_METRICS:
        assert metric in result.values
        assert np.isfinite(result.values[metric]), metric


def test_the_record_carries_its_policy_and_its_choices(result) -> None:
    rec = result.record
    assert rec["risk_policy_hash"]
    assert rec["risk_audit_header"]["config_hash"] == rec["risk_policy_hash"]
    assert rec["k"] > 0.0 and 0.0 < rec["d_star"] <= 0.9
    assert rec["verdict"]["status"] in {"PASS", "INVALID", "UNPROVEN", "FALSIFIED"}


def test_candidates_never_use_a_threshold_below_break_even() -> None:
    from tradebot.train.meta_label import FoldPredictions
    events = pd.DataFrame({"symbol": ["BTCUSDT"] * 3, "event_bar": [10, 11, 12],
                           "exit_bar": [15, 16, 17], "side": [1.0, -1.0, 1.0],
                           "fill_return": [0.05, -0.05, 0.0], "sigma": [0.03] * 3,
                           "target": [1, 0, 0]})
    fold = FoldPredictions(fold_id=0, row_index=np.array([0, 1, 2]),
                           probability=np.array([0.40, 0.55, 0.70]), target=np.array([1, 0, 0]),
                           uniqueness=np.ones(3), n_train=100, purge={},
                           feature_importance=np.ones(19),
                           extras={"calibration_probability": np.linspace(0.3, 0.7, 50),
                                   "train_events_per_week": 5.5})
    cands = trade_candidates(events, [fold], cfg=weekly_meta_config(), cost_rt=0.0013)
    assert (cands["p_trade"] >= cands["p_be"]).all()
    assert (cands["entry_bar"] == events["event_bar"] + 1).all()
    assert (cands["p_low"] < cands["p"]).all()
