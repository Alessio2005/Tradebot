from __future__ import annotations

import numpy as np
import pytest

from tradebot.cv.event_space import event_space_t1
from tradebot.registry.preregistration import StopCriterion
from tradebot.utils.failfast import DataContractError
from tradebot.validation.weekly_verdict import judge

C = [
    StopCriterion("negative_control_shuffle", "max_abs_shuffle_auc_deviation", ">", 0.05, "archive", "r"),
    StopCriterion("negative_control_reversed", "reversed_minus_baseline_sharpe", ">=", 0.0, "archive", "r"),
    StopCriterion("no_edge_after_costs", "ensemble_net_sharpe", "<=", 0.0, "falsify", "r"),
    StopCriterion("filter_adds_nothing", "sharpe_diff_ci_low", "<=", 0.0, "archive", "r"),
    StopCriterion("dsr_below_threshold", "dsr", "<", 0.95, "archive", "r"),
    StopCriterion("insufficient_trades", "n_trades", "<", 100.0, "descope", "r"),
    StopCriterion("holdout_brier_worse", "holdout_brier_diff_ci_low", ">", 0.0, "archive", "r"),
    StopCriterion("promotion_requires_all_clear", "n_binding_stop_criteria", "<=", 0.0, "promote", "r"),
]
GOOD = {"max_abs_shuffle_auc_deviation": 0.01, "reversed_minus_baseline_sharpe": -0.4,
        "ensemble_net_sharpe": 1.4, "sharpe_diff_ci_low": 0.1, "dsr": 0.97, "n_trades": 250.0}


def test_all_clear_is_a_pass() -> None:
    v = judge(C, GOOD, stage="development")
    assert v.status == "PASS" and v.binding == ()


@pytest.mark.parametrize(
    ("metric", "value", "status"),
    [
        ("max_abs_shuffle_auc_deviation", 0.08, "INVALID"),
        ("reversed_minus_baseline_sharpe", 0.1, "INVALID"),
        ("ensemble_net_sharpe", -0.2, "FALSIFIED"),
        ("sharpe_diff_ci_low", -0.1, "UNPROVEN"),
        ("dsr", 0.90, "UNPROVEN"),
        ("n_trades", 40.0, "UNPROVEN"),
    ],
)
def test_each_gate_can_go_red(metric, value, status) -> None:
    v = judge(C, {**GOOD, metric: value}, stage="development")
    assert v.status == status
    assert len(v.binding) == 1


def test_a_missing_measurement_crashes() -> None:
    values = dict(GOOD)
    del values["dsr"]
    with pytest.raises(DataContractError, match="meting"):
        judge(C, values, stage="development")


def test_the_holdout_stage_reads_only_holdout_criteria() -> None:
    assert judge(C, {"holdout_brier_diff_ci_low": -0.01}, stage="holdout").status == "PASS"
    assert judge(C, {"holdout_brier_diff_ci_low": 0.02}, stage="holdout").status == "UNPROVEN"


def test_event_space_t1_points_at_the_last_event_inside_the_label() -> None:
    ev = np.array([0, 2, 3, 7, 9])
    ex = np.array([4, 3, 8, 12, 10])
    assert event_space_t1(ev, ex).tolist() == [2, 2, 3, 4, 4]
    with pytest.raises(DataContractError):
        event_space_t1(np.array([3, 1]), np.array([4, 5]))
