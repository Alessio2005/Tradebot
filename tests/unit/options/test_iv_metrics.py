"""IV rank, IV percentile, de vervangende IV/RV-regel en de realized vol."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from tradebot.options.iv_metrics import (
    IVCondition,
    IVHistory,
    IVMetrics,
    IVRStatus,
    IVRuleSettings,
    check_iv_condition,
    iv_rank,
    iv_rule_settings,
    realized_vol,
)
from tradebot.schemas.options import load_options_config
from tradebot.utils.failfast import DataContractError

D0 = date(2026, 1, 1)
CFG = IVRuleSettings(lookback_days=5)


def history(values, start=D0) -> IVHistory:
    return IVHistory((start + timedelta(days=i), v) for i, v in enumerate(values))


def test_iv_rank_and_percentile():
    assert iv_rank([0.1, 0.2, 0.3], 0.25) == pytest.approx((75.0, 200 / 3))
    assert iv_rank([0.2, 0.2], 0.3) == (50.0, 100.0)  # vlakke historie: IVR 50
    assert iv_rank([0.1, 0.2], 0.05)[0] == 0.0 and iv_rank([0.1, 0.2], 0.5)[0] == 100.0


def test_history_keeps_one_value_per_day_and_caps_its_length():
    h = IVHistory(max_len=3)
    for i in range(5):
        h.record(D0 + timedelta(days=i), 0.2 + i / 100)
    h.record(D0 + timedelta(days=4), 0.5)
    assert len(h) == 3 and h.rows()[-1] == (D0 + timedelta(days=4), 0.5)
    assert h.values(D0 + timedelta(days=3), 10) == [0.22, 0.23]
    with pytest.raises(DataContractError, match="ATM-IV"):
        h.record(D0, 0.0)


def test_metrics_switch_from_fallback_to_ivr_when_history_is_long_enough():
    h = history([0.2, 0.3, 0.4, 0.2])
    today = D0 + timedelta(days=10)
    short = IVMetrics.compute(0.3, h, today, 0.25, CFG)
    assert short.status is IVRStatus.INSUFFICIENT_HISTORY and short.ivr is None
    assert short.iv_rv_ratio == pytest.approx(1.2)
    h.record(D0 + timedelta(days=4), 0.3)
    full = IVMetrics.compute(0.3, h, today, None, CFG, term_slope=0.01, skew_25d=0.04)
    assert full.status is IVRStatus.OK and full.ivr == pytest.approx(50.0)
    assert full.iv_rv_ratio is None and full.term_slope == 0.01 and full.skew_25d == 0.04
    none = IVMetrics.compute(None, h, today, 0.2, CFG)
    assert none.status is IVRStatus.NO_ATM_IV and none.iv_rv_ratio is None


@pytest.mark.parametrize(("ivr", "cond", "ok"), [
    (30.0, IVCondition.HIGH, True), (29.0, IVCondition.HIGH, False),
    (30.0, IVCondition.LOW, True), (31.0, IVCondition.LOW, False),
    (20.0, IVCondition.VERY_LOW, True), (21.0, IVCondition.VERY_LOW, False),
])
def test_conditions_on_ivr(ivr, cond, ok):
    m = IVMetrics(0.2, ivr, 50.0, IVRStatus.OK, 1.0)
    assert check_iv_condition(m, cond, CFG) == (ok, False)


@pytest.mark.parametrize(("ratio", "cond", "ok"), [
    (1.20, IVCondition.HIGH, True), (1.19, IVCondition.HIGH, False),
    (1.00, IVCondition.LOW, True), (1.01, IVCondition.LOW, False),
    (0.90, IVCondition.VERY_LOW, True), (0.91, IVCondition.VERY_LOW, False),
])
def test_conditions_on_the_fallback_ratio_are_marked(ratio, cond, ok):
    m = IVMetrics(0.2, None, None, IVRStatus.INSUFFICIENT_HISTORY, ratio)
    assert check_iv_condition(m, cond, CFG) == (ok, True)


def test_without_ivr_and_without_fallback_nothing_is_met():
    m = IVMetrics(0.2, None, None, IVRStatus.INSUFFICIENT_HISTORY, 2.0)
    off = IVRuleSettings(fallback_enabled=False)
    assert check_iv_condition(m, IVCondition.HIGH, off) == (False, False)
    no_ratio = IVMetrics(0.2, None, None, IVRStatus.INSUFFICIENT_HISTORY, None)
    assert check_iv_condition(no_ratio, IVCondition.HIGH, CFG) == (False, False)


def test_realized_vol_matches_the_generating_vol():
    rng = np.random.default_rng(7)
    n, steps = 400, 50
    path = 100 * np.exp(np.cumsum(rng.normal(0, 0.2 / np.sqrt(252 * steps), n * steps)))
    days = path.reshape(n, steps)
    df = pd.DataFrame({"open": days[:, 0], "high": days.max(axis=1), "low": days.min(axis=1),
                       "close": days[:, -1]},
                      index=pd.date_range("2025-01-01", periods=n, freq="B"))
    rv = realized_vol(df, window=250)
    assert rv == pytest.approx(0.2, abs=0.04)
    assert realized_vol(df.iloc[:20], window=20) is None
    flat = df.assign(open=100.0, high=100.0, low=100.0, close=100.0)
    assert realized_vol(flat, window=20) is None


def test_rule_settings_come_from_the_config():
    s = iv_rule_settings(load_options_config().iv_metrics)
    assert s == IVRuleSettings()
