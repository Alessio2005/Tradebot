"""tests/unit/test_propfirm_governor.py — propfirm governor + cross-account.

Covers the governor-retrofit from artefacts/PROPFIRM_PARALLEL_AUDIT.md:
daily-loss + static-max-DD enforcement with hysteresis, FAIL latching,
challenge/funded vol-target sizing, and the cross-account correlation throttle.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.monitoring.cross_account import CrossAccountConfig, CrossAccountMonitor
from tradebot.risk.daily_loss_governor import (
    GovernorAction,
    PropfirmGovernor,
    PropfirmLimits,
    Regime,
    RegimeConfig,
    target_gross_multiplier,
)

INIT = 100_000.0


def _gov() -> PropfirmGovernor:
    return PropfirmGovernor(PropfirmLimits(initial_balance=INIT))


# ============================================================================
# PropfirmLimits validation
# ============================================================================
def test_limits_validation():
    with pytest.raises(ValueError):
        PropfirmLimits(initial_balance=0)
    with pytest.raises(ValueError):  # daily_hard >= firm_daily_loss
        PropfirmLimits(initial_balance=INIT, daily_hard=0.06, firm_daily_loss=0.05)
    with pytest.raises(ValueError):  # dd_resume >= dd_trigger
        PropfirmLimits(initial_balance=INIT, dd_resume=0.07, dd_trigger=0.06)


# ============================================================================
# Daily-loss governor
# ============================================================================
def test_ok_when_small_loss():
    g = _gov()
    g.start_day(INIT)
    d = g.update(98_500)  # -1.5% day, -1.5% static
    assert d.action is GovernorAction.OK
    assert d.sizing_mult == 1.0 and d.allow_risk_increase


def test_daily_soft_stop():
    g = _gov()
    g.start_day(INIT)
    d = g.update(INIT * (1 - 0.036))  # -3.6% > soft 3.5%, < hard, static < 6%
    assert d.action is GovernorAction.SOFT_STOP
    assert d.sizing_mult == 0.0 and not d.allow_risk_increase


def test_daily_hard_flatten_latches_until_reset():
    g = _gov()
    g.start_day(INIT)
    assert g.update(INIT * (1 - 0.046)).action is GovernorAction.HARD_FLATTEN
    # Recovery within the SAME day stays halted (latched).
    assert g.update(INIT * 0.999).action is GovernorAction.HARD_FLATTEN
    # New day reset clears the daily latch.
    g.reset_day(INIT * 0.999)
    assert g.update(INIT * 0.999).action is GovernorAction.OK


# ============================================================================
# Static max-DD breaker (vs INITIAL balance, with hysteresis)
# ============================================================================
def test_static_dd_trip_and_resume():
    g = _gov()
    # New day at 93.5k so daily loss is tiny; static_dd vs 100k dominates.
    g.start_day(93_500)
    d = g.update(93_000)            # static_dd = 7% >= trigger 6%, daily 0.5%
    assert d.action is GovernorAction.HARD_FLATTEN
    assert d.static_dd == pytest.approx(0.07, abs=1e-6)
    # Still tripped between resume and trigger.
    assert g.update(95_500).action is GovernorAction.HARD_FLATTEN  # static 4.5%
    # Recovers above the resume line (static_dd <= 3%) → resumes.
    assert g.update(97_500).action is GovernorAction.OK            # static 2.5%


def test_fail_latches_permanently():
    g = _gov()
    g.start_day(INIT)
    d = g.update(89_000)  # static_dd 11% >= firm 10%
    assert d.action is GovernorAction.FAIL
    # Even full recovery cannot un-fail a dead account.
    assert g.update(INIT).action is GovernorAction.FAIL


def test_firm_daily_breach_is_fail():
    g = _gov()
    g.start_day(INIT)
    assert g.update(INIT * (1 - 0.051)).action is GovernorAction.FAIL


def test_static_dd_dominates_when_more_severe():
    # daily would be SOFT but static is FAIL-level → FAIL wins.
    g = _gov()
    g.start_day(96_000)
    d = g.update(89_500)  # daily 6.8%>firm? (96->89.5 = 6.77% > 5% firm) → FAIL anyway
    assert d.action is GovernorAction.FAIL


# ============================================================================
# Vol-target / regime sizing
# ============================================================================
def test_target_gross_multiplier_basic():
    m = target_gross_multiplier(0.09, 0.18)  # want 9% vol, realised 18% → 0.5×
    assert m == pytest.approx(0.5)


def test_target_gross_multiplier_zero_vol_returns_floor():
    assert target_gross_multiplier(0.09, 0.0) == 0.0


def test_target_gross_multiplier_capped():
    assert target_gross_multiplier(0.09, 0.001, cap=3.0) == 3.0


def test_challenge_regime_doubles_size():
    funded = target_gross_multiplier(0.09, 0.18, RegimeConfig.funded())
    challenge = target_gross_multiplier(0.09, 0.18, RegimeConfig.challenge())
    assert challenge == pytest.approx(2.0 * funded)
    assert RegimeConfig.challenge().regime is Regime.CHALLENGE


# ============================================================================
# Cross-account correlation throttle
# ============================================================================
def test_cross_account_no_throttle_when_uncorrelated():
    # Deterministic orthogonal patterns: products over each 4-period sum to 0
    # → exact zero correlation (no RNG sampling-noise flakiness).
    a_pat = [1.0, 1.0, -1.0, -1.0]
    b_pat = [1.0, -1.0, -1.0, 1.0]
    mon = CrossAccountMonitor(CrossAccountConfig(corr_window=40, min_obs=10))
    for k in range(40):
        mon.update({"A": a_pat[k % 4], "B": b_pat[k % 4]})
    t = mon.throttles()
    assert t["A"] == 1.0 and t["B"] == 1.0


def test_cross_account_throttles_lower_sharpe():
    mon = CrossAccountMonitor(
        CrossAccountConfig(corr_window=40, corr_threshold=0.4, throttle_factor=0.5, min_obs=10)
    )
    rng = np.random.default_rng(1)
    for _ in range(40):
        shock = rng.normal()
        # A and B share the shock (highly correlated); B has a negative mean
        # (lower Sharpe), so B should be the one throttled.
        mon.update({"A": shock + 0.5, "B": shock - 0.5})
    t = mon.throttles()
    assert t["B"] == 0.5
    assert t["A"] == 1.0


def test_cross_account_needs_min_obs():
    mon = CrossAccountMonitor(CrossAccountConfig(corr_window=20, min_obs=10))
    for _ in range(5):  # below min_obs
        mon.update({"A": 1.0, "B": 1.0})
    assert mon.throttles() == {"A": 1.0, "B": 1.0}


def test_cross_account_config_validation():
    with pytest.raises(ValueError):
        CrossAccountConfig(throttle_factor=1.5)
    with pytest.raises(ValueError):
        CrossAccountConfig(corr_window=5, min_obs=10)
