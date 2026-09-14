"""tests/unit/test_risk.py — Unit tests for risk sub-package."""
from __future__ import annotations

import numpy as np

from tradebot.risk import (
    BreakerState,
    DrawdownBreaker,
    DrawdownConfig,
    PositionLimits,
    check_position_limits,
    gap_risk_kelly_size,
    historical_cvar,
    historical_var,
    kelly_fraction,
    meta_label_kelly,
)

# ── Kelly ─────────────────────────────────────────────────────────────────────

def test_kelly_fraction_basic() -> None:
    f = kelly_fraction(mu=0.01, sigma_sq=0.01)
    assert 0.0 < f <= 0.25


def test_kelly_fraction_zero_variance() -> None:
    assert kelly_fraction(mu=0.01, sigma_sq=0.0) == 0.0


def test_kelly_negative_edge_zero() -> None:
    assert kelly_fraction(mu=-0.01, sigma_sq=0.01) == 0.0


def test_gap_risk_kelly_reduces_size_with_jumps() -> None:
    base = kelly_fraction(mu=0.005, sigma_sq=0.002)
    adj  = gap_risk_kelly_size(expected_alpha=0.005, vol=0.04, jump_sigma=0.10)
    assert adj < base


def test_meta_label_below_threshold_zero() -> None:
    assert meta_label_kelly(0.2, meta_prob=0.3, meta_threshold=0.5) == 0.0


def test_meta_label_above_threshold_positive() -> None:
    size = meta_label_kelly(0.2, meta_prob=0.8, meta_threshold=0.5)
    assert 0 < size <= 0.2


# ── VaR ───────────────────────────────────────────────────────────────────────

def test_historical_var_positive() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, 1000)
    var = historical_var(returns, confidence=0.95)
    assert var > 0


def test_historical_cvar_geq_var() -> None:
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, 1000)
    var = historical_var(returns, confidence=0.95)
    cvar = historical_cvar(returns, confidence=0.95)
    assert cvar >= var


# ── Drawdown breaker ─────────────────────────────────────────────────────────

def test_drawdown_breaker_trips_at_threshold() -> None:
    config = DrawdownConfig(trigger_threshold=0.10, resume_threshold=0.05)
    breaker = DrawdownBreaker(config)
    # Equity falls 15%
    equity = np.array([100.0, 98.0, 95.0, 90.0, 85.0])
    for eq in equity[:-1]:
        breaker.update(equity[: np.where(equity == eq)[0][0] + 1])
    scale = breaker.update(equity)
    assert breaker.state == BreakerState.TRIPPED
    assert scale == config.scale_factor


def test_drawdown_breaker_resumes() -> None:
    config = DrawdownConfig(trigger_threshold=0.10, resume_threshold=0.05, scale_factor=0.0)
    breaker = DrawdownBreaker(config)
    equity_down = np.array([100.0, 85.0])
    breaker.update(equity_down)
    assert breaker.state == BreakerState.TRIPPED
    equity_recover = np.array([100.0, 85.0, 90.0, 96.0])  # DD = 4% < 5% resume
    breaker.update(equity_recover)
    assert breaker.state == BreakerState.OPEN


# ── Position limits ──────────────────────────────────────────────────────────

def test_no_violations_within_limits() -> None:
    # max_net_imbalance=1.0 disables the side-symmetry rule so we test only
    # the per-asset cap and portfolio gross cap in isolation.
    limits = PositionLimits(per_asset_cap=2.0, portfolio_gross_cap=3.0, max_net_imbalance=1.0)
    violations = check_position_limits({"BTC": 1.0, "ETH": 0.5}, limits)
    assert violations == []


def test_per_asset_cap_violation() -> None:
    limits = PositionLimits(per_asset_cap=1.0)
    violations = check_position_limits({"BTC": 2.0}, limits)
    assert any(v.rule == "per_asset_cap" for v in violations)
