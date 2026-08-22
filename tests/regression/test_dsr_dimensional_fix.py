"""Regression guard for the DSR dimensional fix (CHIEF 2026-06-08).

``metrics.deflated_sharpe`` previously subtracted the expected-max of N standard
normals (≈1.67 for N=12, in z units) directly from a per-observation Sharpe
(~0.07), giving z ≈ −68 and DSR ≡ 0 for EVERY realistic strategy — a silently
dead G2 gate.  The fix scales the expected-max by the Sharpe estimator SE
(σ_SR), so z = t_stat − e_max_z (Bailey & López de Prado 2014).

These tests pin the corrected behaviour so the bug cannot silently return.
"""
from __future__ import annotations

import math

from tradebot.backtest.metrics import deflated_sharpe


# 3-sleeve neutral book baseline: per-day SR=0.0658, 1815 daily obs, 12 trials.
_SR_DAY = 0.0658
_N_OBS = 1815


def test_dsr_is_not_dead_for_significant_strategy() -> None:
    """A t≈2.8 strategy must yield a meaningful DSR (~0.87), not ~0."""
    dsr = deflated_sharpe(_SR_DAY, n_trials=12, n_obs=_N_OBS)
    assert 0.80 < dsr < 0.95, dsr


def test_dsr_matches_textbook_t_minus_emax() -> None:
    """z must equal t_stat − e_max_z (the textbook DSR identity)."""
    from scipy import stats

    sr, n, N = _SR_DAY, _N_OBS, 12
    g, e = 0.5772156649015328, math.e
    e_max_z = (1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * e))
    sigma = math.sqrt((1.0 + 0.5 * sr * sr) / (n - 1))  # Gaussian Mertens
    expected = float(stats.norm.cdf(sr / sigma - e_max_z))
    got = deflated_sharpe(sr, n_trials=N, n_obs=n)
    assert abs(got - expected) < 1e-9, (got, expected)


def test_dsr_monotonic_decreasing_in_n_trials() -> None:
    """More tested hypotheses ⇒ harder to clear the deflation benchmark."""
    few = deflated_sharpe(_SR_DAY, n_trials=3, n_obs=_N_OBS)
    many = deflated_sharpe(_SR_DAY, n_trials=2000, n_obs=_N_OBS)
    assert few > many
    assert many < 0.5  # 2000 honest hypotheses should sink a t≈2.8 result


def test_dsr_strong_strategy_passes_gate() -> None:
    """A genuinely strong strategy (t≈5) clears the 0.95 promotion gate."""
    dsr = deflated_sharpe(0.118, n_trials=12, n_obs=_N_OBS)
    assert dsr > 0.95, dsr
