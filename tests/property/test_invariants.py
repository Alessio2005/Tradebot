"""tests/property/test_invariants.py — Property-based invariant tests.

Tests mathematical invariants that must hold for all valid inputs.
Uses hypothesis (if available) or falls back to parameterised numpy RNG.
"""
from __future__ import annotations

import numpy as np
import pytest


def _random_returns(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.02, n)


@pytest.mark.parametrize("seed", range(5))
def test_var_leq_cvar(seed: int) -> None:
    """CVaR must always be >= VaR for the same confidence level."""
    from tradebot.risk import historical_cvar, historical_var
    r = _random_returns(500, seed)
    var  = historical_var(r, confidence=0.95)
    cvar = historical_cvar(r, confidence=0.95)
    assert cvar >= var - 1e-9, f"CVaR ({cvar:.6f}) < VaR ({var:.6f}) for seed={seed}"


@pytest.mark.parametrize("seed", range(5))
def test_gk_nonnegative(seed: int) -> None:
    """GK volatility must be non-negative for any OHLCV input."""
    import pandas as pd

    from tradebot.volatility import get_garman_klass_volatility
    rng = np.random.default_rng(seed)
    n = 100
    close = 10_000 * np.cumprod(1 + rng.normal(0, 0.02, n))
    df = pd.DataFrame({
        "open": close, "high": close * 1.003, "low": close * 0.997, "close": close
    })
    vol = get_garman_klass_volatility(df, window=14)
    assert vol.dropna().ge(0).all()


@pytest.mark.parametrize("kelly_div", [1, 2, 4, 8])
def test_kelly_decreases_with_divisor(kelly_div: int) -> None:
    """Higher Kelly divisor must produce smaller fraction."""
    from tradebot.risk import kelly_fraction
    f_full  = kelly_fraction(mu=0.01, sigma_sq=0.005, kelly_divisor=1)
    f_frac  = kelly_fraction(mu=0.01, sigma_sq=0.005, kelly_divisor=kelly_div)
    assert f_frac <= f_full + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_sb_indices_always_valid(seed: int) -> None:
    """Sequential bootstrap indices must always be in [0, n_samples)."""
    from tradebot.cv import get_sequential_bootstrap_indices
    rng = np.random.default_rng(seed)
    n = 50
    t0 = np.sort(rng.integers(0, 200, n)).astype(np.int64)
    t1 = (t0 + rng.integers(1, 15, n)).astype(np.int64)
    idx = get_sequential_bootstrap_indices(t0, t1, n_draws=n)
    assert (idx >= 0).all() and (idx < n).all(), f"Out-of-range indices for seed={seed}"
