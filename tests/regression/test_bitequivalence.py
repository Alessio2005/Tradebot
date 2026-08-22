"""tests/regression/test_bitequivalence.py — Bit-equivalence regression tests.

These tests compare current output against frozen baselines stored in
tests/regression/baselines/.  They fail LOUDLY when a non-intentional
algorithm change is made.

To regenerate baselines after an intentional change:
    python apps/regenerate_baseline.py

Marked @pytest.mark.regression — run in nightly CI only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_BASELINES = Path(__file__).parent / "baselines"


@pytest.mark.regression
def test_gk_volatility_bitequivalence() -> None:
    """GK volatility must match the frozen baseline within 1e-9."""
    baseline_path = _BASELINES / "gk_volatility.npy"
    if not baseline_path.exists():
        pytest.skip("Baseline not generated yet — run apps/regenerate_baseline.py first.")

    from tradebot.volatility import get_garman_klass_volatility
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 500
    close = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998, "close": close
    })
    current = get_garman_klass_volatility(df, window=14).values
    baseline = np.load(baseline_path)
    np.testing.assert_allclose(current, baseline, atol=1e-9, equal_nan=True)


@pytest.mark.regression
def test_sequential_bootstrap_bitequivalence() -> None:
    """Sequential bootstrap must produce identical indices to baseline."""
    baseline_path = _BASELINES / "sequential_bootstrap_indices.npy"
    if not baseline_path.exists():
        pytest.skip("Baseline not generated yet — run apps/regenerate_baseline.py first.")

    from tradebot.cv import get_sequential_bootstrap_indices

    rng = np.random.default_rng(42)
    t0 = np.arange(400, dtype=np.int64)
    t1 = t0 + rng.integers(5, 20, size=400).astype(np.int64)
    t1 = np.minimum(t1, 499)
    current = get_sequential_bootstrap_indices(t0, t1, n_draws=400, seed=42)
    baseline = np.load(baseline_path)
    np.testing.assert_array_equal(current, baseline)
