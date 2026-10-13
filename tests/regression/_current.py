"""The inputs and the calls behind the frozen baselines in `tests/regression/baselines/`.

`test_bitequivalence.py` compares these functions with the committed arrays and
`regenerate_baselines.py` writes the arrays from them, so the two cannot drift
apart: a baseline is always the output of exactly the call the test repeats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GK_BASELINE = "gk_volatility.npy"
SB_BASELINE = "sequential_bootstrap_indices.npy"


def current_gk_volatility() -> np.ndarray:
    """Garman-Klass volatility (window 14) of 500 seeded synthetic bars."""
    from tradebot.volatility import get_garman_klass_volatility

    rng = np.random.default_rng(42)
    n = 500
    close = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "open": close, "high": close * 1.002, "low": close * 0.998, "close": close
    })
    return get_garman_klass_volatility(df, window=14).to_numpy()


def current_sequential_bootstrap() -> np.ndarray:
    """Sequential-bootstrap indices (400 draws, seed 42) over 400 overlapping events."""
    from tradebot.cv import get_sequential_bootstrap_indices

    rng = np.random.default_rng(42)
    t0 = np.arange(400, dtype=np.int64)
    t1 = t0 + rng.integers(5, 20, size=400).astype(np.int64)
    t1 = np.minimum(t1, 499)
    return get_sequential_bootstrap_indices(t0, t1, n_draws=400, seed=42)
