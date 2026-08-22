# apps/regenerate_baseline.py
"""Utility — regenerate deterministic test baselines for regression tests.

Run this whenever a deliberate algorithm change requires updating the
frozen golden values in tests/regression/baselines/.

Usage:
    python apps/regenerate_baseline.py

The script runs the same smoke pipeline as the regression tests and
serialises the output to Parquet + joblib in tests/regression/baselines/.
It prints a diff of the changed metrics so the author can verify the
change is intentional.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import joblib

logger = logging.getLogger(__name__)

# ── Baseline configuration ────────────────────────────────────────────────────
_SEED = 42
_N_BARS = 500
_BASELINES_DIR = Path(__file__).resolve().parents[1] / "tests" / "regression" / "baselines"


def _generate_smoke_data() -> dict:
    """Generate minimal deterministic smoke data for baseline generation.

    Uses the SAME construction as tests/regression/test_bitequivalence.py
    so that regenerated baselines always match the test harness.
    """
    rng = np.random.default_rng(_SEED)
    close = 30_000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, _N_BARS))
    # Fixed multipliers — identical to what the regression test uses.
    high  = close * 1.002
    low   = close * 0.998
    open_ = close
    volume = np.ones(_N_BARS) * 1_000.0
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def regenerate() -> None:
    """Regenerate all baseline artefacts and report changes."""
    import pandas as pd
    from tradebot.volatility import get_garman_klass_volatility
    from tradebot.cv.bootstrap import get_sequential_bootstrap_indices

    logging.basicConfig(level=logging.INFO)
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    data = _generate_smoke_data()
    df = pd.DataFrame(data)

    # ── Baseline 1: Garman-Klass volatility ──────────────────────────────────
    gk = get_garman_klass_volatility(df, window=14).values
    gk_path = _BASELINES_DIR / "gk_volatility.npy"
    if gk_path.exists():
        old = np.load(gk_path)
        max_diff = float(np.nanmax(np.abs(gk - old)))
        logger.info("GK volatility max delta: %.6e", max_diff)
    np.save(gk_path, gk)
    logger.info("Saved: %s", gk_path)

    # ── Baseline 2: Sequential bootstrap ─────────────────────────────────────
    # Uses default_rng (new-style) for t1 generation — same as the test file.
    rng = np.random.default_rng(_SEED)
    t0 = np.arange(400, dtype=np.int64)
    t1 = t0 + rng.integers(5, 20, size=400).astype(np.int64)
    t1 = np.minimum(t1, 499)
    idx = get_sequential_bootstrap_indices(t0, t1, n_draws=400, seed=_SEED)
    sb_path = _BASELINES_DIR / "sequential_bootstrap_indices.npy"
    if sb_path.exists():
        old_idx = np.load(sb_path)
        changed = int((idx != old_idx).sum())
        logger.info("Sequential bootstrap changed indices: %d / %d", changed, len(idx))
    np.save(sb_path, idx)
    logger.info("Saved: %s", sb_path)

    logger.info("Baseline regeneration complete — commit these files if changes are intentional.")


if __name__ == "__main__":
    regenerate()
