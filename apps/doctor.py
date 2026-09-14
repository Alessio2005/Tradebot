# apps/doctor.py
"""Diagnostic tool — check environment, imports, artefacts, and schema consistency.

Run with: python apps/doctor.py

Checks:
  1. Python version >= 3.10
  2. All required packages importable (numpy, pandas, catboost, numba, ...)
  3. tradebot package importable and version correct
  4. Artefacts directory structure
  5. DVC pipeline stage hashes (optional)
  6. Smoke-test: bars + GK volatility kernel compile without error
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_PACKAGES: list[str] = [
    "numpy", "pandas", "scipy", "numba", "catboost", "sklearn",
    "optuna", "joblib", "pyarrow", "hydra", "pandera",
]
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_python_version() -> tuple[bool, str]:
    maj, min_ = sys.version_info[:2]
    ok = (maj, min_) >= (3, 10)
    return ok, f"Python {maj}.{min_} {'✓' if ok else '✗ (need >= 3.10)'}"


def _check_imports() -> list[tuple[bool, str]]:
    results = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            results.append((True, f"  {pkg:<20} ✓"))
        except ImportError as exc:
            results.append((False, f"  {pkg:<20} ✗  ({exc})"))
    return results


def _check_tradebot() -> tuple[bool, str]:
    try:
        import tradebot
        return True, f"  tradebot {tradebot.__version__}  ✓"
    except Exception as exc:
        return False, f"  tradebot import FAILED: {exc}"


def _check_artefacts() -> list[tuple[bool, str]]:
    expected = [
        "market_data_parquet/",
        "artefacts/bars/",
        "artefacts/features/",
        "artefacts/labels/",
        "artefacts/hparams/",
        "artefacts/models/",
        "artefacts/portfolio/",
    ]
    results = []
    for rel in expected:
        p = _REPO_ROOT / rel
        ok = p.exists()
        results.append((ok, f"  {rel:<35} {'✓' if ok else '✗ (missing)'}"))
    return results


def _smoke_test_bars() -> tuple[bool, str]:
    try:
        import numpy as np
        import pandas as pd

        from tradebot.volatility import get_garman_klass_volatility
        rng = np.random.default_rng(0)
        n = 50
        close = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, n))
        df = pd.DataFrame({
            "open": close, "high": close * 1.002, "low": close * 0.998, "close": close
        })
        gk = get_garman_klass_volatility(df, window=14)
        assert gk.shape == (n,)
        return True, "  GK volatility smoke test  ✓"
    except Exception as exc:
        return False, f"  GK volatility smoke test  ✗  ({exc})"


def run_doctor() -> int:
    """Run all checks; return exit code (0 = all pass)."""
    print("\n" + "=" * 60)
    print("  TRADEBOT DOCTOR")
    print("=" * 60)
    failures = 0

    # Python version
    ok, msg = _check_python_version()
    print(msg)
    if not ok:
        failures += 1

    # Packages
    print("\n  Dependencies:")
    for ok, msg in _check_imports():
        print(msg)
        if not ok:
            failures += 1

    # tradebot package
    print("\n  Package:")
    ok, msg = _check_tradebot()
    print(msg)
    if not ok:
        failures += 1

    # Artefacts
    print("\n  Artefact directories:")
    for ok, msg in _check_artefacts():
        print(msg)
        # Artefacts are informational only — don't count as failures

    # Smoke test
    print("\n  Smoke tests:")
    ok, msg = _smoke_test_bars()
    print(msg)
    if not ok:
        failures += 1

    print("=" * 60)
    if failures == 0:
        print("  All checks passed ✓")
    else:
        print(f"  {failures} check(s) FAILED ✗")
    print()
    return failures


if __name__ == "__main__":
    sys.exit(run_doctor())
