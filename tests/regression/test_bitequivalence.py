"""tests/regression/test_bitequivalence.py — Bit-equivalence regression tests.

These tests compare current output against frozen baselines stored in
tests/regression/baselines/ (committed to git; `.gitignore` re-includes them).
They fail LOUDLY when a non-intentional algorithm change is made, and just as
loudly when a baseline is missing: a measuring stick that is not there is a
red test, not a silent skip (DI-25).

To regenerate baselines after an intentional change:
    python -m tests.regression.regenerate_baselines

Marked @pytest.mark.regression — excluded from the fast run, executed by the
slow job in ci.yml and by the nightly workflow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.regression._current import (
    GK_BASELINE,
    SB_BASELINE,
    current_gk_volatility,
    current_sequential_bootstrap,
)

_BASELINES = Path(__file__).parent / "baselines"


def _load(name: str) -> np.ndarray:
    path = _BASELINES / name
    assert path.is_file(), (
        f"Frozen baseline {name} is missing from {_BASELINES}. "
        "Regenerate it with `python -m tests.regression.regenerate_baselines` "
        "and commit it; the bit-equivalence test has nothing to measure against."
    )
    return np.load(path)


@pytest.mark.regression
def test_gk_volatility_bitequivalence() -> None:
    """GK volatility must match the frozen baseline within 1e-9."""
    np.testing.assert_allclose(
        current_gk_volatility(), _load(GK_BASELINE), atol=1e-9, equal_nan=True)


@pytest.mark.regression
def test_sequential_bootstrap_bitequivalence() -> None:
    """Sequential bootstrap must produce identical indices to baseline."""
    np.testing.assert_array_equal(current_sequential_bootstrap(), _load(SB_BASELINE))


@pytest.mark.regression
def test_a_missing_baseline_is_a_failure_not_a_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check can go red: with no baseline on disk the loader refuses to continue."""
    monkeypatch.setattr(f"{__name__}._BASELINES", tmp_path)
    with pytest.raises(AssertionError, match="is missing"):
        _load(GK_BASELINE)
