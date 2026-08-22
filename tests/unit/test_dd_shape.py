# tests/unit/test_dd_shape.py
"""Guards on the Wave-28 kill-gate re-derivation.

The reference distribution IS a simulation, so these tests check the
properties a gate depends on — determinism, monotonicity, the stated
false-rejection budget — rather than a fitted constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.backtest.dd_shape import (
    calmar_ceiling,
    dd_shape_reference,
    expected_max_dd_over_vol,
    max_dd_over_vol_quantile,
)

TRADING_DAYS = 252
PANEL_YEARS = 5685 / TRADING_DAYS  # the registered cross-asset panel, 22.6y


def test_reference_is_deterministic() -> None:
    """R-5: same inputs, bit-identical distribution."""
    a = dd_shape_reference(0.5, PANEL_YEARS, 0.042)
    b = dd_shape_reference(0.5, PANEL_YEARS, 0.042)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("q", [0.90, 0.95, 0.99])
def test_false_rejection_budget_is_what_it_claims(q: float) -> None:
    """The whole point: a gate at quantile q rejects (1-q) of GOOD strategies.

    The old gate thresholded a realised path against a mean-derived constant
    and so rejected 65-99% of qualifying strategies. This asserts the new
    threshold does what it says.
    """
    sample = dd_shape_reference(0.5, PANEL_YEARS, 0.042)
    cap = max_dd_over_vol_quantile(0.5, PANEL_YEARS, q, 0.042)
    rejected = float((sample > cap).mean())
    assert rejected == pytest.approx(1.0 - q, abs=0.01)


def test_max_drawdown_grows_with_horizon() -> None:
    """The defect that started this: MaxDD is unbounded in T, not stationary."""
    short = expected_max_dd_over_vol(0.5, 5.0, 0.042)
    long = expected_max_dd_over_vol(0.5, 40.0, 0.042)
    assert long > short


def test_max_drawdown_falls_with_skill() -> None:
    prev = np.inf
    for s in (0.4, 0.6, 0.8, 1.0):
        cur = expected_max_dd_over_vol(s, PANEL_YEARS, 0.042)
        assert cur < prev
        prev = cur


def test_stationary_formula_understates_the_maximum() -> None:
    """Regression on the actual bug: sigma^2/(2mu) is NOT the expected maximum.

    ``1/(2S)`` is the mean drawdown at a RANDOM TIME. Using it as a maximum is
    what made a Calmar floor of 0.25 look feasible at Sharpe 0.40.
    """
    for s in (0.4, 0.6, 1.0):
        stationary = 1.0 / (2.0 * s)
        maximum = expected_max_dd_over_vol(s, PANEL_YEARS, 0.042)
        assert maximum > 1.8 * stationary, (
            f"S={s}: maximum {maximum:.2f} vs stationary {stationary:.2f}"
        )


def test_corrected_ceiling_flags_kg_b1_calmar_floor_as_infeasible() -> None:
    """KG-B1 demanded Calmar >= 0.25 at a Sharpe floor of 0.40.

    The old ceiling (2*S^2 = 0.32) said that was satisfiable. It was not.
    """
    ceiling = calmar_ceiling(0.40, PANEL_YEARS, ann_vol=0.042)
    assert ceiling < 0.25, f"corrected ceiling {ceiling:.3f} should be below 0.25"
    assert 2.0 * 0.40**2 > 0.25, "the old ceiling did NOT flag it — that was the bug"


def test_old_gate_rejects_almost_every_qualifying_strategy() -> None:
    """Quantify the pre-registered gate's Type-I error at its own Sharpe floor."""
    sample = dd_shape_reference(0.40, PANEL_YEARS, 0.042)
    calmar = (0.40 - 0.042 / 2.0) / sample
    rejected = float(((sample > 2.5) | (calmar < 0.25)).mean())
    assert rejected > 0.95, (
        f"expected the old KG-B1 shape gate to reject >95% of genuinely "
        f"Sharpe-0.40 strategies, measured {rejected:.1%}"
    )


def test_shape_is_approximately_leverage_invariant_at_moderate_vol() -> None:
    """Scaling a book moves drawdown and vol together.

    Exact only for arithmetic returns; compounding drag breaks it at high vol,
    which is why the reference carries ``ann_vol`` explicitly.
    """
    a = expected_max_dd_over_vol(0.6, PANEL_YEARS, 0.042)
    b = expected_max_dd_over_vol(0.6, PANEL_YEARS, 0.10)
    assert a == pytest.approx(b, rel=0.10)
