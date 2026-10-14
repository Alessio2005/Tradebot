"""Truncation invariance of `HurstComputer.compute_rolling`.

The Hurst value on bar t may only depend on bars <= t. Compute it on a prefix of
a series and on the full series: the shared bars must be bit-identical.

`compute_rolling` used to shrink its window to `max(32, len(prices) // 2)` when
the input was shorter than `window_size`, so on short inputs the value on bar t
depended on how many bars came after t. `feat_hurst_meso` (window 700) runs
through exactly that path whenever fewer than 700 meso bars are available.

Burn-in contract pinned here: the window never changes with the input length;
bars before the first full window carry the neutral value 0.5, the same value
`_rolling_hurst_njit` already uses for every pre-window bar and for degenerate
fits.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.features.regime import HurstComputer

pytestmark = pytest.mark.lookahead

NEUTRAL = 0.5

#: (window_size, cut). The first four are shorter than the window (the regime
#: that used to leak); the last two sit on / past the window (already causal).
CUTS = [
    pytest.param(700, 300, id="meso-300-bars"),
    pytest.param(700, 699, id="meso-one-bar-short"),
    pytest.param(250, 120, id="macro-120-bars"),
    pytest.param(12, 15, id="window-below-the-20-bar-floor"),
    pytest.param(700, 700, id="meso-exactly-one-window"),
    pytest.param(700, 850, id="meso-past-the-window"),
]


def _prices(n: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n)))


def _computer() -> HurstComputer:
    # Same construction as FeaturePipeline.
    return HurstComputer(min_window=700, num_lags=20)


@pytest.mark.parametrize(("window_size", "cut"), CUTS)
def test_value_on_bar_t_does_not_depend_on_later_bars(window_size: int, cut: int) -> None:
    prices = _prices(max(cut, window_size) + 150)
    short = _computer().compute_rolling(prices[:cut], window_size)
    full = _computer().compute_rolling(prices, window_size)
    np.testing.assert_array_equal(
        short, full[:cut],
        err_msg=f"Hurst on prices[:{cut}] differs from the same bars of the "
                f"{len(prices)}-bar series (window_size={window_size}).",
    )


def test_every_prefix_is_a_prefix_of_the_full_series() -> None:
    """Truncate at every possible length, not just a few cut points."""
    window_size = 40
    prices = _prices(120)
    computer = _computer()
    full = computer.compute_rolling(prices, window_size)
    for n in range(1, len(prices) + 1):
        np.testing.assert_array_equal(
            computer.compute_rolling(prices[:n], window_size), full[:n],
            err_msg=f"prefix length {n}",
        )


def test_series_shorter_than_the_window_is_all_burn_in() -> None:
    out = _computer().compute_rolling(_prices(300), 700)
    assert out.shape == (300,)
    assert (out == NEUTRAL).all()


def test_estimates_start_on_the_first_full_window() -> None:
    """Control for the tests above: the truncation comparison is not vacuous.

    On a series longer than the window the burn-in is neutral and real
    estimates follow, so a prefix that ends inside the burn-in is being
    compared against genuine 0.5s, and one that ends past it against estimates.
    """
    window_size = 700
    out = _computer().compute_rolling(_prices(1000), window_size)
    assert (out[: window_size - 1] == NEUTRAL).all()
    assert (out[window_size - 1:] != NEUTRAL).any()
    assert np.isfinite(out).all()
    assert ((out >= 0.0) & (out <= 1.0)).all()
