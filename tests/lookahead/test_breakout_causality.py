"""De richting op bar t leest niets van na t, en de drempel op t leest sigma van t-1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.labeling.breakout import breakout_side
from tradebot.labeling.cusum import directional_cusum_filter


@pytest.fixture
def walk() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 400))), index=idx)
    sigma = pd.Series(rng.uniform(0.015, 0.03, 400), index=idx)
    return close, sigma


@pytest.mark.parametrize("t", [60, 150, 300])
def test_the_side_on_bar_t_ignores_the_future(walk, t) -> None:
    close, sigma = walk
    full = breakout_side(close, sigma, 2.0)
    cut = breakout_side(close.iloc[: t + 1], sigma.iloc[: t + 1], 2.0)
    pd.testing.assert_series_equal(full.iloc[: t + 1], cut)


@pytest.mark.parametrize("t", [60, 150, 300])
def test_the_threshold_on_bar_t_uses_sigma_of_t_minus_one(walk, t) -> None:
    close, sigma = walk
    moved = sigma.copy()
    moved.iloc[t] *= 100.0
    a = breakout_side(close, sigma, 2.0)
    b = breakout_side(close, moved, 2.0)
    pd.testing.assert_series_equal(a.iloc[: t + 1], b.iloc[: t + 1])


def test_the_negative_control_reads_its_own_sigma_and_breaks(walk) -> None:
    """Een drempel op sigma[t] in plaats van sigma[t-1] moet door de tweede test worden gevangen."""
    close, sigma = walk
    p = np.log(close.to_numpy())

    def leaky(s: pd.Series) -> np.ndarray:
        thr = 2.0 * s.to_numpy()
        up, down = directional_cusum_filter(p, thr, thr)
        out = np.zeros(p.size)
        out[up], out[down] = 1.0, -1.0
        return out

    moved = sigma.copy()
    changed = 0
    for t in range(60, 380, 7):
        moved.iloc[:] = sigma.to_numpy()
        moved.iloc[t] *= 100.0
        changed += int(not np.array_equal(leaky(sigma)[: t + 1], leaky(moved)[: t + 1]))
    assert changed > 0
