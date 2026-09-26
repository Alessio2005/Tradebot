"""De richting op bar t leest niets van na t, en de drempel op t leest sigma van t-1.

Beide eigenschappen worden bewezen met een dichte sweep over (bijna) elke bar,
niet met een handvol steekproefpunten: een lek dat maar op een paar specifieke
bars zichtbaar wordt, mag de test niet stilzwijgend passeren. De negatieve
controle (`_leaky_side`) is bewust identiek aan `breakout_side`, op de drempel
na — die leest `sigma[t]` in plaats van `sigma[t-1]` — en moet door precies
dezelfde check (`_sigma_breaks`) worden gevangen die de echte functie vrijpleit.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from tradebot.labeling.breakout import breakout_side
from tradebot.labeling.cusum import directional_cusum_filter

BreakoutFn = Callable[[pd.Series, pd.Series, float], pd.Series]


@pytest.fixture
def walk() -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(5)
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 400))), index=idx)
    sigma = pd.Series(rng.uniform(0.015, 0.03, 400), index=idx)
    return close, sigma


def _leaky_side(close: pd.Series, sigma_daily: pd.Series, k: float) -> pd.Series:
    """Als `breakout_side`, maar de drempel op t leest sigma[t] in plaats van sigma[t-1]."""
    log_p = np.log(close.to_numpy(dtype=np.float64))
    thr = k * sigma_daily.to_numpy(dtype=np.float64)  # LEK: geen .shift(1)
    up, down = directional_cusum_filter(log_p, thr, thr)
    side = np.zeros(len(close), dtype=np.float64)
    side[up] = 1.0
    side[down] = -1.0
    return pd.Series(side, index=close.index, name="side")


def _sigma_breaks(fn: BreakoutFn, close: pd.Series, sigma: pd.Series, ts: range) -> list[int]:
    """Elke t in `ts` waarop het x100 verstoren van sigma[t] `fn(...)[: t + 1]` verandert."""
    base = fn(close, sigma, 2.0)
    breaks = []
    for t in ts:
        moved = sigma.copy()
        moved.iloc[t] *= 100.0
        moved_out = fn(close, moved, 2.0)
        if not base.iloc[: t + 1].equals(moved_out.iloc[: t + 1]):
            breaks.append(t)
    return breaks


def _future_breaks(fn: BreakoutFn, close: pd.Series, sigma: pd.Series, ts: range) -> list[int]:
    """Elke t in `ts` waarop `fn` op de afgeknotte reeks verschilt van de volledige reeks t/m t."""
    full = fn(close, sigma, 2.0)
    breaks = []
    for t in ts:
        cut = fn(close.iloc[: t + 1], sigma.iloc[: t + 1], 2.0)
        if not full.iloc[: t + 1].equals(cut):
            breaks.append(t)
    return breaks


def test_the_side_on_bar_t_ignores_the_future(walk) -> None:
    close, sigma = walk
    assert _future_breaks(breakout_side, close, sigma, range(60, 400, 5)) == []


def test_the_threshold_on_bar_t_uses_sigma_of_t_minus_one(walk) -> None:
    close, sigma = walk
    assert _sigma_breaks(breakout_side, close, sigma, range(61, 380)) == []


def test_the_negative_control_reads_its_own_sigma_and_breaks(walk) -> None:
    """Een drempel op sigma[t] i.p.v. sigma[t-1] moet door dezelfde check worden gevangen."""
    close, sigma = walk
    assert _sigma_breaks(_leaky_side, close, sigma, range(61, 380)) != []
