"""Causaliteit van het robuuste boek: de toekomst verstoren verandert geen besluit tot *t*.

Voor elke sleeve en de combinatie: bouw de doelgewichten op de echte (synthetische)
markt en op dezelfde markt met alles NA bar *t* vervangen door ruis. Tot en met *t*
moeten de gewichten bit-gelijk zijn. De negatieve controle bewijst dat de toets rood
kan worden: een signaal dat de bar erna leest, breekt hem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.systematic_fixtures import config, perturb_after, synthetic_market
from tradebot.systematic.evaluate import regime_masks
from tradebot.systematic.signals import known_carry, trend_score
from tradebot.systematic.sleeves import build_sleeve, combine
from tradebot.utils.failfast import DataContractError

CUTS = (200, 333, 480)
SLEEVES = ("C1_TREND_LS", "C2_TREND_LF", "C3_VOLMAN_CORE", "C4_CARRY_XS")


def _equal_through(a: pd.DataFrame, b: pd.DataFrame, t: int) -> bool:
    x = a.iloc[: t + 1].to_numpy()
    y = b.iloc[: t + 1].to_numpy()
    return bool(np.array_equal(np.isnan(x), np.isnan(y)) and np.allclose(
        np.nan_to_num(x), np.nan_to_num(y), rtol=0.0, atol=0.0))


@pytest.mark.parametrize("t", CUTS)
@pytest.mark.parametrize("name", SLEEVES)
def test_sleeve_weights_do_not_read_the_future(name, t):
    m = synthetic_market()
    cfg = config()
    base = build_sleeve(name, m, cfg).weights
    moved = build_sleeve(name, perturb_after(m, t), cfg).weights
    assert _equal_through(base, moved, t)
    # De verstoring is echt: na t verschillen de gewichten wel.
    assert not _equal_through(base, moved, len(m.index) - 1)


@pytest.mark.parametrize("t", CUTS)
def test_the_combination_does_not_read_the_future(t):
    m = synthetic_market()
    cfg = config()
    mm = perturb_after(m, t)
    a = combine("C5", [build_sleeve(s, m, cfg) for s in SLEEVES], m, cfg).weights
    b = combine("C5", [build_sleeve(s, mm, cfg) for s in SLEEVES], mm, cfg).weights
    assert _equal_through(a, b, t)


@pytest.mark.parametrize("t", CUTS)
def test_regime_masks_do_not_read_the_future(t):
    m = synthetic_market(n_bars=900)
    a = regime_masks(m, m.index)
    b = regime_masks(perturb_after(m, t), m.index)
    for k in a:
        # Het regime van bar t+1 is bekend op de close van t.
        assert a[k].iloc[: t + 2].equals(b[k].iloc[: t + 2]), k


def test_negative_control_a_signal_that_reads_tomorrow_is_caught():
    m = synthetic_market()
    cfg = config()
    t = 333
    leaky = trend_score(m.close.shift(-1), m.sigma_daily, lookbacks=cfg.trend.lookbacks,
                        z_clip=cfg.trend.z_clip)
    mm = perturb_after(m, t)
    leaky_moved = trend_score(mm.close.shift(-1), mm.sigma_daily,
                              lookbacks=cfg.trend.lookbacks, z_clip=cfg.trend.z_clip)
    assert not _equal_through(leaky, leaky_moved, t)


def test_carry_uses_only_settlements_before_the_decision():
    m = synthetic_market()
    listed = m.close.notna()
    t = 300
    base = known_carry(m.funding, listed, window_bars=7, signal_lag_bars=1)
    bumped = m.funding.copy()
    bumped.iloc[t] = bumped.iloc[t] + 1.0  # een enorme afrekening op bar t
    moved = known_carry(bumped, listed, window_bars=7, signal_lag_bars=1)
    assert _equal_through(base, moved, t)
    assert not np.allclose(base.iloc[t + 1].dropna(), moved.iloc[t + 1].dropna())


def test_carry_without_a_lag_is_refused():
    m = synthetic_market()
    with pytest.raises(DataContractError):
        known_carry(m.funding, m.close.notna(), window_bars=7, signal_lag_bars=0)
