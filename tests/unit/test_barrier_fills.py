"""Spec §12: een 1:1-trade vult op het barrièreniveau, niet op de slotkoers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tradebot.labeling.barrier_fills import (
    barrier_fill_returns,
    select_events,
    with_cost_aware_target,
)
from tradebot.labeling.vol_barriers import label_triple_barrier
from tradebot.schemas.config import LabelingConfig

CFG = LabelingConfig(profit_target_sigma=math.sqrt(5.0), stop_loss_sigma=math.sqrt(5.0),
                     horizon_bars=10, entry_lag_bars=1, min_sigma_obs=60)
SIG = 0.02
B = math.sqrt(5.0) * SIG
SLIP = 5.0


def _one_event(o2: float, h2: float, l2: float, c2: float, side: float = 1.0):
    n = 14
    o, h, l, c = (np.full(n, 100.0), np.full(n, 100.2), np.full(n, 99.8), np.full(n, 100.0))
    o[2], h[2], l[2], c[2] = o2, h2, l2, c2
    s = np.zeros(n)
    s[0] = side
    labels = label_triple_barrier(h, l, c, np.full(n, SIG), s, CFG)
    assert len(labels) == 1
    return labels, o, c


def test_a_long_target_fills_at_the_level() -> None:
    labels, o, c = _one_event(100.1, 105.0, 99.9, 104.0)
    assert labels.barrier_outcome[0] == 1
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(B)


def test_a_long_stop_fills_at_the_level_minus_slippage() -> None:
    labels, o, c = _one_event(100.1, 100.1, 95.0, 96.0)
    assert labels.barrier_outcome[0] == -1
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] == pytest.approx((1.0 - B) * (1.0 - SLIP / 1e4) - 1.0)


def test_a_gap_through_the_target_fills_at_the_open() -> None:
    labels, o, c = _one_event(106.0, 107.0, 105.5, 106.0)
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(0.06)


def test_a_gap_through_the_stop_fills_at_the_open_minus_slippage() -> None:
    labels, o, c = _one_event(94.0, 94.5, 93.0, 94.0)
    assert labels.barrier_outcome[0] == -1
    ret = barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)
    assert ret[0] == pytest.approx(0.94 * (1.0 - SLIP / 1e4) - 1.0)


def test_both_barriers_in_one_bar_is_a_stop() -> None:
    labels, o, c = _one_event(100.0, 105.0, 95.0, 100.0)
    assert labels.barrier_outcome[0] == -1
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] < 0.0


def test_the_vertical_barrier_exits_at_the_close() -> None:
    labels, o, c = _one_event(100.0, 100.2, 99.8, 100.0)
    assert labels.barrier_outcome[0] == 0
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(0.0)


def test_a_short_is_the_mirror_image() -> None:
    labels, o, c = _one_event(99.9, 100.1, 95.0, 96.0, side=-1.0)
    assert labels.barrier_outcome[0] == 1
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(B)
    labels, o, c = _one_event(100.1, 105.0, 99.9, 104.0, side=-1.0)
    assert labels.barrier_outcome[0] == -1
    expected = -((1.0 + B) * (1.0 + SLIP / 1e4) - 1.0)
    assert barrier_fill_returns(labels, o, c, CFG, stop_slippage_bps=SLIP)[0] == pytest.approx(expected)


def test_the_target_is_net_of_the_trade_specific_cost() -> None:
    labels, _, _ = _one_event(100.0, 100.2, 99.8, 100.0)
    out = with_cost_aware_target(labels, np.array([0.0012]), costs=np.array([0.0013]))
    assert out.meta_label[0] == 0 and out.realized_return[0] == pytest.approx(0.0012)
    out = with_cost_aware_target(labels, np.array([0.0014]), costs=np.array([0.0013]))
    assert out.meta_label[0] == 1


def test_select_events_keeps_every_field_aligned() -> None:
    labels, _, _ = _one_event(100.0, 100.2, 99.8, 100.0)
    assert len(select_events(labels, np.array([True]))) == 1
    assert len(select_events(labels, np.array([False]))) == 0
