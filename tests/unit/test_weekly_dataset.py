from __future__ import annotations

import numpy as np
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.train.weekly_dataset import build_weekly_dataset

CFG = weekly_meta_config()


@pytest.fixture(scope="module")
def built():
    market = synthetic_market(n=700)
    return market, build_weekly_dataset(market, CFG, k=2.0, d_star=0.4, cost_rt=0.0013)


def test_events_align_row_for_row_with_the_dataset(built) -> None:
    _, wd = built
    ds, ev = wd.dataset, wd.events
    assert len(ev) == len(ds) > 50
    assert np.array_equal(ev["event_bar"].to_numpy(), ds.event_bar)
    assert np.array_equal(ev["exit_bar"].to_numpy(), ds.exit_bar)
    assert np.array_equal(ev["side"].to_numpy(), ds.side)
    assert np.array_equal(ev["target"].to_numpy(), ds.target)
    assert np.array_equal(ds.features["side"].to_numpy(), ds.side)


def test_every_training_row_is_finite(built) -> None:
    _, wd = built
    assert np.isfinite(wd.dataset.features.to_numpy(dtype=float)).all()
    assert sum(wd.n_dropped_nan.values()) > 0  # de burn-in kost events, en dat wordt geteld


def test_the_target_is_net_of_costs(built) -> None:
    _, wd = built
    ev = wd.events
    assert ((ev["fill_return"] - 0.0013 > 0.0).astype(int) == ev["target"]).all()


def test_reversing_the_side_flips_the_events(built) -> None:
    market, wd = built
    rev = build_weekly_dataset(market, CFG, k=2.0, d_star=0.4, cost_rt=0.0013, side_sign=-1.0)
    a = wd.events.set_index(["symbol", "event_bar"])["side"]
    b = rev.events.set_index(["symbol", "event_bar"])["side"]
    common = a.index.intersection(b.index)
    assert len(common) > 0
    assert (a.loc[common] == -b.loc[common]).all()


def test_no_label_reaches_past_a_truncated_market(built) -> None:
    market, _ = built
    cut = market.truncate(market.grid[500])
    wd = build_weekly_dataset(cut, CFG, k=2.0, d_star=0.4, cost_rt=0.0013)
    assert int(wd.dataset.exit_bar.max()) < 500
