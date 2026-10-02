"""Rij t van elke feature is gelijk als alles na t wordt weggelaten."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.features.weekly_set import build_feature_panel


def _zeros(market) -> dict[str, pd.Series]:
    return {s: pd.Series(0.0, index=market.grid) for s in market.symbols}


@pytest.mark.parametrize("t", [230, 320, 410])
def test_row_t_does_not_see_the_future(t) -> None:
    market = synthetic_market(n=460)
    full = build_feature_panel(market, _zeros(market), d_star=0.4, corr_window=60)
    cut_market = market.truncate(market.grid[t] + pd.Timedelta(hours=1))
    cut = build_feature_panel(cut_market, _zeros(cut_market), d_star=0.4, corr_window=60)
    for s in market.symbols:
        a = full[s].iloc[t].to_numpy(dtype=float)
        b = cut[s].iloc[t].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True, rtol=1e-10, atol=1e-12), s


def test_funding_carries_one_bar_of_lag() -> None:
    market = synthetic_market(n=460)
    t = 300
    moved_funding = market.funding.copy()
    moved_funding.iloc[t] += 0.01
    moved = type(market)(**{**market.__dict__, "funding": moved_funding})
    a = build_feature_panel(market, _zeros(market), d_star=0.4, corr_window=60)["BTCUSDT"]
    b = build_feature_panel(moved, _zeros(moved), d_star=0.4, corr_window=60)["BTCUSDT"]
    assert a["funding_z"].iloc[t] == pytest.approx(b["funding_z"].iloc[t])
    assert a["funding_z"].iloc[t + 1] != pytest.approx(b["funding_z"].iloc[t + 1])
