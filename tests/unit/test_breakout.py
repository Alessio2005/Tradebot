from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.labeling.breakout import breakout_side, calibrate_k, events_per_week
from tradebot.labeling.cusum import directional_cusum_filter, symmetric_cusum_filter
from tradebot.utils.failfast import DataContractError


def _walk(n: int = 600, drift: float = 0.0, seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(drift, 0.02, n))), index=idx)
    return close, pd.Series(0.02, index=idx)


def test_the_directional_filter_splits_the_symmetric_one() -> None:
    close, sigma = _walk()
    p = np.log(close.to_numpy())
    thr = np.full(p.size, 0.05)
    up, down = directional_cusum_filter(p, thr, thr)
    assert not set(up) & set(down)
    assert sorted(set(up) | set(down)) == sorted(symmetric_cusum_filter(p, thr))


def test_an_uptrend_breaks_up_more_often() -> None:
    close, sigma = _walk(drift=0.01)
    side = breakout_side(close, sigma, 2.0)
    assert (side > 0).sum() > (side < 0).sum()
    assert set(np.unique(side)) <= {-1.0, 0.0, 1.0}


def test_a_higher_k_gives_no_more_events() -> None:
    close, sigma = _walk()
    counts = [int((breakout_side(close, sigma, k) != 0).sum()) for k in (1.0, 2.0, 3.0, 4.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > counts[-1]


def test_a_gap_after_the_first_valid_bar_crashes() -> None:
    close, sigma = _walk()
    close.iloc[300] = np.nan
    with pytest.raises(DataContractError, match="gat"):
        breakout_side(close, sigma, 2.0)


def test_leading_nans_are_the_listing_not_a_gap() -> None:
    close, sigma = _walk()
    close.iloc[:50] = np.nan
    side = breakout_side(close, sigma, 2.0)
    assert (side.iloc[:51] == 0.0).all()


def test_calibrate_k_picks_the_rate_closest_to_the_target() -> None:
    close, sigma = _walk(n=800)
    closes, sigmas = {"A": close}, {"A": sigma}
    start, end = close.index[0], close.index[-1]
    k, rates = calibrate_k(closes, sigmas, k_grid=(1.0, 2.0, 3.0, 4.0),
                           target_per_week=1.0, start=start, end=end)
    assert k in rates
    assert abs(rates[k] - 1.0) == min(abs(r - 1.0) for r in rates.values())
    assert rates[1.0] > rates[4.0]
    measured = events_per_week({"A": breakout_side(close.loc[close.index < end],
                                                   sigma.loc[sigma.index < end], k)},
                               start=start, end=end)
    assert measured == pytest.approx(rates[k])
