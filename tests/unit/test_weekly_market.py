from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tests.weekly_fixtures import synthetic_market
from tradebot.data.weekly_market import load_weekly_market

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT")


def test_the_certified_market_is_one_gapless_daily_grid() -> None:
    market = load_weekly_market(ROOT, SYMBOLS)
    steps = market.grid[1:] - market.grid[:-1]
    assert (steps == pd.Timedelta(days=1)).all()
    assert str(market.grid.tz) == "UTC"
    for frame in (market.sigma_daily, market.funding, market.open_interest, market.adv_usd):
        assert list(frame.columns) == list(SYMBOLS)
        assert frame.index.equals(market.grid)
    # Drie gecertificeerde reeksen per symbool: ohlcv, funding, open interest.
    assert len(market.source_hashes) == 3 * len(SYMBOLS)


def test_daily_sigma_is_the_annual_ewma_de_annualised() -> None:
    market = load_weekly_market(ROOT, SYMBOLS)
    ratio = (market.sigma_annual / market.sigma_daily).stack().dropna()
    assert np.allclose(ratio.to_numpy(), np.sqrt(365.0))


def test_adv_is_known_one_bar_later() -> None:
    market = synthetic_market()
    turnover = market.ohlcv["BTCUSDT"]["turnover"]
    expected = turnover.rolling(30, min_periods=30).mean().shift(1)
    pd.testing.assert_series_equal(market.adv_usd["BTCUSDT"], expected, check_names=False)


def test_truncate_keeps_only_bars_strictly_before_end() -> None:
    market = synthetic_market()
    end = market.grid[300]
    cut = market.truncate(end)
    assert cut.grid[-1] == market.grid[299]
    pd.testing.assert_frame_equal(cut.sigma_daily, market.sigma_daily.iloc[:300])
    assert len(cut.ohlcv["ETHUSDT"]) == 300
