"""Een synthetische dagmarkt met de vorm van de echte, voor de tests van de wekelijkse strategie."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.data.weekly_market import WeeklyMarket, market_from_frames
from tradebot.schemas.config import VolatilityConfig


def synthetic_market(
    n: int = 500,
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    seed: int = 7,
) -> WeeklyMarket:
    rng = np.random.default_rng(seed)
    grid = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC", name="asof_ts")
    common = rng.normal(0.0, 0.03, n)
    ohlcv = {}
    for s in symbols:
        r = 0.8 * common + 0.6 * rng.normal(0.0, 0.03, n)
        close = 100.0 * np.exp(np.cumsum(r))
        open_ = np.r_[100.0, close[:-1]]
        wick = np.abs(rng.normal(0.0, 0.01, n))
        ohlcv[s] = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * (1.0 + wick),
            "low": np.minimum(open_, close) * (1.0 - wick),
            "close": close,
            "turnover": rng.uniform(1e9, 2e9, n),
        }, index=grid)
    funding = pd.DataFrame(rng.normal(1e-4, 5e-5, (n, len(symbols))), index=grid,
                           columns=list(symbols))
    oi = pd.DataFrame(rng.uniform(1e6, 2e6, (n, len(symbols))), index=grid,
                      columns=list(symbols))
    vol = VolatilityConfig(ewma_lambda=0.94, burn_in_bars=60, annualisation_factor=365.0)
    return market_from_frames(ohlcv, funding, oi, vol=vol, source_hashes={})
