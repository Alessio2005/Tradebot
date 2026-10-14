"""Synthetische markten voor de toetsen van `tradebot.systematic`.

Geen echte data: de toetsen bewijzen eigenschappen van de constructie (causaliteit,
boekhouding), en die moeten op elke markt gelden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.schemas.robust_book import robust_book_config
from tradebot.systematic.market import BookMarket, ewma_vol

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT")


def synthetic_market(n_bars: int = 700, *, seed: int = 7, late_listing: bool = True) -> BookMarket:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="D", tz="UTC", name="asof_ts")
    common = rng.standard_t(4, size=n_bars) * 0.025
    rets = {}
    for i, s in enumerate(SYMBOLS):
        drift = 0.0005 * np.sin(np.arange(n_bars) / (60 + 10 * i))
        rets[s] = 0.8 * common + rng.standard_t(4, size=n_bars) * 0.02 + drift
    ret = pd.DataFrame(rets, index=idx).clip(-0.4, 0.4)
    close = 100.0 * (1.0 + ret).cumprod()
    if late_listing:
        close.loc[close.index[:90], "SOLUSDT"] = np.nan
        close.loc[close.index[:150], "DOTUSDT"] = np.nan
    ret = close.pct_change(fill_method=None)
    listed = close.notna()
    funding = pd.DataFrame(rng.normal(3e-4, 4e-4, size=close.shape), index=idx,
                           columns=list(SYMBOLS)).where(listed, 0.0)
    adv = pd.DataFrame(5e8, index=idx, columns=list(SYMBOLS)).where(
        listed.rolling(30, min_periods=1).sum() >= 30)
    sigma = ewma_vol(ret, span=20)
    return BookMarket(close=close, ret=ret, funding=funding, adv_usd=adv, sigma_daily=sigma,
                      source_hashes={"synthetic": str(seed)})


def perturb_after(market: BookMarket, t: int, *, seed: int = 99) -> BookMarket:
    """Dezelfde markt, met elke prijs en funding NA bar `t` vervangen door ruis."""
    rng = np.random.default_rng(seed)
    close = market.close.copy()
    tail = close.iloc[t + 1:]
    shock = np.exp(rng.normal(0.0, 0.2, size=tail.shape).cumsum(axis=0))
    close.iloc[t + 1:] = tail.to_numpy() * shock
    funding = market.funding.copy()
    funding.iloc[t + 1:] = rng.normal(0.0, 5e-3, size=funding.iloc[t + 1:].shape)
    funding = funding.where(close.notna(), 0.0)
    ret = close.pct_change(fill_method=None)
    sigma = ewma_vol(ret, span=20)
    return BookMarket(close=close, ret=ret, funding=funding, adv_usd=market.adv_usd,
                      sigma_daily=sigma, source_hashes=market.source_hashes)


def config():
    return robust_book_config()
