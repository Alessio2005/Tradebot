"""De gecertificeerde dagmarkt van de wekelijkse strategie, op één raster (spec §4, §17.1).

Elke reeks komt via `load_certified_series`, dus langs het data-register; een
afwijkende hash crasht hier en niet pas in een rapport. Het raster is de unie
van de asof-tijden (sluitmomenten) en moet gatenloos dagelijks zijn. Vóór de
notering van een symbool staat er NaN: dat is geen gat maar het feit dat het
instrument nog niet bestond.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.base import ASOF_INDEX_NAME, DataRegister, load_certified_series
from ..schemas.config import VolatilityConfig, load_config
from ..utils.failfast import DataContractError, require
from ..volatility.ewma import ewma_volatility
from .funding_panel import daily_funding_panel
from .pit_store import PitStore

__all__ = ["OHLCV_COLUMNS", "WeeklyMarket", "load_weekly_market", "market_from_frames"]

OHLCV_COLUMNS = ("open", "high", "low", "close", "turnover")
ADV_WINDOW = 30


@dataclass(frozen=True)
class WeeklyMarket:
    grid: pd.DatetimeIndex
    ohlcv: dict[str, pd.DataFrame]
    sigma_daily: pd.DataFrame
    sigma_annual: pd.DataFrame
    funding: pd.DataFrame
    open_interest: pd.DataFrame
    adv_usd: pd.DataFrame
    source_hashes: dict[str, str]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.ohlcv)

    def truncate(self, end: pd.Timestamp) -> WeeklyMarket:
        """Alles strikt vóór `end`. Elke reeks hier is causaal, dus afkappen verandert niets vóór `end`."""
        grid = self.grid[self.grid < end]
        return WeeklyMarket(
            grid=grid,
            ohlcv={s: f.loc[grid] for s, f in self.ohlcv.items()},
            sigma_daily=self.sigma_daily.loc[grid],
            sigma_annual=self.sigma_annual.loc[grid],
            funding=self.funding.loc[grid],
            open_interest=self.open_interest.loc[grid],
            adv_usd=self.adv_usd.loc[grid],
            source_hashes=dict(self.source_hashes),
        )


def market_from_frames(
    ohlcv: dict[str, pd.DataFrame],
    funding: pd.DataFrame,
    open_interest: pd.DataFrame,
    *,
    vol: VolatilityConfig,
    source_hashes: dict[str, str],
) -> WeeklyMarket:
    """Bouw de afgeleide panels (sigma, ADV) uit ruwe frames op één raster."""
    grid = next(iter(ohlcv.values())).index
    for name, frame in ohlcv.items():
        require(frame.index.equals(grid), "OHLCV-frames delen geen raster.",
                DataContractError, symbol=name)
    closes = pd.DataFrame({s: f["close"] for s, f in ohlcv.items()})
    sigma_annual = pd.DataFrame({
        s: ewma_volatility(closes[s], lam=vol.ewma_lambda,
                           burn_in_bars=vol.burn_in_bars,
                           annualisation_factor=vol.annualisation_factor)
        for s in closes
    })
    turnover = pd.DataFrame({s: f["turnover"] for s, f in ohlcv.items()})
    return WeeklyMarket(
        grid=grid,
        ohlcv=ohlcv,
        sigma_daily=sigma_annual / np.sqrt(vol.annualisation_factor),
        sigma_annual=sigma_annual,
        funding=funding.reindex(columns=list(ohlcv)),
        open_interest=open_interest.reindex(columns=list(ohlcv)),
        # Causaal: de turnover van bar t is pas op zijn close bekend.
        adv_usd=turnover.rolling(ADV_WINDOW, min_periods=ADV_WINDOW).mean().shift(1),
        source_hashes=source_hashes,
    )


def _asof_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        pd.to_datetime(df["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
        name=ASOF_INDEX_NAME)


def load_weekly_market(root: Path, symbols: Sequence[str]) -> WeeklyMarket:
    """Laad OHLCV, funding en open interest langs het data-register."""
    vol = load_config(root / "conf/model/volatility.yaml", VolatilityConfig)
    store = PitStore(root / "data/pit_store")
    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    frames: dict[str, pd.DataFrame] = {}
    oi: dict[str, pd.Series] = {}
    hashes: dict[str, str] = {}
    for sym in symbols:
        df, h = load_certified_series(store, register, asset_class="crypto",
                                      dataset="ohlcv", symbol=sym, granularity="1d")
        frames[sym] = pd.DataFrame(
            {c: df[c].to_numpy(dtype=np.float64) for c in OHLCV_COLUMNS},
            index=_asof_index(df))
        hashes[f"crypto/ohlcv/{sym}/1d"] = h
        odf, oh = load_certified_series(store, register, asset_class="crypto",
                                        dataset="open_interest", symbol=sym,
                                        granularity="1d")
        oi[sym] = pd.Series(odf["open_interest"].to_numpy(dtype=np.float64),
                            index=_asof_index(odf))
        hashes[f"crypto/open_interest/{sym}/1d"] = oh
        hashes[f"crypto/funding/{sym}/8h"] = register.certified_hash(
            "crypto", "funding", sym, "8h")
    grid = pd.DatetimeIndex(sorted(set().union(*[f.index for f in frames.values()])),
                            name=ASOF_INDEX_NAME)
    require(bool((grid[1:] - grid[:-1] == pd.Timedelta(days=1)).all()),
            "Het dagraster heeft gaten.", DataContractError)
    funding = daily_funding_panel(store, register, symbols=list(symbols),
                                  asset_class="crypto", funding_granularity="8h",
                                  bar_index=grid)
    return market_from_frames(
        {s: f.reindex(grid) for s, f in frames.items()},
        funding,
        pd.DataFrame({s: oi[s].reindex(grid) for s in symbols}),
        vol=vol,
        source_hashes=hashes,
    )
