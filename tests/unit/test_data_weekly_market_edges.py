# tests/unit/test_data_weekly_market_edges.py
"""`data/weekly_market.py`: het dagraster, de afgeleide panels en de weigering bij een gat.

`test_weekly_market.py` laadt de echte gecertificeerde store; deze tests vervangen de store/het register door kleine
nepbronnen zodat de foutpaden (verschillende rasters, een gat in het dagraster) aantoonbaar rood worden.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data import weekly_market as wm
from tradebot.schemas.config import VolatilityConfig, load_config
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]
VOL = load_config(ROOT / "conf" / "model" / "volatility.yaml", VolatilityConfig)


def _days(start: str, n: int, drop: tuple[int, ...] = ()) -> pd.DatetimeIndex:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC").as_unit("ns")
    return idx[[i for i in range(n) if i not in drop]]


def _ohlcv(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(index))))
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "turnover": np.full(len(index), 1e6)}, index=index)


class TestMarketFromFrames:
    def test_the_symbols_follow_the_ohlcv_columns(self) -> None:
        idx = _days("2024-01-01", 60)
        market = wm.market_from_frames({"AAA": _ohlcv(idx), "BBB": _ohlcv(idx, 1)}, pd.DataFrame(index=idx),
                                       pd.DataFrame(index=idx), vol=VOL, source_hashes={"k": "h"})
        assert market.symbols == ("AAA", "BBB")
        assert list(market.funding.columns) == ["AAA", "BBB"] and market.source_hashes == {"k": "h"}
        assert market.adv_usd["AAA"].iloc[:30].isna().all() and market.adv_usd["AAA"].iloc[30] == pytest.approx(1e6)

    def test_frames_on_different_grids_are_refused(self) -> None:
        with pytest.raises(DataContractError, match="delen geen raster"):
            wm.market_from_frames({"AAA": _ohlcv(_days("2024-01-01", 40)), "BBB": _ohlcv(_days("2024-01-02", 40))},
                                  pd.DataFrame(), pd.DataFrame(), vol=VOL, source_hashes={})


class _Register:
    def __init__(self, path) -> None:
        self.path = path

    def certified_hash(self, asset_class: str, dataset: str, symbol: str, granularity: str) -> str:
        return f"{asset_class}/{dataset}/{symbol}/{granularity}"


@pytest.fixture()
def fake_store(monkeypatch: pytest.MonkeyPatch):
    """Leveringen per symbool: (asof-index OHLCV, asof-index open interest); funding komt als nullen terug."""
    spec: dict[str, pd.DatetimeIndex] = {}

    def load(store, register, *, asset_class, dataset, symbol, granularity):
        idx = spec[symbol]
        asof = idx.asi8
        if dataset == "ohlcv":
            df = _ohlcv(idx, seed=len(symbol)).reset_index(drop=True).assign(asof_ts_ns=asof)
        else:
            df = pd.DataFrame({"open_interest": np.full(len(idx), 5.0), "asof_ts_ns": asof})
        return df, f"hash:{dataset}:{symbol}:{granularity}"

    monkeypatch.setattr(wm, "load_certified_series", load)
    monkeypatch.setattr(wm, "DataRegister", _Register)
    monkeypatch.setattr(wm, "daily_funding_panel", lambda store, register, *, symbols, bar_index, **kw: pd.DataFrame(
        0.0, index=bar_index, columns=symbols))
    return spec


class TestLoadWeeklyMarket:
    def test_the_grid_is_the_union_of_listings_with_nan_before_a_late_listing(self, fake_store) -> None:
        fake_store["OLD"] = _days("2024-01-01", 70)
        fake_store["NEW"] = _days("2024-02-10", 30)
        market = wm.load_weekly_market(ROOT, ["OLD", "NEW"])
        assert len(market.grid) == 70 and str(market.grid.tz) == "UTC"
        assert market.ohlcv["NEW"]["close"].iloc[:40].isna().all() and market.ohlcv["NEW"]["close"].iloc[40:].notna().all()
        assert market.open_interest["NEW"].iloc[:40].isna().all()
        assert market.source_hashes == {
            "crypto/ohlcv/OLD/1d": "hash:ohlcv:OLD:1d", "crypto/open_interest/OLD/1d": "hash:open_interest:OLD:1d",
            "crypto/funding/OLD/8h": "crypto/funding/OLD/8h",
            "crypto/ohlcv/NEW/1d": "hash:ohlcv:NEW:1d", "crypto/open_interest/NEW/1d": "hash:open_interest:NEW:1d",
            "crypto/funding/NEW/8h": "crypto/funding/NEW/8h"}

    def test_a_missing_day_in_the_union_is_a_contract_violation(self, fake_store) -> None:
        fake_store["AAA"] = _days("2024-01-01", 40, drop=(20,))
        with pytest.raises(DataContractError, match="gaten"):
            wm.load_weekly_market(ROOT, ["AAA"])
