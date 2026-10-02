from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.features.weekly_set import (
    FEATURE_COLUMNS,
    build_feature_panel,
    fit_common_d_star,
    market_features,
)
from tradebot.utils.failfast import DataContractError


def _zeros(market) -> dict[str, pd.Series]:
    return {s: pd.Series(0.0, index=market.grid) for s in market.symbols}


def test_every_symbol_gets_exactly_the_declared_columns() -> None:
    market = synthetic_market()
    panel = build_feature_panel(market, _zeros(market), d_star=0.4, corr_window=60)
    assert set(panel) == set(market.symbols)
    for frame in panel.values():
        assert tuple(frame.columns) == FEATURE_COLUMNS
        assert frame.index.equals(market.grid)
        # Na de langste burn-in (FFD 282 bij d = 0,4 en drempel 1e-4) is alles eindig.
        assert np.isfinite(frame.iloc[300:].to_numpy()).all()


def test_d_star_reads_only_data_before_until() -> None:
    market = synthetic_market(n=600)
    logs = {s: np.log(market.ohlcv[s]["close"]) for s in market.symbols}
    until = market.grid[400]
    d0 = fit_common_d_star(logs, until=until)
    moved = {s: x.where(x.index < until, x + 5.0) for s, x in logs.items()}
    assert fit_common_d_star(moved, until=until) == d0
    assert 0.0 < d0 <= 0.9


def test_d_star_refuses_too_little_history() -> None:
    market = synthetic_market(n=300)
    logs = {s: np.log(market.ohlcv[s]["close"]) for s in market.symbols}
    with pytest.raises(DataContractError, match="historie"):
        fit_common_d_star(logs, until=market.grid[100])


def test_the_average_correlation_recovers_an_equicorrelated_rho() -> None:
    rng = np.random.default_rng(11)
    n, rho = 400, 0.6
    common = rng.normal(size=n)
    data = {f"S{i}": np.sqrt(rho) * common + np.sqrt(1 - rho) * rng.normal(size=n)
            for i in range(4)}
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    out = market_features(pd.DataFrame(data, index=idx) * 0.02, window=60)
    assert out["avg_corr60"].iloc[100:].mean().mean() == pytest.approx(rho, abs=0.1)
