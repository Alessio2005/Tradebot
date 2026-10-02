from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.validation.trend_overlay import summarize, trend_book


def _prices(n: int = 400, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n))), index=idx)


def test_the_position_on_a_bar_only_uses_closes_before_it() -> None:
    close = _prices()
    fund = pd.Series(0.0, index=close.index)
    full = trend_book(close, fund, cost_per_side=0.0006)
    for cut in (150, 250, 399):
        part = trend_book(close.iloc[:cut], fund.iloc[:cut], cost_per_side=0.0006)
        assert part["pos"].equals(full["pos"].iloc[:cut])
    shocked = close.copy()
    shocked.iloc[300:] *= 3.0  # een toekomstschok verandert niets vóór bar 300
    s = trend_book(shocked, fund, cost_per_side=0.0006)
    assert s["pos"].iloc[:301].equals(full["pos"].iloc[:301])


def test_a_flat_book_earns_and_pays_nothing() -> None:
    close = pd.Series(np.linspace(200.0, 100.0, 300),
                      index=pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC"))
    out = trend_book(close, pd.Series(0.001, index=close.index), cost_per_side=0.001)
    assert (out["pos"] == 0.0).all() and (out["net"] == 0.0).all()


def test_costs_are_charged_per_change_and_a_long_pays_positive_funding() -> None:
    close = _prices()
    fund = pd.Series(0.0002, index=close.index)
    out = trend_book(close, fund, cost_per_side=0.0006)
    flips = out["pos"].diff().abs().fillna(out["pos"].abs())
    assert np.isclose(out["cost"].sum(), flips.sum() * 0.0006)
    assert np.isclose(out["funding"].sum(), out["pos"].sum() * 0.0002)
    assert np.allclose(out["net"], out["gross"] - out["cost"] - out["funding"])


def test_annualisation_compounds_the_period_return() -> None:
    x = pd.Series([0.01] * 73)
    s = summarize(x, x)["strategy"]
    assert np.isclose(s["total_return"], 1.01 ** 73 - 1.0)
    assert np.isclose(s["annualised_return"], (1.01 ** 73) ** 5 - 1.0)
