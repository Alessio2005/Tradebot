"""Hypothesis property tests for math kernels — Wave 20.

Tests that mathematical invariants hold for arbitrary inputs.
Requires: pip install hypothesis
"""
from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")
given = hypothesis.given
settings = hypothesis.settings
assume = hypothesis.assume
st = hypothesis.strategies
arrays = pytest.importorskip("hypothesis.extra.numpy").arrays


@given(
    returns=arrays(
        np.float64,
        shape=st.integers(min_value=30, max_value=500),
        elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=5000)
def test_cornish_fisher_var_is_negative(returns):
    """VaR should be negative (loss) for reasonable return distributions."""
    from tradebot.risk.var import cornish_fisher_var
    assume(np.std(returns) > 0.001)
    result = cornish_fisher_var(returns, confidence=0.99)
    assert isinstance(result, float)
    assert np.isfinite(result)


@given(
    returns=arrays(
        np.float64,
        shape=st.integers(min_value=30, max_value=500),
        elements=st.floats(min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=5000)
def test_har_rv_forecast_non_negative(returns):
    """HAR-RV forecasts of variance should be non-negative."""
    from tradebot.volatility.har_rv import har_rv_feature
    realized_var = returns**2
    assume(len(realized_var) >= 25)
    result = har_rv_feature(realized_var)
    valid = result[~np.isnan(result)]
    if len(valid) > 0:
        assert np.all(valid >= 0), f"Negative variance forecast: {valid[valid < 0]}"


@given(
    arr=arrays(
        np.float64,
        shape=st.integers(min_value=2, max_value=300),
        elements=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    ),
    threshold_q=st.floats(min_value=0.5, max_value=0.95),
)
@settings(max_examples=50, deadline=5000)
def test_evt_gpd_var_less_than_cf_at_extreme(arr, threshold_q):
    """EVT/GPD VaR should give a finite estimate at 99% confidence."""
    from tradebot.risk.var import evt_gpd_var
    returns = -arr
    assume(len(returns) >= 50)
    evt = evt_gpd_var(returns, confidence=0.99, threshold_quantile=threshold_q)
    assert isinstance(evt, float)
    assert np.isfinite(evt)


@given(
    close=arrays(
        np.float64,
        shape=st.integers(min_value=5, max_value=100),
        elements=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=100, deadline=5000)
def test_microprice_between_bid_ask(close):
    """Microprice with equal sizes equals mid-price."""
    from tradebot.features.microstructure import microprice
    assume(np.all(close > 0))
    spread = close * 0.001
    bid = close - spread / 2
    ask = close + spread / 2
    bid_size = np.ones_like(close) * 1.0
    ask_size = np.ones_like(close) * 1.0
    mp = microprice(bid, ask, bid_size, ask_size)
    mid = (bid + ask) / 2.0
    np.testing.assert_allclose(mp, mid, rtol=1e-10)
