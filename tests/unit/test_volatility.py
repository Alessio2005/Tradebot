"""tests/unit/test_volatility.py — Unit tests for volatility estimators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import TradebotContractError
from tradebot.volatility import (
    ewma_volatility,
    get_ewma_volatility,
    get_garman_klass_volatility,
    get_parkinson_volatility,
)


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high  = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def test_gk_shape_and_nonnegative() -> None:
    df = _make_ohlcv()
    vol = get_garman_klass_volatility(df, window=14)
    assert vol.shape == (len(df),)
    assert vol.dropna().ge(0).all()


def test_gk_absolute_mode_scales_with_price() -> None:
    df = _make_ohlcv()
    vol_frac = get_garman_klass_volatility(df, window=14, absolute=False)
    vol_abs  = get_garman_klass_volatility(df, window=14, absolute=True)
    ratio = (vol_abs / vol_frac).dropna()
    # absolute = frac * close → ratio should equal close values
    close = df["close"]
    aligned_close = close[ratio.index]
    np.testing.assert_allclose(ratio.values, aligned_close.values, rtol=1e-6)


def test_parkinson_fallback_no_open() -> None:
    df = _make_ohlcv().drop(columns=["open"])
    vol = get_parkinson_volatility(df, window=14)
    assert vol.shape == (len(df),)
    assert vol.dropna().ge(0).all()


def test_ewma_volatility() -> None:
    """Phase 3: lambda en burn-in komen uit conf/, niet uit een default."""
    df = _make_ohlcv()
    burn_in = 10
    vol = ewma_volatility(
        df["close"], lam=0.94, burn_in_bars=burn_in, annualisation_factor=365.0
    )
    assert vol.shape == (len(df),)
    assert vol.dropna().ge(0).all()
    # De burn-in blijft NaN: geen fillna(0.0), geen ffill (DI-12).
    assert vol.iloc[:burn_in].isna().all()
    assert vol.iloc[burn_in:].notna().all()


def test_deprecated_ewma_entrypoint_crashes() -> None:
    """De oude get_ewma_volatility vulde de burn-in met een vol van NUL.

    In Naive Risk Parity levert dat een oneindig gewicht op. Er is geen
    doorgeefpad naar de nieuwe functie: stil vertalen zou de gedragswijziging
    onopgemerkt maken.
    """
    with pytest.raises(TradebotContractError, match="fillna"):
        get_ewma_volatility(_make_ohlcv(), halflife=10)


def test_empty_dataframe_returns_zeros() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    vol = get_garman_klass_volatility(df, window=14)
    assert len(vol) == 0
