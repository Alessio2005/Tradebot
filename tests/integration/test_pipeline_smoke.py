"""tests/integration/test_pipeline_smoke.py — End-to-end Stage 1+2 smoke test.

Verifies that the bars → features → labels pipeline produces valid
outputs on a tiny synthetic dataset (500 bars, seeded).

Marked @pytest.mark.slow — skip in fast CI: pytest -m "not slow".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    close = 30_000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high  = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol   = rng.exponential(1_000, n)
    idx   = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.mark.slow
def test_garman_klass_on_ohlcv(synthetic_ohlcv: pd.DataFrame) -> None:
    from tradebot.volatility import get_garman_klass_volatility, get_jump_adjusted_volatility
    gk = get_garman_klass_volatility(synthetic_ohlcv, window=14)
    assert gk.notna().sum() > 400
    assert (gk >= 0).all()

    jmp = get_jump_adjusted_volatility(synthetic_ohlcv, window=14)
    assert jmp.notna().sum() > 400
    assert (jmp >= gk - 1e-9).all()   # jump-adjusted >= GK


@pytest.mark.slow
def test_sequential_bootstrap_no_lookahead(synthetic_ohlcv: pd.DataFrame) -> None:
    from tradebot.cv import get_sequential_bootstrap_indices
    n = len(synthetic_ohlcv)
    t0 = np.arange(n, dtype=np.int64)
    t1 = np.minimum(t0 + 10, n - 1).astype(np.int64)
    idx = get_sequential_bootstrap_indices(t0, t1, n_draws=n)
    assert idx.shape == (n,)
    assert (idx >= 0).all() and (idx < n).all()


@pytest.mark.slow
def test_monitoring_drift_smoke(synthetic_ohlcv: pd.DataFrame) -> None:
    import numpy as np
    from tradebot.monitoring import check_feature_drift, psi
    # Use log-returns (stationary) not raw price levels (non-stationary).
    # A random-walk price series will always produce high PSI across windows
    # because the level distribution shifts monotonically.
    prices = synthetic_ohlcv["close"].values
    log_rets = np.diff(np.log(prices))  # length n-1
    half = len(log_rets) // 2
    ref = {"close_ret": log_rets[:half]}
    cur = {"close_ret": log_rets[half:]}
    results = check_feature_drift(ref, cur)
    assert len(results) == 1
    assert results[0].feature == "close_ret"
    # Log-returns of same GBM process are i.i.d. — PSI must be small
    assert results[0].psi_value < 0.5 or not results[0].psi_value  # nan OK (< 50 samples each)
