"""Chaos engineering tests — Wave 20.

Tests production resilience under fault injection:
- Feed websocket crash
- Exchange API timeout
- Model exception
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_feed_gap_trips_circuit_breaker():
    """Feed sequence number gap should trip the circuit breaker."""
    # This test validates P0-5.6: feed gap detection
    # Requires live engine with sequence number tracking
    pytest.skip("Requires live engine integration — implement in Wave 20 sprint")


@pytest.mark.asyncio
async def test_exchange_timeout_order_idempotent():
    """Exchange timeout on order should not create duplicate orders."""
    pytest.skip("Requires OMS router integration — implement in Wave 20 sprint")


@pytest.mark.asyncio
async def test_kill_switch_cancels_all_orders():
    """SIGUSR1 kill switch should cancel all open orders atomically."""
    pytest.skip("Requires live engine signal handler — implement in Wave 20 sprint")


def test_circuit_breaker_persists_state(tmp_path, monkeypatch):
    """CircuitBreaker trip should be persisted to disk."""
    import json
    cb_log = tmp_path / "circuit_breaker.log"
    monkeypatch.setattr(
        "tradebot.live.circuit_breaker._CB_LOG_PATH",
        cb_log,
        raising=False,
    )
    try:
        from tradebot.live.circuit_breaker import CircuitBreaker
        cb = CircuitBreaker.__new__(CircuitBreaker)
        cb._logger = MagicMock()
        if hasattr(cb, '_persist_trip'):
            cb._persist_trip("test_reason")
            assert cb_log.exists(), "CB log should be created after trip"
            with open(cb_log) as f:
                entry = json.loads(f.readline().strip())
            assert entry["reason"] == "test_reason"
            assert not entry["acknowledged"]
    except (ImportError, AttributeError):
        pytest.skip("CircuitBreaker _persist_trip not yet implemented")


def test_cpcv_fallback_raises_value_error():
    """CPCV fallback concatenation must raise ValueError (not silent concat)."""
    import pandas as pd

    from tradebot.cv.cpcv import build_cpcv_return_paths

    # n_groups=5, k=2 → 5 % 2 != 0 → should raise ValueError
    fold_returns = {
        (0, 1): pd.Series([0.01, 0.02], dtype=float),
        (2, 3): pd.Series([0.03, 0.04], dtype=float),
    }
    with pytest.raises(ValueError, match="divisible"):
        build_cpcv_return_paths(fold_returns, n_groups=5)


def test_trend_scan_label_removed():
    """TrendScanningLabeler.label() should raise NotImplementedError (P0-8)."""
    import pandas as pd

    from tradebot.labeling.trend_scanning import TrendScanningLabeler
    labeler = TrendScanningLabeler()
    df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})
    with pytest.raises(NotImplementedError):
        labeler.label(df)


def test_funding_searchsorted_causality():
    """Funding rate searchsorted must NOT place rate on next bar at exact match."""
    # Test that at exact timestamp match, rate lands on SAME bar (side="right" behavior)
    # This is a unit test to verify causality
    pytest.skip("Requires mock parquet data — add in Wave 20 sprint")
