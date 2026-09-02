"""tests/integration/test_governor_wiring.py — propfirm governor ↔ LiveEngine.

Proves the config → engine plumbing: when ``propfirm_limits`` is supplied the
engine builds a PropfirmGovernor seeded to the initial equity; when omitted the
engine is unchanged (no governor → no behaviour change for existing runs).
The enforcement logic itself is covered exhaustively in
tests/unit/test_propfirm_governor.py.
"""
from __future__ import annotations

import asyncio

import pytest

from tradebot.live.circuit_breaker import CircuitBreakerConfig
from tradebot.live.engine import LiveEngine, LiveEngineConfig
from tradebot.live.feed import Feed, FeedConfig
from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig
from tradebot.risk.daily_loss_governor import PropfirmLimits, Regime, RegimeConfig


def _engine(tmp_path, *, limits=None, regime=None) -> LiveEngine:
    symbols = ["BTCUSDT"]
    queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    feed = Feed(FeedConfig(symbols=symbols, paper_mode=True), queue, {})
    cfg = LiveEngineConfig(
        symbols=symbols,
        initial_equity=100_000.0,
        mode="paper",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        cb_config=CircuitBreakerConfig(cb_log_path=tmp_path / "cb.log"),
        sr_config=SignalRunnerConfig(),
        propfirm_limits=limits,
        regime=regime,
    )
    sr = SignalRunner(signals=[], config=cfg.sr_config)
    return LiveEngine(config=cfg, signal_runner=sr, feed=feed)


def test_engine_without_limits_has_no_governor(tmp_path):
    eng = _engine(tmp_path)
    assert eng._governor is None  # unchanged behaviour for non-propfirm runs


def test_engine_builds_governor_seeded_to_initial_equity(tmp_path):
    limits = PropfirmLimits(initial_balance=100_000.0)
    eng = _engine(tmp_path, limits=limits, regime=RegimeConfig.challenge())
    assert eng._governor is not None
    assert eng._governor.day_start_balance == pytest.approx(100_000.0)
    # Governor enforces the audit thresholds wired from config: a -4.6% day
    # (>= daily_hard 4.5%, < firm 5%) is a HARD_FLATTEN, not a FAIL.
    d = eng._governor.update(100_000.0 * (1 - 0.046))
    assert d.action.value == "HARD_FLATTEN"
    assert eng._cfg.regime.regime is Regime.CHALLENGE
