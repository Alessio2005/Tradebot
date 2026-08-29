"""tests/e2e/test_paper_trade_smoke.py — 24h paper-trade smoke test.

Acceptance criteria:
  - No exceptions raised
  - Circuit-breaker NOT triggered
  - ≥ 90% of bars processed within 500ms latency budget
  - Equity curve stays > 0 (no complete capital loss)
  - AuditLog receives ≥ 0 complete fill records (no crash)

Marked @pytest.mark.slow — excluded from the fast CI test run.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.live.circuit_breaker import CircuitBreakerConfig
from tradebot.live.engine import LiveEngine, LiveEngineConfig
from tradebot.live.execution_controller import ExecutionControllerConfig
from tradebot.live.feed import Feed, FeedConfig
from tradebot.live.portfolio_controller import PortfolioControllerConfig
from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig


def _make_bars(n: int = 24, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1h OHLCV bars starting 2024-01-01 UTC."""
    rng = np.random.default_rng(seed)
    base = 40_000.0
    closes = base * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    highs = closes * (1 + rng.uniform(0, 0.01, n))
    lows = closes * (1 - rng.uniform(0, 0.01, n))
    opens = np.roll(closes, 1)
    opens[0] = base
    vols = rng.uniform(1_000, 10_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


@pytest.mark.slow
def test_paper_trade_24h_no_crash(tmp_path: Path) -> None:
    """Run LiveEngine in paper-mode over 24 synthetic bars without crash."""
    symbols = ["BTCUSDT"]
    bars_btc = _make_bars(n=24, seed=42)

    queue: asyncio.Queue = asyncio.Queue(maxsize=500)
    feed_cfg = FeedConfig(symbols=symbols, paper_mode=True, replay_delay_s=0.0)
    feed = Feed(config=feed_cfg, queue=queue, historical_bars={"BTCUSDT": bars_btc})

    cb_cfg = CircuitBreakerConfig(
        max_drawdown_pct=0.99,  # effectively disabled for smoke test
        max_daily_loss_pct=0.99,
        feed_timeout_sec=60,
        cb_log_path=tmp_path / "circuit_breaker.log",  # Wave 15 P0-5.5: isolated log
    )
    ec_cfg = ExecutionControllerConfig(model_version="test_v0.1", git_sha="deadbeef")
    # PHASE 5: `constraints` is verplicht. L13 kiest geen risicodrempels meer;
    # ze komen uit `conf/risk/default.yaml` (wiring audit C5).
    from pathlib import Path as _Path

    from tradebot.portfolio.constraints import PortfolioConstraints
    from tradebot.schemas.config import RiskConfig, load_config

    _risk = load_config(
        _Path(__file__).resolve().parents[2] / "conf/risk/default.yaml", RiskConfig)
    pc_cfg = PortfolioControllerConfig(
        # STAGE C-1: expliciet, want `method` heeft geen default meer. HRP is
        # research-gated (fase-6 no-go 15).
        method="erc",
        constraints=PortfolioConstraints.from_risk_config(_risk),
        min_history_bars=5,
    )
    sr_cfg = SignalRunnerConfig(min_confidence=0.0)

    engine_cfg = LiveEngineConfig(
        symbols=symbols,
        initial_equity=100_000.0,
        mode="paper",
        interval="1h",
        audit_log_path=str(tmp_path / "audit.jsonl"),
        cb_config=cb_cfg,
        ec_config=ec_cfg,
        pc_config=pc_cfg,
        sr_config=sr_cfg,
    )
    sr = SignalRunner(signals=[], config=sr_cfg)
    engine = LiveEngine(config=engine_cfg, signal_runner=sr, feed=feed)

    asyncio.run(engine.run())

    # --- Acceptance criteria ---

    # 1. Circuit-breaker must NOT be active
    assert not engine.state.circuit_breaker_active, (
        f"CircuitBreaker tripped: {engine._cb.halt_reason}"
    )

    # 2. Must have processed all 24 bars
    assert engine.bars_processed == 24, (
        f"Expected 24 bars, got {engine.bars_processed}"
    )

    # 3. ≥ 90% bars within 500ms latency
    if engine.latencies:
        within_budget = sum(1 for lat in engine.latencies if lat <= 500.0)
        pct = within_budget / len(engine.latencies)
        assert pct >= 0.90, f"Only {pct:.1%} bars within 500ms budget"

    # 4. Equity > 0
    assert engine.state.equity > 0, f"Equity collapsed: {engine.state.equity}"

    # 5. No fatal errors
    assert len(engine.state.errors) == 0, f"Errors: {engine.state.errors}"
