# tests/unit/test_live_circuit_breaker_edges.py
"""Restgaten van `live/circuit_breaker.py`: sessie-drawdown, positieleeftijd, `halt_reason` en `release_halt`.

De soevereine halt-semantiek (onherroepelijk, uit de policy afgeleid) staat in
`test_live_halt_is_irreversible.py`; deze tests volgen dezelfde opzet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradebot.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, HaltReason
from tradebot.live.state import SystemState
from tradebot.risk.kill_switches import HaltStore
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
NOW = pd.Timestamp("2024-05-06 12:00", tz="UTC")


def _breaker(tmp_path: Path, state: SystemState, **overrides) -> CircuitBreaker:
    cfg = CircuitBreakerConfig.from_risk_config(
        RISK, halt_store_path=tmp_path / "halt.json", cb_log_path=tmp_path / "cb.log", **overrides)
    return CircuitBreaker(cfg, state)


def _state(equity: float = 100_000.0) -> SystemState:
    s = SystemState(equity=equity, equity_peak=equity)
    s.daily_pnl_open = equity
    s.last_feed_ts = NOW
    return s


class TestHaltReason:
    def test_the_reason_is_none_until_the_breaker_trips_and_is_kept_afterwards(self, tmp_path: Path) -> None:
        state = _state()
        breaker = _breaker(tmp_path, state)
        assert breaker.halt_reason is None and breaker.check(NOW) is None
        state.last_feed_ts = NOW - pd.Timedelta(seconds=600)
        assert breaker.check(NOW) == HaltReason.FEED_TIMEOUT
        assert breaker.halt_reason == HaltReason.FEED_TIMEOUT and breaker.is_active
        assert breaker.check(NOW) == HaltReason.FEED_TIMEOUT            # geen herbeoordeling


class TestIntradayDrawdown:
    def test_a_dip_from_the_session_peak_halts_even_far_from_the_lifetime_limit(self, tmp_path: Path) -> None:
        state = _state()
        breaker = _breaker(tmp_path, state, max_intraday_drawdown_pct=0.05)
        assert breaker.check(NOW) is None                                # opent de sessie op 100.000
        state.equity = 94_000.0
        state.daily_pnl_open = 94_000.0                                  # geen dagverlies, alleen sessie-drawdown
        assert state.current_drawdown < RISK.max_drawdown_pct
        assert breaker.check(NOW + pd.Timedelta(seconds=5)) == HaltReason.MAX_DRAWDOWN

    def test_the_session_peak_follows_equity_up_and_resets_at_the_next_utc_midnight(self, tmp_path: Path) -> None:
        state = _state()
        breaker = _breaker(tmp_path, state, max_intraday_drawdown_pct=0.5)
        breaker.check(NOW)
        state.equity = 120_000.0
        state.equity_peak = 120_000.0
        state.daily_pnl_open = 120_000.0
        state.last_feed_ts = NOW + pd.Timedelta(seconds=5)
        breaker.check(NOW + pd.Timedelta(seconds=5))
        assert breaker._session_peak_equity == 120_000.0
        state.equity = 110_000.0
        state.last_feed_ts = NOW + pd.Timedelta(days=1)
        breaker.check(NOW + pd.Timedelta(days=1))
        assert breaker._session_peak_equity == 110_000.0

    def test_a_non_positive_session_peak_means_no_drawdown(self, tmp_path: Path) -> None:
        breaker = _breaker(tmp_path, _state())
        breaker._session_peak_equity = 0.0
        assert breaker._intraday_drawdown() == 0.0


class TestPositionAge:
    def test_an_old_position_halts_and_a_recent_one_does_not(self, tmp_path: Path) -> None:
        state = _state()
        breaker = _breaker(tmp_path, state)
        assert breaker.check(NOW, position_open_ts=NOW - pd.Timedelta(hours=RISK.max_position_age_h - 1)) is None
        stale = NOW - pd.Timedelta(hours=RISK.max_position_age_h + 1)
        assert breaker.check(NOW, position_open_ts=stale) == HaltReason.POSITION_AGE


class TestReleaseHalt:
    def test_a_standing_halt_can_be_released_without_building_a_breaker(self, tmp_path: Path) -> None:
        state = _state()
        state.last_feed_ts = NOW - pd.Timedelta(seconds=600)
        breaker = _breaker(tmp_path, state)
        breaker.check(NOW)
        store_path = tmp_path / "halt.json"
        assert HaltStore(store_path).is_halted()
        with pytest.raises(RuntimeError, match="HALT"):
            _breaker(tmp_path, _state())                                 # weigert te starten zolang de halt staat
        record = CircuitBreaker.release_halt(operator="ops", justification="feed hersteld",
                                             halt_store_path=store_path)
        assert record.reason and not HaltStore(store_path).is_halted()
        assert _breaker(tmp_path, _state()).check(NOW) is None           # nu wel
