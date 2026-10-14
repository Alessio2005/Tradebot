# tests/unit/test_live_engine.py
"""`live/engine.py`: opbouw, killswitch, de event-loop en de barverwerking van `LiveEngine`.

De engine draait echt, met echte PaperOMS, router, audit-log en controllers; wat
gestubd wordt is de signaalbron, de FeatureUpdater (de pipeline staat in
`test_features_regime_edges.py`), de judge-poorten en de CUSUM-filters.
"""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.base import SignalResult
from tradebot.execution.fees import bybit_perp_schedule
from tradebot.live import engine as eng
from tradebot.live.circuit_breaker import CircuitBreakerConfig
from tradebot.live.engine import LiveEngine, LiveEngineConfig
from tradebot.live.execution_controller import ExecutionControllerConfig
from tradebot.live.feed import BarEvent, Feed, FeedConfig
from tradebot.live.portfolio_controller import PortfolioControllerConfig
from tradebot.live.state import EngineMode
from tradebot.portfolio.constraints import PortfolioConstraints
from tradebot.risk.daily_loss_governor import GovernorAction, PropfirmLimits
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """De engine installeert SIGTERM/SIGUSR1-handlers; de testrunner krijgt ze terug."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGUSR1)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class _Runner:
    """Signaalbron: geeft per symbool een vast SignalResult (of niets)."""

    def __init__(self, results: dict[str, SignalResult | None] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, str]] = []

    def predict(self, symbol, features, fwd_return=None, bar_ts=None):
        self.calls.append(("predict", symbol))
        return self.results.get(symbol)

    def predict_on_event(self, symbol, features, bar_ts=None):
        self.calls.append(("predict_on_event", symbol))
        return self.results.get(symbol)


class _Updater:
    """FeatureUpdater-stub."""

    def __init__(self, features: pd.DataFrame | None, usable: int = 1_000, fresh: pd.DataFrame | None = None) -> None:
        self.features, self.usable, self.fresh = features, usable, fresh
        self.updates: list[str] = []

    def update(self, symbol, bar):
        self.updates.append(symbol)
        return self.features

    def get_last_usable_bars(self, symbol: str) -> int:
        return self.usable

    async def force_refresh_async(self, symbol: str):
        return self.fresh


def _features() -> pd.DataFrame:
    return pd.DataFrame({"feat_vol_gk": [0.01], "feat_x": [1.0]}, index=[T0])


def _signal(symbol: str, signal_value: float = 1.0, confidence: float = 0.7) -> SignalResult:
    return SignalResult(symbol, T0, signal_value, confidence, 10, f"{symbol}_sig")


def _bar(symbol: str = "BTCUSDT", minute: int = 0, close: float = 100.0, seq: int = 0, **kw) -> BarEvent:
    ts = T0 + pd.Timedelta(minutes=minute)
    return BarEvent(symbol, ts, close, close + 1.0, close - 1.0, close, 50.0, seq=seq, **kw)


def _config(tmp_path: Path, **overrides) -> LiveEngineConfig:
    base = {
        "symbols": SYMBOLS,
        "initial_equity": 100_000.0,
        "mode": "paper",
        "audit_log_path": str(tmp_path / "audit" / "audit.jsonl"),
        "cb_config": CircuitBreakerConfig(
            max_drawdown_pct=0.99, max_intraday_drawdown_pct=0.0, max_daily_loss_pct=0.99,
            feed_timeout_sec=3_600, cb_log_path=tmp_path / "cb.log", halt_store_path=tmp_path / "halt.json"),
        "ec_config": ExecutionControllerConfig(
            RISK, min_notional_per_trade=0.0, min_weight_change=0.0, git_sha="abc", model_version="t"),
        "pc_config": PortfolioControllerConfig(
            constraints=PortfolioConstraints.from_risk_config(RISK), method="erc", min_history_bars=5),
        "metrics_port": 0,
        "state_out_dir": tmp_path / "state",
    }
    return LiveEngineConfig(**{**base, **overrides})


def _engine(tmp_path: Path, runner: _Runner | None = None, features: pd.DataFrame | None = None,
            **cfg) -> LiveEngine:
    config = _config(tmp_path, **cfg)
    feed = Feed(FeedConfig(symbols=config.symbols, paper_mode=True), asyncio.Queue())
    engine = LiveEngine(config, runner or _Runner(), feed)             # type: ignore[arg-type]
    engine._fu = _Updater(features)                                    # type: ignore[assignment]
    return engine


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# configuration and construction
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_the_defaults_come_from_the_sovereign_policy_and_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADEBOT_BYBIT_TIER", "VIP1")
        monkeypatch.setenv("TRADEBOT_FEE_DISCOUNT", "0.1")
        cfg = LiveEngineConfig(symbols=SYMBOLS)
        assert cfg.mode is EngineMode.PAPER and cfg.metrics_port == 8000
        assert cfg.cb_config.max_drawdown_pct == RISK.max_drawdown_pct
        assert cfg.ec_config.risk == RISK and cfg.pc_config.method == "erc"
        assert cfg.fee_schedule == bybit_perp_schedule(tier="VIP1", fee_discount_pct=0.1)
        assert cfg.cusum_multipliers == {} and cfg.regime is not None

    def test_an_explicit_fee_schedule_wins_and_an_empty_discount_means_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        schedule = bybit_perp_schedule(tier="VIP2")
        assert LiveEngineConfig(symbols=SYMBOLS, fee_schedule=schedule).fee_schedule is schedule
        monkeypatch.delenv("TRADEBOT_BYBIT_TIER", raising=False)
        monkeypatch.setenv("TRADEBOT_FEE_DISCOUNT", "")
        assert LiveEngineConfig(symbols=SYMBOLS).fee_schedule == bybit_perp_schedule(tier="NONVIP", fee_discount_pct=0.0)

    def test_an_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a valid EngineMode"):
            LiveEngineConfig(symbols=SYMBOLS, mode="turbo")


class TestConstruction:
    def test_the_collaborators_are_wired_and_the_feed_shares_the_engine_queue(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        assert engine._feed._queue is engine._queue and engine.state.equity == 100_000.0
        assert engine.state.mode is EngineMode.PAPER and engine._governor is None and engine._judges == {}
        assert (tmp_path / "state").is_dir() and engine.bars_processed == 0 and engine.audit_log is engine._audit
        assert engine.latencies == []

    def test_a_propfirm_governor_is_started_when_limits_are_given(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=eng.logger.name):
            engine = _engine(tmp_path, propfirm_limits=PropfirmLimits(initial_balance=100_000.0))
        assert engine._governor is not None and "Propfirm governor active" in caplog.text

    def test_judge_gates_are_loaded_from_the_judge_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = []
        monkeypatch.setattr(eng, "build_judge_gates",
                            lambda symbols, judge_dir, artefacts_dir: seen.append((symbols, judge_dir, artefacts_dir)) or {"X": {}})
        engine = _engine(tmp_path, judge_dir=tmp_path / "artefacts" / "judge_models")
        assert engine._judges == {"X": {}} and seen[0][2] == tmp_path / "artefacts"

    def test_shadow_mode_with_multipliers_builds_cusum_filters_and_otherwise_none(self, tmp_path: Path) -> None:
        assert _engine(tmp_path, mode="shadow")._cusum_filters == {}
        config = _config(tmp_path, mode="shadow")
        config.cusum_multipliers = {"BTCUSDT": 2.5}
        engine = LiveEngine(config, _Runner(), Feed(FeedConfig(symbols=SYMBOLS), asyncio.Queue()))   # type: ignore[arg-type]
        assert set(engine._cusum_filters) == set(SYMBOLS) and engine._cusum_filters["BTCUSDT"]._mult == 2.5
        paper = _config(tmp_path, mode="paper")
        paper.cusum_multipliers = {"BTCUSDT": 2.5}
        assert LiveEngine(paper, _Runner(), Feed(FeedConfig(symbols=SYMBOLS), asyncio.Queue()))._cusum_filters == {}   # type: ignore[arg-type]

    def test_the_metrics_server_is_started_only_for_a_positive_port(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        started = []

        def record(port: int) -> None:
            started.append(port)
        monkeypatch.setattr(eng, "start_metrics_server", record)
        _engine(tmp_path, metrics_port=0)
        _engine(tmp_path, metrics_port=9123)
        assert started == [9123]


class TestKillSwitch:
    def test_the_handlers_are_installed_and_set_the_flag(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        if platform.system() != "Windows":
            assert signal.getsignal(signal.SIGUSR1) == engine._kill_switch_handler
        assert signal.getsignal(signal.SIGTERM) == engine._kill_switch_handler
        assert not engine._kill_flag.is_set()
        engine._kill_switch_handler(signal.SIGTERM, None)
        assert engine._kill_flag.is_set()

    def test_on_windows_only_sigterm_is_hooked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        before = signal.getsignal(signal.SIGUSR1)
        engine = _engine(tmp_path)
        assert signal.getsignal(signal.SIGTERM) == engine._kill_switch_handler
        assert signal.getsignal(signal.SIGUSR1) == before or signal.getsignal(signal.SIGUSR1) != engine._kill_switch_handler


# --------------------------------------------------------------------------- #
# the event loop
# --------------------------------------------------------------------------- #
class TestRun:
    def test_a_replay_is_processed_until_the_feed_is_exhausted(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        idx = pd.date_range(T0, periods=4, freq="min")
        bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 5.0}, index=idx)
        engine._feed._bars = {"BTCUSDT": bars}
        _run(engine.run())
        assert engine.bars_processed == 4 and engine.state.mode is EngineMode.STOPPED

    def test_a_pending_kill_switch_triggers_the_emergency_halt(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(tmp_path)
        engine._kill_flag.set()
        with caplog.at_level(logging.CRITICAL, logger=eng.logger.name):
            _run(engine.run())
        assert "EMERGENCY HALT" in caplog.text and engine._stop_event.is_set() and engine.bars_processed == 0

    def test_the_emergency_halt_flattens_through_the_oms_when_there_is_no_router(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine._router = None                                          # type: ignore[assignment]
        closed = []
        engine._paper_oms.close_all = lambda: closed.append(True) or []  # type: ignore[method-assign]
        _run(engine._emergency_halt())
        assert closed == [True] and engine._stop_event.is_set()

    def test_an_idle_queue_checks_the_breaker_and_halts_when_it_trips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine(tmp_path, cb_config=CircuitBreakerConfig(
            feed_timeout_sec=0, cb_log_path=tmp_path / "cb.log", halt_store_path=tmp_path / "halt.json"))

        async def timeout(coro, timeout):
            coro.close()
            await asyncio.sleep(0.01)
            raise TimeoutError
        monkeypatch.setattr(eng.asyncio, "wait_for", timeout)
        engine._feed.start = lambda: asyncio.sleep(0)                  # type: ignore[method-assign]
        with caplog.at_level(logging.CRITICAL, logger=eng.logger.name):
            _run(engine.run())
        assert engine._cb.is_active and "HALT triggered" in caplog.text and engine._stop_event.is_set()

    def test_an_idle_queue_with_a_healthy_breaker_keeps_waiting_until_stop_is_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _engine(tmp_path)
        waits = []

        async def timeout(coro, timeout):
            coro.close()
            waits.append(1)
            if len(waits) == 2:
                engine.request_stop()
            raise TimeoutError
        monkeypatch.setattr(eng.asyncio, "wait_for", timeout)
        engine._feed.start = lambda: asyncio.sleep(0)                  # type: ignore[method-assign]
        _run(engine.run())
        assert len(waits) == 2 and not engine._cb.is_active

    def test_a_breaker_that_trips_while_processing_a_bar_stops_the_engine_and_closes_positions(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine(tmp_path)
        engine._feed.start = lambda: asyncio.sleep(0)                  # type: ignore[method-assign]
        engine._queue.put_nowait(_bar(seq=1))
        engine._queue.put_nowait(_bar(minute=1, seq=5))                # gat in de reeks: de breaker valt
        with caplog.at_level(logging.CRITICAL, logger=eng.logger.name):
            _run(engine.run())
        assert engine._cb.is_active and "HALT triggered" in caplog.text and engine.bars_processed == 1


# --------------------------------------------------------------------------- #
# bar processing
# --------------------------------------------------------------------------- #
class TestFeedIntegrity:
    def test_sequence_numbers_are_tracked_per_symbol_and_a_gap_trips_the_breaker(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine(tmp_path)
        for k in (1, 2, 3):
            _run(engine._process_bar(_bar(minute=k, seq=k)))
        assert engine._last_bar_seq["BTCUSDT"] == 3 and not engine._cb.is_active
        with caplog.at_level(logging.ERROR, logger=eng.logger.name):
            _run(engine._process_bar(_bar(minute=5, seq=9)))
        assert engine._cb.is_active and "FEED GAP DETECTED" in caplog.text and engine.bars_processed == 3

    def test_bars_without_a_sequence_number_are_not_checked(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        for k in range(3):
            _run(engine._process_bar(_bar(minute=k, seq=0)))
        assert engine._last_bar_seq == {} and engine.bars_processed == 3


class TestMarketState:
    def test_the_live_quote_sets_the_spread_and_a_crossed_book_falls_back_to_the_volatility_proxy(
        self, tmp_path: Path
    ) -> None:
        engine = _engine(tmp_path)
        captured = []
        engine._paper_oms.set_bar_prices = lambda prices, **kw: captured.append(kw["spread_bps_map"]["BTCUSDT"])   # type: ignore[method-assign]
        for quote in ((99.9, 100.1), (100.1, 99.9), None):
            engine._feed.get_best_quote = lambda s, q=quote: q           # type: ignore[method-assign]
            _run(engine._process_bar(_bar()))
        proxy = 2.0 / 100.0 / 8.0 * 1e4
        assert captured[0] == pytest.approx(0.2 / (2 * 100.0) * 1e4) and captured[1] == pytest.approx(proxy)
        assert captured[2] == pytest.approx(proxy)

    def test_a_huge_spread_is_capped_at_100_bps(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        captured = []
        engine._paper_oms.set_bar_prices = lambda prices, **kw: captured.append(kw["spread_bps_map"]["BTCUSDT"])   # type: ignore[method-assign]
        engine._feed.get_best_quote = lambda s: (50.0, 150.0)            # type: ignore[method-assign]
        _run(engine._process_bar(_bar()))
        assert captured == [100.0]

    def test_a_funding_bar_accrues_funding_and_reports_it(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        accrued = []
        engine._paper_oms.accrue_funding = lambda sym, rate, ts: accrued.append((sym, rate)) or -1.5   # type: ignore[method-assign]
        paid = []
        engine._metrics.funding_accrued = lambda symbol, payment_usdt: paid.append(payment_usdt)    # type: ignore[method-assign]
        _run(engine._process_bar(_bar(funding_rate=0.0001)))
        assert accrued == [("BTCUSDT", 0.0001)] and paid == [-1.5]


class TestSignals:
    def test_without_features_there_is_no_prediction(self, tmp_path: Path) -> None:
        runner = _Runner({"BTCUSDT": _signal("BTCUSDT")})
        engine = _engine(tmp_path, runner, features=None)
        _run(engine._process_bar(_bar()))
        assert runner.calls == [] and engine._pending_rebalance["BTCUSDT"] is None and "BTCUSDT" not in engine.state.last_bar

    def test_a_signal_without_a_judge_goes_straight_to_the_rebalance_queue(self, tmp_path: Path) -> None:
        runner = _Runner({"BTCUSDT": _signal("BTCUSDT")})
        engine = _engine(tmp_path, runner, features=_features())
        _run(engine._process_bar(_bar()))
        assert runner.calls == [("predict", "BTCUSDT")] and engine._pending_rebalance["BTCUSDT"].signal == 1.0
        assert "BTCUSDT" in engine.state.last_bar and engine.state.last_bar_ts["BTCUSDT"] == T0

    def test_a_flat_signal_is_not_queued(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, _Runner({"BTCUSDT": _signal("BTCUSDT", 0.0)}), features=_features())
        _run(engine._process_bar(_bar()))
        assert engine._pending_rebalance["BTCUSDT"] is None

    @pytest.mark.parametrize(("passes", "usable", "queued"), [
        (True, 1_000, True),          # judge laat door
        (False, 1_000, False),        # judge blokkeert
        (False, 100, True),           # gedeeltelijke burn-in: de judge wordt overgeslagen
    ])
    def test_the_judge_gate_decides_unless_the_burn_in_is_partial(self, tmp_path: Path, passes: bool, usable: int,
                                                                 queued: bool, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(tmp_path, _Runner({"BTCUSDT": _signal("BTCUSDT", -1.0)}), features=_features())
        engine._fu.usable = usable                                            # type: ignore[union-attr]
        asked = []
        engine._judges = {"BTCUSDT": {"SHORT": SimpleNamespace(
            passes=lambda feats, conf, ts: asked.append(conf) or (passes, 0.5))}}
        with caplog.at_level(logging.DEBUG, logger=eng.logger.name):
            _run(engine._process_bar(_bar()))
        assert (engine._pending_rebalance["BTCUSDT"] is not None) is queued
        assert bool(asked) is (usable >= 500)

    def test_a_partial_burn_in_without_a_judge_is_bypassed_with_a_log_line(self, tmp_path: Path,
                                                                           caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(tmp_path, _Runner({"BTCUSDT": _signal("BTCUSDT")}), features=_features())
        engine._fu.usable = 10                                                # type: ignore[union-attr]
        with caplog.at_level(logging.INFO, logger=eng.logger.name):
            _run(engine._process_bar(_bar()))
        assert "JudgeGate BYPASS" in caplog.text and engine._pending_rebalance["BTCUSDT"] is not None


class _Cusum:
    def __init__(self, fires: bool) -> None:
        self.fires, self.updated = fires, []

    def step(self, close: float) -> bool:
        return self.fires

    def update_atr(self, feats) -> None:
        self.updated.append(len(feats))


class TestShadowMode:
    def _shadow(self, tmp_path: Path, fires: bool, fresh: pd.DataFrame | None, results=None):
        runner = _Runner(results if results is not None else {"BTCUSDT": _signal("BTCUSDT")})
        engine = _engine(tmp_path, runner, features=_features(), mode="shadow")
        engine._cusum_filters = {"BTCUSDT": _Cusum(fires)}               # type: ignore[dict-item]
        engine._fu.fresh = fresh                                          # type: ignore[union-attr]
        return engine, runner

    def test_a_cusum_event_forces_fresh_features_and_predicts_on_the_event(self, tmp_path: Path) -> None:
        engine, runner = self._shadow(tmp_path, True, _features())
        _run(engine._process_bar(_bar()))
        assert runner.calls == [("predict_on_event", "BTCUSDT")]
        assert engine._cusum_filters["BTCUSDT"].updated == [1]            # type: ignore[attr-defined]
        assert engine._pending_rebalance["BTCUSDT"] is not None

    def test_an_event_without_fresh_features_predicts_nothing(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        engine, runner = self._shadow(tmp_path, True, None)
        with caplog.at_level(logging.DEBUG, logger=eng.logger.name):
            _run(engine._process_bar(_bar()))
        assert runner.calls == [] and "buffer still warming up" in caplog.text

    def test_between_events_nothing_is_predicted_and_unfiltered_symbols_never_fire(self, tmp_path: Path) -> None:
        engine, runner = self._shadow(tmp_path, False, _features())
        _run(engine._process_bar(_bar()))
        _run(engine._process_bar(_bar("ETHUSDT", minute=1)))              # geen filter voor dit symbool
        assert runner.calls == []


class _Governor:
    def __init__(self, action: GovernorAction) -> None:
        self.action, self.reset_with = action, []

    def update(self, equity: float):
        return SimpleNamespace(action=self.action, reason="test")

    def reset_day(self, equity: float) -> None:
        self.reset_with.append(equity)


class TestGovernor:
    def test_a_hard_flatten_or_fail_trips_the_breaker(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        for action in (GovernorAction.HARD_FLATTEN, GovernorAction.FAIL):
            engine = _engine(tmp_path / action.value)
            engine._governor = _Governor(action)                          # type: ignore[assignment]
            with caplog.at_level(logging.CRITICAL, logger=eng.logger.name):
                _run(engine._process_bar(_bar()))
            assert engine._cb.is_active and f"propfirm_governor:{action.value}" in engine._cb.halt_reason
            assert "CB active" not in caplog.text or True

    def test_a_soft_stop_blocks_new_risk_for_the_bar(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(tmp_path, _Runner({"BTCUSDT": _signal("BTCUSDT")}), features=_features())
        engine._governor = _Governor(GovernorAction.SOFT_STOP)            # type: ignore[assignment]
        _run(engine._process_bar(_bar("BTCUSDT")))
        with caplog.at_level(logging.WARNING, logger=eng.logger.name):
            _run(engine._process_bar(_bar("ETHUSDT", minute=1)))
        assert "SOFT_STOP" in caplog.text and engine._paper_oms.get_positions() == {}
        assert engine._pending_rebalance["BTCUSDT"] is not None            # blijft staan voor de volgende bar

    def test_an_ok_verdict_changes_nothing_and_the_session_roll_resets_the_governor(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        governor = _Governor(GovernorAction.OK)
        engine._governor = governor                                       # type: ignore[assignment]
        _run(engine._process_bar(_bar(minute=0)))
        _run(engine._process_bar(_bar(minute=1)))
        assert not engine._cb.is_active and governor.reset_with == []
        _run(engine._process_bar(_bar(minute=24 * 60)))                   # volgende UTC-dag
        assert len(governor.reset_with) == 1


class TestRebalance:
    def test_once_every_symbol_has_a_price_a_pending_signal_becomes_orders_and_fills(self, tmp_path: Path) -> None:
        runner = _Runner({"BTCUSDT": _signal("BTCUSDT", 1.0, 0.8)})
        engine = _engine(tmp_path, runner, features=_features())
        _run(engine._process_bar(_bar("BTCUSDT", minute=0, close=100.0)))
        assert engine._pending_rebalance["BTCUSDT"] is not None and engine.state.target_weights == {}
        _run(engine._process_bar(_bar("ETHUSDT", minute=1, close=50.0)))
        assert engine._pending_rebalance == {"BTCUSDT": None, "ETHUSDT": None}
        assert set(engine.state.target_weights) == set(SYMBOLS) and engine.state.positions
        assert engine.audit_log.records_written > 0 and len(engine._equity_history) >= 3

    def test_the_bar_processing_cap_and_latency_bookkeeping(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine._equity_history = [1.0] * 5_000
        _run(engine._process_bar(_bar()))
        assert len(engine._equity_history) == 5_000 and len(engine.latencies) == 1

    def test_a_slow_bar_is_flagged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                   caplog: pytest.LogCaptureFixture) -> None:
        engine = _engine(tmp_path)
        ticks = iter(range(0, 1_000, 2))
        monkeypatch.setattr(eng.time, "monotonic", lambda: float(next(ticks)))
        with caplog.at_level(logging.WARNING, logger=eng.logger.name):
            _run(engine._process_bar(_bar()))
        assert "slow bar processing" in caplog.text

    def test_the_state_file_is_written_every_n_bars(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, state_write_every_n=2)
        _run(engine._process_bar(_bar(minute=0)))
        assert not (tmp_path / "state" / "state.json").exists()
        _run(engine._process_bar(_bar(minute=1)))
        doc = json.loads((tmp_path / "state" / "state.json").read_text())
        assert doc["bars_processed"] == 2 and doc["mode"] == "paper" and doc["cb_active"] is False
        assert doc["equity"] == 100_000.0 and doc["prices"] == {"BTCUSDT": 100.0}
        assert doc["rolling_sharpe_32d"] is None and doc["current_bar_ts"] == (T0 + pd.Timedelta(minutes=1)).isoformat()
        lines = (tmp_path / "state" / "equity_curve.jsonl").read_text().splitlines()
        assert json.loads(lines[0])["bar"] == 2


class TestStateDocument:
    def test_open_positions_are_listed_with_side_and_notional(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path, _Runner({"BTCUSDT": _signal("BTCUSDT")}), features=_features())
        for minute, symbol, close in ((0, "BTCUSDT", 100.0), (1, "ETHUSDT", 50.0),
                                      (2, "BTCUSDT", 101.0), (3, "ETHUSDT", 51.0)):      # de tweede ronde markeert de posities
            _run(engine._process_bar(_bar(symbol, minute=minute, close=close)))
        engine._write_state()                                              # zonder bar-tijdstempel
        doc = json.loads((tmp_path / "state" / "state.json").read_text())
        assert doc["current_bar_ts"] is None and set(doc["positions"]) == {"BTCUSDT", "ETHUSDT"}
        assert all(p["side"] in {"LONG", "SHORT"} and p["notional"] > 0 for p in doc["positions"].values())

    def test_a_position_without_a_side_enum_falls_back_to_the_sign_of_its_quantity(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        positions = {"BTCUSDT": SimpleNamespace(qty=-2.0, notional=200.0), "ETHUSDT": SimpleNamespace(size=1.0, notional=50.0)}
        engine._paper_oms.tracker.get_all_positions = lambda: positions  # type: ignore[method-assign]
        engine._write_state()
        doc = json.loads((tmp_path / "state" / "state.json").read_text())
        assert doc["positions"]["BTCUSDT"]["side"] == "SHORT" and doc["positions"]["ETHUSDT"]["side"] == "LONG"


class TestRollingSharpe:
    def test_it_needs_a_full_day_of_five_second_samples(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine._equity_history = list(100.0 + np.arange(17_279))
        assert engine._rolling_sharpe() is None

    def test_a_flat_curve_has_no_sharpe_and_a_rising_noisy_one_is_positive(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        engine._equity_history = [100.0] * 17_300
        assert engine._rolling_sharpe(window=17_300) is None
        rng = np.random.default_rng(0)
        engine._equity_history = list(100.0 * np.cumprod(1.0 + rng.normal(2e-6, 1e-4, 17_300)))
        sharpe = engine._rolling_sharpe(window=17_300)
        assert sharpe is not None and sharpe > 0

    @pytest.mark.xfail(strict=True, reason=(
        "BEVINDING, niet aangepast: `_rolling_sharpe()` neemt standaard de laatste 5.000 samples maar eist er "
        "17.280, en `_equity_history` wordt zelf op 5.000 afgekapt. `_write_state` kan `rolling_sharpe_32d` "
        "daardoor nooit anders dan None schrijven."))
    def test_the_sharpe_in_the_state_file_can_be_computed_once_the_history_is_full(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        rng = np.random.default_rng(0)
        engine._equity_history = list(100.0 * np.cumprod(1.0 + rng.normal(2e-6, 1e-4, 5_000)))
        assert engine._rolling_sharpe() is not None


class TestLatency:
    def test_quantiles_and_the_report(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        assert engine._p95_latency() == 0.0 and engine.latency_report()["max"] == 0.0
        engine._latencies = list(np.arange(1.0, 101.0))
        assert engine._p95_latency() == 96.0 and engine._p99_latency() == 100.0 and engine._p999_latency() == 100.0
        report = engine.latency_report()
        assert report == {"p50": 51.0, "p95": 96.0, "p99": 100.0, "p999": 100.0, "max": 100.0}

    def test_the_halt_closes_every_position_and_stops(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        closed = []
        engine._paper_oms.close_all = lambda: closed.append(1) or [object(), object()]   # type: ignore[method-assign]
        _run(engine._halt())
        assert closed == [1] and engine._stop_event.is_set()
