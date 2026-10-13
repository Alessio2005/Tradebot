# tests/unit/test_live_small_edges.py
"""De kleine `live/`-modules: CUSUM-filter, signaalhandlers, exchange-statusmonitor, JudgeGate en SystemState.

Geen netwerk: de statusmonitor krijgt een nep-`aiohttp.ClientSession`, de
JudgeGate een echt maar piepklein CatBoost-model op schijf.
"""
from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
import pytest

from tradebot.live import exchange_status as xs
from tradebot.live import sigterm
from tradebot.live.cusum_filter import CUSUMFilter, build_cusum_filters
from tradebot.live.judge_gate import JudgeGate, build_judge_gates
from tradebot.live.state import SystemState

# --------------------------------------------------------------------------- #
# CUSUMFilter
# --------------------------------------------------------------------------- #


def _feature_parquet(path: Path, **columns: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_parquet(path)
    return path


class TestCusumWarmup:
    def test_a_missing_parquet_is_a_cold_start(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        f = CUSUMFilter(2.0, "BTC")
        with caplog.at_level(logging.WARNING, logger="tradebot.live.cusum_filter"):
            f.warmup_from_parquet(tmp_path / "missing.parquet")
        assert f._last_atr is None and f._prev_close is None and "cold start" in caplog.text

    def test_the_volatility_proxy_and_the_last_close_seed_the_filter(self, tmp_path: Path) -> None:
        path = _feature_parquet(tmp_path / "f.parquet", feat_vol_gk=[0.1, 0.2, 0.3], feat_atr=[9.0, 9.0, 9.0],
                                close=[10.0, 11.0, 12.0])
        f = CUSUMFilter(2.0)
        f.warmup_from_parquet(path)
        assert f._last_atr == pytest.approx(0.3) and f._prev_close == 12.0

    def test_atr_is_the_fallback_and_missing_columns_leave_the_state_empty(self, tmp_path: Path) -> None:
        f = CUSUMFilter(2.0)
        f.warmup_from_parquet(_feature_parquet(tmp_path / "a.parquet", feat_atr=[0.5, 0.6]))
        assert f._last_atr == pytest.approx(0.6) and f._prev_close is None
        g = CUSUMFilter(2.0)
        g.warmup_from_parquet(_feature_parquet(tmp_path / "b.parquet", other=[1.0, 2.0]))
        assert g._last_atr is None and g._prev_close is None

    def test_only_the_last_200_rows_are_read(self, tmp_path: Path) -> None:
        n = 500
        path = _feature_parquet(tmp_path / "f.parquet", feat_vol_gk=list(np.arange(n, dtype=float)),
                                close=list(np.arange(n, dtype=float)))
        f = CUSUMFilter(1.0)
        f.warmup_from_parquet(path)
        assert f._last_atr == n - 1


class TestCusumStep:
    def _filter(self, atr: float = 1.0, mult: float = 2.0, close: float = 100.0) -> CUSUMFilter:
        f = CUSUMFilter(mult)
        f._last_atr, f._prev_close = atr, close
        return f

    def test_a_cold_filter_only_remembers_the_close(self) -> None:
        f = CUSUMFilter(2.0)
        assert f.step(100.0) is False and f._prev_close == 100.0
        f._last_atr = 1.0
        assert f.step(100.5) is False                                  # nu gewapend

    def test_an_upward_drift_fires_once_the_cumulative_gain_reaches_the_threshold(self) -> None:
        f = self._filter()                                             # drempel = 1.0 * 2.0
        assert [f.step(c) for c in (100.8, 101.6, 102.0)] == [False, False, True]
        assert f._s_pos == 0.0 and f._s_neg == 0.0

    def test_a_downward_drift_fires_too(self) -> None:
        f = self._filter()
        assert [f.step(c) for c in (99.0, 98.0)] == [False, True]

    def test_the_sums_are_one_sided(self) -> None:
        f = self._filter(atr=10.0)
        f.step(101.0)
        f.step(100.0)
        assert (f._s_pos, f._s_neg) == (0.0, -1.0)            # +1 en daarna -1: de positieve som is terug op nul

    def test_a_vanishing_atr_still_has_a_positive_threshold(self) -> None:
        f = self._filter(atr=0.0)
        assert f.step(100.0) is False                                 # geen verandering, drempel 1e-8
        assert f.step(100.1) is True


class TestCusumAtrUpdate:
    def test_the_threshold_follows_the_fresh_event_features(self) -> None:
        f = CUSUMFilter(2.0)
        f.update_atr(pd.DataFrame({"feat_vol_gk": [0.1, 0.4]}))
        assert f._last_atr == pytest.approx(0.4)
        f.update_atr(pd.DataFrame({"feat_atr": [0.9]}))
        assert f._last_atr == pytest.approx(0.9)

    def test_missing_columns_and_tiny_values_leave_the_threshold_alone(self) -> None:
        f = CUSUMFilter(2.0)
        f._last_atr = 0.5
        f.update_atr(pd.DataFrame({"close": [1.0]}))
        f.update_atr(pd.DataFrame({"feat_vol_gk": [1e-12]}))
        assert f._last_atr == 0.5


class TestBuildCusumFilters:
    def test_one_warm_started_filter_per_symbol_with_a_default_multiplier(self, tmp_path: Path) -> None:
        _feature_parquet(tmp_path / "features" / "BTCUSDT.parquet", feat_vol_gk=[0.2], close=[50.0])
        filters = build_cusum_filters(["BTCUSDT", "ETHUSDT"], {"BTCUSDT": 2.5}, tmp_path)
        assert filters["BTCUSDT"]._mult == 2.5 and filters["BTCUSDT"]._prev_close == 50.0
        assert filters["ETHUSDT"]._mult == 2.0 and filters["ETHUSDT"]._last_atr is None


# --------------------------------------------------------------------------- #
# signal handlers
# --------------------------------------------------------------------------- #
class _Loop:
    def __init__(self) -> None:
        self.handlers: dict[int, object] = {}
        self.stopped = False

    def add_signal_handler(self, sig: int, callback) -> None:
        self.handlers[sig] = callback

    def stop(self) -> None:
        self.stopped = True


class TestSignalHandlers:
    def test_on_posix_sigterm_and_sigint_stop_the_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sigterm.sys, "platform", "linux")
        loop = _Loop()
        sigterm.setup_signal_handlers(loop)                           # type: ignore[arg-type]
        assert set(loop.handlers) == {signal.SIGTERM, signal.SIGINT}
        loop.handlers[signal.SIGTERM]()                               # type: ignore[operator]
        assert loop.stopped

    def test_on_windows_only_sigterm_gets_a_synchronous_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        installed: dict[int, object] = {}
        monkeypatch.setattr(sigterm.sys, "platform", "win32")
        monkeypatch.setattr(sigterm.signal, "signal", installed.__setitem__)
        loop = _Loop()
        sigterm.setup_signal_handlers(loop)                           # type: ignore[arg-type]
        assert list(installed) == [signal.SIGTERM] and loop.handlers == {}
        installed[signal.SIGTERM](signal.SIGTERM, None)               # type: ignore[operator]
        assert loop.stopped


# --------------------------------------------------------------------------- #
# exchange status monitor
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status, self._payload = status, payload or {}

    async def json(self) -> dict:
        return self._payload

    async def __aenter__(self) -> _Resp:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Session:
    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.urls: list[str] = []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def get(self, url: str, timeout=None):
        self.urls.append(url)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _Breaker:
    def __init__(self) -> None:
        self.trips: list[tuple[str, str]] = []

    def trip(self, reason: str, config_key: str) -> None:
        self.trips.append((reason, config_key))


def _monitor(monkeypatch: pytest.MonkeyPatch, script: list) -> tuple[_Breaker, _Session]:
    session = _Session([*script, asyncio.CancelledError()])          # de laatste poll stopt de lus
    monkeypatch.setattr(xs.aiohttp, "ClientSession", lambda: session)

    async def no_sleep(_s: float) -> None:
        return None
    monkeypatch.setattr(xs.asyncio, "sleep", no_sleep)
    breaker = _Breaker()
    asyncio.run(xs.monitor_exchange_status(breaker, poll_interval=0))   # type: ignore[arg-type]
    return breaker, session


class TestExchangeStatusMonitor:
    def test_a_healthy_exchange_trips_nothing_and_the_probe_url_is_the_public_time_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        breaker, session = _monitor(monkeypatch, [_Resp(200, {"retCode": 0})])
        assert breaker.trips == [] and session.urls == [xs.BYBIT_STATUS_URL] * 2

    def test_a_non_zero_return_code_trips_the_breaker_with_the_message(self, monkeypatch: pytest.MonkeyPatch,
                                                                         caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.CRITICAL, logger=xs.logger.name):
            breaker, _ = _monitor(monkeypatch, [_Resp(200, {"retCode": 10016, "retMsg": "maintenance"})])
        assert breaker.trips == [("exchange_status:bybit:maintenance", "live.exchange_status")]
        assert "tripping CB" in caplog.text

    def test_a_missing_return_code_counts_as_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        breaker, _ = _monitor(monkeypatch, [_Resp(200, {})])
        assert breaker.trips == [("exchange_status:bybit:unknown", "live.exchange_status")]

    def test_a_non_200_status_is_a_warning_and_not_a_trip(self, monkeypatch: pytest.MonkeyPatch,
                                                           caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=xs.logger.name):
            breaker, _ = _monitor(monkeypatch, [_Resp(503)])
        assert breaker.trips == [] and "HTTP 503" in caplog.text

    @pytest.mark.parametrize("error", [xs.aiohttp.ClientError("reset"), TimeoutError(), OSError("down")])
    def test_transient_network_errors_are_logged_and_polling_continues(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error
    ) -> None:
        with caplog.at_level(logging.ERROR, logger=xs.logger.name):
            breaker, session = _monitor(monkeypatch, [error, _Resp(200, {"retCode": 0})])
        assert breaker.trips == [] and len(session.urls) == 3 and "status check failed" in caplog.text

    def test_cancellation_ends_the_monitor_quietly(self, monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=xs.logger.name):
            _monitor(monkeypatch, [])
        assert "monitor cancelled" in caplog.text


# --------------------------------------------------------------------------- #
# JudgeGate
# --------------------------------------------------------------------------- #
N_RAW = 3
TRAIN_COLS = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


@pytest.fixture(scope="module")
def judge_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Twee kleine modellen (LONG en SHORT) met 3 ruwe features + prob_primary + 4 tijdfeatures."""
    root = tmp_path_factory.mktemp("judge")
    rng = np.random.default_rng(0)
    for side in ("LONG", "SHORT"):
        X = rng.normal(size=(300, N_RAW + 5))
        y = (X[:, 0] + X[:, N_RAW] > 0).astype(int)
        model = cb.CatBoostClassifier(iterations=20, depth=3, verbose=False, random_seed=0, allow_writing_files=False)
        model.fit(X, y)
        model.save_model(str(root / f"ETHUSDT_{side}_judge.cbm"))
    return root


@pytest.fixture
def artefacts(tmp_path: Path) -> Path:
    _feature_parquet(tmp_path / "features" / "ETHUSDT.parquet", **{c: [0.0, 1.0] for c in TRAIN_COLS})
    return tmp_path


def _expected(gate: JudgeGate, raw: list[float], cal_prob: float, ts: pd.Timestamp) -> float:
    h, dow = ts.hour, ts.dayofweek
    x = np.array([*raw, cal_prob, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                  np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], dtype=np.float64).reshape(1, -1)
    return cal_prob * float(gate._model.predict_proba(x)[0][1])


TS = pd.Timestamp("2024-03-05 14:00", tz="UTC")


class TestJudgeGate:
    def test_a_missing_model_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="model not found"):
            JudgeGate("ETHUSDT", "LONG", tmp_path)

    def test_the_gate_multiplies_the_primary_probability_with_the_judge_probability(self, judge_dir: Path) -> None:
        gate = JudgeGate("ETHUSDT", "LONG", judge_dir)
        row = pd.DataFrame([[0.5, -0.2, 1.0, 9.0, 9.0]], columns=TRAIN_COLS)
        passed, p = gate.passes(row, 0.7, TS)
        assert p == pytest.approx(_expected(gate, [0.5, -0.2, 1.0], 0.7, TS), rel=1e-5)
        assert passed is bool(p > 0.40) and gate._n_feat == N_RAW + 5

    def test_the_training_layout_reorders_and_zero_fills_the_live_columns(self, judge_dir: Path, artefacts: Path) -> None:
        gate = JudgeGate("ETHUSDT", "SHORT", judge_dir, artefacts)
        assert gate._train_columns == TRAIN_COLS
        live = pd.DataFrame([[2.0, 7.0]], columns=["feat_c", "feat_a"])          # b ontbreekt, volgorde anders
        _, p = gate.passes(live, 0.8, TS)
        assert p == pytest.approx(_expected(gate, [7.0, 0.0, 2.0], 0.8, TS), rel=1e-5)

    def test_a_series_row_is_aligned_on_its_index(self, judge_dir: Path, artefacts: Path) -> None:
        gate = JudgeGate("ETHUSDT", "LONG", judge_dir, artefacts)
        series = pd.Series({"feat_b": 1.5, "feat_a": -1.0, "feat_c": 0.25})
        _, p = gate.passes(series, 0.6, TS)                                       # type: ignore[arg-type]
        assert p == pytest.approx(_expected(gate, [-1.0, 1.5, 0.25], 0.6, TS), rel=1e-5)

    def test_without_a_training_layout_the_row_is_sliced_positionally(self, judge_dir: Path, tmp_path: Path) -> None:
        gate = JudgeGate("ETHUSDT", "LONG", judge_dir, tmp_path)               # geen features-parquet
        assert gate._train_columns is None
        _, p = gate.passes(pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]), 0.5, TS)
        assert p == pytest.approx(_expected(gate, [1.0, 2.0, 3.0], 0.5, TS), rel=1e-5)

    def test_non_finite_values_are_neutralised_before_scoring(self, judge_dir: Path) -> None:
        gate = JudgeGate("ETHUSDT", "LONG", judge_dir)
        _, p = gate.passes(pd.DataFrame([[np.nan, np.inf, -np.inf]]), 0.5, TS)
        assert p == pytest.approx(_expected(gate, [0.0, 0.0, 0.0], 0.5, TS), rel=1e-5)

    def test_build_loads_every_available_model_and_skips_the_rest(self, judge_dir: Path,
                                                                    caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="tradebot.live.judge_gate"):
            gates = build_judge_gates(["ETHUSDT", "XRPUSDT"], judge_dir)
        assert list(gates) == ["ETHUSDT"] and set(gates["ETHUSDT"]) == {"LONG", "SHORT"}
        assert "skipping XRPUSDT/LONG" in caplog.text


# --------------------------------------------------------------------------- #
# SystemState
# --------------------------------------------------------------------------- #
class TestSystemState:
    def test_drawdown_and_daily_pnl_guard_against_a_zero_base(self) -> None:
        s = SystemState()
        s.equity_peak, s.equity = 0.0, 50.0
        assert s.current_drawdown == 0.0
        s.daily_pnl_open, s.daily_pnl = 0.0, 5.0
        assert s.daily_pnl_pct == 0.0

    def test_equity_updates_track_the_peak_and_both_drawdowns(self) -> None:
        s = SystemState()
        base = s.equity
        s.update_equity(base * 1.10)
        s.update_equity(base * 0.99)
        assert s.equity_peak == pytest.approx(base * 1.10) and s.current_drawdown == pytest.approx(0.10)
        assert s.peak_drawdown == pytest.approx(0.10) and s.intraday_peak_drawdown == pytest.approx(0.10)
        s.update_equity(base * 1.20)                                    # nieuwe top: instantaan 0, historie blijft
        assert s.current_drawdown == 0.0 and s.peak_drawdown == pytest.approx(0.10)
        assert s.daily_pnl == pytest.approx(base * 0.20) and s.daily_pnl_pct == pytest.approx(0.20)

    def test_a_zero_peak_skips_the_drawdown_bookkeeping(self) -> None:
        s = SystemState()
        s.equity_peak = s.intraday_peak_equity = -1.0
        s.update_equity(-5.0)
        assert s.peak_drawdown == 0.0 and s.intraday_peak_drawdown == 0.0

    def test_the_daily_reset_restarts_the_session_but_keeps_the_lifetime_worst_case(self) -> None:
        s = SystemState()
        base = s.equity
        s.update_equity(base * 1.10)
        s.update_equity(base * 0.99)
        s.reset_daily()
        assert s.daily_pnl == 0.0 and s.daily_pnl_open == pytest.approx(base * 0.99)
        assert s.intraday_peak_equity == pytest.approx(base * 0.99) and s.intraday_peak_drawdown == 0.0
        assert s.peak_drawdown == pytest.approx(0.10)
