# tests/unit/test_live_signal_runner.py
"""`live/signal_runner.py`: signalen per symbool, richtingsdisambiguatie en de IC-gewogen combinatie."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from tradebot.alpha.base import SignalResult
from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig
from tradebot.utils.failfast import TradebotContractError

TS = pd.Timestamp("2024-01-01 10:00", tz="UTC")
FEATURES = pd.DataFrame({"x": [1.0, 2.0]})


class _Sig:
    """Stub-alpha: vaste uitkomst, optioneel een ander symbool of een fout."""

    def __init__(self, sid: str, signal: float, confidence: float, symbol: str | None = "BTCUSDT",
                 raises: Exception | None = None, on_event: bool = False, accepts_bar_ts: bool = True) -> None:
        self.signal_id = sid
        if symbol is not None:
            self._cfg = SimpleNamespace(symbol=symbol)
        self._result = SignalResult(symbol or "BTCUSDT", TS, signal, confidence, 10, sid)
        self._raises, self.calls = raises, []
        if on_event:
            self.predict_on_event = self._predict_on_event
        if not accepts_bar_ts:
            self.predict = self._predict_without_bar_ts                      # type: ignore[method-assign]
            if on_event:
                self.predict_on_event = self._predict_on_event_without_bar_ts

    def _run(self, kind: str):
        self.calls.append(kind)
        if self._raises is not None:
            raise self._raises
        return self._result

    def predict(self, features, bar_ts=None):
        return self._run("predict")

    def _predict_on_event(self, features, bar_ts=None):
        return self._run("predict_on_event")

    def _predict_without_bar_ts(self, features):
        return self._run("predict")

    def _predict_on_event_without_bar_ts(self, features):
        return self._run("predict_on_event")


class TestConfig:
    def test_defaults(self) -> None:
        cfg = SignalRunnerConfig()
        assert cfg.use_combiner and cfg.ic_lookback == 60 and cfg.min_direction_delta == 0.02


@pytest.mark.parametrize("method", ["predict", "predict_on_event"])
class TestSelectionAndDispatch:
    def _run(self, runner: SignalRunner, method: str, symbol: str = "BTCUSDT"):
        return getattr(runner, method)(symbol, FEATURES, bar_ts=TS)

    def test_no_signal_for_the_symbol_gives_none(self, method: str) -> None:
        runner = SignalRunner([_Sig("a", 0.5, 0.6, symbol="ETHUSDT")])
        assert self._run(runner, method) is None
        assert self._run(SignalRunner([]), method) is None

    def test_signals_without_a_symbol_config_apply_to_every_symbol(self, method: str) -> None:
        runner = SignalRunner([_Sig("a", 0.5, 0.6, symbol=None)])
        assert self._run(runner, method, "XRPUSDT").signal_id == "a"

    def test_a_single_result_is_returned_as_is_and_so_is_the_first_without_the_combiner(self, method: str) -> None:
        a, b = _Sig("a", 0.5, 0.6), _Sig("b", 0.4, 0.9)
        assert self._run(SignalRunner([a]), method).signal_id == "a"
        assert self._run(SignalRunner([a, b], SignalRunnerConfig(use_combiner=False)), method).signal_id == "a"

    def test_a_signal_that_does_not_take_bar_ts_is_called_without_it(self, method: str) -> None:
        sig = _Sig("a", 0.5, 0.6, accepts_bar_ts=False, on_event=True)
        assert self._run(SignalRunner([sig]), method).signal_id == "a"

    def test_conflicting_directions_pick_the_more_convinced_side(self, method: str) -> None:
        longs, shorts = _Sig("long", 0.8, 0.70), _Sig("short", -0.8, 0.60)
        assert self._run(SignalRunner([longs, shorts]), method).signal_id == "long"
        assert self._run(SignalRunner([_Sig("long", 0.8, 0.60), _Sig("short", -0.8, 0.70)]), method).signal_id == "short"

    def test_conflicting_directions_with_nearly_equal_conviction_are_flat(self, method: str) -> None:
        runner = SignalRunner([_Sig("long", 0.8, 0.600), _Sig("short", -0.8, 0.610)])
        assert self._run(runner, method) is None

    def test_agreeing_signals_are_combined_into_one_named_after_the_symbol(self, method: str) -> None:
        runner = SignalRunner([_Sig("a", 0.4, 0.60), _Sig("b", 0.8, 0.75)])
        out = self._run(runner, method)
        assert out.signal_id == "BTCUSDT_combined" and out.confidence == 0.75 and out.symbol == "BTCUSDT"
        assert out.signal == pytest.approx(0.6) and out.horizon_bars == 10       # zonder IC-historie: gelijke gewichten

    def test_the_combiner_is_rebuilt_when_the_signal_set_changes(self, method: str) -> None:
        runner = SignalRunner([_Sig("a", 0.4, 0.60), _Sig("b", 0.8, 0.75)])
        self._run(runner, method)
        first = runner._combiner
        self._run(runner, method)
        assert runner._combiner is first                                # dezelfde set: hergebruikt
        runner._signals = [_Sig("a", 0.4, 0.60), _Sig("c", 0.8, 0.75)]
        self._run(runner, method)
        assert runner._combiner is not first and runner._combiner.signal_names == ["a", "c"]


class TestPredict:
    def test_a_failing_signal_is_a_contract_error_and_never_silently_dropped(self) -> None:
        runner = SignalRunner([_Sig("bad", 0.5, 0.6, raises=ValueError("boom"))])
        with pytest.raises(TradebotContractError, match=r"bad\.predict\(\) faalde: boom"):
            runner.predict("BTCUSDT", FEATURES)

    def test_a_type_error_inside_the_signal_is_retried_once_without_bar_ts_and_then_reported(self) -> None:
        sig = _Sig("bad", 0.5, 0.6, raises=TypeError("nope"))
        with pytest.raises(TradebotContractError):
            SignalRunner([sig]).predict("BTCUSDT", FEATURES, bar_ts=TS)
        assert sig.calls == ["predict", "predict"]

    def test_realised_returns_feed_the_combiner_once_enough_history_exists(self) -> None:
        runner = SignalRunner([_Sig("a", 0.4, 0.60), _Sig("b", 0.8, 0.75)], SignalRunnerConfig(ic_lookback=8))
        fits: list[tuple[int, int]] = []
        for k in range(8):
            runner.predict("BTCUSDT", FEATURES, fwd_return=0.001 * (k - 3))
            combiner = runner._combiner
            if not fits and not hasattr(combiner, "_spied"):
                original = combiner.fit
                combiner._spied = True
                combiner.fit = lambda sigs, rets, _o=original: (fits.append((len(sigs), len(rets))), _o(sigs, rets))[1]
        assert fits and all(5 <= n <= 8 and n == m for n, m in fits)             # lookback 8 begrenst het venster
        assert runner._combiner.lookback == 8
        assert len(runner._return_history) == 8 and set(runner._signal_history) == {"a", "b"}

    def test_without_realised_returns_nothing_is_fitted(self) -> None:
        runner = SignalRunner([_Sig("a", 0.4, 0.60), _Sig("b", 0.8, 0.75)])
        for _ in range(7):
            runner.predict("BTCUSDT", FEATURES)
        assert runner._return_history == [] and len(runner._signal_history["a"]) == 7


class TestPredictOnEvent:
    def test_the_event_entry_point_is_preferred_over_predict(self) -> None:
        sig = _Sig("a", 0.5, 0.6, on_event=True)
        SignalRunner([sig]).predict_on_event("BTCUSDT", FEATURES)
        assert sig.calls == ["predict_on_event"]
        plain = _Sig("p", 0.5, 0.6)
        SignalRunner([plain]).predict_on_event("BTCUSDT", FEATURES)
        assert plain.calls == ["predict"]

    def test_signals_that_return_nothing_are_ignored(self) -> None:
        mute = _Sig("mute", 0.0, 0.5, on_event=True)
        mute._result = None                                             # type: ignore[assignment]
        assert SignalRunner([mute]).predict_on_event("BTCUSDT", FEATURES) is None
        runner = SignalRunner([mute, _Sig("a", 0.5, 0.6, on_event=True)])
        assert runner.predict_on_event("BTCUSDT", FEATURES).signal_id == "a"

    def test_a_failing_signal_is_a_contract_error(self) -> None:
        runner = SignalRunner([_Sig("bad", 0.5, 0.6, raises=RuntimeError("down"), on_event=True)])
        with pytest.raises(TradebotContractError, match=r"predict_on_event\(\) faalde: down"):
            runner.predict_on_event("BTCUSDT", FEATURES)

    def test_a_failing_signal_without_an_id_is_still_reported(self) -> None:
        sig = _Sig("x", 0.5, 0.6, raises=RuntimeError("down"))
        del sig.signal_id
        with pytest.raises(TradebotContractError, match=r"\?\.predict_on_event"):
            SignalRunner([sig]).predict_on_event("BTCUSDT", FEATURES)
