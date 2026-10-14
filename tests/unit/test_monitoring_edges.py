# tests/unit/test_monitoring_edges.py
"""De monitorpaden die de hoofdtests niet raken.

`monitoring/drift.py` had tot nu toe alleen een smoke-test met de `slow`-marker,
en die telt niet in de snelle suite. De drift-maten staan hier daarom met
verdelingen waarvan het antwoord vooraf vaststaat: dezelfde verdeling geeft
geen drift, een verschoven verdeling wel, en te weinig data geeft geen oordeel
(NaN) in plaats van een nul die als "in orde" gelezen zou worden.

Slack wordt niet gebeld: `urlopen` wordt vervangen door een stub die de
HTTP-status teruggeeft die de test kiest.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.monitoring import alerts, drift
from tradebot.monitoring.alerts import Alert, AlertRouter, AlertSeverity
from tradebot.monitoring.cross_account import CrossAccountConfig, CrossAccountMonitor
from tradebot.monitoring.execution_drift import ExecutionDriftMonitor
from tradebot.monitoring.feature_health import HealthIssue, check_feature_health
from tradebot.monitoring.metrics import EngineMetrics
from tradebot.monitoring.prob_calibration import (
    CalibrationResult,
    expected_calibration_error,
    reliability_diagram_data,
)
from tradebot.monitoring.sharpe_monitor import LiveSharpeMonitor
from tradebot.monitoring.vol_forecast_monitor import VolForecastMonitor
from tradebot.oms.order import Fill, Order, OrderSide, OrderType

RNG = np.random.default_rng(23)


@pytest.fixture(autouse=True)
def _fresh_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elke test begint met dezelfde, verse generator, zodat de data niet van
    de testvolgorde afhangt (pytest-randomly schudt die per run)."""
    monkeypatch.setattr(f"{__name__}.RNG", np.random.default_rng(23))


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
class TestDriftMeasures:
    def test_too_little_data_gives_no_verdict(self) -> None:
        few = np.arange(5, dtype=float)
        assert math.isnan(drift.psi(few, few))
        assert all(math.isnan(v) for v in drift.ks_drift(few, few))
        assert math.isnan(drift.wasserstein_drift(few, few))

    def test_the_same_distribution_does_not_drift(self) -> None:
        ref, cur = RNG.normal(size=2000), RNG.normal(size=2000)
        assert drift.psi(ref, cur) < drift.PSI_MODERATE
        assert drift.ks_drift(ref, cur)[1] > 0.01
        assert drift.wasserstein_drift(ref, cur) < 0.2

    def test_a_shifted_distribution_drifts(self) -> None:
        ref, cur = RNG.normal(size=2000), RNG.normal(2.0, 1.0, size=2000)
        assert drift.psi(ref, cur) >= drift.PSI_CRITICAL
        assert drift.ks_drift(ref, cur)[1] < 1e-6
        assert drift.wasserstein_drift(ref, cur) == pytest.approx(2.0, abs=0.2)

    def test_reference_stability_flags_a_regime_break_and_short_windows(self) -> None:
        stable = RNG.normal(size=1000)
        broken = np.concatenate([RNG.normal(size=500), RNG.normal(3.0, 1.0, 500)])
        result = {s.feature: s for s in drift.assess_reference_stability(
            {"stable": stable, "broken": broken, "short": np.ones(10)})}
        assert result["stable"].is_stable is True
        assert result["broken"].is_stable is False
        assert result["short"].is_stable is False and math.isnan(result["short"].intra_psi)

    def test_check_feature_drift_sorts_by_psi_and_warns(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ref = {"calm": RNG.normal(size=1000),
               "moved": RNG.normal(size=1000),
               "broken": np.concatenate([RNG.normal(size=500),
                                         RNG.normal(4.0, 1.0, 500)])}
        cur = {"calm": RNG.normal(size=1000),
               "moved": RNG.normal(0.6, 1.0, size=1000),
               "broken": RNG.normal(size=1000),
               "new": RNG.normal(size=1000)}
        with caplog.at_level(logging.INFO):
            results = drift.check_feature_drift(ref, cur)
        assert [r.feature for r in results][-1] == "new"  # NaN PSI achteraan
        by_name = {r.feature: r for r in results}
        assert by_name["calm"].is_drifted is False
        assert by_name["moved"].is_drifted is True
        assert "unstable reference" in caplog.text
        assert drift.check_feature_drift(ref, cur, assess_reference=False)


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
class _Response:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


class TestSlack:
    def test_a_non_https_webhook_is_refused(self) -> None:
        router = AlertRouter(slack_webhook="file:///tmp/alerts")
        with pytest.raises(ValueError, match="https"):
            router.send(Alert("t", "m", AlertSeverity.INFO))

    @pytest.mark.parametrize("status", [200, 500])
    def test_the_payload_is_posted_and_a_failure_is_logged(
        self, status: int, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        sent: list[object] = []

        def _urlopen(req, timeout):
            sent.append(req)
            return _Response(status)

        monkeypatch.setattr(alerts.urllib_request, "urlopen", _urlopen)
        router = AlertRouter(slack_webhook="https://hooks.example/abc")
        with caplog.at_level(logging.WARNING):
            router.send(Alert("drawdown", "8 %", AlertSeverity.WARNING,
                              metadata={"dd": 0.08}))
        assert sent and sent[0].get_method() == "POST"
        assert ("Slack alert failed" in caplog.text) is (status == 500)

    def test_the_default_router_can_be_configured(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        router = AlertRouter()
        monkeypatch.setattr(alerts, "_default_router", None)
        alerts.configure_default_router(router)
        assert alerts._default_router is router


# --------------------------------------------------------------------------- #
# Kleinere monitors
# --------------------------------------------------------------------------- #
class TestCrossAccount:
    def test_an_invalid_correlation_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="corr_threshold"):
            CrossAccountConfig(corr_threshold=0.0)

    def test_flat_or_short_histories_are_not_throttled(self) -> None:
        monitor = CrossAccountMonitor(CrossAccountConfig(min_obs=3, corr_window=5))
        assert monitor._sharpe(deque([0.01])) == 0.0
        assert monitor._sharpe(deque([0.01, 0.01, 0.01])) == 0.0
        for r in (0.01, 0.02, 0.03, 0.04):
            monitor.update({"flat": 0.0, "live": r})
        assert monitor.throttles() == {"flat": 1.0, "live": 1.0}


def test_execution_drifts_are_kept_and_serialised() -> None:
    order = Order(order_id="o1", symbol="BTC", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, qty_base=1.0, signal_prob=0.6,
                  kelly_fraction=0.1, model_version="v", git_sha="g",
                  feature_hash="f", portfolio_weight=0.1)
    order.created_at = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    fill = Fill(order_id="o1", fill_price=100.5, fill_qty=1.0,
                fill_ts=pd.Timestamp("2024-01-01 00:00:02", tz="UTC"),
                notional_usdt=100.5)
    monitor = ExecutionDriftMonitor()
    record = monitor.record(order, fill, expected_price=100.0).as_record()
    assert record["order_id"] == "o1"
    assert len(monitor.drifts) == 1


class TestFeatureHealth:
    def test_nan_inf_and_constant_columns_are_reported(self) -> None:
        frame = pd.DataFrame({
            "gappy": [np.nan, np.nan, 1.0, 2.0],
            "infinite": [1.0, np.inf, 2.0, 3.0],
            "constant": [5.0, 5.0, 5.0, 5.0],
            "fine": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "c", "d"],
        })
        report = check_feature_health(frame, nan_threshold=0.1)
        kinds = {(i.feature, i.issue_type) for i in report.issues}
        assert kinds == {("gappy", "nan_rate"), ("infinite", "inf_rate"),
                         ("constant", "zero_variance")}
        assert report.n_issues == 3
        assert report.has_critical is True
        assert str(HealthIssue("x", "nan_rate", 0.5, 0.1)) == (
            "HealthIssue(x: nan_rate=0.5000 > 0.1000)")


class TestCalibration:
    def test_one_sample_has_no_ece(self) -> None:
        assert math.isnan(expected_calibration_error([0.5], [1]))

    def test_the_reliability_diagram_marks_empty_bins(self) -> None:
        data = reliability_diagram_data([0.05, 0.1, 0.9, 0.95], [0, 0, 1, 1], n_bins=5)
        assert data["counts"] == [2, 0, 0, 0, 2]
        assert math.isnan(data["accuracy"][2])
        assert data["accuracy"][0] == 0.0 and data["accuracy"][4] == 1.0

    def test_the_result_names_an_alert(self) -> None:
        result = CalibrationResult("BTC", "LONG", ece=0.12, n_samples=500,
                                   n_bins=10, is_alert=True)
        assert str(result) == "CalibrationResult(BTC/LONG: ECE=0.1200 [ALERT])"


def test_the_metrics_facade_drives_every_gauge() -> None:
    metrics = EngineMetrics()
    metrics.bar_processed("BTC", 0.01)
    metrics.equity_updated(100_000.0, 0.02)
    metrics.order_filled("BTC", "BUY")
    metrics.funding_accrued("BTC", -1.5)
    metrics.circuit_breaker_state(True)
    metrics.queue_depth(3)


def test_the_live_sharpe_has_no_value_on_degenerate_input() -> None:
    assert math.isnan(LiveSharpeMonitor._compute_sharpe(np.array([0.01])))
    with np.errstate(invalid="ignore"):  # opzet: een oneindige return
        assert math.isnan(LiveSharpeMonitor._compute_sharpe(np.array([np.inf, 0.01])))


def test_the_vol_forecast_verdict_serialises() -> None:
    monitor = VolForecastMonitor(baseline_qlike=0.5)
    for _ in range(5):
        monitor.record(1e-4, 1e-4)
    record = monitor.verdict().as_record()
    assert record["n_obs"] == 5
    assert record["is_conclusive"] is False
    assert SimpleNamespace(**record).breached == []


def test_a_moderate_shift_is_drift_but_not_critical(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ref = np.random.default_rng(23).normal(size=1000)
    cur = np.random.default_rng(5).normal(0.3, 1.0, size=1000)
    with caplog.at_level(logging.INFO):
        (result,) = drift.check_feature_drift({"f": ref}, {"f": cur},
                                              assess_reference=False)
    assert drift.PSI_MODERATE <= result.psi_value < drift.PSI_CRITICAL
    assert result.is_drifted is True and result.is_critical is False
    assert "Moderate drift" in caplog.text
