# tests/unit/test_features_stationarity_gate_edges.py
"""`features/stationarity_gate.py`: kolomselectie, overgeslagen kolommen en de NaN-p-waarde.

De ADF-test zelf draait echt op een random walk en op witte ruis; alleen de twee
foutpaden (ADF faalt, ADF geeft NaN) gebruiken een stub.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.features import stationarity_gate as sg
from tradebot.features.stationarity_gate import StationarityReport, check_feature_stationarity


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_noise": rng.normal(size=n),
        "feat_walk": np.cumsum(rng.normal(size=n)),
        "feat_const": np.ones(n),
        "feat_trend": np.arange(float(n)),
        "label_y": np.arange(float(n)),
        "feat_text": ["x"] * n,
    })


class TestSelectionAndSkips:
    def test_the_default_selection_drops_identifier_prefixes_and_non_numeric_columns(self) -> None:
        _, report = check_feature_stationarity(_frame())
        assert set(report.p_values) == {"feat_noise", "feat_walk", "feat_trend"}
        assert report.skipped == ["feat_const"]                      # quasi-constant: ADF zinloos
        assert report.non_stationary == ["feat_walk", "feat_trend"]
        assert report.n_features_checked == 3 and report.n_non_stationary == 2 and not report.passed

    def test_an_explicit_column_list_ignores_unknown_and_non_numeric_names(self) -> None:
        _, report = check_feature_stationarity(_frame(), feature_cols=["feat_noise", "missing", "feat_text"])
        assert list(report.p_values) == ["feat_noise"] and report.passed

    def test_a_series_shorter_than_min_obs_is_skipped(self) -> None:
        _, report = check_feature_stationarity(_frame(60), feature_cols=["feat_noise"])
        assert report.skipped == ["feat_noise"] and report.n_features_checked == 0 and report.passed

    def test_a_clean_frame_passes_and_is_returned_untouched(self, caplog: pytest.LogCaptureFixture) -> None:
        df = _frame()[["feat_noise"]]
        with caplog.at_level(logging.INFO, logger=sg.logger.name):
            out, report = check_feature_stationarity(df, drop_if_non_stationary=True)
        assert out is df and report.passed and "gate PASS" in caplog.text


class TestDropping:
    def test_drop_removes_the_offenders_and_keeps_the_rest(self) -> None:
        df = _frame()
        out, report = check_feature_stationarity(df, drop_if_non_stationary=True)
        assert "feat_walk" not in out.columns and "feat_noise" in out.columns
        assert "feat_walk" in df.columns and report.non_stationary == ["feat_walk", "feat_trend"]

    def test_without_drop_the_same_frame_is_returned(self) -> None:
        df = _frame()
        out, _ = check_feature_stationarity(df)
        assert out is df


class TestAdfFailures:
    def test_an_adf_error_skips_the_column_without_flagging_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a, **kw):
            raise np.linalg.LinAlgError("singular")
        monkeypatch.setattr(sg, "adfuller", boom)
        _, report = check_feature_stationarity(_frame(), feature_cols=["feat_noise"])
        assert report.skipped == ["feat_noise"] and report.non_stationary == [] and report.passed

    def test_a_nan_p_value_counts_as_non_stationary_and_still_gets_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(sg, "adfuller", lambda *a, **kw: (0.0, float("nan")))
        with caplog.at_level(logging.WARNING, logger=sg.logger.name):
            out, report = check_feature_stationarity(_frame(), feature_cols=["feat_noise"],
                                                     drop_if_non_stationary=True)
        assert report.non_stationary == ["feat_noise"] and report.skipped == ["feat_noise"]
        assert "non-finite p-value" in caplog.text and "Worst 5" in caplog.text
        assert "feat_noise" not in out.columns


class TestReport:
    def test_the_report_passes_only_without_offenders(self) -> None:
        assert StationarityReport(n_features_checked=3, n_non_stationary=0).passed
        assert not StationarityReport(n_features_checked=3, n_non_stationary=1).passed
