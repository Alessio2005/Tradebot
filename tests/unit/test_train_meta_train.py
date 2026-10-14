# tests/unit/test_train_meta_train.py
"""`train/meta_train.py`: Judge-features, Judge-labels en de trainer (echte CatBoost, klein model)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.train import meta_train
from tradebot.train.meta_train import build_judge_features, make_judge_labels, train_judge

FAST = {"judge_iterations": 20, "judge_depth": 3}


def _inputs(n: int = 120, seed: int = 0, positive_share: float | None = None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"], index=idx)
    probs = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    directions = pd.Series(rng.choice([-1.0, 1.0], n), index=idx)
    returns = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    if positive_share is not None:                              # winst of verlies voor een vaste fractie
        wins = rng.random(n) < positive_share
        returns = pd.Series(np.where(wins, 0.01, -0.01) * directions.to_numpy(), index=idx)
    return X, probs, returns, directions


class TestJudgeFeatures:
    def test_too_little_data_is_refused(self) -> None:
        X, probs, *_ = _inputs(19)
        with pytest.raises(ValueError, match="Onvoldoende data"):
            build_judge_features(X, probs)

    def test_the_primary_probability_and_cyclical_time_features_are_added(self) -> None:
        X, probs, *_ = _inputs(48)
        out = build_judge_features(X, probs)
        assert list(out.columns) == ["f1", "f2", "f3", "prob_primary", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        np.testing.assert_allclose(out["prob_primary"], probs.to_numpy())
        np.testing.assert_allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)
        assert out["hour_cos"].iloc[0] == pytest.approx(1.0)       # 00:00 UTC
        assert "prob_primary" not in X.columns                      # de invoer is niet gemuteerd

    def test_alignment_is_positional_so_repeated_timestamps_do_not_expand(self) -> None:
        X, probs, *_ = _inputs(30)
        dup_index = X.index[:15].append(X.index[:15])             # CPCV hergebruikt tijdstempels
        X.index, probs.index = dup_index, dup_index
        out = build_judge_features(X, probs)
        assert len(out) == 30 and np.allclose(out["prob_primary"], probs.to_numpy())

    def test_unequal_lengths_are_cut_to_the_shorter_one(self) -> None:
        X, probs, *_ = _inputs(60)
        assert len(build_judge_features(X, probs.iloc[:40])) == 40


class TestJudgeLabels:
    def test_a_trade_is_positive_when_the_directed_return_beats_the_cost(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        returns = pd.Series([0.0030, 0.0001, -0.0030, -0.0030], index=idx)
        directions = pd.Series([1.0, 1.0, -1.0, 1.0], index=idx)
        y = make_judge_labels(returns, directions, cost_bps=2.0)
        assert y.tolist() == [1, 0, 1, 0] and y.dtype == np.int32 and y.index.equals(idx)

    def test_unequal_lengths_are_cut_to_the_shorter_one(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        y = make_judge_labels(pd.Series(np.ones(5), index=idx), pd.Series(np.ones(3), index=idx[:3]))
        assert len(y) == 3


class TestTrainJudge:
    def test_it_trains_persists_and_returns_the_in_sample_judge_probabilities(self, tmp_path: Path) -> None:
        X, probs, returns, directions = _inputs(160)
        out = train_judge("BTCUSDT", "LONG", X, probs, returns, directions, tmp_path, **FAST)
        assert out is not None and out.name == "prob_judge" and len(out) == 160
        assert out.between(0.0, 1.0).all() and out.index.equals(X.index)
        assert (tmp_path / "judge_models" / "BTCUSDT_LONG_judge.cbm").is_file()
        stored = pd.read_parquet(tmp_path / "oos_probs" / "BTCUSDT_LONG_judge.parquet")
        np.testing.assert_allclose(stored["prob_judge"], out.to_numpy())

    def test_the_same_seed_gives_the_same_probabilities(self, tmp_path: Path) -> None:
        args = (*_inputs(160), tmp_path)
        a = train_judge("S", "LONG", *args, **FAST)
        b = train_judge("S", "LONG", *args, **FAST)
        pd.testing.assert_series_equal(a, b)                       # type: ignore[arg-type]

    def test_a_feature_build_failure_returns_none(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        X, probs, returns, directions = _inputs(10)
        with caplog.at_level(logging.ERROR, logger=meta_train.logger.name):
            assert train_judge("S", "LONG", X, probs, returns, directions, tmp_path, **FAST) is None
        assert "Judge feature build mislukt" in caplog.text

    def test_fewer_than_thirty_events_return_none(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        X, probs, returns, directions = _inputs(25)
        with caplog.at_level(logging.WARNING, logger=meta_train.logger.name):
            assert train_judge("S", "LONG", X, probs, returns, directions, tmp_path, **FAST) is None
        assert "Onvoldoende Judge events (25)" in caplog.text

    def test_rows_with_nan_are_dropped_and_too_few_clean_rows_return_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        X, probs, returns, directions = _inputs(40)
        X.iloc[::2, 0] = np.nan                                    # de helft van de rijen valt weg
        with caplog.at_level(logging.WARNING, logger=meta_train.logger.name):
            assert train_judge("S", "LONG", X, probs, returns, directions, tmp_path, **FAST) is None
        assert "Te weinig schone events" in caplog.text

    def test_a_skewed_label_distribution_is_warned_about_but_still_trained(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        X, probs, returns, directions = _inputs(120, positive_share=0.98)
        with caplog.at_level(logging.WARNING, logger=meta_train.logger.name):
            out = train_judge("S", "SHORT", X, probs, returns, directions, tmp_path, **FAST)
        assert out is not None and "Scheef label-distributie" in caplog.text
        assert (tmp_path / "judge_models" / "S_SHORT_judge.cbm").is_file()
