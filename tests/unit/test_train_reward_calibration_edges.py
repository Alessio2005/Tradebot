# tests/unit/test_train_reward_calibration_edges.py
"""`train/reward.py`, `train/calibration.py` en de restgaten van `train/meta_label.py`."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.labeling.vol_barriers import BarrierLabels
from tradebot.train.calibration import PathSpecificPlattCalibrator
from tradebot.train.meta_label import (
    MetaLabelDataset,
    build_dataset,
    feature_names,
    walk_forward_fit_predict,
)
from tradebot.train.reward import KalmanImpactObserver, NetAlphaReward
from tradebot.utils.failfast import DataContractError

# --------------------------------------------------------------------------- #
# NetAlphaReward
# --------------------------------------------------------------------------- #


class TestNetAlphaReward:
    def test_a_non_positive_price_gives_a_neutral_result_that_flags_the_impact(self) -> None:
        for entry, exit_ in ((0.0, 100.0), (100.0, -1.0)):
            res = NetAlphaReward().compute(entry, exit_, order_size=1.0, bar_volume=100.0, sigma_per_bar=0.01)
            assert res.reward == 0.0 and res.slippage == 0.0 and res.impact_eats_alpha

    def test_the_reward_is_log_return_minus_round_trip_slippage_and_fees(self) -> None:
        res = NetAlphaReward(eta=0.142, fee_bps=4.0).compute(
            100.0, 102.0, order_size=1.0, bar_volume=10_000.0, sigma_per_bar=0.01)
        assert res.log_return == pytest.approx(math.log(1.02))
        assert res.fees == pytest.approx(4e-4)
        assert res.reward == pytest.approx(res.log_return - res.slippage - res.fees)
        assert res.slippage > 0 and not res.impact_eats_alpha

    def test_a_short_profits_from_a_falling_price(self) -> None:
        long_ = NetAlphaReward().compute(100.0, 98.0, 1.0, 10_000.0, 0.01, side=1)
        short = NetAlphaReward().compute(100.0, 98.0, 1.0, 10_000.0, 0.01, side=-1)
        assert long_.log_return < 0 < short.log_return

    def test_impact_that_eats_the_alpha_is_flagged_even_when_the_price_moved_the_right_way(self) -> None:
        res = NetAlphaReward().compute(100.0, 100.1, order_size=5_000.0, bar_volume=10_000.0, sigma_per_bar=0.02)
        assert res.log_return > 0 and res.reward < 0 and res.impact_eats_alpha

    def test_a_flat_trade_has_no_alpha_to_eat(self) -> None:
        res = NetAlphaReward().compute(100.0, 100.0, 1.0, 10_000.0, 0.01)
        assert res.log_return == 0.0 and res.impact_eats_alpha and res.reward < 0


# --------------------------------------------------------------------------- #
# KalmanImpactObserver
# --------------------------------------------------------------------------- #
class TestKalmanImpactObserver:
    def test_the_initial_eta_is_clipped_to_its_bounds(self) -> None:
        assert KalmanImpactObserver(eta_init=5.0, eta_max=1.0).eta == 1.0
        assert KalmanImpactObserver(eta_init=0.0, eta_min=0.01).eta == 0.01

    @pytest.mark.parametrize(("size", "volume", "sigma"), [
        (0.0, 1_000.0, 0.01),        # geen order
        (-5.0, 1_000.0, 0.01),
        (1e-20, 1.0, 1e-5),          # deelname zo klein dat de noemer onder 1e-12 valt
    ])
    def test_a_degenerate_trade_leaves_eta_untouched(self, size: float, volume: float, sigma: float) -> None:
        obs = KalmanImpactObserver()
        before = obs.eta
        assert obs.update(0.001, sigma, size, volume) == before and obs.history == []

    def test_eta_moves_towards_what_the_trades_imply(self) -> None:
        obs = KalmanImpactObserver(eta_init=0.142)
        # implied eta = slippage / (sigma * sqrt(Q/V)) = 0.0002 / (0.01 * 0.1) = 0.2
        for _ in range(50):
            obs.update(realised_slippage=0.0002, sigma_per_bar=0.01, order_size=10.0, bar_volume=1_000.0)
        assert obs.eta == pytest.approx(0.2, abs=0.01) and len(obs.history) == 50
        assert obs.history[0].eta_observed == pytest.approx(0.2)

    def test_an_absurd_observation_is_clipped_to_the_eta_bounds(self) -> None:
        obs = KalmanImpactObserver(eta_max=1.0)
        obs.update(realised_slippage=10.0, sigma_per_bar=0.01, order_size=10.0, bar_volume=1_000.0)
        assert obs.history[0].eta_observed == 1.0 and obs.eta <= 1.0

    def test_the_history_is_trimmed_to_the_latest_half_after_5000_records(self) -> None:
        obs = KalmanImpactObserver()
        for _ in range(5_001):
            obs.update(0.001, 0.01, 10.0, 1_000.0)
        assert len(obs.history) == 2_500

    def test_predicted_slippage_is_the_square_root_law_with_the_current_eta(self) -> None:
        obs = KalmanImpactObserver(eta_init=0.2)
        assert obs.predicted_slippage(order_size=100.0, bar_volume=10_000.0, sigma_per_bar=0.01) == pytest.approx(
            0.2 * 0.01 * 0.1)
        assert obs.predicted_slippage(-1.0, 100.0, 0.01) == 0.0


# --------------------------------------------------------------------------- #
# PathSpecificPlattCalibrator
# --------------------------------------------------------------------------- #
def _scores(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p).astype(float)
    return p, y, np.repeat([0, 1], n // 2)


class TestPlattFit:
    def test_calibrated_probabilities_follow_the_scores_and_stay_in_the_open_interval(self) -> None:
        p, y, paths = _scores()
        cal = PathSpecificPlattCalibrator().fit(p, y, paths)
        out = cal.predict_proba(np.array([0.1, 0.5, 0.9]), path_id=0)
        assert np.all(np.diff(out) > 0) and out.min() > 0 and out.max() < 1

    def test_an_unknown_or_missing_path_falls_back_to_the_global_fit(self) -> None:
        p, y, paths = _scores()
        cal = PathSpecificPlattCalibrator().fit_per_path(p, y, paths)
        grid = np.array([0.2, 0.8])
        np.testing.assert_array_equal(cal.predict_proba(grid, path_id=99), cal.predict_proba(grid))

    def test_raw_scores_are_used_as_given_when_the_input_is_not_a_probability(self) -> None:
        rng = np.random.default_rng(1)
        f = rng.normal(size=400)
        y = (rng.random(400) < 1 / (1 + np.exp(-2 * f))).astype(float)
        cal = PathSpecificPlattCalibrator(input_is_probability=False).fit(f, y, np.zeros(400, dtype=int))
        out = cal.predict_proba(np.array([-2.0, 0.0, 2.0]))
        assert out[0] < 0.5 < out[2] and cal._global.A > 0.5

    def test_the_volatility_weights_are_optional_and_length_checked(self) -> None:
        p, y, paths = _scores()
        PathSpecificPlattCalibrator().fit(p, y, paths, bar_volatility=np.full(len(p), 0.02))
        with pytest.raises(ValueError, match="bar_volatility"):
            PathSpecificPlattCalibrator().fit(p, y, paths, bar_volatility=np.ones(5))

    def test_mismatching_input_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="shape-mismatch"):
            PathSpecificPlattCalibrator().fit(np.ones(5) * 0.5, np.ones(4), np.zeros(5, dtype=int))

    def test_fitting_on_nothing_marks_the_calibrator_fitted_with_the_identity_prior(self) -> None:
        cal = PathSpecificPlattCalibrator().fit(np.array([]), np.array([]), np.array([], dtype=int))
        assert cal.predict_proba(np.array([0.3])).shape == (1,)

    def test_predicting_before_fitting_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="niet gefit"):
            PathSpecificPlattCalibrator().predict_proba(np.array([0.5]))

    def test_an_inverted_fit_falls_back_to_the_identity(self) -> None:
        p, y, paths = _scores()
        cal = PathSpecificPlattCalibrator().fit(p, 1.0 - y, paths)      # labels tegen de score in
        assert (cal._global.A, cal._global.B) == (1.0, 0.0)
        np.testing.assert_allclose(cal.predict_proba(np.array([0.3, 0.7])), [0.3, 0.7], atol=1e-6)


class TestNewtonPlatt:
    def test_no_samples_give_the_zero_fit(self) -> None:
        assert PathSpecificPlattCalibrator._newton_platt(
            np.array([]), np.array([]), np.array([]), 50, 1e-7) == (0.0, 0.0)

    def test_a_single_class_gives_the_neutral_negative_slope_which_the_guard_then_rejects(self) -> None:
        f = np.linspace(-2, 2, 50)
        assert PathSpecificPlattCalibrator._newton_platt(f, np.ones(50), np.ones(50), 50, 1e-7) == (-1.0, 0.0)
        assert PathSpecificPlattCalibrator._newton_platt(f, np.zeros(50), np.ones(50), 50, 1e-7) == (-1.0, 0.0)

    def test_a_singular_hessian_stops_the_iteration_at_the_prior(self) -> None:
        f = np.zeros(100)                                         # constante score: Hessiaan van rang 1
        y = np.tile([0.0, 1.0], 50)
        a, b = PathSpecificPlattCalibrator._newton_platt(f, y, np.full(100, 1e-9), 50, 1e-7)
        assert a == 0.0 and b == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# meta_label: the three leftover lines
# --------------------------------------------------------------------------- #
def _small_dataset(n_events: int = 120, n_bars: int = 400) -> MetaLabelDataset:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="D", tz="UTC")
    features = pd.DataFrame(rng.normal(size=(n_bars, 3)), columns=["a", "b", "c"], index=idx)
    event = np.arange(n_events)
    labels = BarrierLabels(
        event_idx=event, exit_idx=event + 10,
        barrier_outcome=rng.choice([-1, 0, 1], n_events),
        meta_label=(np.arange(n_events) % 2).astype(np.int64),
        realized_return=rng.normal(0.0, 0.02, n_events),
        side=rng.choice([-1.0, 1.0], n_events), sigma=np.full(n_events, 0.03))
    return build_dataset({"AAA": features}, {"AAA": labels})


class _MajorityModel:
    feature_importance = np.array([0.5, 0.3, 0.2])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(x), 0.4), np.full(len(x), 0.6)])


class TestMetaLabelLeftovers:
    def test_the_dataset_reports_its_positive_ratio_and_feature_names(self) -> None:
        data = _small_dataset()
        assert data.positive_ratio == pytest.approx(0.5) and feature_names(data) == ["a", "b", "c"]

    def test_folds_without_events_in_their_test_window_are_skipped(self) -> None:
        data = _small_dataset(n_events=120, n_bars=400)           # events alleen in de eerste 120 bars
        cv = WalkForwardCV(train_size=60, test_size=30, step=30, mode="rolling", min_train=60, embargo_bars=2)
        folds = walk_forward_fit_predict(data, cv, lambda rows, labels: _MajorityModel(),
                                         n_bars=400, embargo_bars=2)
        assert 0 < len(folds) < len(list(cv.split(400)))
        assert all(f.probability.min() == pytest.approx(0.6) for f in folds)

    def test_a_dataset_without_any_usable_fold_is_an_error(self) -> None:
        data = _small_dataset(n_events=40)
        cv = WalkForwardCV(train_size=300, test_size=30, step=30, mode="rolling", min_train=300, embargo_bars=2)
        with pytest.raises(DataContractError, match="Geen enkele fold"):
            walk_forward_fit_predict(data, cv, lambda rows, labels: _MajorityModel(),
                                     n_bars=400, embargo_bars=2)
