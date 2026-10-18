# tests/unit/test_train_scalers.py
"""`train/_scalers.py`: de rollende robuuste Numba-kernel en `RollingRobustScaler`.

De kernel draait ook als `.py_func` (coverage.py ziet niets binnen gecompileerde
code). De scaler heeft drie paden: batch (`fit` met < 6 rijen of `recalibrate`),
online P² (`fit` met meer rijen en `update`) en rank-modus.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tradebot.train import _scalers as sc
from tradebot.train._scalers import RollingRobustScaler, get_git_hash, numba_rolling_robust_scale


def _x(n: int = 300, k: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, k)).astype(np.float32)


# --------------------------------------------------------------------------- #
# get_git_hash
# --------------------------------------------------------------------------- #
class TestGitHash:
    def test_it_returns_the_short_hash_of_head(self) -> None:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent).decode().strip()
        assert get_git_hash() == head

    def test_outside_a_repository_the_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a, **kw):
            raise subprocess.CalledProcessError(128, "git")
        monkeypatch.setattr(sc.subprocess, "check_output", boom)
        with pytest.raises(subprocess.CalledProcessError):
            get_git_hash()


# --------------------------------------------------------------------------- #
# numba kernel
# --------------------------------------------------------------------------- #
def _reference(X: np.ndarray, window: int, q_low: float, q_high: float) -> np.ndarray:
    out = np.zeros(X.shape, dtype=np.float64)
    for j in range(X.shape[1]):
        for i in range(len(X)):
            w = np.sort(X[max(0, i - window + 1): i + 1, j].astype(np.float64))
            n = len(w)
            median = w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2.0
            iqr = w[int(q_high / 100.0 * (n - 1))] - w[int(q_low / 100.0 * (n - 1))]
            out[i, j] = (X[i, j] - median) / (iqr if iqr >= 1e-9 else 1.0)
    return out


class TestRollingKernel:
    @pytest.mark.parametrize("window", [1, 2, 7, 40])
    def test_compiled_python_source_and_a_plain_numpy_reference_agree(self, window: int) -> None:
        X = _x(120, 2, seed=window)
        compiled = numba_rolling_robust_scale(X, window, 5.0, 95.0)
        source = numba_rolling_robust_scale.py_func(X, window, 5.0, 95.0)
        np.testing.assert_allclose(compiled, source, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(compiled, _reference(X, window, 5.0, 95.0), rtol=1e-4, atol=1e-5)

    def test_a_constant_column_is_centred_with_unit_scale(self) -> None:
        X = np.full((30, 1), 4.0, dtype=np.float32)
        X[-1, 0] = 6.0
        out = numba_rolling_robust_scale.py_func(X, 10, 5.0, 95.0)
        assert (out[:-1] == 0.0).all() and out[-1, 0] == pytest.approx(2.0)

    def test_the_output_at_t_ignores_everything_after_t(self) -> None:
        X = _x(100, 2)
        full = numba_rolling_robust_scale(X, 20, 5.0, 95.0)
        cut = numba_rolling_robust_scale(X[:60], 20, 5.0, 95.0)
        np.testing.assert_array_equal(full[:60], cut)


# --------------------------------------------------------------------------- #
# RollingRobustScaler
# --------------------------------------------------------------------------- #
class TestFit:
    def test_a_one_dimensional_input_is_one_feature(self) -> None:
        s = RollingRobustScaler(window=50).fit(np.arange(4.0))
        assert s.n_features_in_ == 1 and s.center_.shape == (1,)

    def test_up_to_five_rows_keep_the_exact_batch_statistics(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [10.0], [20.0]])
        s = RollingRobustScaler(window=50).fit(X)
        assert s.center_[0] == 3.0
        assert s.scale_[0] == pytest.approx(np.percentile(X, 95) - np.percentile(X, 5))
        assert s._p2_q is not None and s._p2_count == 5            # vanaf vijf rijen staan de P²-markers klaar

    def test_fewer_than_five_rows_leave_the_p2_state_empty(self) -> None:
        s = RollingRobustScaler(window=50).fit(_x(3, 2))
        assert s._p2_q is None and not s._is_full and s._buffer_idx == 3

    def test_a_history_longer_than_the_window_keeps_only_the_last_window_rows(self) -> None:
        X = _x(200, 2)
        s = RollingRobustScaler(window=50).fit(X)
        assert s._is_full and s._buffer_idx == 0
        np.testing.assert_array_equal(s.data_buffer, X[-50:])

    def test_the_fitted_center_and_scale_match_the_batch_statistics(self) -> None:
        X = _x(2000, 3)
        s = RollingRobustScaler(window=2000).fit(X)
        np.testing.assert_allclose(s.center_, np.median(X, axis=0), atol=0.2)
        np.testing.assert_allclose(s.scale_, np.percentile(X, 95, axis=0) - np.percentile(X, 5, axis=0), rtol=0.15)


class TestP2DegenerateMarkers:
    """Samenvallende markers mogen nooit tot een deling door nul leiden; de update gaat dan lineair."""

    def test_coinciding_lower_markers_are_moved_up_in_a_linear_step(self) -> None:
        heights = np.array([0.0, 1.0, 2.0, 3.0, 9.0])
        positions = np.array([1.0, 1.0, 3.0, 5.0, 9.0])
        RollingRobustScaler._p2_update_single(2.5, heights, positions, 0.5, 10)
        assert heights[1] == pytest.approx(1.5) and positions[1] == 2.0
        assert np.isfinite(heights).all()

    def test_coinciding_upper_markers_are_moved_down_in_a_linear_step(self) -> None:
        heights = np.array([0.0, 1.0, 2.0, 3.0, 9.0])
        positions = np.array([1.0, 3.0, 6.0, 6.0, 9.0])
        RollingRobustScaler._p2_update_single(5.0, heights, positions, 0.05, 10)
        assert positions[2] == 5.0 and heights[2] < 2.0 and np.isfinite(heights).all()


class TestRecalibrate:
    def test_without_a_buffer_or_with_an_empty_one_nothing_happens(self) -> None:
        s = RollingRobustScaler(window=10)
        s.recalibrate()
        assert s.center_ is None
        s.data_buffer = np.zeros((10, 2), dtype=np.float32)          # nog niets in de buffer geschreven
        s.recalibrate()
        assert s.center_ is None

    def test_it_recomputes_median_and_spread_over_the_active_buffer(self) -> None:
        X = _x(100, 2)
        s = RollingRobustScaler(window=100).fit(X)
        s.recalibrate()
        np.testing.assert_allclose(s.center_, np.median(X, axis=0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s.scale_, np.percentile(X, 95, axis=0) - np.percentile(X, 5, axis=0), rtol=1e-5)

    def test_a_constant_feature_gets_unit_scale(self) -> None:
        s = RollingRobustScaler(window=20).fit(np.full((4, 1), 3.0))
        assert s.scale_[0] == 1.0 and s.center_[0] == 3.0


class TestUpdate:
    def test_updating_an_unfitted_scaler_fits_it(self) -> None:
        s = RollingRobustScaler(window=20)
        s.update(_x(10, 2))
        assert s.n_features_in_ == 2 and s.center_ is not None

    def test_a_single_row_is_accepted_and_the_buffer_wraps_around(self) -> None:
        s = RollingRobustScaler(window=10).fit(_x(8, 2))
        for row in _x(5, 2, seed=1):
            s.update(row.tolist())
        assert s._is_full and s._buffer_idx == 3 and s._bars_since_calib == 5

    def test_with_a_buffer_but_no_p2_state_the_estimators_are_warm_started_once_five_rows_exist(self) -> None:
        s = RollingRobustScaler(window=50).fit(_x(3, 2))
        assert s._p2_q is None
        s.update(_x(2, 2, seed=1))                                   # 5 rijen in de buffer
        assert s._p2_q is not None and s._p2_count == 5
        s2 = RollingRobustScaler(window=50).fit(_x(3, 2))
        s2.update(_x(10, 2, seed=2))                                 # meer dan vijf: de rest wordt doorgevoerd
        assert s2._p2_count == 13

    def test_with_fewer_than_five_rows_the_warm_start_waits(self) -> None:
        s = RollingRobustScaler(window=50).fit(_x(2, 2))
        s.update(_x(1, 2, seed=1))
        assert s._p2_q is None

    def test_once_warm_every_new_row_moves_the_estimators(self) -> None:
        s = RollingRobustScaler(window=100).fit(_x(50, 2))
        before = s._p2_count
        s.update(_x(4, 2, seed=3))
        assert s._p2_count == before + 4

    def test_feeding_without_p2_state_and_updating_a_missing_buffer_are_no_ops(self) -> None:
        s = RollingRobustScaler(window=10)
        s._p2_feed(np.zeros(2))
        s._update_buffer(np.zeros((1, 2), dtype=np.float32))
        assert s._p2_count == 0 and s.data_buffer is None


class TestTransform:
    def test_a_fitted_scaler_centres_and_scales_row_by_row(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [10.0], [20.0]])
        s = RollingRobustScaler(window=50).fit(X)
        out = s.transform(np.array([3.0]))                           # 1D = een rij
        assert out.shape == (1, 1) and out[0, 0] == pytest.approx(0.0)
        assert s.transform(np.array([[3.0 + s.scale_[0]]]))[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_an_unfitted_scaler_passes_values_through(self) -> None:
        X = _x(5, 2)
        np.testing.assert_array_equal(RollingRobustScaler(window=50).transform(X), X)

    def test_a_long_input_without_a_fitted_buffer_uses_the_rolling_kernel(self) -> None:
        X = _x(60, 2)
        out = RollingRobustScaler(window=20, quantile_range=(10.0, 90.0)).transform(X)
        np.testing.assert_allclose(out, numba_rolling_robust_scale(X, 20, 10.0, 90.0))


class TestRankMode:
    def test_every_call_is_delegated_to_the_rank_scaler(self) -> None:
        s = RollingRobustScaler(mode="rank", rank_n_bins=11, rank_warmup_bars=20)
        X = _x(120, 2)
        assert s.fit(X) is s and s._rank_scaler is not None
        s.update(X[0])                                                # 1D -> een rij
        s.update(X[:4])
        assert s._bars_since_calib == 5
        out = s.transform(X[:6])
        assert out.dtype == np.float32 and out.shape == (6, 2) and out.min() >= 0.0 and out.max() <= 1.0
        assert s.data_buffer is None                                  # het robuuste pad is nooit gebruikt
