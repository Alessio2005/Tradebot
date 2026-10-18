# tests/unit/test_train_quant_arch.py
"""`train/quant_arch.py`: SymmetricQuantileScaler, de P²-kwantielschatter en de dynamische embargo.

De P²-kernel draait ook als `.py_func` (coverage.py ziet niets binnen
gecompileerde code); gecompileerd en bron moeten identiek zijn.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from tradebot.train import quant_arch as qa
from tradebot.train.quant_arch import (
    P2OnlineQuantile,
    SymmetricQuantileScaler,
    dynamic_embargo_bars,
)

# --------------------------------------------------------------------------- #
# SymmetricQuantileScaler
# --------------------------------------------------------------------------- #


def _oscillator(n: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(0.0, 5.0, n), rng.normal(10.0, 20.0, n)])


class TestSymmetricQuantileScaler:
    def test_the_zero_line_is_preserved_and_both_halves_land_in_the_open_unit_interval(self) -> None:
        X = _oscillator()
        out = SymmetricQuantileScaler().fit_transform(X)
        assert out.shape == X.shape and out.min() > -1.0 and out.max() < 1.0
        assert (np.sign(out) == np.sign(X)).all()                   # nul blijft nul, tekens blijven
        zero = SymmetricQuantileScaler().fit(X).transform(np.zeros((1, 2)))
        np.testing.assert_array_equal(zero, 0.0)

    def test_the_mapping_is_monotone_per_column(self) -> None:
        X = _oscillator()
        scaler = SymmetricQuantileScaler().fit(X)
        grid = np.linspace(-30, 30, 200)
        out = scaler.transform(np.column_stack([grid, grid]))
        assert (np.diff(out[:, 0]) >= -1e-12).all() and (np.diff(out[:, 1]) >= -1e-12).all()

    def test_a_one_dimensional_fit_is_treated_as_one_column_and_a_one_dimensional_input_as_one_row(self) -> None:
        scaler = SymmetricQuantileScaler().fit(np.random.default_rng(1).normal(size=200))
        assert scaler.state.n_features == 1
        assert SymmetricQuantileScaler().fit(_oscillator()).transform(np.array([1.0, -1.0])).shape == (1, 2)

    def test_one_sided_columns_fall_back_to_a_default_marker_grid(self) -> None:
        all_positive = np.column_stack([np.abs(np.random.default_rng(2).normal(size=100)) + 1.0] * 2)
        scaler = SymmetricQuantileScaler(n_bins=11).fit(all_positive)
        np.testing.assert_allclose(scaler.state.neg_markers[:, 0], np.linspace(-1.0, 0.0, 11))
        all_negative = -all_positive
        scaler = SymmetricQuantileScaler(n_bins=11).fit(all_negative)
        np.testing.assert_allclose(scaler.state.pos_markers[:, 0], np.linspace(0.0, 1.0, 11))

    def test_passthrough_columns_are_copied_verbatim(self) -> None:
        X = _oscillator()
        X[:, 1] = np.random.default_rng(3).integers(0, 2, len(X))
        scaler = SymmetricQuantileScaler(passthrough_mask=[False, True]).fit(X)
        out = scaler.transform(X)
        np.testing.assert_array_equal(out[:, 1], X[:, 1])
        assert not np.array_equal(out[:, 0], X[:, 0])

    def test_a_passthrough_mask_of_the_wrong_length_is_ignored(self) -> None:
        scaler = SymmetricQuantileScaler(passthrough_mask=[True, True, True]).fit(_oscillator())
        assert not scaler.state.passthrough_mask.any()

    def test_an_empty_fit_warns_and_falls_back_on_default_markers(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=qa.logger.name):
            scaler = SymmetricQuantileScaler().fit(np.empty((0, 2)))
        assert "empty matrix" in caplog.text and scaler.state.fitted

    def test_transforming_before_fitting_passes_the_input_through_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        X = _oscillator(10)
        with caplog.at_level(logging.WARNING, logger=qa.logger.name):
            out = SymmetricQuantileScaler().transform(X)
        np.testing.assert_array_equal(out, X)
        assert out is not X and "before fit" in caplog.text

    def test_a_dimension_mismatch_after_fitting_is_refused(self) -> None:
        scaler = SymmetricQuantileScaler().fit(_oscillator())
        with pytest.raises(ValueError, match="dim-mismatch"):
            scaler.transform(np.ones((3, 5)))

    def test_a_missing_passthrough_mask_in_the_state_is_treated_as_all_false(self) -> None:
        scaler = SymmetricQuantileScaler().fit(_oscillator())
        scaler.state.passthrough_mask = None
        assert scaler.transform(_oscillator(5)).shape == (5, 2)


# --------------------------------------------------------------------------- #
# P2 online quantile
# --------------------------------------------------------------------------- #
def _streams() -> list[tuple[str, float, np.ndarray]]:
    rng = np.random.default_rng(0)
    streams = [
        ("normal", 0.5, rng.normal(size=1500)),
        ("heavy tail", 0.9, rng.standard_cauchy(size=1500)),
        ("ties", 0.3, rng.integers(0, 4, size=1500).astype(float)),
        ("ascending", 0.5, np.arange(1000.0)),
        ("descending", 0.1, np.arange(1000.0)[::-1]),
        ("constant", 0.5, np.full(300, 7.0)),
        ("tiny quantile", 1e-20, rng.normal(size=800)),
        ("quantile near one", 1.0 - 1e-17, rng.normal(size=800)),
    ]
    streams += [(f"mixed {s}", q, rng.standard_t(2, size=900)) for s, q in enumerate((0.05, 0.25, 0.75, 0.95))]
    return streams


def _run(kernel, stream: np.ndarray, quantile: float) -> np.ndarray:
    heights = np.sort(stream[:5]).astype(np.float64)
    positions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    count = 5
    for x in stream[5:]:
        count = kernel(float(x), heights, positions, quantile, count)
    return heights


class TestP2Kernel:
    @pytest.mark.parametrize(("name", "quantile", "stream"), _streams(), ids=[s[0] for s in _streams()])
    def test_the_compiled_kernel_is_its_python_source(self, name: str, quantile: float, stream: np.ndarray) -> None:
        compiled = _run(qa._p2_update_njit, stream, quantile)
        source = _run(qa._p2_update_njit.py_func, stream, quantile)
        np.testing.assert_allclose(compiled, source, rtol=1e-12, atol=1e-12)
        assert np.isfinite(source).all() and (np.diff(source) >= -1e-9).all()      # hoogtes blijven gesorteerd

    def test_markers_that_coincide_are_moved_linearly_instead_of_dividing_by_zero(self) -> None:
        def state():
            return (np.array([0.0, 1.0, 2.0, 3.0, 9.0]), np.array([1.0, 1.0, 3.0, 5.0, 9.0]))
        for kernel in (qa._p2_update_njit, qa._p2_update_njit.py_func):
            heights, positions = state()
            count = kernel(2.5, heights, positions, 0.5, 10)
            assert count == 11 and np.isfinite(heights).all()
            assert heights[1] == pytest.approx(1.0 + (2.0 - 1.0) / 2.0)        # lineaire stap omhoog
            assert positions[1] == 2.0

    def test_coinciding_upper_markers_are_moved_down_in_a_linear_step(self) -> None:
        def state():
            return (np.array([0.0, 1.0, 2.0, 3.0, 9.0]), np.array([1.0, 3.0, 6.0, 6.0, 9.0]))
        for kernel in (qa._p2_update_njit, qa._p2_update_njit.py_func):
            heights, positions = state()
            count = kernel(5.0, heights, positions, 0.05, 10)
            assert count == 11 and np.isfinite(heights).all()
            assert positions[2] == 5.0
            assert heights[2] == pytest.approx(2.0 - (2.0 - heights[1]) / 4.0)  # lineaire stap omlaag


class TestP2OnlineQuantile:
    def test_the_median_estimate_is_close_to_the_sample_median(self) -> None:
        x = np.random.default_rng(1).normal(size=4000)
        est = P2OnlineQuantile(quantile=0.5, n_features=1)
        for v in x:
            est.update([v])
        assert est.value()[0] == pytest.approx(np.median(x), abs=0.05)

    def test_the_estimate_tracks_every_feature_independently(self) -> None:
        rng = np.random.default_rng(2)
        x = np.column_stack([rng.normal(0, 1, 3000), rng.normal(50, 5, 3000)])
        est = P2OnlineQuantile(quantile=0.9, n_features=2)
        for row in x:
            est.update(row)
        np.testing.assert_allclose(est.value(), np.quantile(x, 0.9, axis=0), rtol=0.05, atol=0.15)

    def test_the_estimate_stays_inside_the_observed_range_per_feature(self) -> None:
        rng = np.random.default_rng(2)
        x = np.column_stack([rng.normal(0, 1, 1000), rng.normal(50, 5, 1000)])
        est = P2OnlineQuantile(quantile=0.9, n_features=2)
        for row in x:
            est.update(row)
        assert ((est.value() >= x.min(axis=0)) & (est.value() <= x.max(axis=0))).all()

    def test_there_is_no_value_during_warmup_and_the_first_five_samples_seed_the_markers(self) -> None:
        est = P2OnlineQuantile()
        for v in (5.0, 1.0, 3.0, 2.0):
            est.update([v])
            assert est.value() is None
        est.update([4.0])
        assert est.value()[0] == 3.0                               # middelste van 1..5

    def test_the_quantile_must_lie_strictly_inside_the_unit_interval(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="quantile"):
                P2OnlineQuantile(quantile=bad)

    def test_an_observation_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expects 2 features"):
            P2OnlineQuantile(n_features=2).update([1.0])

    def test_a_zero_feature_count_is_raised_to_one(self) -> None:
        assert P2OnlineQuantile(n_features=0).n_features == 1


# --------------------------------------------------------------------------- #
# dynamic embargo
# --------------------------------------------------------------------------- #
class TestDynamicEmbargo:
    """`quant_arch` hergebruikt de functie uit `cv.purge` (R-3: een implementatie per grootheid);
    de uitgebreide gevallen staan in `test_wave5.py`."""

    def test_it_is_the_canonical_cv_function(self) -> None:
        from tradebot.cv import purge
        assert dynamic_embargo_bars is purge.dynamic_embargo_bars

    def test_the_embargo_is_the_longest_horizon_times_the_safety_factor_rounded_up(self) -> None:
        assert dynamic_embargo_bars([5, 10, 20]) == 50                  # standaardfactor 2,5
        assert dynamic_embargo_bars([10], safety_factor=2.0) == 20
        assert dynamic_embargo_bars([10], safety_factor=2.05) == 21

    def test_a_safety_factor_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="safety_factor must be"):
            dynamic_embargo_bars([10], safety_factor=0.5)

    def test_no_horizons_give_the_minimum_and_the_minimum_is_a_floor(self) -> None:
        assert dynamic_embargo_bars([], min_embargo=3) == 3
        assert dynamic_embargo_bars([1], min_embargo=5) == 5
