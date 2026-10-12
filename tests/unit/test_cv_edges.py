# tests/unit/test_cv_edges.py
"""De cv-paden die de hoofdtests niet raken.

De Numba-kernels (`_sb_kernel`, `_calculate_average_uniqueness`) worden ook via
`.py_func` aangeroepen: `coverage.py` ziet niets binnen gecompileerde code. De
bootstrap geeft gehele indices en is daarom bitgelijk; de uniqueness is een
som van breuken en wordt op afrondingsniveau vergeleken.
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from tradebot.cv import bootstrap as bootstrap_mod
from tradebot.cv.bootstrap import _sb_kernel
from tradebot.cv.cpcv import CombinatorialPurgedCV, build_cpcv_return_paths
from tradebot.cv.purge import dynamic_embargo_bars
from tradebot.cv.uniqueness import (
    _calculate_average_uniqueness,
    get_average_uniqueness,
    get_average_uniqueness_per_fold,
    get_sample_weights,
)
from tradebot.cv.walk_forward import WalkForwardCV


# --------------------------------------------------------------------------- #
# Sequential bootstrap
# --------------------------------------------------------------------------- #
class _TopOfTheUnitInterval:
    """`np` voor `_sb_kernel.py_func`, met een trekking net onder 1.0."""

    class random:
        @staticmethod
        def seed(_seed: int) -> None:
            return None

        @staticmethod
        def random() -> float:
            return float(np.nextafter(1.0, 0.0))

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)


class TestSequentialBootstrapKernel:
    T0 = np.array([0, 3, 6, 9, 12], dtype=np.int64)

    def test_the_compiled_kernel_is_its_python_source(self) -> None:
        t1 = self.T0 + 5
        np.testing.assert_array_equal(_sb_kernel(self.T0, t1, 20, 7),
                                      _sb_kernel.py_func(self.T0, t1, 20, 7))

    def test_without_a_seed_the_global_state_is_left_alone(self) -> None:
        np.random.seed(3)
        expected = np.random.random()
        np.random.seed(3)
        draws = _sb_kernel.py_func(self.T0, self.T0 + 5, 4, -1)
        assert draws.shape == (4,) and ((draws >= 0) & (draws < 5)).all()
        np.random.seed(3)
        assert np.random.random() == expected

    def test_an_inverted_event_that_overlaps_a_draw_loses_all_weight(self) -> None:
        """De wrapper weigert t1 < t0; de kernel zelf zet zo'n event op u = 0
        zodra een getrokken event het overlapt, in plaats van een negatieve
        duur te delen. Event 1 kan daarna niet meer getrokken worden."""
        t0 = np.array([0, 5], dtype=np.int64)
        t1 = np.array([10, 3], dtype=np.int64)
        draws = _sb_kernel.py_func(t0, t1, 30, 0)
        first_valid = int(np.argmax(draws == 0))
        assert (draws[first_valid:] == 0).all()

    def test_a_draw_above_the_rounded_cumulative_sum_stays_in_range(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Zeven gelijke kansen tellen op tot 0.9999999999999998 < 1: een
        trekking daarboven gaf zonder de klem index 7 bij zeven events."""
        assert np.cumsum(np.full(7, 1 / 7))[-1] < np.nextafter(1.0, 0.0)
        monkeypatch.setattr(bootstrap_mod, "np", _TopOfTheUnitInterval())
        t0 = np.arange(0, 70, 10, dtype=np.int64)
        draws = _sb_kernel.py_func(t0, t0 + 2, 1, 0)
        assert draws.tolist() == [6]


# --------------------------------------------------------------------------- #
# CPCV
# --------------------------------------------------------------------------- #
def _cpcv_inputs(
    n: int, horizon: int = 3, tz: str | None = None,
) -> tuple[pd.DatetimeIndex, pd.Series]:
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz=tz)
    t1 = pd.Series(np.minimum(np.arange(n) + horizon, n - 1), index=ts)
    return ts, t1


class TestCpcvSplit:
    def test_a_long_feature_lag_lifts_the_embargo(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ts, t1 = _cpcv_inputs(120)
        plain = CombinatorialPurgedCV(n_groups=4, n_test_groups=1, purge_bars=4)
        lagged = CombinatorialPurgedCV(n_groups=4, n_test_groups=1, purge_bars=4,
                                       feature_max_lag=20)
        with caplog.at_level(logging.INFO):
            lagged_splits = list(lagged.split(ts, t1))
        assert "upgraded by FFD feature_max_lag" in caplog.text
        for (tr_plain, _, _), (tr_lag, _, _) in zip(
            plain.split(ts, t1), lagged_splits, strict=True,
        ):
            assert set(tr_lag) < set(tr_plain)

    def test_a_short_feature_lag_leaves_the_embargo(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ts, t1 = _cpcv_inputs(120)
        with caplog.at_level(logging.INFO):
            list(CombinatorialPurgedCV(n_groups=4, n_test_groups=1, purge_bars=4,
                                       feature_max_lag=1).split(ts, t1))
        assert "upgraded" not in caplog.text

    def test_fewer_events_than_groups_skips_the_empty_groups(self) -> None:
        """Drie events, zes groepen: groepen 0..4 zijn leeg en de laatste
        draagt alles. Combinaties van lege groepen vallen weg."""
        ts, t1 = _cpcv_inputs(3, horizon=1)
        splits = list(CombinatorialPurgedCV(n_groups=6, n_test_groups=2,
                                            purge_bars=1).split(ts, t1))
        assert [groups for _, _, groups in splits] == [(g, 5) for g in range(5)]
        for train, test, _ in splits:
            assert train.size == 0 and test.tolist() == [0, 1, 2]

    @pytest.mark.parametrize("tz", [None, "UTC"])
    def test_calendar_purge_for_daily_studies(self, tz: str | None) -> None:
        """Een tz-naïeve index werd `datetime64[ns]`, en numpy weigert die plus
        een `datetime.timedelta`: de kalenderpurge gaf een UFuncTypeError.
        Alleen tz-bewuste tijdstempels (object-array) werkten."""
        ts, t1 = _cpcv_inputs(60, tz=tz)
        cv = CombinatorialPurgedCV(n_groups=3, n_test_groups=1, purge_days=2,
                                   require_bar_purge=False)
        splits = list(cv.split(ts, t1))
        assert len(splits) == 3
        for train, test, _ in splits:
            assert not set(train) & set(test)
            gap = np.abs(train[:, None] - test[None, :]).min()
            assert gap > 3  # horizon 3, plus purge en embargo


class TestCpcvPaths:
    def test_no_folds_give_no_paths(self) -> None:
        assert build_cpcv_return_paths({}, n_groups=4) == []

    def test_partitions_with_a_missing_fold_are_skipped(self) -> None:
        folds = {
            combo: pd.Series([0.01], index=[pd.Timestamp("2024-01-01") +
                                            pd.Timedelta(days=combo[0])])
            for combo in combinations(range(4), 2) if combo != (0, 1)
        }
        paths = build_cpcv_return_paths(folds, n_groups=4)
        assert len(paths) == 2  # {02,13} en {03,12}; {01,23} mist een fold

    def test_without_a_single_complete_partition_it_refuses(self) -> None:
        folds = {(0, 1): pd.Series([0.01]), (0, 2): pd.Series([0.02])}
        with pytest.raises(ValueError, match="No valid N/k partitions"):
            build_cpcv_return_paths(folds, n_groups=4)


# --------------------------------------------------------------------------- #
# Embargo
# --------------------------------------------------------------------------- #
class TestDynamicEmbargo:
    def test_a_factor_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="safety_factor"):
            dynamic_embargo_bars([5], safety_factor=0.9)

    def test_a_calm_regime_factor_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            assert dynamic_embargo_bars([5], safety_factor=1.2) == 6
        assert "crisis-aware floor" in caplog.text

    def test_the_default_and_the_floor(self) -> None:
        assert dynamic_embargo_bars([5, 2]) == 13  # ceil(5 * 2.5)
        assert dynamic_embargo_bars([], min_embargo=4) == 4
        assert dynamic_embargo_bars([1], min_embargo=9) == 9


class TestAnchoredWalkForward:
    def test_the_previous_test_boundary_is_embargoed(self) -> None:
        """Stap > testlengte: de vorige testgrens valt in het nieuwe
        trainvenster en wordt aan beide kanten uitgesneden."""
        cv = WalkForwardCV(train_size=100, test_size=20, step=50, mode="anchored",
                           min_train=10, embargo_bars=5)
        folds = list(cv.split(300))
        second = folds[1].train_idx
        assert not set(range(115, 125)) & set(second)
        assert {114, 125} <= set(second)
        assert cv.n_splits(300) == len(folds) == 4

    def test_a_heavy_embargo_warns_about_erosion(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        cv = WalkForwardCV(train_size=30, test_size=10, step=40, mode="anchored",
                           min_train=1, embargo_bars=10)
        with caplog.at_level(logging.WARNING):
            list(cv.split(200))
        assert "eroded train set" in caplog.text


# --------------------------------------------------------------------------- #
# Uniqueness en gewichten
# --------------------------------------------------------------------------- #
class TestUniqueness:
    def test_the_kernel_skips_events_outside_the_bar_range(self) -> None:
        t0 = np.array([0, 2, -1, 4, 3], dtype=np.int32)
        t1 = np.array([4, 6, 2, 9, 3], dtype=np.int32)  # 9 > n, 3 == 3
        compiled = _calculate_average_uniqueness(t0, t1, 8)
        source = _calculate_average_uniqueness.py_func(t0, t1, 8)
        np.testing.assert_allclose(compiled, source, rtol=1e-12, atol=0)
        assert source[2] == 0.0 and source[3] == 0.0  # buiten bereik
        assert source[4] == 0.0                       # duur nul
        assert source[0] == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)

    def test_an_event_not_on_the_bar_grid_gets_no_weight(self) -> None:
        bars = pd.date_range("2024-01-01", periods=10, freq="h")
        t1 = pd.Series([3, 5], index=[bars[0], bars[0] - pd.Timedelta(hours=1)])
        assert get_average_uniqueness(bars, t1)[1] == 0.0

    def test_per_fold_uniqueness_only_counts_train_events(self) -> None:
        """Event 1 overlapt event 0, maar zit in de testfold: per fold telt het
        niet mee en is event 0 volledig uniek. Globaal niet."""
        bars = pd.date_range("2024-01-01", periods=10, freq="h")
        t1 = pd.Series([4, 6, 9], index=bars[[0, 2, 7]])
        assert get_average_uniqueness(bars, t1)[0] < 1.0
        per_fold = get_average_uniqueness_per_fold(bars, t1, np.array([0, 2]))
        np.testing.assert_allclose(per_fold, [1.0, 1.0])

    def test_an_empty_train_fold_has_no_uniqueness(self) -> None:
        bars = pd.date_range("2024-01-01", periods=10, freq="h")
        t1 = pd.Series([3, 5], index=bars[:2])
        out = get_average_uniqueness_per_fold(bars, t1, np.array([], dtype=int))
        assert out.shape == (0,) and out.dtype == np.float64

    def test_sample_weights_truncate_to_the_shorter_input(self) -> None:
        ts = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D"))
        w = get_sample_weights(ts, np.ones(3), returns=np.zeros(5))
        assert w.shape == (3,)
        # Gelijke uniqueness, nul-returns: alleen het tijdsverval telt.
        assert w[-1] == pytest.approx(1.0) and w[0] < w[1] < w[2]
