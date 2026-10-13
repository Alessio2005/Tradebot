# tests/unit/test_features_ta_edges.py
"""`features/_ta_kernels.py` en `features/ta.py`.

De kernels draaien ook als `.py_func`: coverage.py ziet niets binnen
gecompileerde code. Causaliteit wordt bewezen door de invoer af te kappen: een
waarde op bar t mag niet veranderen als de bars na t wegvallen.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tradebot.features import _ta_kernels as k
from tradebot.features.ta import FeatureEngineer


def _walk(n: int = 400, seed: int = 0, sigma: float = 0.01) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, sigma, n)))


def _ohlcv(n: int = 400, seed: int = 0, **extra: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = _walk(n, seed)
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low": np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close": close, "volume": rng.lognormal(8, 0.4, n)}, index=idx)
    if "taker" in extra:
        buy = df["volume"] * rng.uniform(0.3, 0.7, n)
        df["taker_buy_volume"], df["taker_sell_volume"] = buy, df["volume"] - buy
    return df


def _same(compiled, source) -> None:
    np.testing.assert_allclose(compiled, source, rtol=1e-12, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# rolling kernels
# --------------------------------------------------------------------------- #
class TestRollingKernels:
    R = np.random.default_rng(1).normal(0, 0.01, 200)

    def test_pearson_autocorrelation_is_its_python_source_and_flat_input_gives_zero(self) -> None:
        _same(k.roll_autocorr_lag1(self.R, 30), k.roll_autocorr_lag1.py_func(self.R, 30))
        out = k.roll_autocorr_lag1.py_func(np.zeros(60), 20)
        assert np.isnan(out[:20]).all() and (out[20:] == 0.0).all()

    def test_a_persistent_series_has_positive_autocorrelation(self) -> None:
        x = np.zeros(300)
        rng = np.random.default_rng(0)
        for t in range(1, 300):
            x[t] = 0.7 * x[t - 1] + rng.normal()
        assert np.nanmean(k.roll_autocorr_lag1.py_func(x, 60)) > 0.4
        assert np.nanmean(k.roll_autocorr_spearman.py_func(x, 60)) > 0.4

    def test_the_average_rank_matches_scipy_including_ties(self) -> None:
        arr = np.array([3.0, 1.0, 3.0, 2.0, 2.0, 2.0, 9.0])
        np.testing.assert_allclose(k._rankdata_average.py_func(arr), stats.rankdata(arr, method="average"))
        np.testing.assert_allclose(k._rankdata_average(arr), stats.rankdata(arr, method="average"))
        assert k._rankdata_average.py_func(np.array([])).size == 0

    def test_spearman_autocorrelation_is_its_python_source_and_flat_input_gives_zero(self) -> None:
        _same(k.roll_autocorr_spearman(self.R, 30), k.roll_autocorr_spearman.py_func(self.R, 30))
        out = k.roll_autocorr_spearman.py_func(np.zeros(60), 20)
        assert (out[20:] == 0.0).all()

    def test_one_outlier_moves_spearman_less_than_pearson(self) -> None:
        x = self.R.copy()
        spiked = x.copy()
        spiked[150] = 0.5
        pearson = np.abs(k.roll_autocorr_lag1(spiked, 30) - k.roll_autocorr_lag1(x, 30))[150:180].max()
        spearman = np.abs(k.roll_autocorr_spearman(spiked, 30) - k.roll_autocorr_spearman(x, 30))[150:180].max()
        assert spearman < pearson

    def test_the_cycle_phase_is_its_python_source_and_skips_flat_or_linear_windows(self) -> None:
        close = _walk(120)
        _same(k.roll_cycle_phase(close, 20), k.roll_cycle_phase.py_func(close, 20))
        flat = k.roll_cycle_phase.py_func(np.full(60, 100.0), 20)
        assert np.isnan(flat).all()                                   # sigma = 0: geen fase
        linear = k.roll_cycle_phase.py_func(np.arange(60.0) + 100.0, 20)
        assert np.isnan(linear[20:]).all()                            # constante stap: sig_d = 0
        valid = k.roll_cycle_phase.py_func(close, 20)
        assert np.nanmax(np.abs(valid)) <= 1.0                        # atan2 / pi

    def test_the_ewma_std_guards_and_grows_with_volatility(self) -> None:
        assert k._ewma_std.py_func(np.array([]), 10.0).size == 0
        assert (k._ewma_std.py_func(np.ones(5), 0.0) == 0.0).all()
        quiet = k._ewma_std.py_func(np.random.default_rng(0).normal(0, 0.01, 300), 50.0)
        loud = k._ewma_std.py_func(np.random.default_rng(0).normal(0, 0.05, 300), 50.0)
        assert loud[-1] > quiet[-1] > 0
        _same(k._ewma_std(self.R, 50.0), k._ewma_std.py_func(self.R, 50.0))


# --------------------------------------------------------------------------- #
# Hilbert phase
# --------------------------------------------------------------------------- #
class TestHilbertPhase:
    CLOSE = _walk(260, seed=2, sigma=0.01)

    def test_short_input_has_no_phase(self) -> None:
        assert k.compute_hilbert_phase(np.array([])).size == 0
        assert np.isnan(k.compute_hilbert_phase(self.CLOSE[:100])).all()   # burn-in 128

    @pytest.mark.parametrize("pre_filter", ["ehlers", "super_smoother", "kalman", "ema", "mystery"])
    def test_every_prefilter_gives_a_bounded_phase_after_the_burn_in(self, pre_filter: str) -> None:
        out = k.compute_hilbert_phase(self.CLOSE, pre_filter=pre_filter)
        assert np.isnan(out[:128]).all() and np.isfinite(out[128:]).all()
        assert np.abs(out[128:]).max() <= math.pi + 1e-12

    def test_the_phase_at_bar_t_does_not_depend_on_later_bars(self) -> None:
        full = k.compute_hilbert_phase(self.CLOSE)
        cut = k.compute_hilbert_phase(self.CLOSE[:200])
        np.testing.assert_allclose(full[:200], cut, equal_nan=True)


# --------------------------------------------------------------------------- #
# entropy
# --------------------------------------------------------------------------- #
class TestShannonEntropy:
    def test_the_compiled_kernel_is_its_python_source(self) -> None:
        close = _walk(150, seed=3)
        _same(k.calculate_shannon_entropy(close, 50, 10), k.calculate_shannon_entropy.py_func(close, 50, 10))

    def test_noise_is_close_to_maximal_and_constant_returns_are_zero(self) -> None:
        noise = _walk(400, seed=4)
        ent = k.calculate_shannon_entropy.py_func(noise, 100, 10)
        assert np.isnan(ent[:99]).all() and 0.9 < np.nanmean(ent) <= 1.0
        flat = k.calculate_shannon_entropy.py_func(np.full(150, 100.0), 100, 10)
        assert (flat[99:] == 0.0).all()

    def test_a_negligible_previous_price_gives_a_zero_return_not_an_error(self) -> None:
        close = _walk(150, seed=5)
        close[40] = 1e-12
        out = k.calculate_shannon_entropy.py_func(close, 50, 10)
        assert np.isfinite(out[49:]).all()

    def test_a_concentrated_distribution_has_lower_entropy_than_a_uniform_one(self) -> None:
        rng = np.random.default_rng(0)
        spread = 100 * np.exp(np.cumsum(rng.uniform(-0.01, 0.01, 300)))
        lumpy = 100 * np.exp(np.cumsum(np.where(rng.random(300) < 0.9, 0.001, rng.normal(0, 0.02, 300))))
        assert np.nanmean(k.calculate_shannon_entropy(lumpy, 100, 10)) < np.nanmean(
            k.calculate_shannon_entropy(spread, 100, 10))


# --------------------------------------------------------------------------- #
# fractional differencing
# --------------------------------------------------------------------------- #
class TestFracDiffKernels:
    def test_the_weights_follow_the_binomial_series_and_truncate_at_the_threshold(self) -> None:
        w = k.get_weights_ffd.py_func(0.5, 1e-4, 1000)
        assert w[-1] == 1.0 and w[-2] == pytest.approx(-0.5) and w[-3] == pytest.approx(-0.125)
        assert abs(w[0]) >= 1e-4 and len(w) < 1000
        np.testing.assert_allclose(k.get_weights_ffd(0.5, 1e-4, 1000), w)

    def test_d_zero_is_the_identity_and_d_one_the_first_difference(self) -> None:
        assert k.get_weights_ffd.py_func(0.0, 1e-4, 50).tolist() == [1.0]
        np.testing.assert_allclose(k.get_weights_ffd.py_func(1.0, 1e-4, 50), [-1.0, 1.0])

    def test_a_limit_below_the_truncation_point_keeps_every_weight(self) -> None:
        w = k.get_weights_ffd.py_func(0.4, 1e-12, 5)
        assert len(w) == 5

    def test_the_mask_decides_which_bars_get_a_value(self) -> None:
        series = np.arange(20.0)
        w = k.get_weights_ffd.py_func(1.0, 1e-4, 20)
        mask = np.ones(20, dtype=bool)
        mask[10] = False
        out = k.apply_frac_diff_mask_fast.py_func(series, w, mask)
        assert np.isnan(out[0]) and np.isnan(out[10]) and out[5] == pytest.approx(1.0)
        _same(k.apply_frac_diff_mask_fast(series, w, mask), out)

    def test_the_dynamic_threshold_widens_in_a_storm_and_stays_inside_its_bounds(self) -> None:
        assert k.compute_dynamic_ffd_threshold(np.zeros(5), base=2e-4) == 2e-4
        assert k.compute_dynamic_ffd_threshold(np.zeros(100), base=2e-4) == 2e-4          # sigma = 0
        rng = np.random.default_rng(0)
        calm = rng.normal(0, 0.005, 400)
        storm = np.concatenate([calm[:300], rng.normal(0, 0.05, 100)])
        tau_calm = k.compute_dynamic_ffd_threshold(calm)
        tau_storm = k.compute_dynamic_ffd_threshold(storm)
        assert tau_storm < tau_calm
        for tau in (tau_calm, tau_storm):
            assert k.DEFAULT_FFD_THRESHOLD_FLOOR <= tau <= k.DEFAULT_FFD_THRESHOLD_CEIL


class TestOptimalD:
    def test_a_stationary_series_needs_no_differencing(self) -> None:
        stationary = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 300))
        assert k.get_optimal_d(stationary, require_kpss=False) == 0.0

    def test_a_random_walk_needs_some_differencing_and_kpss_can_be_switched_off(self) -> None:
        walk = pd.Series(_walk(400, seed=6))
        d_both = k.get_optimal_d(walk, p_thres=0.05)
        d_adf = k.get_optimal_d(walk, p_thres=0.05, require_kpss=False)
        assert 0.0 < d_adf <= d_both <= 1.0

    def test_a_series_too_short_for_any_window_has_no_optimum(self) -> None:
        # d=0 is een random walk (ADF verwerpt niet), d=0.5 met een bijna onbegrensde staart laat < 30 punten over.
        assert math.isnan(k.get_optimal_d(pd.Series(_walk(40, seed=7)), max_d=0.5, step=0.5, ffd_thres=1e-12))

    def test_an_adf_that_never_rejects_gives_nan(self) -> None:
        assert math.isnan(k.get_optimal_d(pd.Series(_walk(300, seed=8)), p_thres=0.0))

    def test_a_degenerate_adf_candidate_is_skipped_not_fatal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}
        real = k.adfuller

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise ValueError("degenerate")
            return real(*a, **kw)

        monkeypatch.setattr(k, "adfuller", flaky)
        out = k.get_optimal_d(pd.Series(_walk(300, seed=9)), p_thres=0.5, require_kpss=False)
        assert calls["n"] > 2 and out >= 0.1

    def test_kpss_that_cannot_be_computed_falls_back_on_the_adf_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(k, "_kpss_pvalue", lambda _s: float("nan"))
        walk = pd.Series(_walk(400, seed=6))
        assert k.get_optimal_d(walk, p_thres=0.05) == k.get_optimal_d(walk, p_thres=0.05, require_kpss=False)

    def test_when_kpss_never_confirms_the_adf_only_candidate_is_the_safety_net(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(k, "_kpss_pvalue", lambda _s: 0.001)
        walk = pd.Series(_walk(400, seed=6))
        with caplog.at_level(logging.INFO, logger=k.logger.name):
            out = k.get_optimal_d(walk, p_thres=0.05)
        assert out == k.get_optimal_d(walk, p_thres=0.05, require_kpss=False)
        assert "fallback op ADF-only" in caplog.text

    def test_the_kpss_p_value_needs_thirty_observations(self) -> None:
        assert math.isnan(k._kpss_pvalue(np.arange(10.0)))
        assert 0.0 <= k._kpss_pvalue(np.random.default_rng(0).normal(size=100)) <= 1.0


# --------------------------------------------------------------------------- #
# FeatureEngineer
# --------------------------------------------------------------------------- #
class TestFeatureEngineerCaches:
    def test_the_volume_column_is_taken_in_order_of_preference(self) -> None:
        eng = FeatureEngineer()
        df = _ohlcv(10)
        assert eng._get_volume_col(df.assign(real_volume=1.0, tick_volume=2.0)) == "real_volume"
        assert eng._get_volume_col(df.assign(tick_volume=2.0)) == "tick_volume"
        assert eng._get_volume_col(df) == "volume"
        with pytest.raises(KeyError, match="volume-kolom"):
            eng._get_volume_col(df.drop(columns="volume"))

    def test_clearing_one_timeframe_leaves_the_others_and_clearing_all_leaves_nothing(self) -> None:
        eng = FeatureEngineer()
        for tf in ("micro", "meso"):
            eng._fixed_d_cache.update({tf: 0.4, f"{tf}_vol": 0.3})
            eng._ffd_thres_cache.update({tf: 1e-4, f"{tf}_vol": 1e-4, f"{tf}_extra": 1e-4})
        eng.clear_d_cache("micro")
        assert set(eng._fixed_d_cache) == {"meso", "meso_vol"}
        assert set(eng._ffd_thres_cache) == {"meso", "meso_vol", "meso_extra"}
        eng.clear_d_cache()
        assert not eng._fixed_d_cache and not eng._ffd_thres_cache

    def test_the_ffd_memory_depth_has_a_conservative_default_and_follows_the_cache(self) -> None:
        eng = FeatureEngineer()
        assert eng.get_ffd_max_lag("micro") == 256 and eng.get_max_feature_lag() == 256
        eng._fixed_d_cache["micro"], eng._ffd_thres_cache["micro"] = 0.5, 1e-4
        eng._fixed_d_cache["meso"], eng._ffd_thres_cache["meso"] = 0.9, 1e-4
        # Hogere d valt sneller af: de geheugendiepte daalt met d.
        assert eng.get_ffd_max_lag("meso") < eng.get_ffd_max_lag("micro")
        assert eng.get_max_feature_lag() == eng.get_ffd_max_lag("micro")

    def test_a_per_fold_calibration_needs_a_close_and_enough_bars(self, caplog: pytest.LogCaptureFixture) -> None:
        eng = FeatureEngineer()
        with pytest.raises(KeyError, match="close"):
            eng.fit_d_on_train_indices(_ohlcv(100).drop(columns="close"), np.arange(50))
        with caplog.at_level(logging.WARNING):
            assert eng.fit_d_on_train_indices(_ohlcv(100), np.arange(10)) == 0.4
        assert "te weinig train-bars" in caplog.text and eng._fixed_d_cache["micro"] == 0.4

    @pytest.mark.parametrize("dynamic", [True, False])
    def test_a_per_fold_calibration_caches_d_and_the_threshold(self, dynamic: bool) -> None:
        eng = FeatureEngineer({"dynamic_ffd_threshold": dynamic, "ffd_thres": 2e-4})
        d = eng.fit_d_on_train_indices(_ohlcv(400, seed=6), np.arange(300), timeframe="meso")
        assert eng._fixed_d_cache["meso"] == d and 0.0 <= d <= 1.0
        if not dynamic:
            assert eng._ffd_thres_cache["meso"] == 2e-4

    def test_a_calibration_that_finds_no_d_falls_back_to_zero_point_four(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from tradebot.features import ta

        monkeypatch.setattr(ta, "get_optimal_d", lambda *a, **kw: float("nan"))
        eng = FeatureEngineer()
        with caplog.at_level(logging.WARNING):
            assert eng.fit_d_on_train_indices(_ohlcv(100), np.arange(60)) == 0.4
        assert "gaf NaN" in caplog.text


class TestAddFeatures:
    @pytest.fixture(scope="class")
    def featured(self) -> pd.DataFrame:
        return FeatureEngineer().add_features(_ohlcv(400, seed=11, taker=""))

    def test_an_empty_frame_comes_back_unchanged(self) -> None:
        empty = _ohlcv(10).iloc[:0]
        assert FeatureEngineer().add_features(empty) is empty

    def test_the_full_feature_set_is_finite_and_complete(self, featured: pd.DataFrame) -> None:
        feats = [c for c in featured.columns if c.startswith("feat_")]
        for name in ("feat_autocorr_60", "feat_hilbert_phase", "feat_entropy_150", "feat_atr", "feat_breakout_pos",
                     "feat_trend_strength", "feat_frac_diff", "feat_ffd_d02", "feat_ffd_d08", "feat_ffd_vol",
                     "feat_vwap_dist", "feat_rvd_z", "feat_cvd_session_z", "feat_cvd_flow_ffd",
                     "feat_jump_intensity", "feat_vol_gk", "feat_vol_gk_jump", "feat_vol_jump_ratio", "feat_hr",
                     "feat_is_ny_open", "feat_bar_is_degenerate"):
            assert name in feats, name
        assert np.isfinite(featured[feats].to_numpy(dtype=float)).all()
        assert featured["feat_vol_jump_ratio"].between(0, 1).all()

    def test_the_d_parameters_are_calibrated_once_and_then_reused(self) -> None:
        eng = FeatureEngineer()
        df = _ohlcv(400, seed=12)
        first = eng.add_features(df)
        cached = dict(eng._fixed_d_cache)
        second = eng.add_features(df)
        assert set(cached) == {"micro", "vol_micro"} and eng._fixed_d_cache == cached
        pd.testing.assert_frame_equal(first, second)

    def test_features_do_not_look_ahead(self) -> None:
        df = _ohlcv(400, seed=13)
        eng_a, eng_b = FeatureEngineer(), FeatureEngineer()
        full = eng_a.add_features(df)
        eng_b._fixed_d_cache.update(eng_a._fixed_d_cache)
        eng_b._ffd_thres_cache.update(eng_a._ffd_thres_cache)
        cut = eng_b.add_features(df.iloc[:300])
        for col in ("feat_atr", "feat_frac_diff", "feat_entropy_150", "feat_hilbert_phase", "feat_vwap_dist",
                    "feat_jump_intensity", "feat_vol_gk"):
            np.testing.assert_allclose(full[col].iloc[:300], cut[col], equal_nan=True, err_msg=col)

    def test_the_config_switches_the_autocorrelation_the_phase_and_the_ffd_threshold(self) -> None:
        df = _ohlcv(400, seed=14)
        base = FeatureEngineer().add_features(df)
        alt = FeatureEngineer({"use_spearman_autocorr": False, "use_hilbert_phase": False,
                               "dynamic_ffd_threshold": False, "cvd_halflife_bars": 0}).add_features(df)
        assert "feat_hilbert_phase" not in alt.columns and "feat_hilbert_phase" in base.columns
        assert not np.allclose(base["feat_autocorr_60"], alt["feat_autocorr_60"])
        assert alt["feat_frac_diff_tau"].iloc[-1] == 1e-4

    def test_exact_taker_flow_replaces_the_signed_volume_proxy(self) -> None:
        df = _ohlcv(400, seed=15, taker="")
        proxy = FeatureEngineer().add_features(df.drop(columns=["taker_buy_volume", "taker_sell_volume"]))
        exact = FeatureEngineer().add_features(df)
        assert not np.allclose(proxy["feat_rvd_z"], exact["feat_rvd_z"])

    def test_a_dataset_too_small_to_calibrate_falls_back_to_the_default_d(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            out = FeatureEngineer().add_features(_ohlcv(30, seed=16))
        assert (out["feat_frac_diff_d"] == 0.4).all() and (out["feat_ffd_vol_d"] == 0.3).all()
        assert "feat_cvd_flow_ffd" not in out.columns
        assert "Dataset te klein" in caplog.text and "dataset te klein" in caplog.text

    def test_zero_return_stretches_use_the_nonzero_pair_rescaling(self) -> None:
        df = _ohlcv(300, seed=17)
        df.loc[df.index[150:156], ["open", "high", "low", "close"]] = df["close"].iloc[149]
        out = FeatureEngineer().add_features(df)
        assert (out["feat_bar_is_degenerate"].iloc[151:156] == 1).all()
        assert np.isfinite(out["feat_jump_intensity"]).all()

    def test_the_index_is_repaired_to_a_utc_datetime_index(self) -> None:
        df = _ohlcv(60, seed=18)
        as_column = df.reset_index().rename(columns={"index": "timestamp"})
        out = FeatureEngineer().add_features(as_column)
        assert str(out.index.tz) == "UTC" and len(out) == 60
        naive = df.copy()
        naive.index = naive.index.tz_localize(None)
        assert str(FeatureEngineer().add_features(naive).index.tz) == "UTC"
        no_stamp = df.reset_index(drop=True)
        no_stamp.index = df.index.astype("int64")
        assert isinstance(FeatureEngineer().add_features(no_stamp).index, pd.DatetimeIndex)

    def test_the_session_flag_marks_the_new_york_open(self) -> None:
        idx = pd.date_range("2024-01-02 13:00", periods=4, freq="30min", tz="UTC")  # 08:00-09:30 NY
        out = FeatureEngineer()._add_session_features(pd.DataFrame(index=idx))
        assert out["feat_hr"].tolist() == [8, 8, 9, 9] and out["feat_is_ny_open"].tolist() == [0.0, 0.0, 0.0, 1.0]
        naive = pd.DataFrame(index=idx.tz_localize(None))
        assert FeatureEngineer()._add_session_features(naive)["feat_is_ny_open"].sum() == 1.0
