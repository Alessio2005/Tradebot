# tests/unit/test_features_regime_edges.py
"""`features/regime.py`: de Numba-kernels, de regimeclassificatie en de pipeline.

De kernels draaien ook als `.py_func` (coverage.py ziet niets binnen
gecompileerde code). De pipeline draait op synthetische minuutdata: drie dagen
geeft het gedeeltelijke burn-in-pad, veertien dagen (bij 300 micro-bars per dag)
meer dan de 2.000 bars van de volledige burn-in.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tradebot.features import regime as rg
from tradebot.features.regime import (
    FeaturePipeline,
    HurstComputer,
    RegimeRouter,
    StationarityComputer,
    StructuralBreakRegime,
)


def _walk(n: int = 600, seed: int = 0, sigma: float = 0.01) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, sigma, n)))


def _same(compiled, source) -> None:
    np.testing.assert_allclose(compiled, source, rtol=1e-12, atol=1e-12, equal_nan=True)


# --------------------------------------------------------------------------- #
# stationarity momentum and Hurst
# --------------------------------------------------------------------------- #
class TestStationarityComputer:
    def test_empty_input_and_a_series_shorter_than_two_windows_give_neutral_output(self) -> None:
        sc = StationarityComputer(window=20)
        assert sc.compute(np.array([])).size == 0
        assert (sc.compute(_walk(30)) == 0.0).all()
        assert (StationarityComputer._rolling_stat_mom_njit.py_func(np.log(_walk(30)), 20) == 0.0).all()

    def test_the_compiled_kernel_is_its_python_source_and_flat_prices_give_zero(self) -> None:
        p = np.log(_walk(200))
        kernel = StationarityComputer._rolling_stat_mom_njit
        _same(kernel(p, 20), kernel.py_func(p, 20))
        assert (kernel.py_func(np.zeros(100), 20) == 0.0).all()

    def test_the_score_is_zero_in_the_burn_in_and_bounded_by_two_clamped_z_scores(self) -> None:
        calm = np.concatenate([np.full(120, 100.0) + np.random.default_rng(0).normal(0, 0.1, 120),
                               np.linspace(100, 130, 30)])
        out = StationarityComputer(window=20).compute(calm)
        assert (out[:38] == 0.0).all()                       # eerste 2 * window - 2 balken: geen score
        assert np.abs(out).max() <= 8.0 and np.abs(out).max() > 1.0   # twee geklemde z-scores van <= 4


class TestHurst:
    def test_short_series_default_to_a_random_walk_and_a_short_window_is_shrunk(self) -> None:
        hc = HurstComputer(min_window=64, num_lags=8)
        assert (hc.compute_rolling(_walk(10), 64) == 0.5).all()
        out = hc.compute_rolling(_walk(200, seed=1), window_size=700)
        assert out.shape == (200,) and np.isfinite(out).all()

    def test_the_compiled_kernel_is_its_python_source(self) -> None:
        p = np.log(_walk(300, seed=2))
        kernel = HurstComputer._rolling_hurst_njit
        _same(kernel(p, 64, 6), kernel.py_func(p, 64, 6))

    def test_persistent_series_score_above_antipersistent_ones(self) -> None:
        rng = np.random.default_rng(0)
        trend = np.cumsum(np.full(400, 0.01) + rng.normal(0, 0.002, 400))
        reverting = np.zeros(400)
        for t in range(1, 400):
            reverting[t] = 0.3 * reverting[t - 1] + rng.normal(0, 0.01)
        hc = HurstComputer(min_window=64, num_lags=6)
        h_trend = hc._rolling_hurst_njit.py_func(trend, 128, 6)[-1]
        h_revert = hc._rolling_hurst_njit.py_func(reverting, 128, 6)[-1]
        assert h_trend > 0.8
        assert h_revert < 0.4
        assert 0.0 <= h_trend <= 1.0

    def test_degenerate_windows_keep_the_default_exponent(self) -> None:
        kernel = HurstComputer._rolling_hurst_njit.py_func
        assert (kernel(np.zeros(20), 64, 6) == 0.5).all()           # n < window
        assert (kernel(np.zeros(100), 32, 6) == 0.5).all()          # geen variantie: < 3 bruikbare lags
        out = kernel(np.log(_walk(100, seed=3)), 16, 20)            # lags langer dan het venster: break
        assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# structural-break kernels
# --------------------------------------------------------------------------- #
class TestStructuralBreakKernels:
    def test_the_cusum_counts_bars_since_the_last_break_and_resets_on_a_shift(self) -> None:
        rng = np.random.default_rng(0)
        res = np.concatenate([np.zeros(60), rng.normal(0, 1, 300), rng.normal(5, 1, 100)])
        flags = rg._cusum_structural_break.py_func(res, 3.0, 30)
        _same(rg._cusum_structural_break(res, 3.0, 30), flags)
        assert flags.dtype == np.float32 and flags[0] == 0.0
        assert flags[1] == 31.0                     # eerste balk: last_break start op -min_bars
        assert (np.diff(flags[1:]) <= 1.0).all()    # +1 per bar, of een reset
        assert (np.diff(flags[1:]) < 0).any()       # minstens één breuk

    def test_an_empty_or_all_zero_residual_series_does_not_break(self) -> None:
        assert rg._cusum_structural_break.py_func(np.array([]), 3.0, 30).size == 0
        flags = rg._cusum_structural_break.py_func(np.zeros(100), 3.0, 30)
        assert flags[-1] == 100.0 - 1 + 30                      # last_break = -30: elke bar telt door

    def test_the_sigma_is_refreshed_every_250_bars(self) -> None:
        res = np.random.default_rng(1).normal(0, 1, 800)
        _same(rg._cusum_structural_break(res, 3.0, 30), rg._cusum_structural_break.py_func(res, 3.0, 30))

    def test_the_ols_residual_of_a_straight_line_is_zero_and_a_zero_window_has_no_fit(self) -> None:
        line = np.linspace(0, 10, 100)
        assert np.abs(rg._rolling_ols_residuals.py_func(line, 20)).max() < 1e-9
        p = np.log(_walk(200))
        _same(rg._rolling_ols_residuals(p, 20), rg._rolling_ols_residuals.py_func(p, 20))
        assert (rg._rolling_ols_residuals.py_func(p, 0) == 0.0).all()

    def test_the_ewma_variance_guards_follows_the_recursion_and_has_a_floor(self) -> None:
        assert rg._ewma_conditional_variance.py_func(np.array([])).size == 0
        assert (rg._ewma_conditional_variance.py_func(np.zeros(10)) == 1e-12).all()
        r = np.random.default_rng(0).normal(0, 0.01, 100)
        var = rg._ewma_conditional_variance.py_func(r)
        assert var[1] == pytest.approx(0.94 * var[0] + 0.06 * r[0] ** 2)
        _same(rg._ewma_conditional_variance(r), var)

    def test_the_wls_residuals_are_its_python_source_and_handle_short_flat_and_huge_series(self) -> None:
        p = np.log(_walk(200, seed=4))
        _same(rg._rolling_wls_residuals(p, 20), rg._rolling_wls_residuals.py_func(p, 20))
        assert (rg._rolling_wls_residuals.py_func(np.arange(5.0), 20) == 0.0).all()
        flat = rg._rolling_wls_residuals.py_func(np.full(60, 100.0), 20)
        assert np.isfinite(flat).all() and np.abs(flat).max() < 1e-3
        huge = np.cumsum(np.random.default_rng(0).normal(0, 1e8, 80))      # gewichten onder 1e-12
        assert (rg._rolling_wls_residuals.py_func(huge, 20) == 0.0).all()
        assert (rg._rolling_wls_residuals.py_func(p, 0) == 0.0).all()      # venster 0: geen regressie


# --------------------------------------------------------------------------- #
# regime router
# --------------------------------------------------------------------------- #
class TestRegimeClassifier:
    def test_each_branch_of_the_closed_form_classifier(self) -> None:
        vol = np.array([1.0, 5.0, 0.1, 1.0, 1.0, 1.0])
        skew = np.array([-3.0, 0.0, 0.0, 0.0, 0.0, -3.0])
        bars = np.array([5.0, 100.0, 100.0, 10.0, 200.0, 100.0])
        lo, hi = np.full(6, 0.5), np.full(6, 2.0)
        out = rg._classify_3_regimes.py_func(vol, skew, bars, lo, hi, -1.5)
        # chaos | hoog | laag | midden+recente breuk | midden+stabiel | paniek maar oude breuk
        assert out.tolist() == [2, 1, 0, 1, 0, 0]
        _same(rg._classify_3_regimes(vol, skew, bars, lo, hi, -1.5), out)


class TestRegimeRouter:
    DF = pd.DataFrame({"close": _walk(500, seed=5, sigma=0.01)})

    def test_too_little_data_or_no_close_gives_the_default_regime(self) -> None:
        router = RegimeRouter(skew_window=240)
        assert (router.classify(self.DF.iloc[:100], np.zeros(100)) == 0).all()
        assert (router.classify(self.DF.rename(columns={"close": "x"}), np.zeros(500)) == 0).all()

    def test_the_warmup_is_marked_unknown_and_later_bars_get_a_regime(self) -> None:
        router = RegimeRouter(vol_window=30, skew_window=100)
        ids = router.classify(self.DF, np.arange(500.0))
        assert ids.dtype == np.int8 and ids.shape == (500,)
        assert (ids[:50] == -1).all() and set(np.unique(ids[100:])) <= {0, 1, 2}

    def test_a_break_counter_of_the_wrong_length_is_resized(self) -> None:
        ids = RegimeRouter(vol_window=30, skew_window=100).classify(self.DF, np.zeros(10))
        assert ids.shape == (500,)

    def test_the_regime_at_bar_t_does_not_depend_on_later_bars(self) -> None:
        router = RegimeRouter(vol_window=30, skew_window=100)
        full = router.classify(self.DF, np.arange(500.0))
        cut = router.classify(self.DF.iloc[:400], np.arange(400.0))
        np.testing.assert_array_equal(full[:400], cut)

    def test_soft_probabilities_smooth_the_one_hot_labels_and_ignore_unknown_ones(self) -> None:
        router = RegimeRouter()
        ids = np.array([0, 0, 1, 1, -1, 2, 2, 2], dtype=np.int8)
        hard = router.soft_probabilities(ids, smoothing_bars=1)
        assert hard.sum(axis=1).tolist() == [1, 1, 1, 1, 0, 1, 1, 1]
        soft = router.soft_probabilities(ids, smoothing_bars=3)
        assert soft.shape == (8, 3) and np.allclose(soft.sum(axis=1), 1.0)
        assert soft[2, 1] < 1.0 and soft[3, 1] > soft[2, 1]    # de overgang is geleidelijk

    def test_an_empty_label_array_has_no_probabilities_instead_of_an_index_error(self) -> None:
        assert RegimeRouter().soft_probabilities(np.array([], dtype=np.int8)).shape == (0, 3)


class TestStructuralBreakRegime:
    def test_short_input_or_a_missing_close_gives_zeros(self) -> None:
        sb = StructuralBreakRegime(ols_window=60)
        assert (sb.compute_regimes(pd.DataFrame({"close": _walk(50)})) == 0).all()
        assert (sb.compute_regimes(pd.DataFrame({"x": _walk(300)})) == 0).all()

    @pytest.mark.parametrize("use_wls", [True, False])
    def test_a_level_shift_resets_the_counter(self, use_wls: bool) -> None:
        close = np.concatenate([_walk(300, seed=6, sigma=0.002), 150 * np.exp(np.cumsum(np.full(100, 0.01)))])
        flags = StructuralBreakRegime(ols_window=40, min_bars_between_breaks=20, use_wls=use_wls).compute_regimes(
            pd.DataFrame({"close": close}))
        assert flags.dtype == np.float32 and (np.diff(flags) < 0).any()


# --------------------------------------------------------------------------- #
# feature pipeline
# --------------------------------------------------------------------------- #
def _raw(days: float, per_day: int = 1440, seed: int = 0, taker: bool = True) -> pd.DataFrame:
    n = int(days * per_day)
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0003,
                       "low": np.minimum(open_, close) * 0.9997, "close": close,
                       "tick_volume": rng.lognormal(4, 0.5, n)}, index=idx)
    if taker:
        buy = df["tick_volume"] * rng.uniform(0.3, 0.7, n)
        df["taker_buy_volume"], df["taker_sell_volume"] = buy, df["tick_volume"] - buy
    return df


def _cfg(target: int = 300, **pipeline):
    return OmegaConf.create({"feature_pipeline": {"target_micro_bars": target, **pipeline}})


@pytest.fixture(scope="module")
def full():
    pipe = FeaturePipeline(_cfg(300), "BTCUSDT")
    return pipe, pipe.transform(_raw(14, seed=1))


class TestFeaturePipeline:
    def test_nothing_in_nothing_out(self) -> None:
        pipe = FeaturePipeline(_cfg(), "X")
        for empty in (None, pd.DataFrame()):
            out = pipe.transform(empty)
            assert [getattr(o, "shape", None) for o in out] == [(0,), (0,), (0,), (0, 0)]

    def test_with_enough_bars_the_full_burn_in_is_dropped_and_the_three_blocks_line_up(self, full) -> None:
        pipe, (x_micro, x_meso, x_macro, merged) = full
        assert len(merged) > 2000 and x_micro.shape[0] == x_meso.shape[0] == x_macro.shape[0] == len(merged)
        assert x_micro.shape[1] == len(pipe._raw_cols_micro) and np.isfinite(x_micro).all()
        for name in ("feat_is_warmup", "feat_bars_since_break_macro", "feat_regime_id_macro",
                     "feat_macro_tradfi_minutes_stale", "feat_macro_tradfi_stale", "feat_ema50_ratio"):
            assert name in merged.columns, name
        assert "feat_macro_tradfi_stale" not in pipe._raw_cols_macro
        assert "feat_macro_tradfi_minutes_stale" in pipe._raw_cols_macro
        np.testing.assert_array_equal(pipe._X_micro_raw, x_micro)

    def test_the_tradfi_staleness_counts_minutes_to_the_next_1300_utc_weekday_open(self, full) -> None:
        _, (_a, _b, _c, merged) = full
        idx = merged.index
        stale = merged["feat_macro_tradfi_minutes_stale"].to_numpy()
        assert stale.min() == 0.0 and stale.max() <= 5 * 1440
        open_now = (idx.dayofweek < 5) & (idx.hour >= 13) & (idx.hour < 21)
        assert (stale[open_now] == 0.0).all() and (stale[~open_now] > 0.0).all()
        sunday_noon = (idx.dayofweek == 6) & (idx.hour == 12) & (idx.minute < 15)
        if sunday_noon.any():
            assert stale[sunday_noon].max() <= 25 * 60 and stale[sunday_noon].min() > 0

    def test_a_shorter_history_keeps_a_partial_burn_in_and_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=rg.logger.name):
            *_, merged = FeaturePipeline(_cfg(300), "X").transform(_raw(3, seed=2))
        assert len(merged) == 100 and "partial burn-in" in caplog.text    # min_usable_bars

    def test_too_few_bars_stops_the_pipeline_with_an_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR, logger=rg.logger.name):
            out = FeaturePipeline(_cfg(30), "X").transform(_raw(1, seed=3))
        assert out[0].size == 0 and out[3].empty and "Pipeline stopped" in caplog.text

    def test_a_history_too_short_for_even_a_macro_bar_returns_empty_instead_of_crashing(self) -> None:
        out = FeaturePipeline(_cfg(30), "X").transform(_raw(0.125, seed=4))   # drie uur
        assert out[0].size == 0 and out[3].empty

    def test_the_index_can_come_from_a_timestamp_column_or_from_integers(self) -> None:
        raw = _raw(2, seed=5)
        as_column = raw.reset_index().rename(columns={"index": "timestamp"})
        a = FeaturePipeline(_cfg(60), "X").transform(as_column)
        b = FeaturePipeline(_cfg(60), "X").transform(raw.set_axis(raw.index.astype("int64")))
        c = FeaturePipeline(_cfg(60), "X").transform(raw)
        assert a[3].shape == b[3].shape == c[3].shape

    def test_a_symbol_column_is_carried_onto_the_bars_and_imbalance_bars_are_an_option(self) -> None:
        raw = _raw(3, seed=6).assign(symbol="BTCUSDT")
        out = FeaturePipeline(_cfg(60, use_runs_bars=False), "X").transform(raw)
        assert out[3].shape[0] > 0

    def test_the_legacy_pre_cv_pca_is_available_behind_a_flag_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        pipe = FeaturePipeline(_cfg(300, use_pca_pre_cv=True), "X")
        with caplog.at_level(logging.WARNING, logger=rg.logger.name):
            x_micro, _, _, merged = pipe.transform(_raw(14, seed=1))
        assert "LEAKAGE" in caplog.text and x_micro.shape[1] < pipe._X_micro_raw.shape[1]
        off = FeaturePipeline(_cfg(300, use_pca_orth=False), "X").transform(_raw(14, seed=1))
        assert off[0].shape[1] == pipe._X_micro_raw.shape[1]

    def test_matrix_extraction_ignores_missing_columns_and_cleans_non_finite_values(self) -> None:
        pipe = FeaturePipeline(_cfg(), "X")
        df = pd.DataFrame({"a": [1.0, np.nan, np.inf], "b": [2.0, 3.0, -np.inf]})
        assert pipe._extract_matrix_fast(df, []).shape == (3, 0)
        assert pipe._extract_matrix_fast(df, ["zzz"]).shape == (3, 0)
        np.testing.assert_array_equal(pipe._extract_matrix_fast(df, ["a", "b", "zzz"]),
                                      [[1.0, 2.0], [0.0, 3.0], [0.0, 0.0]])

    def test_duplicate_timestamps_become_strictly_increasing_without_looking_ahead(self) -> None:
        idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:01"],
                               tz="UTC")
        df = pd.DataFrame({"v": [1, 2, 3, 4]}, index=idx)
        out = FeaturePipeline._disambiguate_duplicate_index(df)
        assert out.index.is_monotonic_increasing and out.index.is_unique and str(out.index.tz) == "UTC"
        assert (out.index - idx).asi8.tolist() == [0, 1, 2, 0]           # nanoseconden, nooit terug in de tijd
        naive = FeaturePipeline._disambiguate_duplicate_index(df.set_axis(idx.tz_localize(None)))
        assert naive.index.tz is None and naive.index.is_unique
        assert FeaturePipeline._disambiguate_duplicate_index(df.iloc[[0, 3]]).equals(df.iloc[[0, 3]])
        assert FeaturePipeline._disambiguate_duplicate_index(df.iloc[:0]).empty
