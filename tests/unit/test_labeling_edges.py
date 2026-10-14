# tests/unit/test_labeling_edges.py
"""CUSUM, fixed-horizon, trend scanning, triple barrier en de volatiliteitsbarrières.

De Numba-kernels draaien ook als `.py_func` (coverage.py ziet niets binnen
gecompileerde code) en worden daarmee vergeleken. Elke kernel-casus is
handgemaakt: vlakke prijzen met één afwijkende bar, zodat het verwachte label
uit de opzet volgt en niet uit de implementatie.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.labeling import cusum, fixed_horizon_labels
from tradebot.labeling import triple_barrier as tb
from tradebot.labeling.trend_scanning import TrendScanningLabeler, _trend_scan_core
from tradebot.labeling.triple_barrier import TripleBarrierLabeler, _triple_barrier_per_event
from tradebot.labeling.vol_barriers import BarrierLabels, label_triple_barrier, labels_to_frame
from tradebot.schemas.config import LabelingConfig
from tradebot.utils.failfast import CausalityViolationError, DataContractError


def _walk(n: int = 300, seed: int = 0, sigma: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.002,
        "low": np.minimum(open_, close) * 0.998, "close": close}, index=idx)


# --------------------------------------------------------------------------- #
# CUSUM
# --------------------------------------------------------------------------- #
class TestCusumKernels:
    P = np.cumsum(np.random.default_rng(1).normal(0, 1, 400)) + 100

    def test_the_symmetric_kernel_is_its_python_source(self) -> None:
        thr = np.full(400, 3.0)
        a = cusum._symmetric_cusum_filter_kernel(self.P, thr)
        b = cusum._symmetric_cusum_filter_kernel.py_func(self.P, thr)
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.int32 and a.size > 5

    def test_a_drop_past_the_threshold_is_an_event_too(self) -> None:
        p = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
        ev = cusum._symmetric_cusum_filter_kernel.py_func(p, np.full(5, 2.5))
        assert ev.tolist() == [3]  # -1, -2, -3 <= -2.5

    def test_the_directional_kernel_splits_up_and_down_events_and_sums_to_the_symmetric_one(self) -> None:
        thr = np.full(400, 3.0)
        up_c, down_c = cusum._directional_cusum_filter_kernel(self.P, thr, thr)
        up_p, down_p = cusum._directional_cusum_filter_kernel.py_func(self.P, thr, thr)
        np.testing.assert_array_equal(up_c, up_p)
        np.testing.assert_array_equal(down_c, down_p)
        merged = np.sort(np.concatenate([up_c, down_c]))
        np.testing.assert_array_equal(merged, cusum._symmetric_cusum_filter_kernel(self.P, thr))

    def test_the_public_wrappers_check_lengths_and_coerce_dtypes(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            cusum.symmetric_cusum_filter(np.arange(5.0), np.arange(4.0))
        with pytest.raises(ValueError, match="even lang"):
            cusum.directional_cusum_filter(np.arange(5.0), np.ones(5), np.ones(4))
        ev = cusum.symmetric_cusum_filter(self.P.astype(np.float32), np.full(400, 3, dtype=np.int64))
        up, down = cusum.directional_cusum_filter(self.P, np.full(400, 3.0), np.full(400, 3.0))
        assert ev.size == up.size + down.size
        assert cusum.cusum_filter is cusum.symmetric_cusum_filter and cusum.dual_cusum_events is cusum.get_cusum_events


class TestGetCusumEvents:
    def test_events_are_timestamps_from_the_frame_with_a_causal_threshold(self) -> None:
        df = _walk()
        ev = cusum.get_cusum_events(df, threshold_multiplier=1.0)
        assert isinstance(ev, pd.DatetimeIndex) and 0 < len(ev) < len(df) and ev.isin(df.index).all()

    def test_a_vol_feature_replaces_the_true_range_atr(self) -> None:
        df = _walk()
        plain = cusum.get_cusum_events(df, 1.0)
        with_vol = cusum.get_cusum_events(df.assign(feat_vol_gk=df["close"] * 0.003), 1.0)
        assert not plain.equals(with_vol)

    def test_zero_events_warns_and_returns_an_empty_index(self, caplog: pytest.LogCaptureFixture) -> None:
        df = _walk()
        with caplog.at_level(logging.WARNING):
            ev = cusum.get_cusum_events(df, threshold_multiplier=1e6)
        assert len(ev) == 0 and "zero events detected" in caplog.text

    def test_degenerate_bars_are_dropped_by_flag_or_by_shape(self) -> None:
        df = _walk()
        base = cusum.get_cusum_events(df, 0.5)
        flagged = df.assign(feat_bar_is_degenerate=np.isin(df.index, base[:3]).astype(np.int8))
        assert len(cusum.get_cusum_events(flagged, 0.5)) == len(base) - 3
        flat = df.copy()
        for ts in base[:2]:
            flat.loc[ts, ["open", "high", "low"]] = flat.loc[ts, "close"]
        assert len(cusum.get_cusum_events(flat, 0.5)) <= len(base)

    def test_a_frame_with_only_a_close_column_skips_the_degenerate_filter_after_the_atr_fallback(self) -> None:
        df = _walk()[["close"]].assign(feat_vol_gk=lambda d: d["close"] * 0.003)
        assert len(cusum.get_cusum_events(df, 1.0)) > 0


class TestGetCusumEventsForShort:
    def test_a_lower_down_multiplier_gives_more_down_events(self) -> None:
        df = _walk()
        same = cusum.get_cusum_events_for_short(df, threshold_multiplier=2.0)
        lower = cusum.get_cusum_events_for_short(df, threshold_multiplier=2.0, short_multiplier_override=0.8)
        assert len(lower) > len(same) > 0 and lower.isin(df.index).all()

    def test_no_down_events_warns_and_degenerate_bars_are_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        df = _walk()
        with caplog.at_level(logging.WARNING):
            assert len(cusum.get_cusum_events_for_short(df, threshold_multiplier=1e6)) == 0
        assert "zero DOWN events" in caplog.text
        keyed = df.assign(feat_vol_gk=0.3)
        ref = cusum.get_cusum_events_for_short(keyed, 0.5)
        flagged = keyed.assign(feat_bar_is_degenerate=np.isin(df.index, ref[:2]).astype(np.int8))
        assert len(cusum.get_cusum_events_for_short(flagged, 0.5)) == len(ref) - 2
        base = cusum.get_cusum_events_for_short(df, 0.5)
        bare = df[["close", "open", "high", "low"]].copy()
        assert len(cusum.get_cusum_events_for_short(bare, 0.5)) > 0
        flat = bare.copy()
        flat.loc[base[:2], ["open", "high", "low"]] = flat.loc[base[:2], "close"].to_numpy()[:, None]
        assert len(cusum.get_cusum_events_for_short(flat, 0.5)) <= len(base)

    def test_a_frame_with_only_a_close_column_and_vol_feature_skips_the_shape_filter(self) -> None:
        df = _walk()[["close"]].assign(feat_vol_gk=lambda d: d["close"] * 0.003)
        assert len(cusum.get_cusum_events_for_short(df, 1.0)) > 0


# --------------------------------------------------------------------------- #
# fixed horizon
# --------------------------------------------------------------------------- #
def test_fixed_horizon_labels_sign_the_forward_return_and_leave_the_tail_undefined() -> None:
    close = pd.Series([100.0, 101.0, 100.0, 99.0, 99.0, 99.0, 101.0, 102.0],
                      index=pd.date_range("2024-01-01", periods=8, tz="UTC"))
    labels = fixed_horizon_labels(close, horizon=2, threshold=0.005)
    assert labels.name == "label"
    assert labels.iloc[:6].tolist() == [0.0, -1.0, -1.0, 0.0, 1.0, 1.0]  # teken boven de drempel, 0 erbinnen
    assert labels.iloc[-2:].isna().all()


# --------------------------------------------------------------------------- #
# trend scanning
# --------------------------------------------------------------------------- #
class TestTrendScanKernel:
    def test_the_compiled_kernel_is_its_python_source(self) -> None:
        close = _walk(150)["close"].to_numpy()
        for a, b in zip(_trend_scan_core(close, 5, 20, 2.0), _trend_scan_core.py_func(close, 5, 20, 2.0),
                        strict=True):
            np.testing.assert_allclose(a, b, equal_nan=True)

    def test_a_clean_uptrend_is_long_a_clean_downtrend_short_and_flat_noise_is_neither(self) -> None:
        rng = np.random.default_rng(0)
        up = 100 + np.arange(60) + rng.normal(0, 0.2, 60)
        down = 200 - np.arange(60) + rng.normal(0, 0.2, 60)
        l_up, t_up, span_up = _trend_scan_core.py_func(up, 5, 20, 3.0)
        l_dn, t_dn, _ = _trend_scan_core.py_func(down, 5, 20, 3.0)
        assert l_up[0] == 1 and t_up[0] > 3.0 and 5 <= span_up[0] <= 20
        assert l_dn[0] == -1 and t_dn[0] < -3.0
        l_flat, _, _ = _trend_scan_core.py_func(rng.normal(0, 1, 60), 5, 20, 50.0)
        assert (l_flat == 0).all()

    def test_degenerate_windows_are_skipped_not_divided_by(self) -> None:
        line = np.arange(40.0)  # perfecte lijn: se_slope = 0 voor elke venstergrootte
        labels, tstats, spans = _trend_scan_core.py_func(line, 3, 10, 2.0)
        assert np.isnan(tstats[:30]).all() and (labels == 0).all() and (spans[:30] == 0).all()
        tiny = _trend_scan_core.py_func(np.arange(10.0) ** 2, 1, 2, 2.0)  # L=1: geen variantie, L=2: geen vrijheidsgraden
        assert np.isnan(tiny[1][:8]).all()

    def test_a_window_that_would_run_past_the_end_stops_the_scan(self) -> None:
        close = 100 + np.arange(12.0) + np.random.default_rng(2).normal(0, 0.1, 12)
        _, tstats, spans = _trend_scan_core.py_func(close, 4, 30, 2.0)
        assert spans[-8] <= 12 - 4  # nooit een venster langer dan wat er nog is


class TestTrendScanningLabeler:
    DF = _walk(250, seed=3, sigma=0.012)

    def test_the_lookahead_apis_are_closed(self) -> None:
        scan = TrendScanningLabeler()
        with pytest.raises(NotImplementedError, match="REMOVED"):
            scan.label(self.DF)
        with pytest.raises(RuntimeError, match="LOOKAHEAD"):
            scan.label_exploration_only(self.DF)

    def test_the_unsafe_escape_hatch_is_explicit_and_handles_empty_input(self) -> None:
        scan = TrendScanningLabeler(t_min=5, t_max=15, min_tstat=2.0)
        out = scan._label_exploration_only_unsafe(self.DF)
        assert list(out.columns) == ["label", "tstat", "best_span"] and len(out) == len(self.DF)
        assert scan._label_exploration_only_unsafe(self.DF.iloc[:0]).empty
        assert scan._label_exploration_only_unsafe(self.DF[["open"]]).empty

    def test_production_labelling_demands_event_timestamps(self) -> None:
        with pytest.raises(ValueError, match="AUDIT B-2"):
            TrendScanningLabeler().label_data(self.DF)

    def test_labels_are_scoped_to_the_events_and_missing_events_get_neutral_values(self) -> None:
        scan = TrendScanningLabeler(t_min=5, t_max=15, min_tstat=1.0)
        events = self.DF.index[[10, 50, 100]].append(pd.DatetimeIndex(["2030-01-01"], tz="UTC"))
        labels, t1, rets = scan.label_data(self.DF, event_timestamps=events, side="LONG",
                                           spread=0.0004, use_fixed_horizon=True)
        assert list(labels.index) == list(events)
        assert labels.iloc[-1] == 0 and t1.iloc[-1] == -1 and rets.iloc[-1] == 0.0
        assert labels.isin([0, 1]).all()

    def test_the_whole_frame_is_labelled_when_events_are_optional_and_the_trend_horizon_is_used(self) -> None:
        scan = TrendScanningLabeler(t_min=5, t_max=15, min_tstat=1.0)
        labels, t1, rets = scan.label_data(self.DF, require_events=False)
        assert len(labels) == len(t1) == len(rets) == len(self.DF)
        assert (t1 >= 0).all() and (t1 <= len(self.DF) - 1).all()

    def test_short_events_use_the_negative_trend_and_the_side_specific_barriers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        scan = TrendScanningLabeler(t_min=5, t_max=15, min_tstat=1.0)
        common = dict(df=self.DF, require_events=False, side="SHORT")
        with caplog.at_level(logging.INFO):
            tight, *_ = scan.label_data(**common, pt_width_short=0.5, sl_width_short=2.0)
        assert "Side-specifieke barriers" in caplog.text
        plain, *_ = scan.label_data(**common, pt_width=2.0, sl_width=1.0)
        assert tight.sum() >= plain.sum()
        only_pt, *_ = scan.label_data(**common, pt_width_short=0.5)
        assert only_pt.sum() >= plain.sum()

    def test_bid_ask_columns_and_a_vol_feature_are_used_when_present(self) -> None:
        df = self.DF.assign(bid_high=self.DF["high"], bid_low=self.DF["low"], ask_high=self.DF["high"],
                            ask_low=self.DF["low"], feat_vol_gk=self.DF["close"] * 0.004)
        scan = TrendScanningLabeler(t_min=5, t_max=15, min_tstat=1.0)
        labels, *_ = scan.label_data(df, require_events=False)
        assert len(labels) == len(df)

    def test_a_frame_without_open_high_and_low_still_labels_through_its_bid_ask_quartet(self) -> None:
        df = self.DF.assign(bid_high=self.DF["high"], bid_low=self.DF["low"], ask_high=self.DF["high"],
                            ask_low=self.DF["low"], feat_vol_gk=self.DF["close"] * 0.004)
        df = df.drop(columns=["open", "high", "low"])
        labels, *_ = TrendScanningLabeler(t_min=5, t_max=15, min_tstat=1.0).label_data(df, require_events=False)
        assert len(labels) == len(df)


# --------------------------------------------------------------------------- #
# triple barrier kernel
# --------------------------------------------------------------------------- #
N = 40


def _flat_market():
    """Alles op 100; geen barrière wordt geraakt tenzij de test één bar verschuift."""
    flat = np.full(N, 100.0)
    return {
        "open": flat.copy(), "bid_high": np.full(N, 100.5), "bid_low": np.full(N, 99.5),
        "ask_high": np.full(N, 100.5), "ask_low": np.full(N, 99.5), "close": flat.copy(),
        "atr": np.ones(N), "jump": np.zeros(N),
    }


def _run(m, events, horizons, *, side=1, pt=2.0, sl=1.0, half_spread=0.0, delay=1, sl_jump=0.0,
         min_future=1, py=True):
    kernel = _triple_barrier_per_event.py_func if py else _triple_barrier_per_event
    return kernel(
        m["open"], m["bid_high"], m["bid_low"], m["ask_high"], m["ask_low"], m["close"], m["atr"],
        np.asarray(events, dtype=np.int32), np.asarray(horizons, dtype=np.int32),
        pt, sl, side, half_spread, delay, m["jump"], sl_jump, min_future)


class TestTripleBarrierKernel:
    def test_the_compiled_kernel_is_its_python_source(self) -> None:
        df = _walk(200, seed=5)
        hi, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        atr = (hi - lo) * 3
        args = (df["open"].to_numpy(), hi, lo, hi, lo, c, atr, np.arange(5, 150, 7, dtype=np.int32),
                np.full(21, 8, dtype=np.int32), 2.0, 1.0, 1, 0.01, 1, np.zeros(200), 0.0)
        for side in (1, -1):
            a = _triple_barrier_per_event(*args[:11], side, *args[12:])
            b = _triple_barrier_per_event.py_func(*args[:11], side, *args[12:])
            for x, y in zip(a, b, strict=True):
                np.testing.assert_allclose(x, y)

    def test_a_long_profit_target_stop_loss_and_timeout(self) -> None:
        m = _flat_market()
        m["ask_high"][5] = 103.0     # event 2: entry bar 4, target 102 geraakt op bar 5
        m["bid_low"][13] = 98.5      # event 10: stop 99 geraakt op bar 13
        labels, t1, ret = _run(m, [2, 10, 20], [5, 5, 5])
        assert labels.tolist() == [1, 0, 0] and t1.tolist() == [5, 13, 26]
        assert ret.tolist() == [0.0, 0.0, 0.0]  # close blijft 100

    def test_a_short_profit_target_and_stop_loss(self) -> None:
        m = _flat_market()
        m["bid_low"][5] = 97.0       # short target 98 geraakt
        m["ask_high"][13] = 101.5    # short stop 101 geraakt
        labels, t1, _ = _run(m, [2, 10], [5, 5], side=-1)
        assert labels.tolist() == [1, 0] and t1.tolist() == [5, 13]

    def test_the_realised_return_is_signed_by_direction_and_costs_the_half_spread(self) -> None:
        m = _flat_market()
        m["bid_low"][:] = 99.6      # de stop (99,5 na de half-spread) blijft buiten bereik
        m["ask_high"][:] = 100.4    # en de short-stop (100,5) ook
        m["close"][26] = 101.0
        _, _, long_ret = _run(m, [20], [5], half_spread=0.5)
        _, _, short_ret = _run(m, [20], [5], side=-1, half_spread=0.5)
        assert long_ret[0] == pytest.approx((101.0 - 100.5) / 100.5)
        assert short_ret[0] == pytest.approx((99.5 - 101.0) / 99.5)

    def test_events_that_cannot_be_judged_are_rejected_with_the_sentinel(self) -> None:
        m = _flat_market()
        m["atr"][15] = np.nan
        m["atr"][16] = 0.0
        m["open"][32] = -1.0                                   # event 30: entry onder nul
        labels, t1, _ = _run(m, [35, -1, 15, 16, 30], [5, 5, 5, 5, 3])
        assert labels.tolist() == [-9] * 5 and t1.tolist() == [-1] * 5

    def test_an_entry_beyond_the_series_is_rejected_even_when_the_timeout_looks_fine(self) -> None:
        m = _flat_market()
        labels, t1, _ = _run(m, [N - 2], [-5])  # negatieve horizon: timeout vóór de entry
        assert labels.tolist() == [-9] and t1.tolist() == [-1]

    def test_the_stop_widens_with_the_previous_bars_wick_when_the_adaptive_stop_is_on(self) -> None:
        m = _flat_market()
        m["bid_low"][13] = 98.8                                 # stop 99 raakt, 98.5 niet
        plain, _, _ = _run(m, [10], [5])
        m["jump"][9] = 5.0                                      # wick-ratio 5 -> score 1 -> stop 1,5 ATR
        wide, _, _ = _run(m, [10], [5], sl_jump=0.5)
        assert plain.tolist() == [0] and wide.tolist() == [0]
        # De timeout-exit verschilt van de stop-exit: t1 zegt welke barrière raakte.
        t1_plain = _run(m, [10], [5])[1][0]
        t1_wide = _run(m, [10], [5], sl_jump=0.5)[1][0]
        assert t1_plain == 13 and t1_wide == 10 + 2 + 5 - 1

    def test_a_jump_array_shorter_than_the_event_bar_means_no_wick_adjustment(self) -> None:
        m = _flat_market()
        m["jump"] = np.zeros(3)
        m["bid_low"][13] = 98.8
        assert _run(m, [10], [5], sl_jump=0.5)[1][0] == 13


# --------------------------------------------------------------------------- #
# triple barrier labeler
# --------------------------------------------------------------------------- #
class TestTripleBarrierLabeler:
    DF = _walk(200, seed=7)
    EVENTS = np.arange(5, 150, 6, dtype=np.int32)
    HORIZONS = np.full(len(EVENTS), 8, dtype=np.int32)
    ATR = (DF["high"] - DF["low"]).to_numpy() * 2

    def test_the_result_is_indexed_on_the_event_timestamps(self) -> None:
        out = TripleBarrierLabeler().label(self.DF, self.EVENTS, self.HORIZONS, self.ATR)
        assert list(out.columns) == ["barrier_label", "t1_idx", "realized_return"]
        assert out.index.equals(self.DF.index[self.EVENTS]) and out["barrier_label"].isin([0, 1]).all()

    def test_events_without_enough_future_bars_are_dropped_with_a_log_line(self, caplog: pytest.LogCaptureFixture) -> None:
        events = np.array([10, 197], dtype=np.int32)
        with caplog.at_level(logging.INFO, logger=tb._logger.name):
            out = TripleBarrierLabeler().label(self.DF, events, np.full(2, 8, dtype=np.int32), self.ATR)
        assert len(out) == 1 and "rejected" in caplog.text

    def test_an_explicit_bid_ask_quartet_is_used_as_is(self) -> None:
        df = self.DF.assign(bid_high=self.DF["high"] - 0.01, bid_low=self.DF["low"] - 0.01,
                            ask_high=self.DF["high"] + 0.01, ask_low=self.DF["low"] + 0.01)
        out = TripleBarrierLabeler(sl_jump_max_mult=0.5).label(df, self.EVENTS, self.HORIZONS, self.ATR, side=-1)
        assert len(out) == len(self.EVENTS)

    def test_a_per_bar_half_spread_must_match_the_frame_and_spikes_are_clamped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        labeler = TripleBarrierLabeler(half_spread=0.001)
        with pytest.raises(ValueError, match="half_spread_arr length"):
            labeler.label(self.DF, self.EVENTS, self.HORIZONS, self.ATR, half_spread_arr=np.ones(3))
        hs = np.full(len(self.DF), 0.001)
        hs[50] = 50.0                                              # kapotte print
        with caplog.at_level(logging.WARNING, logger=tb._logger.name):
            out = labeler.label(self.DF, self.EVENTS, self.HORIZONS, self.ATR, half_spread_arr=hs)
        assert "clamped 1 half_spread" in caplog.text and len(out) == len(self.EVENTS)

    def test_a_frame_with_only_a_close_still_labels_with_a_neutral_wick_and_no_spread(self) -> None:
        df = self.DF[["close"]]
        out = TripleBarrierLabeler().label(df, self.EVENTS, self.HORIZONS, self.ATR)
        assert len(out) == len(self.EVENTS)

    def test_all_zero_spreads_give_a_zero_median_cap(self) -> None:
        out = TripleBarrierLabeler().label(self.DF, self.EVENTS, self.HORIZONS, self.ATR,
                                           half_spread_arr=np.zeros(len(self.DF)))
        assert len(out) == len(self.EVENTS)


# --------------------------------------------------------------------------- #
# vol barriers
# --------------------------------------------------------------------------- #
CFG = LabelingConfig(profit_target_sigma=2.0, stop_loss_sigma=2.0, horizon_bars=5, entry_lag_bars=1)


def _bars(n: int = 40):
    close = np.full(n, 100.0)
    return close.copy() * 1.001, close.copy() * 0.999, close, np.full(n, 0.01)


class TestVolBarriers:
    def test_the_empty_label_set_has_a_nan_ratio_and_a_summary(self) -> None:
        empty = BarrierLabels(*(np.array([]) for _ in range(7)))
        assert len(empty) == 0 and np.isnan(empty.positive_ratio)
        d = empty.as_dict()
        assert d["n_events"] == 0 and np.isnan(d["mean_holding_bars"])

    def test_a_profit_target_stop_loss_and_vertical_exit_are_distinguished(self) -> None:
        high, low, close, sigma = _bars()
        high[4] = 103.0          # event 2, entry 3, target 102 op bar 4
        low[14] = 97.0           # event 12, entry 13, stop 98 op bar 14
        side = np.zeros(40)
        side[[2, 12, 22]] = 1.0
        out = label_triple_barrier(high, low, close, sigma, side, CFG)
        assert out.event_idx.tolist() == [2, 12, 22]
        assert out.barrier_outcome.tolist() == [1, -1, 0] and out.exit_idx.tolist() == [4, 14, 28]
        d = out.as_dict()
        assert (d["n_profit_target"], d["n_stop_loss"], d["n_vertical"]) == (1, 1, 1)
        assert d["mean_holding_bars"] == pytest.approx((2 + 2 + 6) / 3)

    def test_when_both_barriers_fall_inside_one_bar_the_stop_wins(self) -> None:
        high, low, close, sigma = _bars()
        high[4], low[4] = 103.0, 97.0       # beide barrières in dezelfde bar: volgorde onbekend
        side = np.zeros(40)
        side[2] = 1.0
        out = label_triple_barrier(high, low, close, sigma, side, CFG)
        assert out.barrier_outcome.tolist() == [-1] and out.exit_idx.tolist() == [4]

    def test_a_short_wins_when_the_price_falls(self) -> None:
        high, low, close, sigma = _bars()
        low[4] = 97.0
        close[4] = 97.5
        side = np.zeros(40)
        side[2] = -1.0
        out = label_triple_barrier(high, low, close, sigma, side, CFG)
        assert out.barrier_outcome.tolist() == [1] and out.meta_label.tolist() == [1]
        assert out.realized_return[0] > 0

    def test_unusable_bars_produce_no_event(self) -> None:
        high, low, close, sigma = _bars()
        side = np.full(40, 1.0)
        side[0] = 0.0                       # geen mening
        sigma[1] = np.nan                   # geen sigma
        sigma[2] = 0.0
        side[3] = np.nan                    # richting onbekend
        close[5] = np.nan                   # entry-prijs van event 4 onbruikbaar
        close[9] = 0.0                      # en die van event 8 niet positief
        out = label_triple_barrier(high, low, close, sigma, side, CFG)
        assert not {0, 1, 2, 3, 4, 8} & set(out.event_idx.tolist())
        assert 5 in out.event_idx

    def test_a_missing_window_price_ends_the_scan_and_a_missing_exit_price_drops_the_event(self) -> None:
        high, low, close, sigma = _bars()
        side = np.zeros(40)
        side[10] = 1.0
        high[13] = np.nan                   # venster breekt af: verticale exit
        out = label_triple_barrier(high, low, close, sigma, side, CFG)
        assert out.barrier_outcome.tolist() == [0]
        close[16] = np.nan                  # exit-bar (10 + 1 + 5): geen slotkoers
        assert len(label_triple_barrier(high, low, close, sigma, side, CFG)) == 0

    def test_the_inputs_must_be_consistent_causal_and_directional(self) -> None:
        high, low, close, sigma = _bars()
        with pytest.raises(DataContractError, match="ongelijke lengte"):
            label_triple_barrier(high[:-1], low, close, sigma, np.zeros(40), CFG)
        with pytest.raises(DataContractError, match="andere waarden"):
            label_triple_barrier(high, low, close, sigma, np.full(40, 0.5), CFG)
        sneaky = LabelingConfig.model_construct(profit_target_sigma=2.0, stop_loss_sigma=2.0,
                                                horizon_bars=5, entry_lag_bars=0, min_sigma_obs=60)
        with pytest.raises(CausalityViolationError, match="entry_lag_bars"):
            label_triple_barrier(high, low, close, sigma, np.zeros(40), sneaky)

    def test_the_labels_become_a_frame_on_the_event_timestamps(self) -> None:
        high, low, close, sigma = _bars()
        high[4] = 103.0
        side = np.zeros(40)
        side[[2, 12]] = 1.0
        labels = label_triple_barrier(high, low, close, sigma, side, CFG)
        index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        frame = labels_to_frame(labels, index)
        assert frame.index.tolist() == [index[2], index[12]]
        assert frame["holding_bars"].tolist() == [2, 6] and frame["exit_ts"].iloc[0] == index[4]
        with pytest.raises(DataContractError, match="korter dan de hoogste exit"):
            labels_to_frame(labels, index[:10])
        assert labels_to_frame(BarrierLabels(*(np.array([], dtype=np.int64) for _ in range(7))), index).empty
