# tests/unit/test_bars_edges.py
"""De informatiebars: de drie Numba-kernels en de pandas-laag eromheen.

Coverage.py ziet niets binnen gecompileerde code, dus elke kernel draait ook als
`.py_func` en wordt daarmee vergeleken: de datums en de indices bit-voor-bit, de
floats op afrondingsniveau (LLVM is niet bit-identiek aan numpy).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.bars import dollar, generate_imbalance_bars, generate_runs_bars, imbalance, runs
from tradebot.bars._kernels import numba_dollar_bars, numba_imbalance_bars, numba_runs_bars
from tradebot.bars.dollar import generate_dollar_bars

N_WEEKS = 3


def _ticks(n: int = 24 * 7 * N_WEEKS, freq: str = "h", seed: int = 0, **extra: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.002, n)) * close
    vol = rng.lognormal(7.0, 0.5, n)
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                       "low": np.minimum(open_, close) - spread, "close": close,
                       "tick_volume": vol}, index=idx)
    if "taker" in extra:
        buy = vol * rng.uniform(0.3, 0.7, n)
        df["taker_buy_volume"] = buy
        df["taker_sell_volume"] = vol - buy
    return df


def _arrays(df: pd.DataFrame):
    c = df["close"].to_numpy(np.float64)
    return (df.index.astype(np.int64).to_numpy(), df["open"].to_numpy(np.float64),
            df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64), c)


def _assert_same(compiled, source) -> None:
    for a, b in zip(compiled, source, strict=True):
        if a.dtype.kind == "i":
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)


def _assert_bar_continuity(out) -> None:
    ts, o, h, l, c = out[:5]
    assert (np.diff(ts) > 0).all() and (h >= l).all()
    assert (h >= np.maximum(o, c) - 1e-9).all() and (l <= np.minimum(o, c) + 1e-9).all()
    np.testing.assert_allclose(o[1:], c[:-1])  # de volgende bar opent op de vorige close


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
class TestImbalanceKernel:
    def _args(self, thresholds: float | np.ndarray = 4_000.0):
        df = _ticks(300)
        dates, o, h, l, c = _arrays(df)
        vol = df["tick_volume"].to_numpy(np.float64)
        sign = np.sign(np.diff(c, prepend=c[0]))
        sign[sign == 0] = 1.0
        imb = vol * sign
        thr = np.full(len(df), thresholds) if np.isscalar(thresholds) else thresholds
        return (dates, o, h, l, c, vol, imb, thr, c, c, c, c, c, c, c, c)

    def test_the_compiled_kernel_is_its_python_source_and_the_bars_chain(self) -> None:
        args = self._args()
        compiled, source = numba_imbalance_bars(*args), numba_imbalance_bars.py_func(*args)
        _assert_same(compiled, source)
        _assert_bar_continuity(source)
        assert len(source[0]) > 5

    def test_a_negative_run_closes_a_bar_and_carries_its_residual(self) -> None:
        n = 4
        z = np.full(n, 100.0)
        ones = np.ones(n)
        out = numba_imbalance_bars.py_func(
            np.arange(n, dtype=np.int64), z, z, z, z, ones, np.array([-3.0, -3.0, -3.0, -3.0]),
            np.full(n, 5.0), z, z, z, z, z, z, z, z)
        assert len(out[0]) == 2 and out[0].tolist() == [1, 3]  # |−6| ≥ 5, rest 1, dan |−7|...
        assert out[5][0] == pytest.approx(2.0 * 5.0 / 6.0)    # volume naar rato van de drempel

    def test_a_non_positive_threshold_falls_back_to_the_default(self) -> None:
        args = self._args(thresholds=0.0)
        _assert_same(numba_imbalance_bars(*args), numba_imbalance_bars.py_func(*args))


class TestRunsKernel:
    def _args(self, thresholds: float = 3_000.0, buy_share: float = 0.5):
        df = _ticks(300)
        dates, o, h, l, c = _arrays(df)
        vol = df["tick_volume"].to_numpy(np.float64)
        buy = vol * buy_share
        sell = vol - buy
        return (dates, o, h, l, c, vol, buy, sell, np.full(len(df), thresholds),
                c, c, c, c, c, c, c, c)

    @pytest.mark.parametrize("share", [0.8, 0.2])
    def test_the_dominant_side_closes_the_bar_for_both_sides(self, share: float) -> None:
        args = self._args(buy_share=share)
        compiled, source = numba_runs_bars(*args), numba_runs_bars.py_func(*args)
        _assert_same(compiled, source)
        _assert_bar_continuity(source)
        buy, sell = source[6], source[7]
        assert (np.maximum(buy, sell) >= 3_000.0 - 1e-6).all()
        # De dominante kant sluit de bars; de andere kant rolt als residu mee.
        assert buy.sum() > sell.sum() if share > 0.5 else sell.sum() > buy.sum()

    def test_a_non_positive_threshold_falls_back_to_the_default(self) -> None:
        args = self._args(thresholds=0.0)
        _assert_same(numba_runs_bars(*args), numba_runs_bars.py_func(*args))


class TestDollarKernel:
    def _args(self, thresholds: float = 200_000.0):
        df = _ticks(300)
        dates, o, h, l, c = _arrays(df)
        vol = df["tick_volume"].to_numpy(np.float64)
        return (dates, o, h, l, c, vol, np.full(len(df), thresholds), c, c, c, c, c, c, c, c)

    def test_the_compiled_kernel_is_its_python_source_and_each_bar_reaches_the_threshold(self) -> None:
        args = self._args()
        compiled, source = numba_dollar_bars(*args), numba_dollar_bars.py_func(*args)
        _assert_same(compiled, source)
        _assert_bar_continuity(source)
        assert (source[6] >= 200_000.0 - 1e-6).all()

    def test_a_non_positive_threshold_falls_back_to_the_default(self) -> None:
        args = self._args(thresholds=0.0)
        _assert_same(numba_dollar_bars(*args), numba_dollar_bars.py_func(*args))

    def test_a_carried_residual_can_close_the_next_bar_on_a_tick_without_notional(self) -> None:
        # Eén tick van 3x de drempel: het residu (2x) sluit de volgende bar op een tick zonder volume.
        n = 3
        price = np.full(n, 10.0)
        vol = np.array([30.0, 0.0, 0.0])
        out = numba_dollar_bars.py_func(
            np.arange(n, dtype=np.int64), price, price, price, price, vol, np.full(n, 100.0),
            price, price, price, price, price, price, price, price)
        # Bar 0: 300; residu 200 (20 stuks). Bar 1: die 200 zonder nieuwe tick; residu 100, 0 stuks. Bar 2: 100.
        assert out[0].tolist() == [0, 1, 2] and out[5].tolist() == [30.0, 20.0, 0.0]


def _with_a_silent_hour(**extra: str) -> pd.DataFrame:
    """Elk uur 03:00 zonder handel: die uur-van-de-week-emmers hebben geen geschiedenis."""
    df = _ticks(**extra)
    quiet = df.index.hour == 3
    for col in ("tick_volume", "taker_buy_volume", "taker_sell_volume"):
        if col in df:
            df.loc[quiet, col] = 0.0
    return df


# --------------------------------------------------------------------------- #
# imbalance bars
# --------------------------------------------------------------------------- #
class TestImbalanceBars:
    def test_empty_input_gives_an_empty_frame(self) -> None:
        assert generate_imbalance_bars(pd.DataFrame()).empty

    def test_exact_taker_flow_drops_the_unused_bid_ask_columns(self) -> None:
        bars = generate_imbalance_bars(_ticks(taker=""), imbalance_threshold=3_000.0)
        assert not bars.empty and "bid_open" not in bars and bars["tick_volume"].gt(0).all()

    def test_without_taker_flow_the_tick_rule_signs_the_volume(self) -> None:
        df = _ticks().rename(columns={"tick_volume": "volume"})
        bars = generate_imbalance_bars(df, imbalance_threshold=3_000.0)
        assert not bars.empty and {"bid_open", "ask_close"} <= set(bars.columns)

    def test_the_series_prefers_real_volume_and_the_tick_rule_carries_the_last_sign(self) -> None:
        df = _ticks(taker="").assign(real_volume=5.0)
        vol, imb = imbalance._calculate_imbalance_series(df)
        assert (vol == 5.0).all()
        flat = df.assign(close=100.0).drop(columns=["taker_buy_volume", "taker_sell_volume", "real_volume"])
        vol2, imb2 = imbalance._calculate_imbalance_series(flat)
        assert (imb2 == vol2).all()  # nul-verschil: de tickregel valt terug op +1

    def test_bid_and_ask_ohlc_flow_through_to_the_bars(self) -> None:
        df = _ticks()
        for side in ("bid", "ask"):
            for col in ("open", "high", "low", "close"):
                df[f"{side}_{col}"] = df[col]
        bars = generate_imbalance_bars(df, imbalance_threshold=3_000.0)
        np.testing.assert_allclose(bars["bid_close"], bars["close"])

    def test_a_non_datetime_index_is_converted_to_utc(self) -> None:
        df = _ticks()
        df.index = df.index.astype(np.int64)
        bars = generate_imbalance_bars(df, imbalance_threshold=3_000.0)
        assert str(bars.index.tz) == "UTC"

    def test_dynamic_thresholds_use_the_same_hour_of_week_in_earlier_weeks(self) -> None:
        df = _ticks(taker="")
        _, imb = imbalance._calculate_imbalance_series(df)
        thr = imbalance._calculate_dynamic_thresholds(df, imb, target_per_day=12)
        assert thr.shape == (len(df),) and np.isfinite(thr).all() and (thr > 0).all()
        bars = generate_imbalance_bars(df, daily_bar_target=12)
        assert not bars.empty
        assert imbalance._calculate_dynamic_thresholds(df.iloc[:0], imb[:0], 12).size == 0

    def test_an_hour_of_week_without_history_falls_back_to_the_daily_median(self) -> None:
        df = _with_a_silent_hour(taker="")
        _, imb = imbalance._calculate_imbalance_series(df)
        thr = imbalance._calculate_dynamic_thresholds(df, imb, 12)
        assert np.isfinite(thr).all() and (thr > 0).all()

    def test_the_dynamic_threshold_is_lookahead_safe(self) -> None:
        df = _ticks(taker="")
        _, imb = imbalance._calculate_imbalance_series(df)
        full = imbalance._calculate_dynamic_thresholds(df, imb, 12)
        cut = 24 * 7 * 2
        part = imbalance._calculate_dynamic_thresholds(df.iloc[:cut], imb[:cut], 12)
        np.testing.assert_allclose(full[:cut - 24], part[:cut - 24])


# --------------------------------------------------------------------------- #
# runs bars
# --------------------------------------------------------------------------- #
class TestRunsBars:
    def test_empty_input_gives_an_empty_frame(self) -> None:
        assert generate_runs_bars(pd.DataFrame()).empty

    def test_exact_taker_flow_builds_buy_and_sell_volumes_that_cover_the_total(self) -> None:
        df = _ticks(taker="")
        tick, buy, sell = runs._calculate_runs_series(df)
        np.testing.assert_allclose(buy + sell, tick)
        bars = generate_runs_bars(df, runs_threshold=2_000.0)
        assert not bars.empty and "bid_open" not in bars
        assert {"runs_imbalance", "feat_run_length_avg_20", "feat_run_burst_5",
                "feat_run_imbalance_zscore_50"} <= set(bars.columns)

    def test_a_buy_only_flow_derives_the_sell_side_as_the_remainder(self) -> None:
        df = _ticks(taker="").drop(columns="taker_sell_volume")
        tick, buy, sell = runs._calculate_runs_series(df)
        np.testing.assert_allclose(sell, np.maximum(tick - buy, 0.0))

    def test_sell_volume_never_drops_below_the_unexplained_remainder(self) -> None:
        df = _ticks(taker="")
        df["taker_sell_volume"] = 0.0
        tick, buy, sell = runs._calculate_runs_series(df)
        np.testing.assert_allclose(sell, tick - buy)

    def test_book_depth_signs_the_volume(self) -> None:
        df = _ticks().assign(ask_depth=np.tile([2.0, 1.0, 1.0], 168), bid_depth=np.tile([1.0, 2.0, 1.0], 168))
        tick, buy, sell = runs._calculate_runs_series(df)
        assert buy[0] == tick[0] and sell[1] == tick[1] and buy[2] == sell[2] == 0.0

    def test_the_tick_rule_on_crypto_data_warns_and_on_other_data_only_informs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        crypto = _ticks()
        with caplog.at_level(logging.INFO, logger=runs.logger.name):
            runs._calculate_runs_series(crypto)
        assert "do NOT use in production" in caplog.text
        caplog.clear()
        forex = _ticks().assign(spread=0.0001).rename(columns={"tick_volume": "volume"})
        with caplog.at_level(logging.INFO, logger=runs.logger.name):
            runs._calculate_runs_series(forex)
        assert "do NOT use" not in caplog.text and "tick-rule fallback (no orderflow" in caplog.text

    def test_a_flat_price_carries_the_previous_tick_direction(self) -> None:
        df = _ticks().assign(close=100.0, open=100.0, high=100.0, low=100.0)
        tick, buy, sell = runs._calculate_runs_series(df)
        np.testing.assert_allclose(buy, tick)  # begin op +1, nooit een tekenwissel

    def test_an_hour_of_week_without_history_falls_back_to_the_daily_median(self) -> None:
        df = _with_a_silent_hour(taker="")
        _, buy, sell = runs._calculate_runs_series(df)
        thr = runs._calculate_runs_thresholds(df, buy, sell, 12)
        assert np.isfinite(thr).all() and (thr > 0).all()

    def test_dynamic_thresholds_are_positive_and_lookahead_safe(self) -> None:
        df = _ticks(taker="")
        _, buy, sell = runs._calculate_runs_series(df)
        full = runs._calculate_runs_thresholds(df, buy, sell, 12)
        cut = 24 * 7 * 2
        part = runs._calculate_runs_thresholds(df.iloc[:cut], buy[:cut], sell[:cut], 12)
        assert (full > 0).all() and np.isfinite(full).all()
        np.testing.assert_allclose(full[:cut - 24], part[:cut - 24])
        assert not generate_runs_bars(df, daily_bar_target=12).empty

    def test_bid_ask_and_a_non_datetime_index_flow_through(self) -> None:
        df = _ticks()
        for side in ("bid", "ask"):
            for col in ("open", "high", "low", "close"):
                df[f"{side}_{col}"] = df[col]
        df.index = df.index.astype(np.int64)
        bars = generate_runs_bars(df, runs_threshold=2_000.0)
        assert str(bars.index.tz) == "UTC" and "bid_close" in bars

    def test_the_aggregate_features_are_causal_and_have_a_fallback_without_volume(self) -> None:
        frame = pd.DataFrame({"runs_imbalance": np.random.default_rng(0).uniform(-1, 1, 80),
                              "tick_volume": np.random.default_rng(1).uniform(1, 10, 80)})
        out = runs._add_run_aggregate_features(frame)
        cut = runs._add_run_aggregate_features(frame.iloc[:60])
        for col in ("feat_run_length_avg_20", "feat_run_burst_5", "feat_run_imbalance_zscore_50"):
            np.testing.assert_allclose(out[col].iloc[:60], cut[col], equal_nan=True)
        assert out["feat_run_imbalance_zscore_50"].abs().max() <= 5.0
        bare = runs._add_run_aggregate_features(frame[["runs_imbalance"]])
        assert (bare["feat_run_burst_5"] == 0.0).all()
        via_buy_vol = runs._add_run_aggregate_features(frame.rename(columns={"tick_volume": "runs_buy_vol"}))
        assert via_buy_vol["feat_run_burst_5"].notna().all()
        untouched = pd.DataFrame({"x": [1.0]})
        assert runs._add_run_aggregate_features(untouched) is untouched


# --------------------------------------------------------------------------- #
# dollar bars
# --------------------------------------------------------------------------- #
class TestDollarBars:
    def test_empty_input_gives_an_empty_frame(self) -> None:
        assert generate_dollar_bars(pd.DataFrame()).empty

    def test_each_bar_carries_at_least_the_threshold_in_notional(self) -> None:
        bars = generate_dollar_bars(_ticks(), threshold=300_000.0)
        assert not bars.empty and (bars["bar_notional"] >= 300_000.0 - 1e-6).all()
        assert {"bid_open", "ask_close"} <= set(bars.columns)

    def test_taker_flow_drops_the_unused_bid_ask_columns_and_volume_is_a_fallback_column(self) -> None:
        with_taker = generate_dollar_bars(_ticks(taker=""), threshold=300_000.0)
        assert "bid_open" not in with_taker
        plain = _ticks().rename(columns={"tick_volume": "volume"})
        assert not generate_dollar_bars(plain, threshold=300_000.0).empty

    def test_bid_ask_ohlc_and_a_non_datetime_index_flow_through(self) -> None:
        df = _ticks()
        for side in ("bid", "ask"):
            for col in ("open", "high", "low", "close"):
                df[f"{side}_{col}"] = df[col]
        df.index = df.index.astype(np.int64)
        bars = generate_dollar_bars(df, threshold=300_000.0)
        assert str(bars.index.tz) == "UTC"
        np.testing.assert_allclose(bars["ask_close"], bars["close"])

    def test_dynamic_thresholds_target_a_bar_count_and_stay_lookahead_safe(self) -> None:
        df = _ticks()
        full = dollar._calculate_dynamic_dollar_thresholds(df, target_per_day=12)
        cut = 24 * 7 * 2
        part = dollar._calculate_dynamic_dollar_thresholds(df.iloc[:cut], target_per_day=12)
        assert (full > 0).all() and np.isfinite(full).all()
        np.testing.assert_allclose(full[:cut - 24], part[:cut - 24])
        assert not generate_dollar_bars(df, daily_bar_target=12).empty
        assert dollar._calculate_dynamic_dollar_thresholds(df.iloc[:0], 12).size == 0

    def test_an_hour_of_week_without_history_falls_back_to_the_daily_median(self) -> None:
        thr = dollar._calculate_dynamic_dollar_thresholds(_with_a_silent_hour(), target_per_day=12)
        assert np.isfinite(thr).all() and (thr > 0).all()

    def test_a_volume_column_is_taken_when_tick_volume_is_absent_for_the_thresholds(self) -> None:
        df = _ticks().rename(columns={"tick_volume": "volume"})
        thr = dollar._calculate_dynamic_dollar_thresholds(df, target_per_day=12)
        assert (thr > 0).all()
        assert dollar.numba_dollar_bars is numba_dollar_bars
