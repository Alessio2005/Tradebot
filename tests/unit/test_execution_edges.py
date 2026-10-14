# tests/unit/test_execution_edges.py
"""De executiemodules die vooral door integratietests werden geraakt: het
impactmodel en zijn smoothers, de LOB-simulator, de spread-schatters, slippage,
fees en de kalibratie.

De LOB-simulator wordt met handgemaakte bars gevoed zodat elke fill-regel
(aanraking, queue, expiratie, maker versus taker) een eigen, leesbare casus is.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from tradebot.execution import market_impact as mi
from tradebot.execution.context import MarketSlice, build_context
from tradebot.execution.fees import (
    BYBIT_PERP_DEFAULT,
    BYBIT_PERP_TIERS,
    FeeSchedule,
    bybit_perp_schedule,
)
from tradebot.execution.impact_calibration import (
    REQUIRED_DATASETS,
    CalibrationEvidence,
    _permanent_share,
    calibrate_impact,
    missing_calibration_datasets,
    range_implied_eta,
    render_calibration_report,
)
from tradebot.execution.impact_model import ImpactStatus
from tradebot.execution.order_router import Order, OrderRouter, RejectReason
from tradebot.execution.simulator import FillRecord, LOBSimulator, OrderRecord
from tradebot.execution.slippage import (
    SlippageModel,
    compute_slippage,
    fixed_bps_slippage,
    vol_proportional_slippage,
)
from tradebot.execution.spread import (
    compute_annualised_sharpe,
    compute_dynamic_spread_arr,
    corwin_schultz_spread,
    detect_spread,
)
from tradebot.utils.failfast import ConfigContractError, DataContractError

from ..integration._engine_fixtures import impact_params, spread_model, venue_spec


def _hourly(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h", tz="UTC")


# --------------------------------------------------------------------------- #
# market_impact: impact, gate, inverse
# --------------------------------------------------------------------------- #
class TestSquareRootImpact:
    def test_impact_scales_with_the_square_root_of_participation(self) -> None:
        a = mi.square_root_impact(1.0, 100.0, 0.02)
        b = mi.square_root_impact(4.0, 100.0, 0.02)
        assert b.impact_fraction == pytest.approx(2 * a.impact_fraction)
        assert a.impact_bps == pytest.approx(a.impact_fraction * 1e4)
        assert a.order_participation == pytest.approx(0.01)

    def test_a_zero_order_has_no_impact_and_the_permanent_share_is_clipped(self) -> None:
        r = mi.square_root_impact(-5.0, 100.0, 0.02, permanent_share=1.7)
        assert r.impact_fraction == 0.0 and r.permanent_share == 1.0

    def test_the_crisis_multiplier_scales_eta_but_never_below_calm(self) -> None:
        calm = mi.square_root_impact(1.0, 100.0, 0.02)
        crisis = mi.square_root_impact(1.0, 100.0, 0.02, crisis_multiplier=3.0)
        discount = mi.square_root_impact(1.0, 100.0, 0.02, crisis_multiplier=0.1)
        assert crisis.impact_fraction == pytest.approx(3 * calm.impact_fraction)
        assert discount.impact_fraction == calm.impact_fraction

    def test_a_zero_volume_bar_is_guarded_by_the_minimum(self) -> None:
        assert math.isfinite(mi.square_root_impact(1.0, 0.0, 0.02).impact_fraction)

    @pytest.mark.parametrize(("skew", "mult"), [(-2.0, 3.0), (-1.5, 3.0), (0.2, 1.0)])
    def test_crisis_eta_follows_the_skew_threshold(self, skew: float, mult: float) -> None:
        assert mi.crisis_eta_from_skew(skew) == mult

    def test_the_gate_demands_alpha_above_impact_times_the_margin(self) -> None:
        ok, impact = mi.trade_passes_impact_gate(0.01, 1.0, 100.0, 0.02)
        blocked, _ = mi.trade_passes_impact_gate(impact.impact_fraction, 1.0, 100.0, 0.02)
        assert ok and not blocked

    def test_the_max_size_inverts_the_model_and_respects_the_participation_cap(self) -> None:
        q = mi.max_size_for_alpha(0.001, 1000.0, 0.02, cap_participation=1.0)
        back = mi.square_root_impact(q, 1000.0, 0.02).impact_fraction * 1.5
        assert back == pytest.approx(0.001, rel=1e-6)
        assert mi.max_size_for_alpha(1.0, 1000.0, 0.02) == pytest.approx(100.0)  # 10 % van het volume
        assert mi.max_size_for_alpha(0.0, 1000.0, 0.02) == 0.0


# --------------------------------------------------------------------------- #
# market_impact: covariance shrinkage
# --------------------------------------------------------------------------- #
class TestLedoitWolf:
    R = np.random.default_rng(0).normal(0, 0.01, (200, 4))

    def test_a_non_matrix_or_too_short_a_sample_gets_the_identity(self) -> None:
        cov, delta = mi.ledoit_wolf_shrunk_cov(np.zeros(10))
        assert cov.shape == (1, 1) and delta == 0.0
        cov, delta = mi.ledoit_wolf_shrunk_cov(self.R[:10])
        assert np.allclose(cov, np.eye(4)) and delta == 1.0

    @pytest.mark.parametrize("target", ["const_corr", "identity"])
    def test_both_targets_give_a_symmetric_psd_matrix_and_a_delta_in_the_unit_interval(self, target: str) -> None:
        R = self.R.copy()
        R[3, 1] = np.nan
        cov, delta = mi.ledoit_wolf_shrunk_cov(R, target=target)
        assert 0.0 <= delta <= 1.0
        assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > -1e-12

    def test_a_single_asset_shrinks_towards_itself_and_a_perfect_target_does_not_shrink(self) -> None:
        one, delta = mi.ledoit_wolf_shrunk_cov(self.R[:, :1])
        assert delta == 0.0 and one.shape == (1, 1)
        two = self.R[:, :2]
        cov, delta = mi.ledoit_wolf_shrunk_cov(two)  # n=2: de const-corr doel IS de sample
        assert delta == 0.0
        assert cov[0, 1] == pytest.approx(np.cov(two.T, bias=True)[0, 1])

    def test_cov_to_corr_is_unit_diagonal_and_bounded(self) -> None:
        corr = mi.cov_to_corr(np.cov(self.R.T))
        assert np.allclose(np.diag(corr), 1.0) and np.abs(corr).max() <= 1.0
        assert mi.cov_to_corr(np.zeros((0, 0))).shape == (0, 0)

    def test_the_corr_wrapper_returns_the_delta_of_the_cov(self) -> None:
        corr, delta = mi.ledoit_wolf_shrunk_corr(self.R)
        _, delta_cov = mi.ledoit_wolf_shrunk_cov(self.R)
        assert delta == delta_cov and np.allclose(np.diag(corr), 1.0)


# --------------------------------------------------------------------------- #
# market_impact: crisis multiplier, smoothers, scaler, vwpf
# --------------------------------------------------------------------------- #
class TestCrisisMultiplier:
    CALM = np.random.default_rng(1).normal(0, 0.01, 200)
    CRASH = np.concatenate([np.random.default_rng(2).normal(0, 0.002, 190), np.full(10, -0.1)])

    def test_too_short_or_too_sparse_or_flat_series_keep_the_calm_state(self) -> None:
        assert mi.negative_skew_crisis_multiplier(np.zeros(10)) == (1.0, False)
        assert mi.negative_skew_crisis_multiplier(np.full(100, np.nan), state_active=True) == (1.0, True)
        assert mi.negative_skew_crisis_multiplier(np.full(100, 0.01)) == (1.0, False)

    def test_a_negatively_skewed_tail_switches_the_overlay_on_and_hysteresis_holds_it(self) -> None:
        mult, active = mi.negative_skew_crisis_multiplier(self.CRASH)
        assert (mult, active) == (1.5, True)
        # Zolang de skew onder de herstelgrens blijft, blijft de overlay aan.
        assert mi.negative_skew_crisis_multiplier(self.CRASH, state_active=True) == (1.5, True)

    def test_recovery_above_the_resume_threshold_switches_it_off(self) -> None:
        assert mi.negative_skew_crisis_multiplier(self.CALM, state_active=True) == (1.0, False)
        assert mi.negative_skew_crisis_multiplier(self.CALM, state_active=False) == (1.0, False)


class TestSmoothers:
    def test_the_super_smoother_is_causal_and_handles_tiny_inputs(self) -> None:
        assert mi.ehlers_super_smoother(np.array([])).size == 0
        assert mi.ehlers_super_smoother(np.array([3.0])).tolist() == [3.0]
        assert mi.ehlers_super_smoother(np.array([3.0, 4.0])).tolist() == [3.0, 4.0]
        price = np.linspace(100, 110, 50) + np.random.default_rng(0).normal(0, 1, 50)
        full = mi.ehlers_super_smoother(price)
        cut = mi.ehlers_super_smoother(price[:30])
        assert np.array_equal(full[:30], cut)

    def test_the_kalman_smoother_tracks_a_level_shift_causally(self) -> None:
        assert mi.kalman_smoother_1d(np.array([])).size == 0
        price = np.concatenate([np.full(30, 100.0), np.full(30, 110.0)])
        out = mi.kalman_smoother_1d(price, process_var=1e-2, obs_var=1.0)
        assert out[0] == 100.0 and 109.0 < out[-1] <= 110.0
        assert np.array_equal(out[:40], mi.kalman_smoother_1d(price[:40], process_var=1e-2, obs_var=1.0))


class TestRankQuantileScaler:
    def test_before_fit_the_transform_is_the_constant_midpoint(self) -> None:
        out = mi.RankQuantileScaler().transform(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert (out == 0.5).all() and out.shape == (2, 2)

    def test_a_fitted_scaler_maps_onto_the_unit_interval_and_clips(self) -> None:
        X = np.random.default_rng(0).normal(size=(500, 2))
        sc = mi.RankQuantileScaler(n_bins=21)
        out = sc.fit_transform(X)
        assert out.min() >= sc.clip_low and out.max() <= sc.clip_high
        assert abs(float(np.median(out)) - 0.5) < 0.05
        far = sc.transform(np.array([[100.0, -100.0]]))
        assert far.tolist() == [[sc.clip_high, sc.clip_low]]

    def test_a_one_dimensional_fit_and_transform_are_promoted_to_columns(self) -> None:
        sc = mi.RankQuantileScaler().fit(np.arange(100.0))
        assert sc.transform(np.array([50.0])).shape == (1, 1)

    def test_partial_fit_buffers_through_the_warmup_and_then_fits(self) -> None:
        rng = np.random.default_rng(0)
        sc = mi.RankQuantileScaler(warmup_bars=20)
        sc.partial_fit(rng.normal(size=(12, 2)))
        assert sc._sorted is None and (sc.transform(np.zeros((1, 2))) == 0.5).all()
        sc.partial_fit(rng.normal(size=(12, 2)))
        assert sc._sorted is not None and sc._buffer is None

    def test_a_one_dimensional_partial_fit_is_a_single_row(self) -> None:
        sc = mi.RankQuantileScaler(warmup_bars=5)
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            sc.partial_fit(np.array([v, v * 2]))
        assert sc._sorted is not None and sc._n_features == 2

    def test_after_the_warmup_the_markers_drift_towards_new_data_and_extend_at_the_edges(self) -> None:
        sc = mi.RankQuantileScaler(warmup_bars=10).fit(np.linspace(0, 1, 100).reshape(-1, 1))
        lo, hi = float(sc._sorted[0, 0]), float(sc._sorted[-1, 0])
        sc.partial_fit(np.array([[5.0], [-5.0]]))
        # Beide uitschieters verleggen hun kant van de markers, en de volgorde blijft monotoon.
        assert sc._sorted[-1, 0] > hi and sc._sorted[0, 0] < lo
        assert (np.diff(sc._sorted[:, 0]) >= 0).all()


def test_vwpf_accepts_only_a_realistic_participation_band() -> None:
    assert mi.vwpf_check(0.0, 100.0) == (False, 0.0)
    assert mi.vwpf_check(10.0, 0.0) == (False, 0.0)
    assert mi.vwpf_check(1.0, 100.0) == (False, 0.01)    # te klein voor een fill
    assert mi.vwpf_check(10.0, 100.0) == (True, 0.1)
    assert mi.vwpf_check(50.0, 100.0) == (False, 0.5)    # eigen impact


# --------------------------------------------------------------------------- #
# LOB simulator
# --------------------------------------------------------------------------- #
def _bars(lows, highs=None, vols=None, start: str = "2024-01-01") -> pd.DataFrame:
    n = len(lows)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs if highs is not None else lows + 2.0, dtype=float)
    return pd.DataFrame(
        {"open": lows + 1.0, "high": highs, "low": lows, "close": lows + 1.0,
         "volume": np.asarray(vols if vols is not None else np.full(n, 100.0), dtype=float)},
        index=_hourly(n, start),
    )


def _order(side: str = "BUY", limit: float = 100.0, queue: float | None = 50.0,
           entry: str = "2024-01-01") -> OrderRecord:
    return OrderRecord("o1", "BTC", side, limit, 2.0, pd.Timestamp(entry, tz="UTC"), queue)


class TestLOBSimulator:
    def test_an_order_fills_only_after_enough_volume_traded_at_its_level(self) -> None:
        # Bar 1 raakt de limiet niet, bar 2 en 3 wel: 40 + 40 >= 50.
        bars = _bars(lows=[101.0, 99.0, 99.0], vols=[100.0, 100.0, 100.0])
        fill = LOBSimulator().simulate_order(_order(), bars)
        assert isinstance(fill, FillRecord)
        assert fill.bars_in_queue == 3 and fill.fill_price == 100.0
        assert fill.fill_size == 2.0 and fill.market_impact_bps == 0.0

    def test_a_sell_touches_on_the_high(self) -> None:
        bars = _bars(lows=[90.0, 90.0], highs=[99.0, 101.0], vols=[200.0, 200.0])
        fill = LOBSimulator().simulate_order(_order("SELL", queue=10.0), bars)
        assert fill is not None and fill.bars_in_queue == 2

    def test_the_queue_is_proxied_from_the_first_bar_volume_when_unknown(self) -> None:
        bars = _bars(lows=[99.0] * 4, vols=[100.0] * 4)
        sim = LOBSimulator(queue_vol_multiplier=2.0)  # queue = 200, 40 per bar
        assert sim.simulate_order(_order(queue=None), bars) is None  # 5 bars nodig, er zijn er 4
        long_bars = _bars(lows=[99.0] * 6, vols=[100.0] * 6)
        assert sim.simulate_order(_order(queue=None), long_bars).bars_in_queue == 5

    def test_no_bars_no_touch_and_expiry_all_leave_the_order_unfilled(self, caplog: pytest.LogCaptureFixture) -> None:
        sim = LOBSimulator(max_wait_bars=2)
        assert sim.simulate_order(_order(), _bars(lows=[]).iloc[0:0]) is None
        assert sim.simulate_order(_order(), _bars(lows=[110.0, 110.0])) is None
        slow = _bars(lows=[99.0] * 5, vols=[1.0] * 5)
        with caplog.at_level(logging.DEBUG):
            assert sim.simulate_order(_order(queue=1e6), slow) is None
        assert "expired" in caplog.text

    def test_a_taker_fill_pays_impact_a_maker_fill_does_not(self) -> None:
        bars = _bars(lows=[99.0, 99.0], vols=[1_000.0, 1_000.0])
        sim = LOBSimulator(adv_window_bars=2)
        maker = sim.simulate_order(_order(queue=1.0), bars, treat_as_maker=True)
        taker_buy = sim.simulate_order(_order("BUY", queue=1.0), bars, treat_as_maker=False)
        taker_sell = sim.simulate_order(_order("SELL", queue=1.0), bars.assign(high=101.0), treat_as_maker=False)
        assert maker.market_impact_bps == 0.0
        assert taker_buy.market_impact_bps > 0 and taker_buy.fill_price > 100.0
        assert taker_sell.fill_price < 100.0

    def test_impact_can_be_switched_off_for_taker_fills(self) -> None:
        bars = _bars(lows=[99.0], vols=[1_000.0])
        fill = LOBSimulator(apply_market_impact=False).simulate_order(_order(queue=1.0), bars, treat_as_maker=False)
        assert fill.market_impact_bps == 0.0 and fill.fill_price == 100.0

    def test_the_impact_formula_guards_and_caps(self) -> None:
        sim = LOBSimulator()
        assert sim._compute_market_impact_bps("BUY", 1.0, 100.0, 10.0, adv=0.0, ann_vol=0.8) == 0.0
        assert sim._compute_market_impact_bps("BUY", 1.0, 0.0, 10.0, adv=10.0, ann_vol=0.8) == 0.0
        huge = sim._compute_market_impact_bps("BUY", 1e12, 100.0, 10.0, adv=1.0, ann_vol=50.0, bars_per_year=1.0)
        assert huge == 200.0

    def test_bars_missing_high_and_low_fall_back_to_the_close(self) -> None:
        bars = _bars(lows=[99.0, 99.0]).drop(columns=["high", "low"])  # close = 100
        assert LOBSimulator().simulate_order(_order(queue=1.0), bars) is not None

    def test_simulate_fills_slices_after_the_entry_bar_and_skips_orders_without_a_future(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bars = _bars(lows=[99.0] * 6)
        ok = _order(queue=1.0, entry="2024-01-01")
        offgrid = OrderRecord("o2", "BTC", "BUY", 100.0, 1.0, pd.Timestamp("2024-01-01 00:30", tz="UTC"), 1.0)
        last = OrderRecord("o3", "BTC", "BUY", 100.0, 1.0, bars.index[-1], 1.0)
        with caplog.at_level(logging.INFO):
            fills = LOBSimulator().simulate_fills(bars, [ok, offgrid, last])
        assert [f.order_id for f in fills] == ["o1", "o2"]
        assert fills[0].fill_ts == bars.index[1]
        assert "3 orders" in caplog.text and "2 fills" in caplog.text

    def test_the_fill_probability_and_the_wait_cost(self) -> None:
        assert LOBSimulator.estimate_fill_probability(0.0, 0.4, 100.0, 3) == 1.0
        assert LOBSimulator.estimate_fill_probability(100.0, 0.5, 100.0, 1) == pytest.approx(0.5)
        assert LOBSimulator.estimate_fill_probability(1e9, 0.4, 100.0, 1) == pytest.approx(0.0, abs=1e-6)
        assert LOBSimulator.adjust_cost_for_queue_wait(2.0, 10.0) == pytest.approx(3.0)

    def test_the_constructor_clips_its_parameters(self) -> None:
        sim = LOBSimulator(fill_fraction=5.0, queue_vol_multiplier=-1.0, bars_per_year=0.0)
        assert sim.fill_fraction == 1.0 and sim.queue_vol_multiplier == 0.0 and sim.bars_per_year == 1.0


# --------------------------------------------------------------------------- #
# spread
# --------------------------------------------------------------------------- #
def _ohlc(n: int = 80, seed: int = 0, price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.uniform(1_000, 2_000, n),
    }, index=_hourly(n))


class TestSpread:
    def test_bid_ask_data_already_carries_the_cost(self) -> None:
        assert detect_spread(_ohlc().assign(bid_close=1.0, ask_close=1.0), 0.002) == 0.0

    def test_mid_price_data_gets_a_floor_of_the_round_trip_fee(self) -> None:
        assert detect_spread(_ohlc(), 0.002) >= 0.0010

    def test_an_empty_frame_and_an_unusable_median_return_the_configured_spread(self) -> None:
        assert detect_spread(_ohlc().iloc[0:0], 0.0033) == 0.0033
        assert math.isnan(detect_spread(_ohlc(), float("nan")))

    def test_the_per_bar_spread_is_bounded_between_the_fee_and_five_times_the_fallback(self) -> None:
        arr = compute_dynamic_spread_arr(_ohlc(), fallback=0.002)
        assert arr.shape == (80,) and arr.min() >= 0.0010 and arr.max() <= 0.010
        assert compute_dynamic_spread_arr(_ohlc().iloc[0:0]).size == 0

    def test_a_live_fee_schedule_replaces_the_default_round_trip_fee(self) -> None:
        deep = _ohlc().assign(tick_volume=1e6)  # genoeg diepte: de cap bindt niet
        default = compute_dynamic_spread_arr(deep)
        scheduled = compute_dynamic_spread_arr(deep, fee_schedule=FeeSchedule(taker_bps=2.0))
        # 2 bps taker, round trip = 4 bps in plaats van de 10 bps van de VIP0-default.
        np.testing.assert_allclose(default - scheduled, 0.0006, atol=1e-12)

    def test_the_sigma_and_depth_proxies_follow_the_available_columns(self) -> None:
        base = _ohlc()
        gk = compute_dynamic_spread_arr(base.assign(feat_vol_gk=0.05), ref_trade_size=5.0)
        taker = compute_dynamic_spread_arr(base.assign(taker_buy_volume=10.0, taker_sell_volume=10.0))
        tick = compute_dynamic_spread_arr(base.assign(tick_volume=50.0))
        plain = compute_dynamic_spread_arr(base)
        assert gk.mean() > plain.mean()          # grotere sigma en Q, kleine diepte
        assert taker.shape == tick.shape == plain.shape

    def test_a_notional_trade_size_is_converted_with_the_median_price(self) -> None:
        cheap = compute_dynamic_spread_arr(_ohlc(price=1.0), ref_trade_notional_usd=10_000.0)
        dear = compute_dynamic_spread_arr(_ohlc(price=10_000.0), ref_trade_notional_usd=10_000.0)
        assert cheap.mean() > dear.mean()

    def test_a_volume_column_in_quote_units_is_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            compute_dynamic_spread_arr(_ohlc().assign(tick_volume=1e12))
        assert "USDT instead of base asset" in caplog.text

    def test_the_sharpe_pads_idle_days_with_zero_and_annualises_with_365_25(self) -> None:
        ts = np.array(["2024-01-01", "2024-01-02", "2024-01-04"], dtype="datetime64[ns]")
        sharpe = compute_annualised_sharpe(np.array([0.01, -0.005, 0.02]), ts)
        daily = np.array([0.01, -0.005, 0.0, 0.02])  # 3 januari zonder trade
        assert sharpe == pytest.approx(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))

    def test_the_sharpe_of_no_trades_and_of_a_flat_book_is_zero(self) -> None:
        assert compute_annualised_sharpe(np.array([]), np.array([], dtype="datetime64[ns]")) == 0.0
        ts = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        assert compute_annualised_sharpe(np.array([0.0, 0.0]), ts) == 0.0


class TestCorwinSchultz:
    def test_the_estimate_is_non_negative_and_the_first_bar_is_undefined(self) -> None:
        df = _ohlc(200)
        cs = corwin_schultz_spread(df["high"], df["low"])
        assert cs.name == "cs_spread" and np.isnan(cs.iloc[0])
        assert (cs.dropna() >= 0.0).all()

    def test_a_wider_range_than_the_two_bar_range_implies_gives_a_positive_spread(self) -> None:
        idx = _hourly(4)
        high = pd.Series([101.0, 101.0, 101.0, 101.0], index=idx)
        low = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
        assert corwin_schultz_spread(high, low).iloc[1:].gt(0).all()

    def test_mismatched_axes_and_non_positive_prices_are_refused(self) -> None:
        idx = _hourly(3)
        ok = pd.Series([2.0, 2.0, 2.0], index=idx)
        with pytest.raises(DataContractError, match="tijdas"):
            corwin_schultz_spread(ok, pd.Series([1.0, 1.0, 1.0], index=_hourly(3, "2025-01-01")))
        with pytest.raises(DataContractError, match="niet-positieve"):
            corwin_schultz_spread(ok, pd.Series([1.0, 0.0, 1.0], index=idx))


# --------------------------------------------------------------------------- #
# slippage
# --------------------------------------------------------------------------- #
class TestSlippage:
    def test_fixed_slippage_is_bps_of_the_price_and_never_negative(self) -> None:
        assert fixed_bps_slippage(5.0, 200.0) == pytest.approx(0.1)
        assert fixed_bps_slippage(-5.0, 200.0) == 0.0

    def test_vol_slippage_uses_the_larger_of_vol_and_the_floor(self) -> None:
        assert vol_proportional_slippage(0.01, 100.0) == pytest.approx(1.0)
        assert vol_proportional_slippage(0.0, 100.0, floor_bps=5.0) == pytest.approx(0.05)
        assert vol_proportional_slippage(0.0, 100.0, floor_bps=5.0, observed_spread_bps=20.0) == pytest.approx(0.2)

    def test_each_mode_dispatches_and_every_mode_respects_the_observed_floor(self) -> None:
        fixed = SlippageModel(mode="fixed", fixed_bps=1.0, floor_bps=5.0)
        assert compute_slippage(fixed, 100.0) == pytest.approx(0.05)
        assert compute_slippage(fixed, 100.0, observed_spread_bps=30.0) == pytest.approx(0.3)
        vol = SlippageModel(mode="vol_linear", vol_scale=2.0)
        assert compute_slippage(vol, 100.0, sigma_bar=0.01) == pytest.approx(2.0)
        sq = SlippageModel(mode="sqrt_impact", impact_eta=0.1)
        small = compute_slippage(sq, 100.0, order_size=1e-6, bar_volume=1e6, sigma_bar=0.01)
        big = compute_slippage(sq, 100.0, order_size=500.0, bar_volume=1000.0, sigma_bar=0.05)
        crisis = compute_slippage(sq, 100.0, order_size=500.0, bar_volume=1000.0, sigma_bar=0.05,
                                  crisis_multiplier=3.0)
        assert small == pytest.approx(0.05)  # de vloer bindt
        assert big > small and crisis == pytest.approx(3 * big)

    def test_an_unknown_mode_falls_back_to_fixed_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        bogus = SlippageModel(mode="mystery")  # type: ignore[arg-type]
        with caplog.at_level(logging.WARNING):
            assert compute_slippage(bogus, 100.0) == pytest.approx(0.05)
        assert "Unknown SlippageModel.mode" in caplog.text


# --------------------------------------------------------------------------- #
# fees
# --------------------------------------------------------------------------- #
class TestFees:
    def test_maker_and_taker_fees_are_bps_of_the_notional_and_never_negative(self) -> None:
        assert BYBIT_PERP_DEFAULT.compute(10_000.0, "taker") == pytest.approx(5.5)
        assert BYBIT_PERP_DEFAULT.compute(10_000.0, "maker") == pytest.approx(2.0)
        assert BYBIT_PERP_DEFAULT.compute(-10_000.0) == 0.0

    def test_funding_accrues_per_eight_hour_period(self) -> None:
        fs = FeeSchedule(funding_bps_per_8h=2.0)
        assert fs.funding_cost(10_000.0, 16, bars_per_8h=8) == pytest.approx(4.0)
        assert fs.funding_cost(10_000.0, 4, bars_per_8h=0) == pytest.approx(8.0)  # nul wordt 1

    def test_tier_aliases_and_unknown_tiers_resolve_to_non_vip(self, caplog: pytest.LogCaptureFixture) -> None:
        base = bybit_perp_schedule("NONVIP")
        assert bybit_perp_schedule("vip0") == base and bybit_perp_schedule("default") == base
        with caplog.at_level(logging.WARNING):
            assert bybit_perp_schedule("GALAXY") == base
        assert "unknown tier" in caplog.text
        assert bybit_perp_schedule(next(k for k in BYBIT_PERP_TIERS if k != "NONVIP")).taker_bps <= base.taker_bps


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
def _slice() -> MarketSlice:
    return MarketSlice(
        ts=pd.Timestamp("2024-01-02", tz="UTC"),
        marks={"ETH": 2000.0, "BTC": 40_000.0}, sigma_hat={"ETH": 0.7, "BTC": 0.5},
        sigma_daily={"ETH": 0.03, "BTC": 0.02}, adv_notional={"ETH": 1e8, "BTC": 1e9},
        bar_volume_notional={"ETH": 1e7, "BTC": 1e8}, clusters={"ETH": "a", "BTC": "a"},
        funding_rate={},
    )


class TestContext:
    def test_a_slice_lists_its_symbols_sorted(self) -> None:
        assert _slice().symbols == ("BTC", "ETH")

    @pytest.mark.parametrize("kind", ["backtest", "live"])
    def test_both_contexts_expose_the_same_slice_and_state(self, kind: str) -> None:
        market = _slice()
        ctx = build_context(kind, market, equity=1e6, high_water_mark=1e6, day_start_equity=1e6,
                            positions={"BTC": 1.0}, next_fill_ts=pd.Timestamp("2024-01-03", tz="UTC"))
        assert ctx.slice() is market and ctx.kind == kind
        assert ctx.marks() == market.marks and ctx.positions() == {"BTC": 1.0}
        assert ctx.audit_header()["n_symbols"] == 2

    def test_an_unknown_kind_and_a_fill_that_is_not_in_the_future_are_refused(self) -> None:
        kw = dict(equity=1.0, high_water_mark=1.0, day_start_equity=1.0, positions={})
        with pytest.raises(DataContractError, match="Onbekend soort"):
            build_context("paper", _slice(), next_fill_ts=pd.Timestamp("2024-01-03", tz="UTC"), **kw)
        with pytest.raises(DataContractError, match="strikt na"):
            build_context("live", _slice(), next_fill_ts=_slice().ts, **kw)


# --------------------------------------------------------------------------- #
# order router: a bar nobody traded
# --------------------------------------------------------------------------- #
def test_an_order_into_a_bar_without_volume_is_rejected_for_lack_of_liquidity() -> None:
    router = OrderRouter(venue=venue_spec(), spread=spread_model(), impact=impact_params(),
                         risk_config_hash="h")
    order = Order("BTC", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"), 1.0, "h")
    report = router.execute(order, arrival_price=100.0, bar_volume_notional=0.0, adv_notional=1e6,
                            sigma_daily=0.02, participation_cap=0.01)
    assert report.is_rejected and report.reject_reason is RejectReason.NO_LIQUIDITY


# --------------------------------------------------------------------------- #
# impact calibration
# --------------------------------------------------------------------------- #
def _panel(n: int = 60, seed: int = 0, sym_vol: float = 0.02) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, sym_vol, n)))
    frame = pd.DataFrame({"high": close * 1.02, "low": close * 0.98, "close": close}, index=idx)
    return frame, pd.Series(np.full(n, sym_vol), index=idx)


class TestImpactCalibration:
    def test_missing_datasets_are_reported_in_the_required_order(self) -> None:
        assert missing_calibration_datasets([]) == REQUIRED_DATASETS
        assert missing_calibration_datasets(["trades"]) == ("orderbook_l2",)
        assert missing_calibration_datasets(["trades", "orderbook_l2", "x"]) == ()

    def test_range_implied_eta_drops_days_without_a_vol_estimate(self) -> None:
        frame, sigma = _panel(10)
        sigma.iloc[3] = 0.0
        sigma.iloc[4] = np.nan
        assert len(range_implied_eta(frame, sigma)) == 8

    def test_range_implied_eta_validates_its_inputs(self) -> None:
        frame, sigma = _panel(10)
        with pytest.raises(DataContractError, match="high, low en close"):
            range_implied_eta(frame.drop(columns=["low"]), sigma)
        with pytest.raises(DataContractError, match="tijdas"):
            range_implied_eta(frame, sigma.iloc[1:])

    def test_the_permanent_share_defaults_to_one_without_data_or_range(self) -> None:
        frame, _ = _panel(10)
        assert _permanent_share(frame.iloc[:1]) == 1.0
        flat = frame.assign(high=frame["close"], low=frame["close"])
        assert _permanent_share(flat) == 1.0
        assert 0.0 <= _permanent_share(frame) <= 1.0

    def test_calibration_without_orderbook_data_yields_an_uncalibrated_upper_bound(self) -> None:
        a, sa = _panel(60, 0)
        b, sb = _panel(60, 1, 0.03)
        params, evidence = calibrate_impact(
            {"BTC": a, "ETH": b}, {"BTC": sa, "ETH": sb},
            available_datasets=[], data_hashes={"BTC": "h1", "ETH": "h2"},
        )
        assert params.status is ImpactStatus.IMPACT_UNCALIBRATED and not params.is_calibrated
        assert params.instruments == ("BTC", "ETH") and params.sample_size == 120
        assert evidence.missing_datasets == REQUIRED_DATASETS
        assert params.eta_ci_low <= params.eta <= params.eta_ci_high
        record = evidence.as_record()
        assert record["status"] == "IMPACT_UNCALIBRATED" and set(record["per_symbol"]) == {"BTC", "ETH"}
        assert params.as_record()["eta"] == params.eta

    def test_calibration_refuses_empty_input_and_missing_vol(self) -> None:
        frame, sigma = _panel(10)
        with pytest.raises(DataContractError, match="zonder instrumenten"):
            calibrate_impact({}, {}, available_datasets=[], data_hashes={})
        with pytest.raises(DataContractError, match="Ontbrekende dagvolatiliteit"):
            calibrate_impact({"BTC": frame}, {}, available_datasets=[], data_hashes={})
        zero = pd.Series(0.0, index=sigma.index)
        with pytest.raises(DataContractError, match="Nul bruikbare"):
            calibrate_impact({"BTC": frame}, {"BTC": zero}, available_datasets=[], data_hashes={})

    def test_the_report_cites_the_status_the_quantiles_and_the_sources(self) -> None:
        a, sa = _panel(60, 0)
        params, evidence = calibrate_impact({"BTC": a}, {"BTC": sa}, available_datasets=["trades"],
                                            data_hashes={"BTC": "abc123"})
        report = render_calibration_report(params, evidence, git_sha="deadbeef")
        assert isinstance(evidence, CalibrationEvidence)
        for needle in ("IMPACT_UNCALIBRATED", "deadbeef", "`orderbook_l2` | **nee**",
                       "`trades` | ja", "`abc123`", "p95 (gekozen bovengrens)"):
            assert needle in report
        assert f"{params.eta:.4f}" in report


def test_the_config_error_type_is_the_one_the_router_uses() -> None:
    router = OrderRouter(venue=venue_spec(), spread=spread_model(), impact=impact_params(),
                         risk_config_hash="h")
    order = Order("BTC", pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC"), 1.0, "h")
    with pytest.raises(ConfigContractError, match="participation_cap"):
        router.execute(order, arrival_price=100.0, bar_volume_notional=1e6, adv_notional=1e6,
                       sigma_daily=0.02, participation_cap=0.0)
