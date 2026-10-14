# tests/unit/test_backtest_edges.py
"""De backtest-paden buiten de grote integratietests: metrics, PBO/SPA, de
drawdown-nul, de boekhouding, de event-loop en de Numba-kernel.

De kernel (`calc_non_overlapping_stats`) wordt ook via `.py_func` aangeroepen:
`coverage.py` ziet niets binnen gecompileerde code.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot import backtest as backtest_pkg
from tradebot.backtest import engine as engine_mod
from tradebot.backtest._kernels import calc_non_overlapping_stats
from tradebot.backtest.accounting import Fill, Ledger
from tradebot.backtest.barrier_book import BookInputs, SizingRule, run_barrier_book
from tradebot.backtest.dd_shape import (
    dd_shape_reference,
    expected_max_dd_over_vol,
    max_dd_over_vol_quantile,
)
from tradebot.backtest.engine import build_slices, exposures_from_frame, iter_equity
from tradebot.backtest.metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from tradebot.backtest.pbo import compute_pbo
from tradebot.backtest.phase5_baseline import LayerResult, summarise
from tradebot.backtest.spa import _autocorr_block_size, spa_test
from tradebot.execution.order_router import OrderRouter
from tradebot.risk.engine import RiskEngine
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

from ..integration._engine_fixtures import (
    impact_params,
    make_engine,
    make_replay,
    make_router,
    spread_model,
    venue_spec,
)

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
BPY = 365.0


# --------------------------------------------------------------------------- #
# package
# --------------------------------------------------------------------------- #
class TestPackage:
    def test_the_engine_names_load_lazily_and_unknown_names_are_refused(self) -> None:
        assert backtest_pkg.EventDrivenEngine is engine_mod.EventDrivenEngine
        assert backtest_pkg.build_slices is engine_mod.build_slices
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            backtest_pkg.nope  # noqa: B018


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class TestMetrics:
    R = np.array([0.01, -0.02, 0.015, 0.0, 0.005, -0.01, 0.02])

    def test_empty_and_non_finite_inputs_have_neutral_answers(self) -> None:
        nan = np.array([np.nan, np.inf])
        assert annualized_return(np.array([]), bars_per_year=BPY) == 0.0
        assert annualized_return(nan, bars_per_year=BPY) == 0.0
        assert annualized_vol(np.array([0.01]), bars_per_year=BPY) == 0.0
        assert sharpe_ratio(np.array([0.01]), bars_per_year=BPY) == 0.0
        assert sortino_ratio(np.array([0.01]), bars_per_year=BPY) == 0.0
        assert max_drawdown(np.array([])) == (0.0, 0, 0)
        assert calmar_ratio(np.array([100.0]), bars_per_year=BPY) == 0.0

    def test_annualisation_uses_the_bars_per_year_it_is_given(self) -> None:
        assert annualized_vol(self.R, bars_per_year=BPY) == pytest.approx(
            np.std(self.R, ddof=1) * np.sqrt(BPY))
        assert annualized_return(self.R, bars_per_year=len(self.R)) == pytest.approx(
            np.prod(1 + self.R) - 1)

    def test_sharpe_and_sortino_subtract_the_per_bar_risk_free_rate(self) -> None:
        rf = 0.05
        rf_bar = (1 + rf) ** (1 / BPY) - 1
        excess = self.R - rf_bar
        assert sharpe_ratio(self.R, risk_free=rf, bars_per_year=BPY) == pytest.approx(
            excess.mean() / excess.std(ddof=1) * np.sqrt(BPY))
        downside = excess[excess < 0]
        assert sortino_ratio(self.R, risk_free=rf, bars_per_year=BPY) == pytest.approx(
            excess.mean() / downside.std(ddof=1) * np.sqrt(BPY))

    def test_flat_series_and_a_single_loss_give_zero_not_a_division(self) -> None:
        assert sharpe_ratio(np.full(10, 0.01), bars_per_year=BPY) == 0.0
        assert sortino_ratio(np.array([0.01, 0.02, -0.01, 0.03]), bars_per_year=BPY) == 0.0  # 1 neerwaartse
        assert sortino_ratio(np.array([-0.01, -0.01, 0.02]), bars_per_year=BPY) == 0.0       # nul spreiding

    def test_the_drawdown_names_its_peak_and_trough_and_calmar_needs_one(self) -> None:
        eq = np.array([100.0, 120.0, 90.0, 95.0, 130.0])
        assert max_drawdown(eq) == (pytest.approx(0.25), 1, 2)
        assert calmar_ratio(np.array([100.0, 101.0, 102.0, 103.0]), bars_per_year=BPY) == 0.0  # geen drawdown
        assert calmar_ratio(eq, bars_per_year=BPY) > 0


# --------------------------------------------------------------------------- #
# PBO, SPA, drawdown shape
# --------------------------------------------------------------------------- #
class TestStatisticalGates:
    def test_pbo_refuses_an_odd_subset_count_and_too_few_periods(self) -> None:
        returns = np.random.default_rng(0).normal(size=(40, 5))
        with pytest.raises(ValueError, match="even and >= 4"):
            compute_pbo(returns, n_subsets=5)
        with pytest.raises(ValueError, match="Too few time periods"):
            compute_pbo(returns[:6], n_subsets=8)

    def test_the_block_size_estimate_has_a_floor_for_short_or_flat_series(self) -> None:
        assert _autocorr_block_size(np.arange(10.0)) == 5
        assert _autocorr_block_size(np.zeros(100)) == 5

    def test_the_block_size_follows_the_autocorrelation_decay(self) -> None:
        rng = np.random.default_rng(0)
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.9 * x[t - 1] + rng.normal()
        assert 5 < _autocorr_block_size(x) <= 60
        trend = np.cumsum(np.ones(200)) + rng.normal(0, 1e-3, 200)  # nooit onder de band
        assert _autocorr_block_size(trend, max_lag=10) == 10 or _autocorr_block_size(trend, max_lag=10) == 5

    def test_spa_calibrates_its_own_block_size_when_none_is_given(self, caplog: pytest.LogCaptureFixture) -> None:
        rng = np.random.default_rng(1)
        bench = rng.normal(0, 0.01, 200)
        strategies = rng.normal(0, 0.01, (200, 3))
        with caplog.at_level(logging.INFO):
            out = spa_test(strategy_returns_matrix=strategies, benchmark_returns=bench, n_bootstrap=50)
        assert "auto-calibrated block_size" in caplog.text
        assert 0.0 <= out["p_value_consistent"] <= 1.0

    def test_the_drawdown_reference_validates_and_orders_its_quantiles(self) -> None:
        with pytest.raises(ValueError, match="years"):
            dd_shape_reference(1.0, 0.0)
        with pytest.raises(ValueError, match="ann_vol"):
            dd_shape_reference(1.0, 2.0, ann_vol=0.0)
        with pytest.raises(ValueError, match="q must"):
            max_dd_over_vol_quantile(1.0, 2.0, q=0.4)
        lo = max_dd_over_vol_quantile(1.0, 2.0, q=0.6)
        hi = max_dd_over_vol_quantile(1.0, 2.0, q=0.95)
        assert 0 < lo < hi
        assert expected_max_dd_over_vol(1.0, 2.0) > 0


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #
class TestNonOverlappingKernel:
    def _inputs(self, n: int = 12):
        rng = np.random.default_rng(3)
        pl = np.ascontiguousarray(rng.uniform(0.3, 0.8, n))
        ps = np.ascontiguousarray(rng.uniform(0.3, 0.8, n))
        rl = np.ascontiguousarray(rng.normal(0, 0.02, n))
        rs = np.ascontiguousarray(rng.normal(0, 0.02, n))
        t1 = np.minimum(np.arange(n) + 2, n - 1).astype(np.int32)
        sp = np.full(n, 0.001)
        th = np.full(n, 0.5)
        return ps, pl, rs, rl, t1.copy(), t1.copy(), sp, th, th.copy()

    def test_the_compiled_kernel_is_its_python_source(self) -> None:
        args = self._inputs()
        for compiled, source in zip(calc_non_overlapping_stats(*args),
                                    calc_non_overlapping_stats.py_func(*args), strict=True):
            np.testing.assert_array_equal(compiled, source)

    def test_trades_do_not_overlap_and_each_pays_half_the_spread(self) -> None:
        ps, pl, rs, rl, t1s, t1l, sp, th_l, th_s = self._inputs()
        rets, sides, idx = calc_non_overlapping_stats.py_func(ps, pl, rs, rl, t1s, t1l, sp, th_l, th_s)
        assert (np.diff(idx) >= 3).all()  # t1 = i + 2, dan pas bar i + 3
        for r, side, i in zip(rets, sides, idx, strict=True):
            raw = rl[i] if side == 2 else rs[i]
            assert r == pytest.approx(raw - 0.0005)

    def test_a_conflict_is_decided_by_the_larger_probability_over_threshold_ratio(self) -> None:
        one = lambda v: np.array([v], dtype=np.float64)  # noqa: E731
        t1 = np.array([0], dtype=np.int32)
        long_wins = calc_non_overlapping_stats.py_func(
            one(0.7), one(0.9), one(0.01), one(0.02), t1, t1, one(0.0), one(0.5), one(0.5))
        short_wins = calc_non_overlapping_stats.py_func(
            one(0.9), one(0.7), one(0.01), one(0.02), t1, t1, one(0.0), one(0.5), one(0.5))
        assert long_wins[1].tolist() == [2] and short_wins[1].tolist() == [0]

    def test_bars_below_both_thresholds_stay_flat(self) -> None:
        n = 5
        z = np.zeros(n)
        t1 = np.arange(n, dtype=np.int32)
        low = np.full(n, 0.1)
        rets, sides, idx = calc_non_overlapping_stats.py_func(low, low, z, z, t1, t1, z,
                                                              np.full(n, 0.5), np.full(n, 0.5))
        assert rets.size == sides.size == idx.size == 0

    def test_a_trade_with_a_stale_exit_index_advances_one_bar(self) -> None:
        n = 4
        z = np.zeros(n)
        probs = np.full(n, 0.9)
        t1 = np.zeros(n, dtype=np.int32)  # exit vóór de entry
        rets, _, idx = calc_non_overlapping_stats.py_func(z, probs, z, np.full(n, 0.01), t1, t1,
                                                          z, np.full(n, 0.5), np.full(n, 2.0))
        assert idx.tolist() == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #
class TestLedgerReadouts:
    TS = pd.Timestamp("2024-01-01", tz="UTC")

    def test_the_readouts_reflect_fills_marks_and_costs(self) -> None:
        ledger = Ledger(10_000.0)
        ledger.apply_fill(Fill("BTC", self.TS, 2.0, 100.0, 1.0, "taker", "o1"))
        assert ledger.initial_equity == 10_000.0
        assert ledger.fees_paid == 1.0 and ledger.funding_paid == 0.0 and ledger.realized_pnl == 0.0
        assert ledger.cash == pytest.approx(10_000.0 - 200.0 - 1.0)
        assert len(ledger.fills) == 1 and ledger.position("BTC").qty == 2.0
        assert ledger.position("ETH").qty == 0.0 and set(ledger.positions()) == {"BTC"}
        assert ledger.unrealized_pnl({"BTC": 110.0}) == pytest.approx(20.0)
        assert ledger.position_value({"BTC": 110.0}) == pytest.approx(220.0)
        assert ledger.equity({"BTC": 110.0}) == pytest.approx(10_000.0 - 1.0 + 20.0)

    def test_funding_is_booked_and_reported(self) -> None:
        ledger = Ledger(10_000.0)
        ledger.apply_fill(Fill("BTC", self.TS, 1.0, 100.0, 0.0, "maker", "o1"))
        ledger.apply_funding("BTC", rate=0.001, mark_price=100.0, ts=self.TS)
        assert ledger.funding_paid == pytest.approx(0.1)

    def test_valuing_without_a_mark_for_an_open_position_is_refused(self) -> None:
        ledger = Ledger(10_000.0)
        ledger.apply_fill(Fill("BTC", self.TS, 1.0, 100.0, 0.0, "maker", "o1"))
        with pytest.raises(DataContractError, match="markprijs"):
            ledger.equity({})


# --------------------------------------------------------------------------- #
# event-driven engine
# --------------------------------------------------------------------------- #
class _SlowRouter(OrderRouter):
    """Een router wiens orders pas twee bars na het besluit mogen vullen."""

    def build_orders(self, *args, **kwargs):
        return [dataclasses.replace(o, ts_earliest_fill=o.ts_earliest_fill + pd.Timedelta(days=1))
                for o in super().build_orders(*args, **kwargs)]


class TestEventDrivenEngine:
    def test_the_engine_exposes_its_ledger_and_policy_hash(self) -> None:
        engine = make_engine()
        assert engine.ledger.initial_equity > 0
        assert engine.risk_policy_hash and len(engine.risk_policy_hash) > 8

    def test_bars_without_a_view_do_nothing(self) -> None:
        replay = make_replay(n_bars=8)
        only_early = {ts: e for ts, e in replay.exposures.items() if ts == replay.index[0]}
        result = make_engine().run(replay.slices, only_early)
        assert result.n_orders > 0
        assert make_engine().run(replay.slices, {}).n_orders == 0

    def test_an_order_whose_fill_lies_two_bars_ahead_stays_pending_one_more_bar(self) -> None:
        replay = make_replay(n_bars=8)
        engine_base = make_engine()
        risk = RiskEngine(RISK)
        router = _SlowRouter(venue=venue_spec(), spread=spread_model(), impact=impact_params(),
                             risk_config_hash=risk.config_hash)
        engine = engine_mod.EventDrivenEngine(risk_engine=risk, router=router,
                                              initial_equity=engine_base.ledger.initial_equity)
        first = {replay.index[0]: replay.exposures[replay.index[0]]}
        result = engine.run(replay.slices, first)
        fill_ts = {r.order.ts_earliest_fill for r in result.reports}
        assert fill_ts == {replay.index[2]}  # bar 0 besluit, bar 2 vult, bar 1 houdt vast

    def test_iter_equity_pairs_each_bar_with_its_equity(self) -> None:
        replay = make_replay(n_bars=6)
        result = make_engine().run(replay.slices, replay.exposures)
        pairs = list(iter_equity(result))
        assert [ts for ts, _ in pairs] == list(result.equity_curve.index)
        assert [e for _, e in pairs] == list(result.equity_curve)

    def test_slices_skip_bars_where_no_symbol_has_a_price(self) -> None:
        replay = make_replay(n_bars=6)
        prices = replay.prices.copy()
        prices.iloc[2] = np.nan
        ones = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        slices = build_slices(prices, ones, ones, ones * 1e8, ones * 1e8, {})
        assert len(slices) == 5 and replay.index[2] not in [s.ts for s in slices]

    def test_slice_panels_must_share_the_price_axis_and_columns(self) -> None:
        prices = make_replay(n_bars=4).prices
        ones = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        with pytest.raises(DataContractError, match="tijdas"):
            build_slices(prices, ones.iloc[1:], ones, ones, ones, {})
        with pytest.raises(DataContractError, match="andere kolommen"):
            build_slices(prices, ones.iloc[:, :2], ones, ones, ones, {})

    def test_an_all_nan_exposure_row_gives_no_entry(self) -> None:
        frame = pd.DataFrame({"A": [np.nan, 0.5], "B": [np.nan, -0.5]},
                             index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
        out = exposures_from_frame(frame)
        assert list(out) == [frame.index[1]] and out[frame.index[1]] == {"A": 0.5, "B": -0.5}


# --------------------------------------------------------------------------- #
# barrier book
# --------------------------------------------------------------------------- #
def _market():
    m = synthetic_market(n=200)
    m.sigma_annual.loc[:, :] = 0.05
    m.sigma_daily.loc[:, :] = 0.05 / np.sqrt(365.0)
    m.adv_usd.loc[:, :] = 1e12
    m.funding.loc[:, :] = 0.0
    return m


def _cand(**over) -> pd.DataFrame:
    row = dict(symbol="BTCUSDT", entry_bar=100, exit_bar=104, side=1.0, fill_return=0.05,
               barrier=0.05, p=0.6, p_low=0.58, p_trade=0.52)
    row.update(over)
    return pd.DataFrame([row])


FIXED = SizingRule(mode="fixed", kelly_multiple=0.25, fixed_risk_fraction=0.01,
                   resize_band=0.25, cost_rt=0.0)


class TestBarrierBook:
    def test_with_an_impact_model_every_leg_pays_square_root_impact(self) -> None:
        m = _market()
        m.adv_usd.loc[:, :] = 1e6  # dun genoeg dat de impact telt
        corr = pd.DataFrame(0.0, index=m.grid, columns=list(m.symbols))
        with_impact = BookInputs(market=m, avg_corr=corr, cost_rate=0.0, impact=impact_params())
        res = run_barrier_book(_cand(), with_impact, RiskEngine(RISK), FIXED, equity0=100_000.0)
        assert res.total_impact > 0.0
        change = res.equity.iloc[-1] - 100_000.0
        assert change == pytest.approx(res.total_pnl - res.total_fees - res.total_funding
                                       - res.total_impact, rel=1e-9)

    def test_a_candidate_the_risk_layer_refuses_opens_nothing(self) -> None:
        m = _market()
        entry = float(m.ohlcv["BTCUSDT"]["close"].iloc[100])
        frame = m.ohlcv["BTCUSDT"].copy()
        frame.iloc[101:, frame.columns.get_loc("close")] = entry * 0.6
        m = type(m)(**{**m.__dict__, "ohlcv": {**m.ohlcv, "BTCUSDT": frame}})
        corr = pd.DataFrame(0.0, index=m.grid, columns=list(m.symbols))
        inputs = BookInputs(market=m, avg_corr=corr, cost_rate=0.0, impact=None)
        big = SizingRule(mode="fixed", kelly_multiple=0.25, fixed_risk_fraction=0.9,
                         resize_band=0.25, cost_rt=0.0)
        # Op bar 101 breekt de halt de BTC-positie; de ETH-kandidaat van diezelfde bar
        # krijgt van de gehalteerde risicolaag exposure nul en wordt niet geopend.
        both = pd.concat([_cand(exit_bar=150), _cand(symbol="ETHUSDT", entry_bar=101, exit_bar=150)],
                         ignore_index=True)
        res = run_barrier_book(both, inputs, RiskEngine(RISK), big, equity0=1e5)
        assert res.halted
        assert res.trades["symbol"].tolist() == ["BTCUSDT"]


# --------------------------------------------------------------------------- #
# phase 5 baseline
# --------------------------------------------------------------------------- #
def test_the_layer_summary_has_one_row_per_track_and_layer() -> None:
    idx = pd.date_range("2024-01-01", periods=3, tz="UTC")
    s = pd.Series([0.0, 0.01, -0.01], index=idx)

    def layer(name: str, track: str) -> LayerResult:
        return LayerResult(name, track, (1 + s).cumprod(), s, s, {"net_sharpe": 1.5},
                           {"fees": 0.002}, {"n_bars": 3})

    table = summarise([layer("L10", "a"), layer("L10", "b")])
    assert list(table["track"]) == ["a", "b"]
    assert table.loc[0, "net_sharpe"] == 1.5 and table.loc[0, "cost_fees"] == 0.002 and table.loc[0, "n_bars"] == 3


def test_a_router_for_the_slow_engine_test_is_built_like_the_regular_one() -> None:
    risk = RiskEngine(RISK)
    regular = make_router(risk)
    assert regular.risk_config_hash == risk.config_hash
