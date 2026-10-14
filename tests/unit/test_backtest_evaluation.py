# tests/unit/test_backtest_evaluation.py
"""`backtest/evaluation.py` (realisme-helpers) en de baseline-keten
(`baseline_runner`, `baseline_report`).

De helpers in `evaluation.py` zijn pure functies; elke test legt één regel vast.
De baseline-keten draait op een synthetisch close-panel (1.300 dagbars, zes
symbolen): ruim genoeg voor de zes walk-forward folds uit `conf/validation/`.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest import baseline_report as br
from tradebot.backtest import evaluation as ev
from tradebot.backtest.baseline_runner import (
    BASELINE_TRACKS,
    BaselineResult,
    CostModel,
    build_weight_tracks,
    returns_matrix,
    run_baseline_tracks,
)
from tradebot.features.base import CertifiedPanel
from tradebot.registry.preregistration import PreRegistration, StopCriterion
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# evaluation: git commits and the deflated-Sharpe penalty
# --------------------------------------------------------------------------- #
class TestGitCommitCount:
    def test_the_repository_has_commits(self) -> None:
        assert ev.count_git_commits() > 0
        assert ev.count_git_commits(ROOT) > 0

    @pytest.mark.parametrize("error", [subprocess.CalledProcessError(128, "git"), FileNotFoundError("git")])
    def test_a_failing_git_is_refused_not_replaced_by_a_made_up_count(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        def boom(*_a, **_k):
            raise error

        monkeypatch.setattr(ev.subprocess, "check_output", boom)
        with pytest.raises(DataContractError, match="GEEN fallback"):
            ev.count_git_commits(fallback=99)

    def test_an_unparsable_answer_is_refused_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ev.subprocess, "check_output", lambda *_a, **_k: b"not-a-number")
        with pytest.raises(DataContractError):
            ev.count_git_commits()


_SHARPES = (0.5, 1.0, 1.5, 0.2, 0.8)


class TestDeflatedSharpePenalty:
    def test_few_observations_a_flat_spread_or_a_single_trial_cost_nothing(self) -> None:
        assert ev.deflated_sharpe_penalty([1.0], n_optuna_trials=100) == 0.0
        assert ev.deflated_sharpe_penalty([1.0, 1.0, 1.0], n_optuna_trials=100) == 0.0
        assert ev.deflated_sharpe_penalty(_SHARPES, n_optuna_trials=1) == 0.0

    def test_the_penalty_grows_with_trials_and_is_capped(self) -> None:
        small = ev.deflated_sharpe_penalty(_SHARPES, n_optuna_trials=5)
        large = ev.deflated_sharpe_penalty(_SHARPES, n_optuna_trials=500)
        sigma = float(np.std(_SHARPES))
        assert 0 < small < large <= 5.0 * sigma
        capped = ev.deflated_sharpe_penalty(_SHARPES, n_optuna_trials=10**9, cap_factor=1.0)
        assert capped == pytest.approx(sigma)
        assert ev.deflated_sharpe_penalty is ev._deflated_sharpe_penalty_legacy

    def test_prior_experiments_count_towards_the_effective_trials(self) -> None:
        base = ev.deflated_sharpe_penalty(_SHARPES, n_optuna_trials=10)
        more = ev.deflated_sharpe_penalty(_SHARPES, n_optuna_trials=10, n_prior_experiments=400)
        assert more > base


class TestDeflatedSharpeProbability:
    def test_guards_return_zero(self) -> None:
        assert ev.deflated_sharpe_ratio_p(1.0, [1.0], 10, 100) == 0.0
        assert ev.deflated_sharpe_ratio_p(1.0, _SHARPES, 1, 100) == 0.0
        assert ev.deflated_sharpe_ratio_p(1.0, _SHARPES, 10, 1) == 0.0
        assert ev.deflated_sharpe_ratio_p(1.0, [0.7, 0.7], 10, 100) == 0.0

    def test_a_strong_sharpe_over_many_observations_is_significant_and_a_weak_one_is_not(self) -> None:
        strong = ev.deflated_sharpe_ratio_p(6.0, _SHARPES, 10, 5000)
        weak = ev.deflated_sharpe_ratio_p(0.1, _SHARPES, 10, 5000)
        assert strong > 0.95 and weak < 0.5


# --------------------------------------------------------------------------- #
# evaluation: gap-risk Kelly
# --------------------------------------------------------------------------- #
class TestGapRiskKelly:
    def test_the_empirical_cvar_handles_empty_and_non_finite_input(self) -> None:
        assert ev._empirical_cvar(np.array([])) == 0.0
        assert ev._empirical_cvar(np.array([np.nan, np.inf])) == 0.0
        r = np.concatenate([np.full(95, 0.01), np.full(5, -0.10)])
        assert ev._empirical_cvar(r, alpha=0.05) == pytest.approx(0.10)

    def test_gap_multiples_resolve_exact_prefix_and_default(self) -> None:
        assert ev.get_gap_multiple_for_asset("SOLUSDT") == 2.5
        assert ev.get_gap_multiple_for_asset("SOL-PERP") == 2.5
        assert ev.get_gap_multiple_for_asset("DOGEUSDT") == 1.5

    def test_a_cold_start_uses_the_conservative_cvar_prior(self) -> None:
        r = ev.gap_risk_kelly_size(0.01, 2.0, 100.0, 2.0, max_leverage=10.0)
        assert r.cvar_factor == 1.3 and not r.capped
        assert r.gap_risk_factor == pytest.approx(2.5)

    def test_the_symbol_overrides_the_default_multiple_but_an_explicit_one_wins(self) -> None:
        by_symbol = ev.gap_risk_kelly_size(0.01, 2.0, 100.0, 2.0, 10.0, symbol="SOLUSDT")
        explicit = ev.gap_risk_kelly_size(0.01, 2.0, 100.0, 2.0, 10.0, symbol="SOLUSDT", gap_atr_multiple=0.5)
        assert by_symbol.gap_risk_factor == pytest.approx(3.5)
        assert explicit.gap_risk_factor == pytest.approx(1.5)

    def test_enough_history_switches_to_the_empirical_tail_and_the_cap_binds(self) -> None:
        heavy = np.concatenate([np.full(80, 0.0), np.full(20, -0.2)])
        r = ev.gap_risk_kelly_size(0.5, 0.1, 100.0, 1.0, max_leverage=1.0, historical_returns=heavy,
                                   kelly_divisor=1.0)
        assert r.cvar_factor > 1.0
        mild = ev.gap_risk_kelly_size(0.5, 0.1, 100.0, 1.0, max_leverage=1.0,
                                      historical_returns=np.full(30, 0.01), kelly_divisor=1.0)
        assert mild.cvar_factor == 1.0 and mild.capped and mild.leverage == 1.0


# --------------------------------------------------------------------------- #
# evaluation: bootstrap
# --------------------------------------------------------------------------- #
def _path(n: int = 60, seed: int = 0, start: str = "2024-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.01, n),
                     index=pd.date_range(start, periods=n, freq="D", tz="UTC"))


class TestBootstrap:
    def test_the_index_matrix_has_the_requested_shape_and_stays_in_range(self) -> None:
        idx = ev.stationary_block_bootstrap_indices(20, 4.0, 7, np.random.default_rng(0))
        assert idx.shape == (7, 20) and idx.min() >= 0 and idx.max() < 20
        assert ev.stationary_block_bootstrap_indices(0, 4.0, 7, np.random.default_rng(0)).shape == (7, 0)

    def test_degenerate_path_sets_return_a_zero_summary(self) -> None:
        zero = {"sharpe_mean": 0.0, "sharpe_std": 0.0, "sharpe_lower_5": 0.0, "sharpe_upper_95": 0.0}
        empty = ev.block_bootstrap_path_sharpes([])
        assert {k: empty[k] for k in zero} == zero and empty["effective_dof"] == 0.0
        assert ev.block_bootstrap_path_sharpes([pd.Series(dtype=float)])["effective_dof"] == 0.0
        one_day = ev.block_bootstrap_path_sharpes([_path(1)])
        assert one_day["effective_dof"] == 1.0 and one_day["sharpe_mean"] == 0.0

    def test_a_flat_path_has_no_defined_sharpe_distribution(self) -> None:
        flat = pd.Series(0.0, index=pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"))
        out = ev.block_bootstrap_path_sharpes([flat], avg_block_size=5.0)
        assert out["sharpe_mean"] == 0.0 and out["effective_dof"] == pytest.approx(30 / 5.0)

    def test_a_real_path_gives_an_ordered_interval_and_is_seeded(self) -> None:
        paths = [_path(120, 0), _path(120, 1, "2024-02-01")]
        a = ev.block_bootstrap_path_sharpes(paths, n_boot=100)
        assert a["sharpe_lower_5"] <= a["sharpe_mean"] <= a["sharpe_upper_95"] and a["sharpe_std"] > 0
        assert a == ev.block_bootstrap_path_sharpes(paths, n_boot=100)


class TestMonteCarloSharpe:
    def test_a_sample_below_two_has_no_interval(self) -> None:
        assert ev.monte_carlo_sharpe_ci(np.array([0.01])) == (0.0, 0.0, 0.0)
        assert ev.monte_carlo_sharpe_ci(np.array([np.nan, 0.01])) == (0.0, 0.0, 0.0)

    def test_the_interval_brackets_the_point_estimate_for_both_block_choices(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        r = np.random.default_rng(0).normal(0.002, 0.01, 200)
        with caplog.at_level(logging.INFO, logger=ev.logger.name):
            auto = ev.monte_carlo_sharpe_ci(r, n_sims=200)
            fixed = ev.monte_carlo_sharpe_ci(r, n_sims=200, block_size=5)
        for lo, point, hi in (auto, fixed):
            assert lo < point < hi
        assert "auto=True" in caplog.text and "auto=False" in caplog.text

    def test_a_constant_return_series_has_a_zero_sharpe(self) -> None:
        assert ev.monte_carlo_sharpe_ci(np.full(40, 0.01), n_sims=20) == (0.0, 0.0, 0.0)

    def test_the_block_size_estimate_has_floor_and_ceiling(self) -> None:
        assert ev._estimate_block_size(np.arange(10.0)) == 3
        assert ev._estimate_block_size(np.zeros(100)) == 3
        rng = np.random.default_rng(0)
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.9 * x[t - 1] + rng.normal()
        assert 3 < ev._estimate_block_size(x) <= 60
        assert ev._estimate_block_size(np.cumsum(np.ones(200)), max_lag=8) >= 3


# --------------------------------------------------------------------------- #
# evaluation: timeouts and collisions
# --------------------------------------------------------------------------- #
class TestTimeoutFraction:
    def test_no_events_no_fraction(self) -> None:
        out = ev.compute_timeout_fraction(np.array([]), np.array([]), np.array([]), n_total_bars=10)
        assert out == {"n_events": 0.0, "n_timeouts": 0.0, "timeout_fraction": 0.0}

    def test_a_timeout_is_an_exit_on_the_expected_horizon_bar(self, caplog: pytest.LogCaptureFixture) -> None:
        events = np.array([0, 10, 20, 30])
        horizons = np.array([5, 5, 5, 5])
        t1 = events + 2 + horizons - 1          # exact op de horizon
        t1[3] = events[3] + 1                    # eerder verlaten
        with caplog.at_level(logging.INFO, logger=ev.logger.name):
            out = ev.compute_timeout_fraction(events, horizons, t1, n_total_bars=100, warn_threshold=0.9)
        assert out["n_timeouts"] == 3.0 and out["timeout_fraction"] == pytest.approx(0.75)
        assert "within 90% target" in caplog.text

    def test_a_high_rate_warns_and_the_max_horizon_variant_uses_the_tolerance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        events = np.array([0, 10, 20])
        t1 = np.array([5, 15, 22])
        with caplog.at_level(logging.WARNING, logger=ev.logger.name):
            out = ev.compute_timeout_fraction(events, np.zeros(3), t1, n_total_bars=100,
                                              max_horizon=6, tolerance=1)
        assert out["n_timeouts"] == 2.0  # 5 en 5 >= 6 - 1; 2 niet
        assert "Hanging-barrier alert" in caplog.text


_COLLISION_KW = dict(prob_long=0.7, prob_short=0.6, threshold_long=0.5, threshold_short=0.5,
                     horizon_long=10, horizon_short=6, ret_long=0.02, ret_short=0.01)


class TestCollisionResolution:
    def test_nothing_active_is_flat(self) -> None:
        r = ev.resolve_long_short_collision(long_active=False, short_active=False, **_COLLISION_KW)
        assert (r.side, r.net_weight, r.horizon) == ("FLAT", 0.0, 0)

    def test_a_single_side_takes_its_own_ratio_horizon_and_return(self) -> None:
        long = ev.resolve_long_short_collision(long_active=True, short_active=False, **_COLLISION_KW)
        short = ev.resolve_long_short_collision(long_active=False, short_active=True, **_COLLISION_KW)
        assert (long.side, long.weight_long, long.horizon, long.expected_return) == ("LONG", 1.4, 10, 0.02)
        assert (short.side, short.net_weight, short.horizon, short.expected_return) == ("SHORT", -1.2, 6, 0.01)

    def test_both_sides_net_by_ratio_and_the_horizon_follows_duration_neutrality(self) -> None:
        both = ev.resolve_long_short_collision(long_active=True, short_active=True, **_COLLISION_KW)
        assert both.side == "LONG" and both.net_weight == pytest.approx((1.4 - 1.2) / 2.6)
        assert both.horizon == 6
        assert both.expected_return == pytest.approx((1.4 * 0.02 - 1.2 * 0.01) / 2.6)
        assert ev.resolve_long_short_collision(long_active=True, short_active=True,
                                               duration_neutral=False, **_COLLISION_KW).horizon == 10
        short_heavy = {**_COLLISION_KW, "prob_long": 0.55, "prob_short": 0.9}
        assert ev.resolve_long_short_collision(long_active=True, short_active=True, **short_heavy).side == "SHORT"

    def test_equal_ratios_net_to_zero_and_zero_ratios_go_flat(self) -> None:
        even = {**_COLLISION_KW, "prob_long": 0.6, "prob_short": 0.6}
        assert ev.resolve_long_short_collision(long_active=True, short_active=True, **even).side == "NET"
        zero = {**_COLLISION_KW, "prob_long": 0.0, "prob_short": 0.0}
        assert ev.resolve_long_short_collision(long_active=True, short_active=True, **zero).side == "FLAT"


# --------------------------------------------------------------------------- #
# evaluation: bar-by-bar mark-to-market
# --------------------------------------------------------------------------- #
def _mtm_inputs(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    return dict(
        close=close, bar_index=index,
        trade_entry_idx=np.array([5, 20, 40]), trade_exit_idx=np.array([30, 50, 90]),
        trade_sides=np.array([1.0, -1.0, 1.0]), trade_leverage=np.array([1.0, 1.5, 2.0]),
    )


class TestBarByBarMtm:
    def test_no_trades_or_no_bars_give_the_flat_start_equity(self) -> None:
        kw = _mtm_inputs()
        r = ev.compute_bar_by_bar_mtm(**{**kw, "trade_entry_idx": np.array([], dtype=int),
                                         "trade_exit_idx": np.array([], dtype=int),
                                         "trade_sides": np.array([]), "trade_leverage": np.array([])})
        assert r.n_bars == 0 and r.max_drawdown == 0.0 and r.diagnostics == {"reason": 1.0}
        assert r.equity_curve.iloc[0] == 100_000.0

    def test_exposure_starts_the_bar_after_entry_and_ends_on_the_exit_bar(self) -> None:
        kw = _mtm_inputs()
        r = ev.compute_bar_by_bar_mtm(**kw)
        assert r.n_bars == 120 and r.equity_curve.name == "mtm_equity"
        assert r.bar_returns.iloc[:6].eq(0.0).all()          # eerste exposure op bar 6
        assert r.bar_returns.iloc[6] != 0.0
        assert r.diagnostics["max_exposure"] == pytest.approx(3.5 + 0.0, abs=2.0)
        assert r.max_drawdown <= 0.0 <= r.calmar or r.calmar <= 0.0

    def test_the_portfolio_cap_scales_or_drops_overlapping_trades(self) -> None:
        kw = _mtm_inputs()
        kw.update(trade_entry_idx=np.array([5, 6, 7, 8]), trade_exit_idx=np.array([40, 40, 40, 40]),
                  trade_sides=np.array([1.0, 1.0, -1.0, 1.0]), trade_leverage=np.array([1.0, 1.0, 1.0, 0.0]))
        capped = ev.compute_bar_by_bar_mtm(**kw, max_portfolio_leverage=1.5)
        assert capped.diagnostics["n_portfolio_capped"] == 2.0   # de tweede wordt geschaald, de derde valt weg
        assert capped.diagnostics["max_portfolio_leverage_used"] == 1.5
        assert capped.diagnostics["max_exposure"] <= 1.5 + 1e-9

    def test_a_trade_that_ends_before_it_starts_is_skipped(self) -> None:
        kw = _mtm_inputs()
        kw.update(trade_entry_idx=np.array([50]), trade_exit_idx=np.array([10]),
                  trade_sides=np.array([1.0]), trade_leverage=np.array([1.0]))
        r = ev.compute_bar_by_bar_mtm(**kw, max_portfolio_leverage=2.0)
        assert r.diagnostics["max_exposure"] == 0.0 and r.sharpe_annualised == 0.0

    def test_funding_and_rebalance_costs_reduce_the_return(self) -> None:
        kw = _mtm_inputs()
        base = ev.compute_bar_by_bar_mtm(**kw)
        funding = np.zeros(120)
        funding[::8] = 0.0005
        paid = ev.compute_bar_by_bar_mtm(**kw, funding_rates=funding, rebalance_cost_bps=5.0)
        assert paid.equity_curve.iloc[-1] < base.equity_curve.iloc[-1]
        assert paid.diagnostics["total_funding_cost"] > 0 and paid.diagnostics["total_rebalance_cost"] > 0

    def test_a_dense_funding_series_is_flagged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            ev.compute_bar_by_bar_mtm(**_mtm_inputs(), funding_rates=np.full(120, 0.0001))
        assert "DENSE" in caplog.text

    def test_an_account_without_capital_has_no_cagr(self) -> None:
        r = ev.compute_bar_by_bar_mtm(**_mtm_inputs(), account_size=0.0)
        assert r.diagnostics["cagr"] == 0.0 and r.calmar == 0.0 and (r.equity_curve == 0.0).all()

    def test_a_single_bar_has_no_cagr_and_a_losing_book_has_a_negative_calmar(self) -> None:
        one = ev.compute_bar_by_bar_mtm(
            close=np.array([100.0]), bar_index=pd.date_range("2024-01-01", periods=1, tz="UTC"),
            trade_entry_idx=np.array([0]), trade_exit_idx=np.array([0]),
            trade_sides=np.array([1.0]), trade_leverage=np.array([1.0]))
        assert one.diagnostics["cagr"] == 0.0 and one.calmar == 0.0
        kw = _mtm_inputs()
        kw["close"] = 100 * np.exp(-np.linspace(0, 1.0, 120)) * (1 + 0.01 * np.sin(np.arange(120)))
        kw.update(trade_entry_idx=np.array([0]), trade_exit_idx=np.array([119]),
                  trade_sides=np.array([1.0]), trade_leverage=np.array([1.0]))
        down = ev.compute_bar_by_bar_mtm(**kw)
        assert down.max_drawdown < 0 and down.calmar < 0


class TestInstitutionalReport:
    def test_the_report_combines_penalty_bootstrap_and_optional_blocks(self) -> None:
        paths = [_path(90, 0), _path(90, 1, "2024-02-01")]
        sharpes = [0.4, 0.9, 1.3, 0.2, 0.7]
        bare = ev.institutional_evaluation_report(
            path_returns=paths, fold_sharpes_for_dsr=sharpes, n_optuna_trials=50, n_prior_experiments=10)
        assert bare["deflated_sharpe"] == pytest.approx(bare["raw_mean_sharpe"] - bare["dsr_penalty"])
        assert "timeout_fraction" not in bare and "mtm_sharpe" not in bare
        full = ev.institutional_evaluation_report(
            path_returns=paths, fold_sharpes_for_dsr=sharpes, n_optuna_trials=50, n_prior_experiments=10,
            timeout_stats={"timeout_fraction": 0.2}, mtm=ev.compute_bar_by_bar_mtm(**_mtm_inputs()))
        assert full["timeout_fraction"] == 0.2
        assert {"mtm_sharpe", "mtm_calmar", "mtm_max_drawdown", "mtm_intra_drawdown_p95"} <= set(full)

    def test_no_fold_sharpes_means_a_zero_mean(self) -> None:
        out = ev.institutional_evaluation_report(
            path_returns=[], fold_sharpes_for_dsr=[], n_optuna_trials=10, n_prior_experiments=0)
        assert out["raw_mean_sharpe"] == 0.0 and out["dsr_penalty"] == 0.0


# --------------------------------------------------------------------------- #
# baseline chain
# --------------------------------------------------------------------------- #
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]


def _panel(n: int = 1300, seed: int = 11) -> CertifiedPanel:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC", name="asof_ts")
    values = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n))) for s in SYMBOLS},
        index=index, dtype="float64")
    return CertifiedPanel(values=values, data_hashes=(("crypto/ohlcv/BTCUSDT/1d", "0" * 32),))


@pytest.fixture(scope="module")
def configs() -> dict:
    return br.load_baseline_configs(ROOT)


@pytest.fixture(scope="module")
def chain(configs):
    panel = _panel()
    unit = build_cross_sectional_momentum(configs["alpha"])
    tracks, exposures = build_weight_tracks(
        panel, unit, lam=configs["vol"].ewma_lambda, vol_burn_in_bars=configs["vol"].burn_in_bars,
        annualisation_factor=configs["vol"].annualisation_factor, gross_target=1.0, git_sha="abc1234")
    cost = CostModel(taker_fee_bps=5.5, half_spread_bps=2.0, is_provisional=True)
    val = configs["val"]
    results = run_baseline_tracks(
        panel, tracks, cost=cost, train_bars=val.train_bars, test_bars=val.test_bars,
        embargo_bars=val.embargo_bars, min_splits=val.n_splits,
        warmup_bars=max(unit.burn_in_period, configs["vol"].burn_in_bars))
    return panel, unit, tracks, exposures, cost, results


class TestBaselineRunner:
    def test_the_four_registered_tracks_are_built_on_the_panel_axis(self, chain) -> None:
        panel, _unit, tracks, exposures, _cost, _results = chain
        assert set(tracks) == set(BASELINE_TRACKS)
        assert all(t.index.equals(panel.values.index) for t in tracks.values())
        assert list(exposures.columns) == SYMBOLS

    def test_the_cost_model_is_per_side_and_reports_its_provisional_status(self, chain) -> None:
        cost = chain[4]
        assert cost.per_side == pytest.approx(7.5e-4)
        assert cost.as_dict() == {"taker_fee_bps": 5.5, "half_spread_bps": 2.0,
                                  "per_side_decimal": pytest.approx(7.5e-4), "is_provisional": True}

    def test_results_cover_the_same_out_of_sample_bars_and_net_is_below_gross(self, chain) -> None:
        results = chain[5]
        lengths = {r.n_oos_bars for r in results.values()}
        assert len(lengths) == 1 and all(len(r.net_returns) == r.n_oos_bars for r in results.values())
        r = results["xs_momentum_risk_parity"]
        assert r.n_folds >= 6 and r.n_embargoed_bars == r.n_folds * 5
        assert r.net_returns.sum() <= r.gross_returns.sum()

    def test_the_result_reports_its_metrics(self, chain) -> None:
        m = chain[5]["long_only_equal_weight"].metrics(bars_per_year=365)
        assert set(m) == {"gross_sharpe", "net_sharpe", "net_total_return", "max_drawdown",
                          "calmar", "avg_daily_turnover", "n_oos_bars", "n_folds"}

    def test_an_empty_result_has_zero_total_return(self) -> None:
        idx = pd.DatetimeIndex([], tz="UTC")
        empty = BaselineResult("t", pd.Series(dtype=float, index=idx), pd.Series(dtype=float, index=idx),
                               pd.Series(dtype=float, index=idx), pd.DataFrame(index=idx), 0, 0, 0)
        assert empty.metrics(bars_per_year=365)["net_total_return"] == 0.0

    def test_walk_forward_refuses_a_too_short_series_too_few_folds_and_a_misaligned_track(self, chain) -> None:
        panel, unit, tracks, _e, cost, _r = chain
        kw = dict(cost=cost, train_bars=500, test_bars=100, embargo_bars=5, min_splits=6, warmup_bars=61)
        short = CertifiedPanel(values=panel.values.iloc[:550], data_hashes=panel.data_hashes)
        with pytest.raises(DataContractError, match="Te weinig bars"):
            run_baseline_tracks(short, {k: v.iloc[:550] for k, v in tracks.items()}, **kw)
        with pytest.raises(DataContractError, match="Minder walk-forward folds"):
            run_baseline_tracks(panel, tracks, **{**kw, "min_splits": 50})
        with pytest.raises(DataContractError, match="tijdas"):
            run_baseline_tracks(panel, {"x": tracks["long_only_equal_weight"].iloc[1:]}, **kw)
        with pytest.raises(DataContractError, match="Geen enkele track"):
            run_baseline_tracks(panel, {}, **kw)

    def test_a_fold_whose_embargo_swallows_the_test_window_adds_no_oos_bars(self, chain) -> None:
        panel, _u, tracks, _e, cost, _r = chain
        out = run_baseline_tracks(panel, tracks, cost=cost, train_bars=500, test_bars=100,
                                  embargo_bars=100, min_splits=1, warmup_bars=61)
        assert all(r.n_oos_bars == 0 for r in out.values())

    def test_the_returns_matrix_is_time_by_track_and_demands_equal_lengths(self, chain) -> None:
        results = chain[5]
        m = returns_matrix(results, ["long_only_equal_weight", "long_only_risk_parity"])
        assert m.shape == (results["long_only_equal_weight"].n_oos_bars, 2)
        with pytest.raises(DataContractError, match="Geen tracks"):
            returns_matrix(results, [])
        cut = BaselineResult(**{**results["long_only_equal_weight"].__dict__,
                                "net_returns": results["long_only_equal_weight"].net_returns.iloc[1:]})
        with pytest.raises(DataContractError, match="verschillend aantal"):
            returns_matrix({"a": cut, "b": results["long_only_risk_parity"]}, ["a", "b"])


# --------------------------------------------------------------------------- #
# baseline report
# --------------------------------------------------------------------------- #
def _prereg() -> PreRegistration:
    def crit(name: str, metric: str, op: str, thr: float, action: str) -> StopCriterion:
        return StopCriterion(name, metric, op, thr, action, rationale="test")

    return PreRegistration(
        wave="w", title="t", hypothesis="h", null_hypothesis="n", universe=("BTCUSDT",),
        granularity="1d", period_start="2021-01-01", period_end="2025-01-01",
        evaluation_start="2022-01-01", primary_metric="net_oos_sharpe", parameters={},
        planned_trials=4,
        stop_criteria=(
            crit("no_edge", "net_oos_sharpe", "<=", 0.0, "falsify"),
            crit("dsr", "dsr_probability", "<", 0.95, "archive"),
            crit("cost_drag", "cost_drag_fraction", ">", 0.5, "archive"),
            crit("promote", "n_binding_stop_criteria", "<=", 0.0, "promote"),
        ),
        data_hashes=(("crypto/ohlcv/BTCUSDT/1d", "0" * 32),),
    )


class TestBaselineReport:
    def test_the_per_bar_sharpe_needs_variance_and_observations(self) -> None:
        with pytest.raises(DataContractError, match="Te weinig"):
            br.per_bar_sharpe(np.array([0.01, np.nan]))
        with pytest.raises(DataContractError, match="Nul-variantie"):
            br.per_bar_sharpe(np.zeros(10))
        r = np.array([0.01, -0.01, 0.02, 0.0])
        assert br.per_bar_sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))

    def test_track_metrics_covers_every_track(self, chain) -> None:
        out = br.track_metrics(chain[5], bars_per_year=365)
        assert set(out) == set(BASELINE_TRACKS) and "net_sharpe" in out["long_only_equal_weight"]

    def test_the_dsr_gate_demands_an_honest_trial_count_and_reports_its_approximation(self, chain) -> None:
        result = chain[5]["long_only_equal_weight"]
        with pytest.raises(DataContractError, match="eerlijk aantal trials"):
            br.run_dsr_gate(result, n_trials=1, dsr_alpha=0.05, bars_per_year=365.0)
        out = br.run_dsr_gate(result, n_trials=20, dsr_alpha=0.05, bars_per_year=365.0)
        assert out["dsr_approximation"] == "normal"
        assert out["dsr_p_value"] == pytest.approx(1.0 - out["dsr_probability"])
        assert out["passes"] is (out["dsr_probability"] >= 0.95)

    def test_the_spa_gate_compares_candidates_with_the_benchmark(self, chain) -> None:
        results = chain[5]
        kw = dict(benchmark_track="long_only_equal_weight", n_bootstrap=50, block_length=5, alpha=0.05)
        out = br.run_spa_gate(results, candidate_tracks=["xs_momentum_risk_parity", "long_only_risk_parity"], **kw)
        assert out["best_candidate"] in out["candidate_tracks"] and 0.0 <= out["spa_p_value"] <= 1.0
        assert out["passes"] is (out["spa_p_value"] < 0.05)
        with pytest.raises(DataContractError, match="benchmark-track"):
            br.run_spa_gate(results, **{**kw, "benchmark_track": "nope"}, candidate_tracks=["x"])
        with pytest.raises(DataContractError, match="zonder kandidaten"):
            br.run_spa_gate(results, candidate_tracks=[], **kw)

    def test_stop_criteria_bind_clear_or_cannot_be_judged_and_promotion_follows_the_count(self) -> None:
        bad = br.evaluate_stop_criteria(_prereg(), {"net_oos_sharpe": -0.2, "dsr_probability": 0.4,
                                                   "cost_drag_fraction": float("nan")})
        status = {v.name: v.status for v in bad.verdicts}
        assert status == {"no_edge": "BINDING", "dsr": "BINDING", "cost_drag": "NOT_APPLICABLE",
                          "promote": "BINDING"}
        assert bad.n_binding == 2 and not bad.promoted
        good = br.evaluate_stop_criteria(_prereg(), {"net_oos_sharpe": 1.0, "dsr_probability": 0.99,
                                                    "cost_drag_fraction": 0.1})
        assert good.promoted and good.n_binding == 0 and good.verdicts[-1].status == "NOT_BINDING"
        record = good.as_dict()
        assert record["promoted_to_candidate"] is True and record["verdicts"][0]["metric"] == "net_oos_sharpe"

    def test_a_missing_measurement_is_not_applicable_not_a_pass(self) -> None:
        out = br.evaluate_stop_criteria(_prereg(), {})
        assert [v.status for v in out.verdicts[:3]] == ["NOT_APPLICABLE"] * 3 and out.promoted

    def test_a_registration_needs_exactly_one_promotion_criterion(self) -> None:
        prereg = _prereg()
        no_promote = PreRegistration(**{**prereg.__dict__, "stop_criteria": prereg.stop_criteria[:3]})
        with pytest.raises(DataContractError, match="precies een promotiecriterium"):
            br.evaluate_stop_criteria(no_promote, {})

    def test_non_finite_floats_become_none_for_strict_json(self) -> None:
        out = br.json_safe({"a": float("nan"), "b": [1.0, float("inf"), (2, float("-inf"))], "c": "x", "d": 3})
        assert out == {"a": None, "b": [1.0, None, [2, None]], "c": "x", "d": 3}

    def test_the_configs_come_from_the_repository_domains(self, configs) -> None:
        assert set(configs) == {"data", "alpha", "vol", "val", "exec", "bt"}

    def test_the_whole_wave_runs_end_to_end_on_a_stubbed_store(
        self, monkeypatch: pytest.MonkeyPatch, configs
    ) -> None:
        import tradebot.features.base as fb

        panel = _panel()
        seen: dict = {}

        def fake_loader(store, register, *, symbols, granularity, asset_class):
            seen.update(symbols=list(symbols), granularity=granularity, asset_class=asset_class)
            return panel

        monkeypatch.setattr(fb, "load_certified_close_panel", fake_loader)
        payload, gate = br.execute_baseline_wave(ROOT, _prereg(), m_trials=20, git_sha="abc1234")
        assert seen == {"symbols": list(configs["data"].symbols), "granularity": "1d", "asset_class": "crypto"}
        assert payload["primary_track"] == "xs_momentum_risk_parity"
        assert payload["gate"] == gate.as_dict() and payload["m_trials"] == 20
        assert set(payload["metrics"]) == set(BASELINE_TRACKS)
        assert set(payload["measured"]) == {"net_oos_sharpe", "dsr_probability", "spa_p_value",
                                            "cost_drag_fraction"}
        assert payload["data_hashes"] == dict(panel.data_hashes)
