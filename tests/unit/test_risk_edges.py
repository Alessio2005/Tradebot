# tests/unit/test_risk_edges.py
"""De risicopaden die de hoofdtests niet raken.

Elk onderdeel hieronder heeft een eigen hoofdtest. Hier staan de randen: een
te korte reeks, een leeg boek, een asset die niet in de koersdata zit, een
toestand die de code zelf nooit zet, en het water-filling-randgeval dat alleen
op de numerieke tolerantie bestaat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.risk import beta_hedge
from tradebot.risk.binding_audit import StepEffect
from tradebot.risk.daily_loss_governor import PropfirmGovernor, PropfirmLimits
from tradebot.risk.drawdown import (
    BreakerState,
    DrawdownBreaker,
    DrawdownConfig,
    compute_current_drawdown,
)
from tradebot.risk.engine import RiskEngine, RiskState
from tradebot.risk.factor_risk import FactorRiskModel, _compute_loadings
from tradebot.risk.hmm_regime import HMMRegimeDetector, Regime
from tradebot.risk.kelly import meta_label_kelly
from tradebot.risk.kill_switches import HaltStore
from tradebot.risk.limits import water_filling_limit
from tradebot.risk.liquidity_risk import liquidity_adjusted_var
from tradebot.risk.position_limits import PositionLimits, check_position_limits
from tradebot.risk.stress_test import ScenarioOutcome, StressTestSuite
from tradebot.risk.var import (
    best_var,
    cornish_fisher_var,
    evt_gpd_var,
    historical_cvar,
    historical_var,
    rolling_cvar,
    rolling_var,
    rolling_var_causal,
)
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(17)


@pytest.fixture(autouse=True)
def _fresh_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elke test begint met dezelfde, verse generator.

    Een gedeelde generator op moduleniveau maakte de data afhankelijk van de
    testvolgorde: onder pytest-randomly kreeg `_prices()` per run andere
    koersen, en de stresstest hieronder faalde of slaagde daarop.
    """
    monkeypatch.setattr(f"{__name__}.RNG", np.random.default_rng(17))


# --------------------------------------------------------------------------- #
# Beta-hedge
# --------------------------------------------------------------------------- #
class TestBetaHedge:
    def test_too_few_observations_give_no_beta(self) -> None:
        y = np.array([0.01, np.nan, 0.02])
        assert np.isnan(beta_hedge._huber_beta(y, y))

    def test_a_clean_linear_relation_recovers_its_slope(self) -> None:
        x = RNG.normal(0.0, 0.02, 200)
        y = 1.5 * x + RNG.normal(0.0, 0.002, 200)
        y[::25] += 0.2  # uitschieters: Huber weegt ze af
        assert beta_hedge._huber_beta(y, x) == pytest.approx(1.5, abs=0.05)

    def test_a_singular_fit_is_refused_and_not_replaced_by_ols(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _singular(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(beta_hedge.np.linalg, "lstsq", _singular)
        x = RNG.normal(0.0, 0.02, 50)
        with pytest.raises(DataContractError, match="singulier"):
            beta_hedge._huber_beta(x, x)

    def test_rolling_betas_are_causal_and_btc_is_one(self) -> None:
        index = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")
        btc = RNG.normal(0.0, 0.01, 60)
        returns = pd.DataFrame({"BTC": btc, "ALT": 2.0 * btc}, index=index)
        betas = beta_hedge.compute_rolling_betas(
            returns, btc_col="BTC", window_days=1, bar_hours=1)
        assert (betas["BTC"] == 1.0).all()
        assert betas["ALT"].iloc[:24].isna().all()
        assert betas["ALT"].iloc[24:].to_numpy() == pytest.approx(2.0, abs=1e-6)

    def test_without_a_btc_column_every_beta_is_one(self) -> None:
        returns = pd.DataFrame({"ALT": [0.01, -0.02]})
        betas = beta_hedge.compute_rolling_betas(returns, btc_col="BTC")
        assert (betas == 1.0).all().all()

    def test_the_hedge_only_acts_outside_the_band(self) -> None:
        betas = {"ALT": 1.2, "NOISE": float("nan")}
        assert beta_hedge.compute_portfolio_beta(
            {"ALT": 0.5, "NOISE": 1.0, "UNKNOWN": 0.1}, betas) == pytest.approx(0.6)
        assert beta_hedge.compute_btc_hedge_size({"ALT": 0.05}, betas) == 0.0
        assert beta_hedge.compute_btc_hedge_size({"ALT": 0.5}, betas) == pytest.approx(-0.6)


# --------------------------------------------------------------------------- #
# Kleine randgevallen
# --------------------------------------------------------------------------- #
def test_a_step_on_a_flat_book_shrinks_nothing() -> None:
    step = StepEffect(name="gross_cap", changed=False, recorded=False,
                      gross_before=0.0, gross_after=0.0)
    assert step.shrink == 0.0


class TestPropfirmGovernor:
    def test_failure_is_readable_and_a_day_needs_positive_equity(self) -> None:
        gov = PropfirmGovernor(PropfirmLimits(initial_balance=100_000.0))
        assert gov.failed is False
        with pytest.raises(ValueError, match="equity must be > 0"):
            gov.start_day(0.0)


class TestDrawdown:
    def test_the_resume_level_must_sit_below_the_trigger(self) -> None:
        with pytest.raises(ValueError, match="resume_threshold"):
            DrawdownConfig(trigger_threshold=0.1, resume_threshold=0.2)

    def test_degenerate_equity_curves_have_no_drawdown(self) -> None:
        assert compute_current_drawdown(np.array([])) == 0.0
        assert compute_current_drawdown(np.array([0.0, 0.0])) == 0.0

    def test_trips_reset_and_an_unknown_state_counts_as_open(self) -> None:
        breaker = DrawdownBreaker(DrawdownConfig(
            trigger_threshold=0.1, resume_threshold=0.05, scale_factor=0.0))
        assert breaker.update(np.array([100.0, 85.0])) == 1.0
        assert breaker.trips == 1
        assert breaker.update(np.array([100.0, 85.0])) == 0.0
        breaker.reset()
        assert breaker.state is BreakerState.OPEN
        # RESUME bestaat in de enum maar wordt door `update` nooit gezet; een
        # breaker die er toch in staat, gedraagt zich als OPEN.
        breaker._state = BreakerState.RESUME
        assert breaker.update(np.array([100.0, 50.0])) == 1.0


class TestRiskEngineHaltStore:
    @staticmethod
    def _config() -> RiskConfig:
        return load_config(ROOT / "conf/risk/default.yaml", RiskConfig)

    def test_hydrate_without_a_store_or_a_record_changes_nothing(
        self, tmp_path,
    ) -> None:
        state = RiskState(equity=100.0, high_water_mark=100.0,
                          day_start_equity=100.0)
        bare = RiskEngine(self._config())
        assert bare.halt_store is None
        assert bare.hydrate(state) is state
        store = HaltStore(tmp_path / "halt.json")
        with_store = RiskEngine(self._config(), halt_store=store)
        assert with_store.halt_store is store
        assert with_store.hydrate(state) is state


class TestFactorRisk:
    def test_a_short_history_falls_back_to_unit_beta(self) -> None:
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        exposure = _compute_loadings(
            pd.Series(RNG.normal(0, 0.01, 10), index=index, name="BTC"),
            pd.DataFrame({"market": RNG.normal(0, 0.01, 10)}, index=index))
        assert exposure.market_beta == 1.0
        assert exposure.r_squared == 0.0

    def test_an_explicit_market_series_and_an_unknown_asset(self) -> None:
        index = pd.date_range("2024-01-01", periods=150, freq="D")
        returns = pd.DataFrame({"A": RNG.normal(0, 0.02, 150),
                                "B": RNG.normal(0, 0.02, 150)}, index=index)
        model = FactorRiskModel(factor_window=60)
        model.fit(returns, btc_returns=returns["A"].iloc[10:])
        risk = model.portfolio_factor_risk(pd.Series({"A": 0.5, "Z": 0.5}))
        assert risk["total_vol_ann"] > 0.0
        assert set(model.exposures().index) == {"A", "B"}


def test_meta_label_kelly_can_scale_below_the_threshold() -> None:
    assert meta_label_kelly(0.2, 0.25, meta_threshold=0.5,
                            scale_below_threshold=True) == pytest.approx(0.1)
    assert meta_label_kelly(0.2, 0.25, meta_threshold=0.5) == 0.0


class TestWaterFilling:
    def test_no_consistent_fixed_point_is_refused(self) -> None:
        with pytest.raises(ConfigContractError, match="geen consistent vast punt"):
            water_filling_limit([1.0, 1.0, 1.0], 0.2, key="cap")

    def test_the_tolerance_edge_also_ends_in_a_refusal(self) -> None:
        """`k * cap` binnen `_TOL` van 1 en een nul in het boek: de lus breekt
        af in plaats van door te rekenen op een noemer van ~0."""
        magnitudes = [0.0, 541461.2202490917, 299711.8905373848,
                      422687221.1976584]
        with pytest.raises(ConfigContractError):
            water_filling_limit(magnitudes, 0.33333333333316667, key="cap")


def test_liquidity_var_skips_flat_positions() -> None:
    weights = pd.Series({"BTC": 0.6, "ETH": 0.4})
    frame = liquidity_adjusted_var(
        weights, positions_usd=pd.Series({"BTC": 1e6, "ETH": 0.0}),
        adv_map=pd.Series({"BTC": 5e8}), spread_map=pd.Series({"BTC": 2.0}),
        var_map=pd.Series({"BTC": 300.0}))
    assert "ETH" not in set(frame.get("symbol", pd.Series(dtype=str)))


class TestPositionLimits:
    def test_an_empty_book_has_no_violations(self) -> None:
        assert check_position_limits({}, PositionLimits()) == []

    def test_the_portfolio_gross_cap(self) -> None:
        violations = check_position_limits(
            {"A": 1.5, "B": 1.5, "C": 1.0}, PositionLimits())
        assert any(v.rule == "portfolio_gross_cap" for v in violations)


def test_the_bear_regime_halves_the_long_cap() -> None:
    detector = HMMRegimeDetector()
    assert detector.get_long_cap(int(Regime.BEAR)) == 0.5
    assert detector.get_long_cap(int(Regime.BULL)) == 1.0


# --------------------------------------------------------------------------- #
# Stresstests
# --------------------------------------------------------------------------- #
def _prices() -> pd.DataFrame:
    index = pd.date_range("2019-06-01", "2023-06-01", freq="D", tz="UTC")
    steps = RNG.normal(0.0, 0.03, size=(len(index), 3))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)), index=index,
                        columns=["BTC", "ETH", "SOL"])


class TestStressSuite:
    def test_an_unrequested_book_has_no_reduction(self) -> None:
        outcome = ScenarioOutcome(name="empty", description="", requested={},
                                  decision=None)  # type: ignore[arg-type]
        assert outcome.gross_reduction == 0.0

    def test_an_unknown_scenario_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            StressTestSuite(_prices()).run_historical(pd.Series({"BTC": 1.0}), "dotcom")

    def test_weights_outside_the_price_universe_give_an_empty_result(self) -> None:
        suite = StressTestSuite(_prices())
        weights = pd.Series({"DOGE": 1.0})
        assert suite.run_historical(weights, "covid_crash").portfolio_return == 0.0
        assert suite.run_monte_carlo(weights, n_paths=10).portfolio_return == 0.0
        assert suite.marginal_var(weights).empty

    def test_tail_correlation_and_t_shocks_make_the_tail_heavier(self) -> None:
        suite = StressTestSuite(_prices())
        weights = pd.Series({"BTC": 0.5, "ETH": 0.3, "SOL": 0.2})
        gaussian = suite.run_monte_carlo(weights, n_paths=2000, seed=3)
        heavy = suite.run_monte_carlo(weights, n_paths=2000, seed=3,
                                      tail_correlation=0.95, t_copula_df=3.0)
        # `var_99` is hier een (negatief) rendement: zwaarder is lager.
        assert heavy.var_99 <= gaussian.var_99

    def test_the_tail_correlation_floor_lifts_negative_correlations_upward(
        self,
    ) -> None:
        """De vloer modelleert corr -> 1 in een crisis. Hij behield het TEKEN:
        een correlatie van -0.3 werd -0.95. Drie paren op -0.95 is geen geldige
        correlatiematrix (eigenwaarde 1 - 2 * 0.95 < 0), en een zwak negatief
        paar werd een bijna perfecte hedge -- de 'stress' maakte de staart
        LICHTER. Na de fix gaan alle drie naar +0.95 en wordt hij zwaarder."""
        corr = np.full((3, 3), -0.3)
        np.fill_diagonal(corr, 1.0)
        rng = np.random.default_rng(5)
        steps = rng.multivariate_normal(np.zeros(3), corr * 0.03 ** 2, size=600)
        index = pd.date_range("2021-01-01", periods=600, freq="D", tz="UTC")
        prices = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, axis=0)),
                              index=index, columns=["BTC", "ETH", "SOL"])
        sample = prices.pct_change().dropna().tail(252).corr().to_numpy()
        assert (sample[np.triu_indices(3, 1)] < 0).all()

        suite = StressTestSuite(prices)
        weights = pd.Series({"BTC": 1 / 3, "ETH": 1 / 3, "SOL": 1 / 3})
        plain = suite.run_monte_carlo(weights, n_paths=1000, seed=3)
        floored = suite.run_monte_carlo(weights, n_paths=1000, seed=3,
                                        tail_correlation=0.95)
        # Diversificatie verdwijnt: de 1%-staart wordt minstens 2x dieper.
        assert floored.var_99 < 2.0 * plain.var_99 < 0.0

    def test_a_floor_that_breaks_positive_definiteness_is_refused(self) -> None:
        """Een geldige correlatiematrix kan na het optillen naar 0.95 ongeldig
        worden (hier: kleinste eigenwaarde -4e-4). Dan wordt er niet getrokken.
        De returns zijn exact gewit, zodat de steekproefcorrelatie precies deze
        matrix is."""
        target = np.array([[1.0, 0.999, 0.954, 0.964],
                           [0.999, 1.0, 0.949, 0.974],
                           [0.954, 0.949, 1.0, 0.893],
                           [0.964, 0.974, 0.893, 1.0]])
        assert np.linalg.eigvalsh(target).min() > 0.0
        z = np.random.default_rng(9).normal(size=(252, 4))
        q, _ = np.linalg.qr(z - z.mean(axis=0))
        returns = q @ np.linalg.cholesky(target).T * 0.01
        cols = ["BTC", "ETH", "SOL", "XRP"]
        prices = pd.DataFrame(
            100.0 * np.vstack([np.ones(4), np.cumprod(1.0 + returns, axis=0)]),
            index=pd.date_range("2022-01-01", periods=253, freq="D", tz="UTC"),
            columns=cols)
        np.testing.assert_allclose(prices.pct_change().dropna().corr(), target,
                                   atol=1e-9)
        suite = StressTestSuite(prices)
        with pytest.raises(ConfigContractError, match="niet positief semidefiniet"):
            suite.run_monte_carlo(pd.Series(0.25, index=cols), n_paths=10,
                                  tail_correlation=0.95)


# --------------------------------------------------------------------------- #
# VaR
# --------------------------------------------------------------------------- #
class TestVar:
    def test_too_little_history_has_no_var(self) -> None:
        assert historical_var(np.array([0.01])) == 0.0
        assert historical_cvar(np.array([np.nan, 0.01])) == 0.0

    def test_cvar_is_never_smaller_than_var(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        assert historical_cvar(returns, confidence=0.99) >= historical_var(
            returns, confidence=0.99)

    def test_rolling_variants_leave_the_burn_in_empty(self) -> None:
        returns = RNG.normal(0.0, 0.01, 40)
        trailing = rolling_var(returns, window=10)
        causal = rolling_var_causal(returns, window=10)
        cvar = rolling_cvar(returns, window=10)
        assert np.isnan(trailing[:9]).all() and np.isfinite(trailing[9:]).all()
        assert np.isnan(causal[:10]).all() and np.isfinite(causal[10:]).all()
        # De causale variant op bar i is de gewone variant van bar i - 1.
        assert causal[10:] == pytest.approx(trailing[9:-1])
        assert np.isnan(cvar[:9]).all() and np.isfinite(cvar[9:]).all()

    def test_short_samples_fall_back_to_simpler_estimators(self) -> None:
        short = RNG.normal(0.0, 0.01, 20)
        assert cornish_fisher_var(short) == pytest.approx(
            float(np.quantile(short, 0.01)))
        assert cornish_fisher_var(np.array([])) == 0.0
        assert evt_gpd_var(short) == cornish_fisher_var(short)

    def test_best_var_dispatches_on_method_and_sample_size(self) -> None:
        sample = RNG.standard_t(4, 300) * 0.01
        assert best_var(sample, method="evt") == evt_gpd_var(sample, 0.99)
        assert best_var(sample, method="cornish_fisher") == cornish_fisher_var(sample, 0.99)
        assert best_var(sample, method="historical") == pytest.approx(
            float(np.quantile(sample, 0.01)))
        assert best_var(np.array([]), method="historical") == 0.0


class TestEvtEdges:
    def test_a_confidence_inside_the_threshold_falls_back(self) -> None:
        """`p_exceed >= 1`: de gevraagde kans ligt niet voorbij de drempel."""
        sample = RNG.standard_t(4, 300) * 0.01
        assert evt_gpd_var(sample, confidence=0.9, threshold_quantile=0.9) == (
            cornish_fisher_var(sample, 0.9))

    def test_a_zero_shape_uses_the_exponential_limit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from types import SimpleNamespace

        from tradebot.risk import var as var_module

        monkeypatch.setattr(var_module, "_genpareto", SimpleNamespace(
            fit=lambda losses, floc: (0.0, 0.0, 0.01)))
        sample = RNG.standard_t(4, 300) * 0.01
        u = float(np.quantile(sample, 0.90))
        expected = u - 0.01 * np.log(1.0 / ((1.0 - 0.99) / (1.0 - 0.90)))
        assert evt_gpd_var(sample, 0.99) == pytest.approx(expected)


def test_stress_var_never_scales_below_the_baseline() -> None:
    from tradebot.risk.var import stress_test_var

    assert stress_test_var(0.02, stress_multiplier=2.0, vol_ratio=1.5) == pytest.approx(0.06)
    assert stress_test_var(0.02, stress_multiplier=0.5, vol_ratio=0.5) == 0.02
