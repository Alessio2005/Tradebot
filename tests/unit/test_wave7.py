"""tests/unit/test_wave7.py — Wave 7 smoke tests.

Covers: portfolio construction (HRP, BL, ERC, MVO), TCA, risk extensions.
All tests use synthetic data — no external dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture()
def returns_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "BTCUSDT": rng.normal(0.001, 0.03, n),
        "ETHUSDT": rng.normal(0.001, 0.04, n),
        "SOLUSDT": rng.normal(0.002, 0.05, n),
    }, index=idx)


@pytest.fixture()
def prices_df(returns_df: pd.DataFrame) -> pd.DataFrame:
    p0 = pd.Series({"BTCUSDT": 30_000.0, "ETHUSDT": 2_000.0, "SOLUSDT": 100.0})
    return (1 + returns_df).cumprod() * p0


# ============================================================================
# HRP
# ============================================================================

class TestHRP:

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import hrp_weights

        w = hrp_weights(returns_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_weights_positive(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import hrp_weights

        w = hrp_weights(returns_df)
        assert (w >= 0).all()

    def test_single_asset(self) -> None:
        from tradebot.portfolio import hrp_weights

        df = pd.DataFrame({"BTC": np.random.default_rng(0).normal(0, 0.01, 50)})
        w = hrp_weights(df)
        assert w["BTC"] == pytest.approx(1.0)

    def test_hrp_optimizer(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import HRPOptimizer

        opt = HRPOptimizer()
        w = opt.optimize(returns_df)
        assert opt.weights is not None
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ============================================================================
# Black-Litterman
# ============================================================================

class TestBlackLitterman:

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import BLViews, black_litterman_weights

        views = BLViews(returns_df.columns.tolist())
        views.add_absolute_view("BTCUSDT", 0.10, 0.01)
        w = black_litterman_weights(returns_df, views=views)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_no_views_equal_weight(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import black_litterman_weights

        w = black_litterman_weights(returns_df, views=None)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_relative_view(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import BLViews, black_litterman_weights

        views = BLViews(returns_df.columns.tolist())
        views.add_relative_view("BTCUSDT", "SOLUSDT", 0.05, 0.005)
        w = black_litterman_weights(returns_df, views=views)
        assert (w >= 0).all()


# ============================================================================
# ERC
# ============================================================================

class TestERC:

    def test_weights_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import erc_weights

        w = erc_weights(returns_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_positive(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import erc_weights

        w = erc_weights(returns_df)
        assert (w >= 0).all()


# ============================================================================
# MVO / MinVar
# ============================================================================

class TestMVO:

    def test_minvar_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import min_variance_weights

        w = min_variance_weights(returns_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_mvo_sum_to_one(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import mvo_weights

        w = mvo_weights(returns_df)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# Optimizer (unified API)
# ============================================================================

def _sovereign_constraints():
    """Constraints uit de SOEVEREINE policy — de enige toegestane herkomst.

    Sinds Phase 5 kiest L8 geen risicodrempels meer; `max_weight` en
    `max_leverage` komen uit `conf/risk/default.yaml` via
    `PortfolioConstraints.from_risk_config()`. Zie
    `reports/phase5_sovereign_wiring_audit.md` D1/D2.
    """
    from pathlib import Path

    from tradebot.portfolio import PortfolioConstraints
    from tradebot.schemas.config import RiskConfig, load_config

    root = Path(__file__).resolve().parents[2]
    return PortfolioConstraints.from_risk_config(
        load_config(root / "conf/risk/default.yaml", RiskConfig))


class TestOptimizer:

    @pytest.mark.parametrize("method", ["hrp", "erc", "minvar", "mvo"])
    def test_methods(self, returns_df: pd.DataFrame, method: str) -> None:
        from tradebot.portfolio import optimize

        # PHASE 5: `constraints` is verplicht. De oude default gaf L8 zijn
        # eigen concentratie- en leveragelimiet (wiring audit D1/D2).
        w = optimize(returns_df, constraints=_sovereign_constraints(),
                     method=method)  # type: ignore[arg-type]
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= 0).all()

    def test_constraint_applied(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import PortfolioConstraints, optimize

        cfg = _sovereign_constraints()
        w = optimize(returns_df, method="hrp", constraints=cfg)
        assert w.max() <= cfg.max_weight + 1e-6


# ============================================================================
# Constraints
# ============================================================================

class TestConstraints:

    def test_apply_max_weight(self, returns_df: pd.DataFrame) -> None:
        from tradebot.portfolio import PortfolioConstraints, apply_constraints

        raw = pd.Series({"BTCUSDT": 0.9, "ETHUSDT": 0.05, "SOLUSDT": 0.05})
        cfg = PortfolioConstraints(max_weight=0.5, max_leverage=1.0)
        w = apply_constraints(raw, cfg)
        assert w.max() <= 0.5 + 1e-6
        assert w.sum() == pytest.approx(1.0, abs=1e-6)

    def test_compute_turnover(self) -> None:
        from tradebot.portfolio import compute_turnover

        old = pd.Series({"A": 0.5, "B": 0.5})
        new = pd.Series({"A": 0.3, "B": 0.7})
        to = compute_turnover(new, old)
        assert to == pytest.approx(0.2, abs=1e-9)


# ============================================================================
# Rebalance
# ============================================================================

class TestRebalance:

    def test_rebalance_on_large_drift(self) -> None:
        from tradebot.portfolio import RebalanceConfig, should_rebalance

        cur = pd.Series({"A": 0.6, "B": 0.4})
        tgt = pd.Series({"A": 0.3, "B": 0.7})
        cfg = RebalanceConfig(band_width=0.05)
        dec = should_rebalance(cur, tgt, cfg)
        assert dec.rebalance is True

    def test_no_rebalance_on_small_drift(self) -> None:
        from tradebot.portfolio import RebalanceConfig, should_rebalance

        cur = pd.Series({"A": 0.50, "B": 0.50})
        tgt = pd.Series({"A": 0.502, "B": 0.498})
        cfg = RebalanceConfig(band_width=0.05)
        dec = should_rebalance(cur, tgt, cfg)
        assert dec.rebalance is False


# ============================================================================
# TCA — Pre-trade
# ============================================================================

class TestPreTrade:

    def test_cost_estimate_positive(self) -> None:
        from tradebot.tca import estimate_pre_trade_cost

        est = estimate_pre_trade_cost(
            notional=100_000.0,
            volatility=0.03,
            adv=10_000_000.0,
            bid_ask_spread=0.0002,
        )
        assert est.spread_cost >= 0.0
        assert est.impact_cost >= 0.0
        assert est.total_cost == pytest.approx(est.spread_cost + est.impact_cost)
        assert est.total_bps > 0.0

    def test_small_order_negligible_impact(self) -> None:
        from tradebot.tca import estimate_pre_trade_cost

        est = estimate_pre_trade_cost(
            notional=1_000.0,
            volatility=0.03,
            adv=1_000_000_000.0,  # huge ADV
        )
        assert est.impact_cost < 0.01  # near-zero impact


# ============================================================================
# TCA — IS decomposition
# ============================================================================

class TestISDecomposition:

    def test_decomposition_identity(self) -> None:
        from tradebot.tca import decompose_implementation_shortfall

        d = decompose_implementation_shortfall(
            symbol="BTCUSDT",
            decision_price=30_000.0,
            arrival_price=30_010.0,
            execution_price=30_020.0,
            signed_qty=1.0,
        )
        expected_total = d.delay_component + d.market_impact + d.timing_component
        assert d.total_is == pytest.approx(expected_total, abs=1e-9)

    def test_buy_order_negative_is(self) -> None:
        from tradebot.tca import decompose_implementation_shortfall

        d = decompose_implementation_shortfall(
            symbol="BTCUSDT",
            decision_price=30_000.0,
            arrival_price=30_000.0,
            execution_price=30_010.0,
            signed_qty=1.0,
        )
        assert d.market_impact < 0  # executed higher than arrival = cost


# ============================================================================
# TCA — Arrival price benchmark
# ============================================================================

class TestArrivalPrice:

    def test_arrival_price_benchmark(self) -> None:
        from tradebot.tca import compute_arrival_price_benchmark

        idx = pd.date_range("2022-01-01", periods=50, freq="1h", tz="UTC")
        prices = pd.Series(30_000.0 + np.arange(50), index=idx)

        result = compute_arrival_price_benchmark(
            symbol="BTCUSDT",
            decision_time=idx[25],
            execution_price=30_025.5,
            signed_qty=1.0,
            price_series=prices,
        )
        assert result.arrival_price == pytest.approx(30_025.0)
        assert isinstance(result.slippage_bps, float)


# ============================================================================
# TCA — Post-trade + Report
# ============================================================================

class TestPostTradeReport:

    def test_post_trade_record(self) -> None:
        from tradebot.tca import analyse_execution

        record = analyse_execution(
            symbol="BTCUSDT",
            execution_time=pd.Timestamp("2022-06-01", tz="UTC"),
            signed_qty=0.5,
            execution_price=30_010.0,
            decision_price=30_000.0,
            arrival_price=30_005.0,
            volatility=0.03,
            adv=500_000_000.0,
        )
        assert record.symbol == "BTCUSDT"
        assert isinstance(record.cost_vs_expected_bps, float)

    def test_tca_report_summary(self) -> None:
        from tradebot.tca import analyse_execution, build_tca_report

        records = [
            analyse_execution(
                symbol="BTCUSDT",
                execution_time=pd.Timestamp(f"2022-{m:02d}-01", tz="UTC"),
                signed_qty=float(m) * 0.1,
                execution_price=30_000.0 + m * 10,
                decision_price=30_000.0,
                arrival_price=30_000.0 + m * 5,
                volatility=0.03,
                adv=500_000_000.0,
            )
            for m in range(1, 6)
        ]
        report = build_tca_report(records)
        summary = report.summary()
        assert "mean_is_bps" in summary.index
        assert summary["n_trades"] == 5


# ============================================================================
# Risk — Factor risk
# ============================================================================

class TestFactorRisk:

    def test_factor_risk_decomposition(self, returns_df: pd.DataFrame) -> None:
        from tradebot.risk import compute_factor_risk

        weights = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2})
        result = compute_factor_risk(returns_df, weights)

        assert "total_vol_ann" in result
        assert result["total_vol_ann"] > 0.0
        assert result["diversification"] >= 0.0

    def test_factor_model_exposures(self, returns_df: pd.DataFrame) -> None:
        from tradebot.risk import FactorRiskModel

        model = FactorRiskModel()
        model.fit(returns_df)
        exp_df = model.exposures()
        assert "market_beta" in exp_df.columns
        assert len(exp_df) == 3


# ============================================================================
# Risk — Stress tests
# ============================================================================

class TestStressTests:

    def test_run_all_returns_dict(self, prices_df: pd.DataFrame) -> None:
        from tradebot.risk import StressTestSuite

        suite = StressTestSuite(prices_df)
        weights = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2})
        results = suite.run_all(weights)

        assert "covid_crash" in results
        assert "synthetic_3sigma" in results

    def test_monte_carlo_deterministic(self, prices_df: pd.DataFrame) -> None:
        from tradebot.risk import StressTestSuite

        suite = StressTestSuite(prices_df)
        weights = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2})
        r1 = suite.run_monte_carlo(weights, n_paths=100, seed=42)
        r2 = suite.run_monte_carlo(weights, n_paths=100, seed=42)
        assert r1.portfolio_return == pytest.approx(r2.portfolio_return, abs=1e-9)

    def test_marginal_var(self, prices_df: pd.DataFrame) -> None:
        from tradebot.risk import StressTestSuite

        suite = StressTestSuite(prices_df)
        weights = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": 0.3, "SOLUSDT": 0.2})
        mvar = suite.marginal_var(weights)
        assert len(mvar) == 3


# ============================================================================
# Risk — Liquidity risk
# ============================================================================

class TestLiquidityRisk:

    def test_assessment_values(self) -> None:
        from tradebot.risk import assess_liquidity_risk

        result = assess_liquidity_risk(
            symbol="BTCUSDT",
            notional=1_000_000.0,
            adv=500_000_000.0,
            bid_ask_spread_bps=3.0,
            var_99_bps=200.0,
        )
        assert 0.0 <= result.liquidity_score <= 1.0
        assert result.liquidation_days > 0.0
        assert result.stressed_lvar_bps > result.normal_lvar_bps

    def test_liquidity_adjusted_var_portfolio(self) -> None:
        from tradebot.risk import liquidity_adjusted_var

        weights = pd.Series({"BTC": 0.6, "ETH": 0.4})
        positions = pd.Series({"BTC": 600_000.0, "ETH": 400_000.0})
        adv_map   = pd.Series({"BTC": 5e8, "ETH": 3e8})
        spread    = pd.Series({"BTC": 3.0, "ETH": 5.0})
        var_map   = pd.Series({"BTC": 200.0, "ETH": 300.0})

        df = liquidity_adjusted_var(weights, positions, adv_map, spread, var_map)
        assert not df.empty
        assert "lvar_bps" in df.columns
