# tests/unit/test_portfolio_edges.py
"""De portfolio-paden die de hoofdtests niet raken: solver-uitval, te weinig
rijen, de rebalance-baten en de constraint-volgorde.

Een SLSQP-solve laat zich laten falen met een onhaalbare cap: drie assets met
`max_weight=0.2` kunnen nooit tot 1 sommeren.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.portfolio import hrp as hrp_mod
from tradebot.portfolio.black_litterman import BLViews, black_litterman_weights
from tradebot.portfolio.constraints import PortfolioConstraints, apply_constraints
from tradebot.portfolio.hrp import HrpResearchGate
from tradebot.portfolio.markowitz import _cov_lw, min_variance_weights, mvo_weights
from tradebot.portfolio.optimizer import optimize
from tradebot.portfolio.rebalance import RebalanceConfig, should_rebalance
from tradebot.portfolio.risk_parity import (
    erc_weights,
    inverse_volatility_long_only,
    risk_contributions,
    sized_by_risk_parity,
)
from tradebot.utils.failfast import ConfigContractError, DataContractError

ASSETS = ["A", "B", "C"]


def _returns(n: int = 120, seed: int = 0, cols: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cols or ASSETS
    base = rng.normal(0.0004, 0.01, (n, 1))
    data = base + rng.normal(0.0, 0.01, (n, len(cols))) * np.linspace(1.0, 2.0, len(cols))
    return pd.DataFrame(data, columns=cols)


def _gappy(n: int = 6) -> pd.DataFrame:
    """Elke rij mist één waarde: na `dropna` blijft er niets over."""
    df = _returns(n)
    for i in range(n):
        df.iloc[i, i % 3] = np.nan
    return df


def _constraints(**kw) -> PortfolioConstraints:
    base = dict(max_weight=0.6, max_leverage=1.0)
    base.update(kw)
    return PortfolioConstraints(**base)


GATE = HrpResearchGate(
    reason="vergelijking met inverse volatility, geregistreerd onder de HRP-pre-registratie",
    preregistration_id="PR-test-1",
)


# --------------------------------------------------------------------------- #
# markowitz
# --------------------------------------------------------------------------- #
class TestMarkowitz:
    def test_gappy_returns_are_mean_filled_when_too_few_rows_survive(self) -> None:
        cov = _cov_lw(_gappy())
        assert cov.shape == (3, 3) and np.isfinite(cov).all()

    def test_min_variance_sums_to_one_and_respects_the_cap(self) -> None:
        w = min_variance_weights(_returns(), max_weight=0.5)
        assert w.sum() == pytest.approx(1.0) and (w <= 0.5 + 1e-9).all() and (w >= 0).all()

    def test_an_infeasible_cap_falls_back_to_equal_weight_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            w = min_variance_weights(_returns(), max_weight=0.2)
        assert "using equal-weight" in caplog.text
        assert w.tolist() == pytest.approx([1 / 3] * 3)

    def test_mvo_takes_explicit_expected_returns_and_a_turnover_penalty(self) -> None:
        rets = _returns()
        mu = pd.Series({"A": 0.01, "B": 0.0, "C": -0.01, "ZZZ": 5.0})  # ZZZ valt buiten het universum
        free = mvo_weights(rets, expected_returns=mu, risk_aversion=1.0)
        sticky = mvo_weights(rets, expected_returns=mu, risk_aversion=1.0,
                             prev_weights=np.array([0.0, 0.0, 1.0]), turnover_penalty=50.0)
        assert free["A"] > free["C"]
        assert sticky["C"] > free["C"]  # de straf trekt naar de vorige gewichten

    def test_the_turnover_penalty_needs_both_a_previous_book_and_a_positive_lambda(self) -> None:
        rets = _returns()
        plain = mvo_weights(rets)
        with_prev_only = mvo_weights(rets, prev_weights=np.array([0.0, 0.0, 1.0]), turnover_penalty=0.0)
        pd.testing.assert_series_equal(plain, with_prev_only)

    def test_a_failed_mvo_solve_falls_back_to_min_variance_and_then_equal_weight(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            w = mvo_weights(_returns(), max_weight=0.2)
        assert "min-var fallback" in caplog.text
        assert w.tolist() == pytest.approx([1 / 3] * 3)


# --------------------------------------------------------------------------- #
# risk parity
# --------------------------------------------------------------------------- #
class TestRiskParity:
    def test_a_zero_variance_book_has_equal_risk_contributions(self) -> None:
        rc = risk_contributions(np.array([0.5, 0.5]), np.zeros((2, 2)))
        assert rc.tolist() == [0.5, 0.5]

    def test_gappy_returns_are_mean_filled_for_the_covariance(self) -> None:
        w = erc_weights(_gappy())
        assert w.sum() == pytest.approx(1.0)

    def test_a_failed_erc_solve_falls_back_to_inverse_volatility(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rets = _returns()
        with caplog.at_level(logging.WARNING):
            w = erc_weights(rets, max_weight=0.2)
        assert "inverse-vol fallback" in caplog.text
        assert w.sum() == pytest.approx(1.0)
        # De minst volatiele (A) krijgt het meeste gewicht.
        assert w.idxmax() == "A"

    def test_inverse_volatility_refuses_an_empty_universe(self) -> None:
        with pytest.raises(DataContractError, match="Leeg universum"):
            inverse_volatility_long_only(pd.DataFrame(index=range(3)), gross_target=1.0)

    def test_inverse_volatility_gives_unusable_cells_no_position(self) -> None:
        vol = pd.DataFrame({"A": [0.1, np.nan, 0.0], "B": [0.2, 0.2, 0.2]})
        w = inverse_volatility_long_only(vol, gross_target=1.0)
        assert w.loc[0, "A"] == pytest.approx(2 / 3) and w.loc[0, "B"] == pytest.approx(1 / 3)
        assert w.loc[1, "A"] == 0.0 and w.loc[2, "A"] == 0.0

    def test_a_view_without_a_vol_estimate_is_refused_with_the_offending_cells(self) -> None:
        idx = pd.RangeIndex(2)
        expo = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, -0.5]}, index=idx)
        vol = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, np.nan]}, index=idx)
        with pytest.raises(DataContractError, match="vol-schatting") as exc:
            sized_by_risk_parity(expo, vol, gross_target=1.0)
        assert "B" in str(exc.value)

    def test_a_missing_vol_is_fine_where_there_is_no_view(self) -> None:
        idx = pd.RangeIndex(2)
        expo = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.0]}, index=idx)
        vol = pd.DataFrame({"A": [0.1, 0.1], "B": [0.2, np.nan]}, index=idx)
        w = sized_by_risk_parity(expo, vol, gross_target=1.0)
        assert w.loc[1, "B"] == 0.0 and w.loc[1, "A"] == pytest.approx(1.0)

    def test_the_two_panels_must_describe_the_same_universe_and_time_axis(self) -> None:
        expo = pd.DataFrame({"A": [0.5]}, index=[0])
        with pytest.raises(DataContractError, match="verschillend universum"):
            sized_by_risk_parity(expo, pd.DataFrame({"B": [0.1]}, index=[0]), gross_target=1.0)
        with pytest.raises(DataContractError, match="tijdas"):
            sized_by_risk_parity(expo, pd.DataFrame({"A": [0.1]}, index=[1]), gross_target=1.0)


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #
class TestBlackLitterman:
    def test_views_are_built_from_cpcv_oos_predictions_and_unknown_symbols_are_skipped(self) -> None:
        views = BLViews(ASSETS)
        views.from_cpcv_oos(pd.DataFrame({
            "symbol": ["A", "NOPE", "C"],
            "expected_return": [0.10, 0.50, -0.05],
            "prob_std": [0.1, 0.1, 0.2],
        }))
        assert views.P.shape == (2, 3)
        assert views.Q.tolist() == [0.10, -0.05]
        assert np.diag(views.Omega).tolist() == pytest.approx([0.01, 0.04])

    def test_missing_columns_use_their_defaults(self) -> None:
        views = BLViews(ASSETS)
        views.from_cpcv_oos(pd.DataFrame({"symbol": ["B"]}))
        assert views.Q.tolist() == [0.0]
        assert np.diag(views.Omega).tolist() == pytest.approx([1e-4])

    def test_no_views_means_the_equal_weight_prior(self) -> None:
        w = black_litterman_weights(_returns(), views=None)
        assert w.tolist() == pytest.approx([1 / 3] * 3)
        assert black_litterman_weights(_returns(), views=BLViews(ASSETS)).tolist() == pytest.approx([1 / 3] * 3)

    def test_gappy_returns_are_mean_filled_and_a_bullish_view_tilts_the_book(self) -> None:
        views = BLViews(ASSETS)
        views.add_absolute_view("A", 0.5, 1e-4)
        gappy = black_litterman_weights(_gappy(), views=views)
        assert gappy.sum() == pytest.approx(1.0)
        tilted = black_litterman_weights(_returns(), views=views)
        assert tilted["A"] > 1 / 3

    def test_a_relative_view_is_supported(self) -> None:
        views = BLViews(ASSETS)
        views.add_relative_view("A", "C", 0.2, 1e-3)
        assert black_litterman_weights(_returns(), views=views, tau=0.05).sum() == pytest.approx(1.0)

    def test_views_that_push_every_asset_negative_leave_the_equal_weight_prior(self) -> None:
        views = BLViews(ASSETS)
        for a in ASSETS:
            views.add_absolute_view(a, -50.0, 1e-8)
        w = black_litterman_weights(_returns(), views=views)
        assert w.tolist() == pytest.approx([1 / 3] * 3)

    @pytest.mark.parametrize(("fail_on", "match"), [(1, "posterior-solve"), (2, "MVO-solve")])
    def test_a_singular_solve_is_refused_not_replaced_by_the_prior(
        self, monkeypatch: pytest.MonkeyPatch, fail_on: int, match: str
    ) -> None:
        views = BLViews(ASSETS)
        views.add_absolute_view("A", 0.1, 1e-3)
        real = np.linalg.solve
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == fail_on:
                raise np.linalg.LinAlgError("singular")
            return real(*a, **k)

        monkeypatch.setattr(np.linalg, "solve", flaky)
        with pytest.raises(DataContractError, match=match):
            black_litterman_weights(_returns(), views=views)


# --------------------------------------------------------------------------- #
# hrp
# --------------------------------------------------------------------------- #
def test_hrp_mean_fills_gaps_when_too_few_clean_rows_survive() -> None:
    cov = hrp_mod._cov_ledoit_wolf(_gappy())
    assert cov.shape == (3, 3) and np.isfinite(cov).all()


# --------------------------------------------------------------------------- #
# optimizer
# --------------------------------------------------------------------------- #
class TestOptimize:
    @pytest.mark.parametrize("method", ["bl", "erc", "mvo", "minvar", "hrp"])
    def test_every_method_returns_constrained_weights(self, method: str) -> None:
        w = optimize(_returns(), _constraints(), method, hrp_research_gate=GATE)  # type: ignore[arg-type]
        assert w.sum() == pytest.approx(1.0) and (w <= 0.6 + 1e-9).all()

    def test_mvo_and_bl_use_the_views_and_expected_returns_they_are_given(self) -> None:
        views = BLViews(ASSETS)
        views.add_absolute_view("A", 0.5, 1e-4)
        bl = optimize(_returns(), _constraints(max_weight=1.0), "bl", views=views)
        mvo = optimize(_returns(), _constraints(max_weight=1.0), "mvo",
                       expected_returns=pd.Series({"A": 0.05, "B": 0.0, "C": 0.0}))
        assert bl["A"] > 1 / 3 and mvo["A"] > 1 / 3

    def test_too_few_rows_gives_the_constrained_equal_weight_book(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            w = optimize(_returns(3), _constraints(), "erc")
        assert "Insufficient returns" in caplog.text
        assert w.tolist() == pytest.approx([1 / 3] * 3)

    def test_hrp_without_a_research_gate_is_refused(self) -> None:
        with pytest.raises(ConfigContractError, match="hrp_research_gate"):
            optimize(_returns(), _constraints(), "hrp")

    def test_an_unknown_method_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown optimisation method"):
            optimize(_returns(), _constraints(), "magic")  # type: ignore[arg-type]

    def test_constraints_are_mandatory(self) -> None:
        with pytest.raises(ConfigContractError, match="PortfolioConstraints"):
            optimize(_returns(), None, "erc")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# rebalance
# --------------------------------------------------------------------------- #
class TestRebalance:
    CUR = pd.Series({"A": 0.5, "B": 0.5})
    TGT = pd.Series({"A": 0.3, "B": 0.5, "C": 0.2})

    def test_the_default_config_rebalances_once_the_band_is_breached(self) -> None:
        d = should_rebalance(self.CUR, self.TGT)
        assert d.rebalance and d.max_drift == pytest.approx(0.2) and d.turnover == pytest.approx(0.2)
        assert d.estimated_cost == pytest.approx(0.2 * 5.0 / 10_000)

    def test_a_drift_inside_the_band_holds_still(self) -> None:
        d = should_rebalance(self.CUR, self.CUR + pd.Series({"A": 0.01, "B": -0.01}))
        assert not d.rebalance and "< band" in d.reason

    def test_a_positive_net_benefit_threshold_blocks_a_rebalance_that_does_not_pay(self) -> None:
        d = should_rebalance(self.CUR, self.TGT, RebalanceConfig(min_net_benefit=0.01))
        assert not d.rebalance and "net_benefit" in d.reason

    def test_a_small_threshold_lets_a_paying_rebalance_through_and_says_why(self) -> None:
        d = should_rebalance(self.CUR, self.TGT, RebalanceConfig(min_net_benefit=0.0001))
        assert d.rebalance and "net_benefit=" in d.reason


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
class TestConstraintOrder:
    def test_a_book_that_dust_removal_empties_is_spread_equally(self) -> None:
        w = apply_constraints(pd.Series({"A": 0.01, "B": 0.01}), _constraints(min_weight=0.05))
        assert w.tolist() == pytest.approx([0.5, 0.5])

    def test_the_cap_is_closed_again_after_dust_removal_shifts_the_proportions(self) -> None:
        raw = pd.Series({"A": 0.4, "B": 0.4, "C": 0.1, "D": 0.05, "E": 0.05})
        w = apply_constraints(raw, _constraints(max_weight=0.4, min_weight=0.06))
        assert w[["D", "E"]].tolist() == [0.0, 0.0]
        assert w.max() <= 0.4 + 1e-9 and w.sum() == pytest.approx(1.0)

    def test_leverage_below_one_scales_the_whole_book_down(self) -> None:
        w = apply_constraints(pd.Series({"A": 0.5, "B": 0.5}), _constraints(max_leverage=0.8))
        assert w.sum() == pytest.approx(0.8)

    def test_manual_constraints_are_refused_when_provenance_is_enforced(self) -> None:
        with pytest.raises(ConfigContractError, match="soevereine"):
            apply_constraints(pd.Series({"A": 1.0}), _constraints(), enforce_provenance=True)
