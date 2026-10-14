# tests/unit/test_systematic_small_edges.py
"""De kleine randen van `systematic/`: marktbouw, gatenoverbrugging, vol-doel zonder covariantie,
onbekende sleeves en de twee rapportagefuncties van de evaluatie.

De causaliteit en de boekhouding staan in `test_systematic_book.py`, `test_systematic_programme.py`
en `tests/lookahead/test_breadth_causality.py`; hier staat wat die niet raken.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.systematic_fixtures import config, synthetic_market
from tests.weekly_fixtures import synthetic_market as weekly_market
from tradebot.schemas.robust_book_v2 import robust_book_v2_config
from tradebot.systematic.book import CostSpec
from tradebot.systematic.breadth import (
    BreadthMarket,
    bridge_data_holes,
    scale_to_vol,
    sleeve_targets,
)
from tradebot.systematic.evaluate import dsr_record, regime_breakdown, regime_masks
from tradebot.systematic.market import BARS_PER_YEAR, from_weekly_market
from tradebot.systematic.programme import build_trial, run_targets
from tradebot.systematic.sleeves import build_sleeve, portfolio_scale
from tradebot.utils.failfast import DataContractError

COSTS = CostSpec(taker_fee=5.5e-4, half_spread=1e-4, impact=None, aum_usd=1e5)


# --------------------------------------------------------------------------- #
# market
# --------------------------------------------------------------------------- #
class TestFromWeeklyMarket:
    def test_the_weekly_panels_become_the_book_market_on_the_same_grid(self) -> None:
        wm = weekly_market(n=120)
        book = from_weekly_market(wm)
        assert book.symbols == wm.symbols and book.index.equals(wm.grid)
        pd.testing.assert_frame_equal(book.sigma_daily, wm.sigma_daily)
        assert book.source_hashes == wm.source_hashes
        # Funding wordt vóór de notering op nul gezet en nergens NaN.
        assert not book.funding.isna().any().any()
        np.testing.assert_allclose(
            book.ret.iloc[1:].to_numpy(),
            np.column_stack([wm.ohlcv[s]["close"].pct_change().iloc[1:] for s in wm.symbols]))

    def test_a_missing_close_after_the_listing_is_refused_not_bridged(self) -> None:
        wm = weekly_market(n=120)
        wm.ohlcv["ETHUSDT"].iloc[50, wm.ohlcv["ETHUSDT"].columns.get_loc("close")] = np.nan
        with pytest.raises(DataContractError, match="Ontbrekende close na de notering"):
            from_weekly_market(wm)

    def test_a_coin_listed_later_is_nan_before_its_listing_and_has_no_funding_there(self) -> None:
        wm = weekly_market(n=120)
        wm.ohlcv["SOLUSDT"].iloc[:30, wm.ohlcv["SOLUSDT"].columns.get_loc("close")] = np.nan
        book = from_weekly_market(wm)
        assert book.close["SOLUSDT"].iloc[:30].isna().all() and book.close["SOLUSDT"].iloc[30:].notna().all()
        assert (book.funding["SOLUSDT"].iloc[:30] == 0.0).all()


# --------------------------------------------------------------------------- #
# sleeves
# --------------------------------------------------------------------------- #
class TestPortfolioScale:
    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame([[0.5, -0.5], [0.0, 0.0], [0.5, -0.5]], columns=["A", "B"])

    def test_a_row_is_scaled_to_the_target_with_that_bars_covariance(self) -> None:
        cov = np.tile(np.array([[1e-4, 0.0], [0.0, 1e-4]]), (3, 1, 1))
        out = portfolio_scale(self._raw(), cov, vol_target=0.5, max_scale=10.0)
        var = (0.25 * 1e-4 + 0.25 * 1e-4) * BARS_PER_YEAR
        assert out.iloc[0, 0] == pytest.approx(0.5 * 0.5 / np.sqrt(var))
        assert (out.iloc[1] == 0.0).all()

    def test_a_bar_without_a_usable_covariance_gets_no_position(self) -> None:
        cov = np.tile(np.array([[1e-4, 0.0], [0.0, 1e-4]]), (3, 1, 1))
        cov[0, 0, 0] = np.nan                                           # nog geen covariantie
        out = portfolio_scale(self._raw(), cov, vol_target=0.5, max_scale=10.0)
        assert (out.iloc[0] == 0.0).all() and out.iloc[2, 0] != 0.0

    def test_a_zero_variance_bar_gets_no_position(self) -> None:
        cov = np.zeros((3, 2, 2))
        out = portfolio_scale(self._raw(), cov, vol_target=0.5, max_scale=10.0)
        assert (out.to_numpy() == 0.0).all()

    def test_the_scale_is_capped(self) -> None:
        cov = np.tile(np.array([[1e-8, 0.0], [0.0, 1e-8]]), (3, 1, 1))
        out = portfolio_scale(self._raw(), cov, vol_target=0.5, max_scale=2.0)
        assert out.iloc[0, 0] == pytest.approx(0.5 * 2.0)


def test_an_unknown_sleeve_is_refused() -> None:
    with pytest.raises(DataContractError, match="Onbekende sleeve 'X0_NOPE'"):
        build_sleeve("X0_NOPE", synthetic_market(n_bars=300), config())


# --------------------------------------------------------------------------- #
# breadth
# --------------------------------------------------------------------------- #
def _breadth_market(n_bars: int = 200) -> BreadthMarket:
    book = synthetic_market(n_bars=n_bars)
    close = book.close
    return BreadthMarket(
        book=book, high=close * 1.01, low=close * 0.99, quote_volume=book.adv_usd.fillna(1e8),
        half_spread=pd.DataFrame(2e-4, index=close.index, columns=close.columns),
        universe=close.notna())


class TestBreadthMarket:
    def test_a_subset_restricts_every_panel_to_the_chosen_coins(self) -> None:
        m = _breadth_market()
        sub = m.subset(("ETHUSDT", "BTCUSDT"))
        for frame in (sub.high, sub.low, sub.quote_volume, sub.half_spread, sub.universe,
                      sub.book.close, sub.book.ret, sub.book.funding):
            assert list(frame.columns) == ["ETHUSDT", "BTCUSDT"]
        pd.testing.assert_series_equal(sub.high["BTCUSDT"], m.high["BTCUSDT"])

    def test_an_unknown_sleeve_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="Onbekende sleeve 'X9_NOPE'"):
            sleeve_targets("X9_NOPE", _breadth_market(), robust_book_v2_config())


class TestBridgeDataHoles:
    def test_short_holes_inside_the_listing_are_bridged_and_long_ones_are_not(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        close = pd.DataFrame({"A": np.arange(1.0, 31.0), "EMPTY": np.nan}, index=idx)
        close.loc[idx[:3], "A"] = np.nan                                # voor de notering: blijft leeg
        close.loc[idx[10:12], "A"] = np.nan                             # kort gat: 2 bars
        close.loc[idx[20:29], "A"] = np.nan                             # lang gat: 9 bars
        out = bridge_data_holes(close, close, max_gap=7)
        assert out["A"].iloc[:3].isna().all()
        assert out["A"].iloc[10] == 10.0 and out["A"].iloc[11] == 10.0   # laatst bekende waarde
        assert out["A"].iloc[20:29].isna().all()
        assert out["EMPTY"].isna().all()                                 # een lege kolom wordt overgeslagen


class TestScaleToVol:
    def test_a_coin_without_enough_history_makes_the_row_undecided(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        rng = np.random.default_rng(1)
        ret = pd.DataFrame(rng.normal(0.0, 0.02, (60, 2)), index=idx, columns=["A", "B"])
        ret.loc[idx[:50], "B"] = np.nan                                  # B bestaat pas sinds bar 50
        raw = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        reb = pd.Series(False, index=idx)
        reb.iloc[[40, 59]] = True
        out = scale_to_vol(raw, ret, reb, span=30, vol_target=0.2, max_scale=3.0)
        assert out.iloc[40].isna().all()                                 # B heeft nog 0 van de 10 waarnemingen
        assert out.iloc[59].notna().all()                                # B heeft er nu precies 10: geschaald
        assert out.iloc[[i for i in range(60) if i not in (40, 59)]].isna().all().all()

    def test_a_row_without_positions_is_flat_and_a_zero_variance_row_is_undecided(self) -> None:
        idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
        ret = pd.DataFrame(0.0, index=idx, columns=["A", "B"])           # niets beweegt: variantie 0
        raw = pd.DataFrame({"A": 0.5, "B": -0.5}, index=idx)
        raw.iloc[30] = 0.0                                               # geen positie
        reb = pd.Series(True, index=idx)
        out = scale_to_vol(raw, ret, reb, span=30, vol_target=0.2, max_scale=3.0)
        assert (out.iloc[30] == 0.0).all()
        assert out.iloc[35].isna().all()

    def test_a_decision_row_is_scaled_to_the_target_volatility(self) -> None:
        idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
        rng = np.random.default_rng(5)
        ret = pd.DataFrame(rng.normal(0.0, 0.02, (300, 2)), index=idx, columns=["A", "B"])
        raw = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
        reb = pd.Series(False, index=idx)
        reb.iloc[299] = True
        out = scale_to_vol(raw, ret, reb, span=60, vol_target=0.2, max_scale=50.0)
        w = out.iloc[299].to_numpy()
        realised = float(np.sqrt(w @ np.cov(ret.iloc[-200:].T.to_numpy()) @ w * BARS_PER_YEAR))
        assert realised == pytest.approx(0.2, rel=0.35)                  # EWMA-schatter, geen steekproefcov


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
class TestRegimeBreakdown:
    def test_every_regime_reports_its_share_and_a_sharpe(self) -> None:
        m = synthetic_market(n_bars=800)
        res = run_targets(build_trial("B1_BTC_HOLD", m, config()), m, COSTS, lag=1)
        a, b = m.index[250], m.index[780]
        out = regime_breakdown(res.window(a, b), m)
        masks = regime_masks(m, pd.DatetimeIndex(res.window(a, b).frame.index))
        assert set(out) == set(masks) == {
            "bull_btc_above_ma200", "bear_btc_below_ma200", "high_vol", "low_vol",
            "high_correlation", "low_correlation"}
        for name, rec in out.items():
            assert rec["share_of_bars"] == pytest.approx(float(masks[name].mean()))
            assert set(rec) == {"share_of_bars", "sharpe", "ann_return_arith"}
        assert out["bull_btc_above_ma200"]["share_of_bars"] + out["bear_btc_below_ma200"]["share_of_bars"] <= 1.0


class TestDsrRecord:
    R = pd.Series(np.random.default_rng(3).normal(0.0006, 0.01, 800))

    def test_more_trials_demand_more_from_the_same_returns(self) -> None:
        sharpes = [0.01, 0.03, 0.05, 0.02]
        few = dsr_record(self.R, n_trials=2, trial_sharpes_per_bar=sharpes)
        many = dsr_record(self.R, n_trials=500, trial_sharpes_per_bar=sharpes)
        assert many["dsr"] <= few["dsr"]
        assert many["required_annual_sharpe"] > few["required_annual_sharpe"]
        assert few["sr_variance_empirical"] == pytest.approx(float(np.var(sharpes, ddof=1)))

    def test_without_two_trial_sharpes_the_empirical_variance_is_zero_and_the_floor_applies(self) -> None:
        rec = dsr_record(self.R, n_trials=7, trial_sharpes_per_bar=[0.02])
        assert rec["sr_variance_empirical"] == 0.0
        assert np.isfinite(rec["dsr"]) and rec["required_annual_sharpe"] > 0.0
