# tests/unit/test_alpha_edges.py
"""De alpha-paden die de hoofdtests niet raken.

Elke signaalklasse levert bij onvoldoende data of een onbruikbare parameter een
"geen mening" (signaal 0, vertrouwen 0.5). Die uitstappen zijn hier één voor één
vastgelegd, naast de rekenpaden die een echt signaal geven.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import adaptive_wf as awf
from tradebot.alpha import base as base_mod
from tradebot.alpha.adaptive_wf import AdaptiveWalkForward, AdaptiveWFConfig
from tradebot.alpha.base import AlphaOutput, AlphaSignal, AlphaUnit, SignalResult
from tradebot.alpha.carry import FundingCarry
from tradebot.alpha.combination import ICWeightedCombiner, _ic_damp
from tradebot.alpha.csm_volume_clock import (
    CSMVolumeClockSignal,
    compute_csm_volume_clock_signals,
    rank_normalize_cross_section,
)
from tradebot.alpha.kalman_ou import KalmanOUMeanReversion
from tradebot.alpha.macro_regime import MacroRegimeOverlay, RegimeLabel
from tradebot.alpha.mean_reversion import OUMeanReversion, _fit_ou_mle
from tradebot.alpha.microstructure import OFISignal
from tradebot.alpha.momentum import CrossSectionalMomentum, CSMomentum, TSMomentum
from tradebot.alpha.research_harness import HarnessResult, run_signal_harness
from tradebot.utils.failfast import DataContractError, TradebotContractError


def _bars(close, **cols) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D", tz="UTC")
    return pd.DataFrame({"close": close, **cols}, index=idx)


def _ar1(n: int, phi: float = 0.9, sigma: float = 0.01, seed: int = 3, mu: float = 4.6) -> np.ndarray:
    """Prijzen waarvan de log een AR(1) rond `mu` is: een echte mean-reverter."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, sigma)
    return np.exp(x)


def _walk(n: int, seed: int = 1, drift: float = 0.0005, sigma: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))


def _is_no_view(r: SignalResult) -> bool:
    return r.signal == 0.0 and r.confidence == 0.5


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class TestBase:
    def test_the_protocol_bodies_are_plain_no_ops(self) -> None:
        df = _bars([1.0, 2.0])
        assert AlphaSignal.fit(object(), df) is None
        assert AlphaSignal.predict(object(), df) is None
        assert AlphaSignal.feature_names(object()) is None

    def test_a_relative_import_without_a_module_is_not_a_layer(self) -> None:
        assert base_mod._imported_layers("from . import sibling", "x.py") == set()

    def test_forbidden_layers_are_found_whichever_way_they_are_imported(self) -> None:
        src = (
            "import risk\n"
            "from tradebot.portfolio.hrp import hrp_weights\n"
            "from ..oms import router\n"
            "from tradebot.cv import cpcv\n"
        )
        assert base_mod._imported_layers(src, "x.py") == {"risk", "portfolio", "oms"}

    def test_a_unit_exposes_its_symbols_burn_in_and_identity(self) -> None:
        class _Unit(AlphaUnit):
            name = "edge_unit"

            @property
            def burn_in_period(self) -> int:
                return 2

            def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
                return features.astype("float64")

        unit = _Unit(signal_floor=-1.0, signal_cap=1.0, params={"k": 3})
        assert unit.unit_id == "edge_unit(k=3,signal_cap=1.0,signal_floor=-1.0)"
        frame = pd.DataFrame({"A": [np.nan, np.nan, 0.1, 0.2], "B": [np.nan, np.nan, 0.3, 0.4]},
                             index=pd.date_range("2024-01-01", periods=4, tz="UTC"))
        out = AlphaOutput("edge_unit", frame, 2, unit.params, ())
        assert out.symbols == ("A", "B")
        assert len(out.drop_burn_in()) == 2


# --------------------------------------------------------------------------- #
# carry
# --------------------------------------------------------------------------- #
class TestFundingCarry:
    def _frame(self, n: int = 60, funding: float = 0.0005) -> pd.DataFrame:
        return _bars(_walk(n), funding_rate=np.full(n, funding))

    def test_a_missing_funding_column_is_no_view(self) -> None:
        assert _is_no_view(FundingCarry("BTC").predict(_bars(_walk(40))))

    def test_a_lag_longer_than_the_history_is_no_view(self) -> None:
        assert _is_no_view(FundingCarry("BTC").predict(self._frame(10), lag_bars=10))

    def test_without_lag_the_whole_close_history_feeds_the_vol(self) -> None:
        r = FundingCarry("BTC", min_carry_vol_ratio=0.0).predict(self._frame(), lag_bars=0)
        assert r.signal < 0  # positieve funding: perp is duur, carry is short de perp

    def test_one_bar_of_close_history_is_no_view(self) -> None:
        assert _is_no_view(FundingCarry("BTC").predict(self._frame(1), lag_bars=0))

    def test_fewer_returns_than_the_vol_window_is_no_view(self) -> None:
        assert _is_no_view(FundingCarry("BTC", vol_window=21).predict(self._frame(10), lag_bars=0))

    def test_a_carry_below_the_noise_floor_is_no_view(self) -> None:
        r = FundingCarry("BTC", min_carry_vol_ratio=50.0).predict(self._frame(), lag_bars=0)
        assert _is_no_view(r)

    def test_negative_funding_is_a_long_signal_and_the_features_are_listed(self) -> None:
        sig = FundingCarry("BTC", min_carry_vol_ratio=0.0)
        r = sig.predict(self._frame(funding=-0.0005))
        assert r.signal > 0 and r.confidence > 0.5
        assert sig.feature_names() == ["close", "funding_rate"]
        assert sig.fit(self._frame()) is None


# --------------------------------------------------------------------------- #
# combination
# --------------------------------------------------------------------------- #
class TestICWeightedCombiner:
    def test_the_damp_refuses_to_trust_two_observations_or_a_nan(self) -> None:
        assert _ic_damp(0.5, 2) == 0.0
        assert _ic_damp(float("nan"), 100) == 0.0
        # Alleen teken en bereik worden hier vastgelegd. De grootte is NIET wat de
        # docstring belooft ("shrinks towards zero"): zie de PR-notitie over _ic_damp.
        assert 0 < _ic_damp(0.5, 100) <= 1.0
        assert -1.0 <= _ic_damp(-0.5, 100) < 0

    def test_a_signal_with_too_few_finite_pairs_gets_zero_weight(self) -> None:
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=60, tz="UTC")
        good = pd.Series(rng.normal(size=60), index=idx)
        sparse = pd.Series(np.nan, index=idx)
        sparse.iloc[:5] = rng.normal(size=5)
        fwd = good * 0.5 + pd.Series(rng.normal(0, 0.1, 60), index=idx)
        comb = ICWeightedCombiner(["good", "sparse"], min_ic_abs=0.0)
        comb.fit(pd.DataFrame({"good": good, "sparse": sparse}), fwd)
        table = comb.ic_table()
        assert table.loc["sparse", "weight"] == 0.0
        assert table.loc["good", "weight"] == pytest.approx(1.0)

    def test_predict_and_ic_table_refuse_to_run_before_fit(self) -> None:
        comb = ICWeightedCombiner(["a"])
        with pytest.raises(RuntimeError, match="fit"):
            comb.predict(pd.DataFrame({"a": [0.1]}))
        with pytest.raises(RuntimeError, match="fit"):
            comb.ic_table()


# --------------------------------------------------------------------------- #
# microstructure
# --------------------------------------------------------------------------- #
class TestOFISignal:
    def _flow(self, n: int = 30, buy: float = 3.0, sell: float = 1.0) -> pd.DataFrame:
        return _bars(_walk(n), taker_buy_volume=np.full(n, buy), taker_sell_volume=np.full(n, sell))

    def test_without_taker_volumes_there_is_no_view(self) -> None:
        assert _is_no_view(OFISignal("BTC").predict(_bars(_walk(20))))

    def test_a_lag_longer_than_the_history_is_no_view(self) -> None:
        assert _is_no_view(OFISignal("BTC").predict(self._flow(5), lag_bars=5))

    def test_buy_pressure_is_long_and_the_features_are_listed(self) -> None:
        sig = OFISignal("BTC")
        r = sig.predict(self._flow())
        assert r.signal > 0.9 and r.confidence > 0.9
        assert sig.feature_names() == ["taker_buy_volume", "taker_sell_volume"]
        assert sig.fit(self._flow()) is None


# --------------------------------------------------------------------------- #
# mean reversion
# --------------------------------------------------------------------------- #
class TestOUMeanReversion:
    def test_a_short_sample_falls_back_to_no_reversion(self) -> None:
        theta, mu, sigma = _fit_ou_mle(np.log(_walk(8)))
        assert theta == 0.0 and np.isfinite(mu) and sigma >= 0

    def test_a_constant_price_has_a_singular_design_and_no_reversion(self) -> None:
        theta, mu, _sigma = _fit_ou_mle(np.ones(50))
        assert theta == 0.0 and mu == 1.0

    def test_an_unfitted_signal_is_no_view(self) -> None:
        assert _is_no_view(OUMeanReversion("BTC").predict(_bars(_ar1(80))))

    def test_a_random_walk_is_too_slow_to_revert(self) -> None:
        sig = OUMeanReversion("BTC", fit_window=200)
        df = _bars(_walk(200, drift=0.0))
        sig.fit(df)
        assert _is_no_view(sig.predict(df))

    def test_a_dislocated_price_is_faded_against_the_fitted_mean(self) -> None:
        # De MLE geeft voor log-prijzen ver van nul theta ~ 0 (zie de PR-notitie
        # over _fit_ou_mle), dus de toestand wordt hier rechtstreeks gezet: dit
        # test het signaalpad, niet de schatter.
        sig = OUMeanReversion("BTC", fit_window=300, min_halflife_bars=1, max_halflife_bars=63)
        sig._theta, sig._mu, sig._sigma_eq = 0.1, float(np.log(100.0)), 0.05
        above = sig.predict(_bars([100.0] * 5 + [110.0]))
        below = sig.predict(_bars([100.0] * 5 + [90.0]))
        assert above.signal < 0 < below.signal
        assert above.horizon_bars == int(np.log(2) / 0.1)
        assert sig.feature_names() == ["close"]

    def test_a_half_life_outside_the_band_is_no_view(self) -> None:
        sig = OUMeanReversion("BTC", min_halflife_bars=5, max_halflife_bars=63)
        sig._theta, sig._mu, sig._sigma_eq = 0.001, float(np.log(100.0)), 0.05
        assert _is_no_view(sig.predict(_bars([100.0] * 5 + [110.0])))


# --------------------------------------------------------------------------- #
# Kalman-OU
# --------------------------------------------------------------------------- #
class TestKalmanOU:
    def _fitted(self, **kw) -> tuple[KalmanOUMeanReversion, pd.DataFrame]:
        sig = KalmanOUMeanReversion("BTC", init_window=200, judge_threshold=0.55, **kw)
        df = _bars(_ar1(260))
        sig.fit(df.iloc[:200])
        # De MLE-initialisatie geeft voor log-prijzen ver van nul phi ~ 1 (zie de
        # PR-notitie over _fit_ou_mle); de toestand wordt daarom op een echte
        # mean-reverter gezet (halfwaardetijd 6,6 bars) om het signaalpad te testen.
        sig._phi, sig._mu, sig._sigma_eq = 0.9, 4.6, 0.05
        return sig, df

    def test_an_unfitted_filter_is_no_view(self) -> None:
        assert _is_no_view(KalmanOUMeanReversion("BTC").predict(_bars(_ar1(50))))

    def test_a_short_history_initialises_from_the_price_mean_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        sig = KalmanOUMeanReversion("BTC")
        with caplog.at_level(logging.WARNING):
            sig.fit(_bars(_ar1(12)))
        assert "too few bars" in caplog.text
        assert sig._initialized and sig._phi == 0.9

    def test_the_judge_gate_suppresses_the_signal(self) -> None:
        sig, df = self._fitted()
        assert _is_no_view(sig.predict(df.iloc[:220], prob_judge=0.2))
        assert not _is_no_view(sig.predict(df.iloc[:220], prob_judge=0.9))

    def test_the_filter_updates_once_per_new_bar_and_is_idempotent_for_repeats(self) -> None:
        sig, df = self._fitted()
        first = sig.predict(df.iloc[:230])
        x_after_first = sig._x
        again = sig.predict(df.iloc[:230])
        assert again == first and sig._x == x_after_first
        sig.predict(df.iloc[:231])
        assert sig._x != x_after_first

    @pytest.mark.parametrize("phi", [0.999, 0.1])
    def test_a_half_life_outside_the_band_is_no_view(self, phi: float) -> None:
        sig, df = self._fitted()
        sig._phi = phi
        assert _is_no_view(sig.predict(df.iloc[:220]))

    def test_the_diagnostics_and_feature_names(self) -> None:
        sig, df = self._fitted()
        assert sig.current_halflife() > 0
        assert np.isfinite(sig.current_zscore(float(np.log(df["close"].iloc[-1]))))
        assert sig.feature_names() == ["close"]


# --------------------------------------------------------------------------- #
# macro regime
# --------------------------------------------------------------------------- #
class TestMacroRegime:
    @staticmethod
    def _scenario(drift: float, sigma: float, n_tail: int = 40) -> np.ndarray:
        rng = np.random.default_rng(2)
        calm = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.002, 100)))
        tail = calm[-1] * np.exp(np.cumsum(rng.normal(drift, sigma, n_tail)))
        return np.concatenate([calm, tail])

    @staticmethod
    def _calm_trend(drift: float) -> np.ndarray:
        rng = np.random.default_rng(2)
        rng.normal(0.0, 0.002, 100)  # zelfde stroom als _scenario
        return 100 * np.exp(np.cumsum(np.full(150, drift) + rng.normal(0, 0.0005, 150)))

    def test_fit_is_a_no_op(self) -> None:
        assert MacroRegimeOverlay("BTC").fit(_bars(_walk(5))) is None

    def test_too_little_history_is_neutral(self) -> None:
        ov = MacroRegimeOverlay("BTC")
        assert ov.regime_state(_bars([100.0, 101.0]), lag_bars=1).label is RegimeLabel.NEUTRAL
        assert ov.regime_state(_bars(_walk(10))).label is RegimeLabel.NEUTRAL
        assert ov.regime_state(_bars([100.0]), lag_bars=0).label is RegimeLabel.NEUTRAL

    def test_a_calm_uptrend_is_risk_on_with_full_multiplier(self) -> None:
        ov = MacroRegimeOverlay("BTC", vol_window=10, trend_window=30)
        state = ov.regime_state(_bars(self._calm_trend(0.001)), lag_bars=0)
        assert state.label is RegimeLabel.RISK_ON and state.multiplier == 1.0

    def test_a_volatile_selloff_is_risk_off_and_switches_the_book_off(self) -> None:
        ov = MacroRegimeOverlay("BTC", vol_window=10, trend_window=30)
        state = ov.regime_state(_bars(self._scenario(-0.03, 0.05)), lag_bars=0)
        assert state.label is RegimeLabel.RISK_OFF
        assert state.multiplier == 0.0

    def test_high_vol_without_a_trend_break_is_neutral(self) -> None:
        ov = MacroRegimeOverlay("BTC", vol_window=10, trend_window=30)
        state = ov.regime_state(_bars(self._scenario(0.03, 0.05)), lag_bars=0)
        assert state.label is RegimeLabel.NEUTRAL and state.multiplier == 0.7

    def test_a_trend_break_in_calm_markets_is_neutral(self) -> None:
        ov = MacroRegimeOverlay("BTC", vol_window=10, trend_window=30)
        state = ov.regime_state(_bars(self._calm_trend(-0.001)), lag_bars=0)
        assert state.label is RegimeLabel.NEUTRAL

    def test_a_short_vol_history_uses_the_median_percentile(self) -> None:
        close = 100 * np.exp(np.cumsum(np.random.default_rng(6).normal(0, 0.01, 14)))
        state = MacroRegimeOverlay("BTC", vol_window=10, trend_window=63).regime_state(
            _bars(close), lag_bars=0
        )
        # vol_pct = 0.5 < 0.8 en geen trendbreuk (te weinig bars): risk-on.
        assert state.label is RegimeLabel.RISK_ON

    def test_a_positive_fear_reading_tips_the_score_and_adds_a_factor(self) -> None:
        up = self._calm_trend(0.001)
        kw = dict(vol_window=10, trend_window=30)
        calm = _bars(up, fear=np.full(len(up), 1.0))
        no_fear = MacroRegimeOverlay("BTC", **kw).regime_state(calm, lag_bars=0)
        fear = MacroRegimeOverlay("BTC", fear_col="fear", **kw).regime_state(calm, lag_bars=0)
        relaxed = MacroRegimeOverlay("BTC", fear_col="fear", **kw).regime_state(
            calm.assign(fear=-1.0), lag_bars=0
        )
        assert no_fear.label is RegimeLabel.RISK_ON
        assert fear.label is RegimeLabel.NEUTRAL  # 1 van 3 factoren
        assert relaxed.label is RegimeLabel.RISK_ON

    def test_predict_maps_the_multiplier_onto_the_signal_and_lists_features(self) -> None:
        ov = MacroRegimeOverlay("BTC", fear_col="fear")
        r = ov.predict(_bars(_walk(5), fear=np.zeros(5)))
        assert r.signal == pytest.approx(0.7 * 2.0 - 1.0)
        assert ov.feature_names() == ["close", "fear"]
        assert MacroRegimeOverlay("BTC").feature_names() == ["close"]


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #
class TestTSMomentum:
    def test_a_non_positive_past_price_is_no_view(self) -> None:
        close = _walk(30)
        close[-11] = 0.0  # lookback 20, skip 0: p_past = close[-(20+1)]
        close[-21] = 0.0
        assert _is_no_view(TSMomentum("BTC", lookback_bars=20, skip_bars=0).predict(_bars(close)))

    def test_without_lag_the_vol_covers_every_return(self) -> None:
        sig = TSMomentum("BTC", lookback_bars=20, skip_bars=2)
        df = _bars(_walk(60, drift=0.01))
        assert sig.predict(df, lag_bars=0).signal > 0

    def test_a_lag_longer_than_the_history_falls_back_to_all_returns(self) -> None:
        sig = TSMomentum("BTC", lookback_bars=1, skip_bars=0)
        r = sig.predict(_bars([100.0, 110.0]), lag_bars=5)
        assert r.signal == pytest.approx(1.0)  # één return, nul vol-spreiding: elke stijging saturates

    def test_a_single_bar_has_no_returns_at_all_and_a_floor_vol(self) -> None:
        r = TSMomentum("BTC", lookback_bars=0, skip_bars=0).predict(_bars([100.0]), lag_bars=0)
        assert r.signal == 0.0

    def test_the_unscaled_variant_and_the_feature_list(self) -> None:
        sig = TSMomentum("BTC", lookback_bars=20, skip_bars=2, vol_scale=False)
        assert -1 <= sig.predict(_bars(_walk(60))).signal <= 1
        assert sig.feature_names() == ["close"]
        assert sig.fit(_bars(_walk(60))) is None


class TestCSMomentum:
    def test_too_short_a_history_is_no_view(self) -> None:
        assert _is_no_view(CSMomentum("BTC", lookback_bars=20).predict(_bars(_walk(15))))

    def test_a_non_positive_past_price_is_no_view(self) -> None:
        close = _walk(60)
        close[-22] = 0.0
        assert _is_no_view(CSMomentum("BTC", lookback_bars=20).predict(_bars(close)))
        close0 = _walk(60)
        close0[-21] = -1.0
        assert _is_no_view(CSMomentum("BTC", lookback_bars=20).predict(_bars(close0), lag_bars=0))

    def test_a_calibration_window_without_room_for_a_return_is_no_view(self) -> None:
        # Genoeg bars voor de enkele return, maar het venster (lag 3 weg) is te kort.
        sig = CSMomentum("BTC", lookback_bars=20)
        assert _is_no_view(sig.predict(_bars(_walk(22)), lag_bars=1))

    def test_too_few_calibration_returns_is_no_view(self) -> None:
        assert _is_no_view(CSMomentum("BTC", lookback_bars=20).predict(_bars(_walk(24)), lag_bars=0))

    def test_a_strong_recent_rise_scores_above_its_own_history(self) -> None:
        sig = CSMomentum("BTC", lookback_bars=10)
        close = np.concatenate([_walk(90, drift=0.0, sigma=0.005), 100 * np.exp(np.linspace(0, 0.8, 20))])
        r = sig.predict(_bars(close), lag_bars=0)
        assert r.signal > 0 and r.confidence > 0.5
        assert sig.feature_names() == ["close"]
        assert sig.fit(_bars(close)) is None

    def test_with_a_lag_the_latest_bar_cannot_move_the_signal(self) -> None:
        sig = CSMomentum("BTC", lookback_bars=10)
        close = _walk(120)
        base = sig.predict(_bars(close), lag_bars=1)
        bumped = close.copy()
        bumped[-1] *= 3
        assert sig.predict(_bars(bumped), lag_bars=1).signal == base.signal


class TestCrossSectionalMomentumCalendar:
    def _unit(self, every: int) -> CrossSectionalMomentum:
        return CrossSectionalMomentum(
            lookback_bars=5, skip_bars=1, min_assets=2, rebalance_every_bars=every,
            signal_floor=-1.0, signal_cap=1.0,
        )

    def test_the_view_is_held_between_rebalances_and_clipped_to_the_band(self) -> None:
        idx = pd.date_range("2024-01-01", periods=7, tz="UTC")
        feats = pd.DataFrame(
            {"A": [0.5, 0.9, 2.0, -0.4, np.nan, 0.3, -3.0], "B": np.linspace(-1, 1, 7)}, index=idx
        )
        out = self._unit(3)._generate(feats)
        # Rebalance op i = 0, 3, 6; daartussen wordt de laatste beslissing vastgehouden.
        assert out["A"].tolist() == [0.5, 0.5, 0.5, -0.4, -0.4, -0.4, -1.0]
        assert out["B"].between(-1, 1).all()

    def test_with_a_daily_calendar_the_features_pass_through(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, tz="UTC")
        feats = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [-0.1, -0.2, -0.3]}, index=idx)
        pd.testing.assert_frame_equal(self._unit(1)._generate(feats), feats)


# --------------------------------------------------------------------------- #
# CSM op volume-clock bars
# --------------------------------------------------------------------------- #
class TestRankNormalize:
    def test_ranks_map_linearly_onto_the_band_and_nan_gets_zero(self) -> None:
        out = rank_normalize_cross_section({"a": 3.0, "b": 1.0, "c": 2.0, "d": float("nan")})
        assert out == {"a": 1.0, "b": -1.0, "c": 0.0, "d": 0.0}

    def test_a_universe_below_the_minimum_gets_zeros(self) -> None:
        assert rank_normalize_cross_section({"a": 1.0}, min_universe=2) == {"a": 0.0}

    def test_a_single_asset_universe_is_allowed_to_be_neutral(self) -> None:
        assert rank_normalize_cross_section({"a": 5.0}, min_universe=1) == {"a": 0.0}


class TestCSMVolumeClock:
    def test_fit_needs_a_close_column(self) -> None:
        with pytest.raises(ValueError, match="close"):
            CSMVolumeClockSignal("BTC").fit(pd.DataFrame({"open": [1.0]}))

    def test_fit_caches_the_closes(self) -> None:
        sig = CSMVolumeClockSignal("BTC")
        sig.fit(_bars([1.0, 2.0, 3.0]))
        assert sig._fitted and list(sig._macro_closes) == [1.0, 2.0, 3.0]

    def test_a_short_history_is_no_view(self) -> None:
        assert _is_no_view(CSMVolumeClockSignal("BTC", macro_window_bars=20).predict(_bars(_walk(10))))

    def test_a_low_judge_probability_suppresses_the_trade(self, caplog: pytest.LogCaptureFixture) -> None:
        sig = CSMVolumeClockSignal("BTC", macro_window_bars=10)
        df = _bars(_walk(40, drift=0.01))
        with caplog.at_level(logging.DEBUG):
            assert _is_no_view(sig.predict(df, prob_judge=0.1))
        assert not _is_no_view(sig.predict(df, prob_judge=0.9))

    @pytest.mark.parametrize("bad_position", [-3, -13])
    def test_a_non_positive_price_at_either_end_of_the_window_is_no_view(self, bad_position: int) -> None:
        sig = CSMVolumeClockSignal("BTC", macro_window_bars=10, skip_bars=2)
        close = _walk(40)
        close[bad_position] = 0.0
        assert _is_no_view(sig.predict(_bars(close)))

    def test_the_vol_scaled_and_raw_variants_agree_on_direction(self) -> None:
        df = _bars(_walk(80, drift=0.01))
        scaled = CSMVolumeClockSignal("BTC", macro_window_bars=20, vol_scale=True).predict(df)
        raw = CSMVolumeClockSignal("BTC", macro_window_bars=20, vol_scale=False).predict(df)
        assert scaled.signal > 0 and raw.signal > 0
        assert CSMVolumeClockSignal("BTC").feature_names() == ["close"]

    def test_the_ranker_lags_each_frame_but_stamps_the_latest_bar(self) -> None:
        sigs = {s: CSMVolumeClockSignal(s, macro_window_bars=10) for s in ("A", "B", "C")}
        dfs = {
            "A": _bars(_walk(60, seed=1, drift=0.02)),
            "B": _bars(_walk(60, seed=2, drift=0.0)),
            "C": _bars(_walk(60, seed=3, drift=-0.02)),
        }
        out = compute_csm_volume_clock_signals(sigs, dfs)
        assert out["A"].signal == 1.0 and out["C"].signal == -1.0
        assert all(r.timestamp == dfs[s].index[-1] for s, r in out.items())
        assert out["B"].confidence == pytest.approx(0.5 + 0.5 * out["B"].signal)

    def test_without_lag_a_missing_symbol_is_skipped_and_judge_probs_apply(self) -> None:
        sigs = {s: CSMVolumeClockSignal(s, macro_window_bars=10) for s in ("A", "B", "GONE")}
        dfs = {"A": _bars(_walk(60, seed=1, drift=0.02)), "B": _bars(_walk(60, seed=2, drift=-0.02))}
        out = compute_csm_volume_clock_signals(sigs, dfs, judge_probs={"B": 0.1}, lag_bars=0)
        assert set(out) == {"A", "B"}
        assert out["B"].signal == -1.0  # rank-onderaan: het judge-nulsignaal is de laagste

    def test_a_history_no_longer_than_the_lag_is_not_truncated(self) -> None:
        sigs = {"A": CSMVolumeClockSignal("A", macro_window_bars=10)}
        out = compute_csm_volume_clock_signals(sigs, {"A": _bars(_walk(1))}, lag_bars=1)
        assert out["A"].signal == 0.0


# --------------------------------------------------------------------------- #
# research harness
# --------------------------------------------------------------------------- #
class _Replay:
    """Een signaal dat een vaste reeks signaalwaarden afspeelt."""

    signal_id = "replay"

    def __init__(self, values=None, fail_on_fit: bool = False) -> None:
        self._values = values
        self._i = 0
        self._fail = fail_on_fit

    def fit(self, df: pd.DataFrame) -> None:
        if self._fail:
            raise ValueError("no fit today")

    def predict(self, df: pd.DataFrame) -> SignalResult:
        value = 0.0 if self._values is None else float(self._values[self._i % len(self._values)])
        self._i += 1
        return SignalResult("X", df.index[-1], value, 0.5 + 0.5 * value, 5, self.signal_id)

    def feature_names(self) -> list[str]:
        return ["close"]


class TestResearchHarness:
    DF = _bars(_walk(400, seed=9))

    def test_the_ic_series_and_summary_are_populated_on_a_long_run(self) -> None:
        rng = np.random.default_rng(0)
        sig = _Replay(rng.normal(size=200))
        res = run_signal_harness(sig, self.DF, fit_window=60, step_bars=3, horizon_bars=3, min_fit_bars=30)
        assert isinstance(res, HarnessResult)
        assert len(res.predictions) > 21 and len(res.ic_series) > 0
        assert list(res.ic_table().columns) == ["ic"]
        assert 0.0 <= res.hit_rate <= 1.0
        assert np.isfinite(res.icir)

    def test_a_constant_signal_has_a_zero_ic_not_a_nan(self) -> None:
        res = run_signal_harness(_Replay(), self.DF, fit_window=60, step_bars=3, horizon_bars=3, min_fit_bars=30)
        assert res.mean_ic == 0.0 and res.icir == 0.0

    def test_a_failing_fit_aborts_instead_of_skipping_the_bar(self) -> None:
        with pytest.raises(TradebotContractError, match="Bars overslaan"):
            run_signal_harness(_Replay(fail_on_fit=True), self.DF, fit_window=60, step_bars=3,
                               horizon_bars=3, min_fit_bars=30)

    def test_too_few_predictions_return_an_empty_summary(self) -> None:
        res = run_signal_harness(_Replay(), self.DF, fit_window=300, step_bars=50, horizon_bars=21)
        assert len(res.predictions) < 5
        assert res.mean_ic == 0.0 and res.ic_series.empty

    def test_between_five_and_the_ic_window_the_full_sample_ic_is_the_mean(self) -> None:
        rng = np.random.default_rng(1)
        res = run_signal_harness(_Replay(rng.normal(size=50)), self.DF, fit_window=60,
                                 step_bars=30, horizon_bars=5, min_fit_bars=30)
        assert 5 <= len(res.predictions) <= 21
        assert res.ic_series.empty and res.icir == 0.0
        assert np.isfinite(res.mean_ic)

    def test_a_non_positive_price_skips_that_forward_return(self) -> None:
        df = self.DF.copy()
        df.iloc[100, df.columns.get_loc("close")] = 0.0
        full = run_signal_harness(_Replay(), self.DF, fit_window=60, step_bars=10, horizon_bars=5, min_fit_bars=30)
        holed = run_signal_harness(_Replay(), df, fit_window=60, step_bars=10, horizon_bars=5, min_fit_bars=30)
        assert len(holed.predictions) == len(full.predictions) - 1

    def test_nan_signals_leave_too_few_pairs_for_both_ic_estimates(self) -> None:
        res = run_signal_harness(_Replay([np.nan]), self.DF, fit_window=60, step_bars=3,
                                 horizon_bars=3, min_fit_bars=30)
        assert res.mean_ic == 0.0 and res.ic_series.empty


# --------------------------------------------------------------------------- #
# adaptive walk-forward
# --------------------------------------------------------------------------- #
def _panel(n_assets: int = 16, n_days: int = 760, seed: int = 7, with_btc: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-06-01", periods=n_days, freq="D", tz="UTC")
    names = (["BTCUSDT"] if with_btc else []) + [f"A{i}USDT" for i in range(n_assets - int(with_btc))]
    frames = []
    for s in names:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.04, n_days)))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": s,
            "open": close * (1 + rng.normal(0, 0.005, n_days)),
            "high": close * (1 + np.abs(rng.normal(0, 0.01, n_days))),
            "low": close * (1 - np.abs(rng.normal(0, 0.01, n_days))),
            "close": close, "volume": rng.lognormal(10, 1, n_days),
        }))
    return pd.concat(frames, ignore_index=True)


def _wf_cfg(**kw) -> AdaptiveWFConfig:
    base = dict(wf_start="2023-01-01", refit_days=180, min_train_rows=500, min_hist_bars=300,
                mda_keep=99, recency_halflife_days=200.0, seed=0)
    base.update(kw)
    cfg = AdaptiveWFConfig(**base)
    cfg.catboost_params = {"iterations": 30, "depth": 3, "learning_rate": 0.1, "l2_leaf_reg": 4,
                           "loss_function": "Logloss", "verbose": 0}
    return cfg


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _panel()


@pytest.fixture(scope="module")
def run_book(ohlcv: pd.DataFrame):
    eng = AdaptiveWalkForward(_wf_cfg()).build_panel(ohlcv)
    return eng, eng.run(eng.build_features_labels())


class TestAdaptiveWalkForward:
    def test_funding_enters_as_a_lagged_z_score(self, ohlcv: pd.DataFrame) -> None:
        dates = pd.DatetimeIndex(sorted(ohlcv["date"].unique()))
        syms = sorted(ohlcv["symbol"].unique())
        funding = pd.DataFrame(np.random.default_rng(0).normal(0, 1e-4, (len(dates), len(syms))),
                               index=dates, columns=syms)
        eng = AdaptiveWalkForward(_wf_cfg()).build_panel(ohlcv, funding=funding)
        z = eng._panels["fund_z"]
        assert z.iloc[:30].isna().all().all()          # 30 bars om te schatten, +1 voor de lag
        assert np.isfinite(z.iloc[100:].to_numpy()).all()

    def test_a_panel_without_btc_uses_the_market_return_as_the_factor(self) -> None:
        eng = AdaptiveWalkForward(_wf_cfg()).build_panel(_panel(n_assets=6, with_btc=False))
        pd.testing.assert_series_equal(eng._panels["btc_ret"], eng._panels["mkt"])

    def test_a_short_training_history_keeps_every_feature(self, ohlcv: pd.DataFrame) -> None:
        eng = AdaptiveWalkForward(_wf_cfg(wf_start="2021-11-01", mda_keep=5, min_train_rows=10**9))
        eng.build_panel(ohlcv)
        P = eng.build_features_labels()
        out = eng.run(P)
        assert eng.frozen_features_ == eng.features_
        # Geen enkel blok haalt min_train_rows: er wordt niets voorspeld.
        assert out["__p"].isna().all() and eng.refit_dates_ == []

    def test_mda_freezes_the_top_features_on_the_first_window(self, ohlcv: pd.DataFrame) -> None:
        eng = AdaptiveWalkForward(_wf_cfg(mda_keep=5)).build_panel(ohlcv)
        P = eng.build_features_labels()
        frozen = eng._freeze_features(P, pd.Timestamp("2023-01-01", tz="UTC"))
        assert len(frozen) == 5 and set(frozen) <= set(eng.features_)

    def test_an_empty_mda_result_is_an_error_not_a_silent_full_set(
        self, ohlcv: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tradebot.selection import mda

        monkeypatch.setattr(mda, "causal_mda", lambda *a, **k: pd.DataFrame())
        eng = AdaptiveWalkForward(_wf_cfg(mda_keep=5)).build_panel(ohlcv)
        P = eng.build_features_labels()
        with pytest.raises(ValueError, match="empty MDA"):
            eng._freeze_features(P, pd.Timestamp("2023-01-01", tz="UTC"))

    def test_blocks_with_too_little_training_data_are_skipped(self, ohlcv: pd.DataFrame) -> None:
        eng = AdaptiveWalkForward(_wf_cfg(wf_start="2021-09-01", min_train_rows=3000)).build_panel(ohlcv)
        out = eng.run(eng.build_features_labels())
        assert 0 < len(eng.refit_dates_) < 5
        assert out["__p"].notna().any()

    def test_a_singular_neutralisation_regression_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*_a, **_k):
            raise np.linalg.LinAlgError("SVD did not converge")

        idx = pd.date_range("2024-01-01", periods=2, tz="UTC")
        cols = [f"S{i}" for i in range(20)]
        rng = np.random.default_rng(0)
        sig = pd.DataFrame(rng.normal(size=(2, 20)), index=idx, columns=cols)
        logvol = pd.DataFrame(rng.normal(size=(2, 20)), index=idx, columns=cols)
        betaf = pd.DataFrame(rng.normal(size=(2, 20)), index=idx, columns=cols)
        monkeypatch.setattr(np.linalg, "lstsq", _boom)
        with pytest.raises(DataContractError, match="singulier"):
            AdaptiveWalkForward._neutralise(sig, logvol, betaf)

    def test_evaluate_reports_the_deflated_and_overfit_statistics(self, run_book) -> None:
        eng, P = run_book
        res = eng.evaluate(P)
        assert res["n_trials"] == 8
        assert 0.0 <= res["pbo"] <= 1.0
        assert np.isfinite(res["deflated_sharpe"])
        assert res["dsr_approximation"] == "empirical"
        assert res["n_refits"] == len(eng.refit_dates_)
        assert res["n_obs"] == len(res["pnl"])
        assert 2023 in res["per_year"]

    def test_module_constants(self) -> None:
        assert awf.DAYS == 365.0
