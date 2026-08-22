"""tests/unit/test_wave5.py — Smoke tests for Wave 5 migrated modules.

Covers the minimal "does it run and produce sane output" contract for every
new public symbol introduced in Wave 5.  No heavy fixtures — all inputs are
synthesised in-process.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# train.quant_arch — SymmetricQuantileScaler
# ---------------------------------------------------------------------------

class TestSymmetricQuantileScaler:
    def _fitted(self, n_bins: int = 21):
        from tradebot.train.quant_arch import SymmetricQuantileScaler
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (500, 3)).astype(np.float64)
        scaler = SymmetricQuantileScaler(n_bins=n_bins)
        scaler.fit(X)
        return scaler, X

    def test_transform_shape_preserved(self) -> None:
        scaler, X = self._fitted()
        out = scaler.transform(X)
        assert out.shape == X.shape

    def test_transform_returns_finite(self) -> None:
        scaler, X = self._fitted()
        out = scaler.transform(X)
        assert np.all(np.isfinite(out))

    def test_zero_column_maps_near_zero(self) -> None:
        """Zero-crossing-aware scaler: a column of zeros should map near 0."""
        scaler, X = self._fitted()
        zeros = np.zeros((10, 3), dtype=np.float64)
        out = scaler.transform(zeros)
        assert np.all(np.abs(out) < 0.1)

    def test_fit_transform_matches_fit_then_transform(self) -> None:
        from tradebot.train.quant_arch import SymmetricQuantileScaler
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, (200, 2)).astype(np.float64)
        s1 = SymmetricQuantileScaler(n_bins=21)
        s2 = SymmetricQuantileScaler(n_bins=21)
        out1 = s1.fit_transform(X)
        s2.fit(X)
        out2 = s2.transform(X)
        np.testing.assert_array_almost_equal(out1, out2)


# ---------------------------------------------------------------------------
# train.quant_arch — P2OnlineQuantile
# ---------------------------------------------------------------------------

class TestP2OnlineQuantile:
    def test_value_none_before_warmup(self) -> None:
        """P² needs 5 seed observations — value() must return None before that."""
        from tradebot.train.quant_arch import P2OnlineQuantile
        est = P2OnlineQuantile(quantile=0.5, n_features=1)
        assert est.value() is None
        for _ in range(4):                       # 4 obs — still in warmup
            est.update(np.array([1.0]))
        assert est.value() is None

    def test_value_returns_array_post_warmup(self) -> None:
        """After 5+ observations value() must return a finite ndarray."""
        from tradebot.train.quant_arch import P2OnlineQuantile
        est = P2OnlineQuantile(quantile=0.5, n_features=1)
        rng = np.random.default_rng(42)
        for _ in range(50):
            est.update(rng.normal(0, 1, (1,)))
        val = est.value()
        assert isinstance(val, np.ndarray)
        assert val.shape == (1,)
        assert np.all(np.isfinite(val))

    def test_multi_feature_shape(self) -> None:
        from tradebot.train.quant_arch import P2OnlineQuantile
        est = P2OnlineQuantile(quantile=0.9, n_features=3)
        rng = np.random.default_rng(7)
        for _ in range(20):
            est.update(rng.normal(0, 1, (3,)))
        val = est.value()
        assert val is not None
        assert val.shape == (3,)
        assert np.all(np.isfinite(val))


# ---------------------------------------------------------------------------
# train.quant_arch — dynamic_embargo_bars
# ---------------------------------------------------------------------------

class TestDynamicEmbargoBars:
    def test_positive_output(self) -> None:
        from tradebot.train.quant_arch import dynamic_embargo_bars
        result = dynamic_embargo_bars(horizons_bars=[10, 20, 30])
        assert result > 0

    def test_longer_horizons_yield_more_bars(self) -> None:
        from tradebot.train.quant_arch import dynamic_embargo_bars
        short = dynamic_embargo_bars(horizons_bars=[5])
        long_ = dynamic_embargo_bars(horizons_bars=[5, 50, 100])
        assert long_ >= short

    def test_min_embargo_respected(self) -> None:
        from tradebot.train.quant_arch import dynamic_embargo_bars
        result = dynamic_embargo_bars(horizons_bars=[1], min_embargo=42)
        assert result >= 42

    def test_returns_int(self) -> None:
        from tradebot.train.quant_arch import dynamic_embargo_bars
        result = dynamic_embargo_bars(horizons_bars=[15, 30])
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# train.schema_guard — FeatureSchemaGuard
# ---------------------------------------------------------------------------

class TestFeatureSchemaGuard:
    def test_stamp_and_check_ok(self) -> None:
        from tradebot.train.schema_guard import FeatureSchemaGuard
        guard = FeatureSchemaGuard()
        features = ["feat_rsi", "feat_macd", "feat_bb_upper"]
        guard.stamp(features)
        guard.check(features)  # must not raise

    def test_stamped_property_none_before_stamp(self) -> None:
        from tradebot.train.schema_guard import FeatureSchemaGuard
        guard = FeatureSchemaGuard()
        assert guard.stamped is None

    def test_stamped_property_set_after_stamp(self) -> None:
        from tradebot.train.schema_guard import FeatureSchemaGuard
        guard = FeatureSchemaGuard()
        fp = guard.stamp(["feat_a", "feat_b"])
        assert guard.stamped is not None
        assert guard.stamped.feature_count == 2

    def test_check_detects_addition(self) -> None:
        from tradebot.train.schema_guard import FeatureSchemaGuard, SchemaMismatchError
        guard = FeatureSchemaGuard()
        guard.stamp(["feat_a", "feat_b"])
        with pytest.raises(SchemaMismatchError):
            guard.check(["feat_a", "feat_b", "feat_c"])

    def test_check_detects_removal(self) -> None:
        from tradebot.train.schema_guard import FeatureSchemaGuard, SchemaMismatchError
        guard = FeatureSchemaGuard()
        guard.stamp(["feat_a", "feat_b", "feat_c"])
        with pytest.raises(SchemaMismatchError):
            guard.check(["feat_a", "feat_b"])

    def test_check_detects_reorder(self) -> None:
        """Order matters — SHA-256 hash is order-sensitive."""
        from tradebot.train.schema_guard import FeatureSchemaGuard, SchemaMismatchError
        guard = FeatureSchemaGuard()
        guard.stamp(["feat_a", "feat_b"])
        with pytest.raises(SchemaMismatchError):
            guard.check(["feat_b", "feat_a"])

    def test_check_before_stamp_raises(self) -> None:
        from tradebot.train.schema_guard import FeatureSchemaGuard
        guard = FeatureSchemaGuard()
        with pytest.raises(Exception):
            guard.check(["feat_x"])


# ---------------------------------------------------------------------------
# train.schema_guard — EntropyGate
# ---------------------------------------------------------------------------

class TestEntropyGate:
    def test_decision_has_required_fields(self) -> None:
        from tradebot.train.schema_guard import EntropyGate
        gate = EntropyGate()
        decision = gate.evaluate([0.6, 0.4])
        assert hasattr(decision, "pass_through")
        assert hasattr(decision, "entropy")
        assert hasattr(decision, "threshold")
        assert hasattr(decision, "action")

    def test_low_entropy_passes_through_during_warmup(self) -> None:
        from tradebot.train.schema_guard import EntropyGate
        gate = EntropyGate(history_size=500, tau_percentile=80.0)
        # Very confident prediction → low entropy → should pass through
        decision = gate.evaluate([0.99, 0.01])
        assert decision.pass_through

    def test_high_entropy_gated_post_warmup(self) -> None:
        from tradebot.train.schema_guard import EntropyGate
        gate = EntropyGate(history_size=5, tau_percentile=50)
        # Feed many very-low-entropy 3-class distributions to anchor threshold low
        for _ in range(30):
            gate.evaluate([0.999, 0.0005, 0.0005])
        # Max-entropy 3-class input (log3 ≈ 1.099) >> anchored threshold → blocked
        decision = gate.evaluate([1/3, 1/3, 1/3])
        assert not decision.pass_through

    def test_entropy_value_is_finite(self) -> None:
        from tradebot.train.schema_guard import EntropyGate
        gate = EntropyGate()
        decision = gate.evaluate([0.7, 0.2, 0.1])
        assert math.isfinite(decision.entropy)
        assert decision.entropy >= 0.0


# ---------------------------------------------------------------------------
# train.reward — KalmanImpactObserver
# ---------------------------------------------------------------------------

class TestKalmanImpactObserver:
    def test_update_returns_finite_eta(self) -> None:
        from tradebot.train.reward import KalmanImpactObserver
        obs = KalmanImpactObserver()
        eta = obs.update(
            realised_slippage=0.002,
            sigma_per_bar=0.01,
            order_size=100.0,
            bar_volume=10_000.0,
        )
        assert math.isfinite(eta)
        assert eta > 0

    def test_predicted_slippage_positive(self) -> None:
        from tradebot.train.reward import KalmanImpactObserver
        obs = KalmanImpactObserver()
        slip = obs.predicted_slippage(
            order_size=200.0, bar_volume=20_000.0, sigma_per_bar=0.01
        )
        assert math.isfinite(slip)
        assert slip >= 0.0

    def test_eta_bounded_after_updates(self) -> None:
        from tradebot.train.reward import KalmanImpactObserver
        obs = KalmanImpactObserver(eta_min=0.01, eta_max=1.0)
        for _ in range(50):
            eta = obs.update(
                realised_slippage=0.05,
                sigma_per_bar=0.01,
                order_size=500.0,
                bar_volume=10_000.0,
            )
        assert 0.01 <= eta <= 1.0


# ---------------------------------------------------------------------------
# train.reward — NetAlphaReward
# ---------------------------------------------------------------------------

class TestNetAlphaReward:
    def test_compute_returns_netalpharesult(self) -> None:
        from tradebot.train.reward import NetAlphaReward, NetAlphaResult
        reward = NetAlphaReward(fee_bps=4.0)
        result = reward.compute(
            price_entry=100.0,
            price_exit=101.0,
            order_size=1_000.0,
            bar_volume=50_000.0,
            sigma_per_bar=0.01,
            side=1,
        )
        assert isinstance(result, NetAlphaResult)
        assert math.isfinite(result.reward)

    def test_winning_trade_positive_reward(self) -> None:
        from tradebot.train.reward import NetAlphaReward
        reward = NetAlphaReward(fee_bps=2.0)
        result = reward.compute(
            price_entry=100.0,
            price_exit=102.0,  # +2%
            order_size=100.0,
            bar_volume=1_000_000.0,  # huge vol → near-zero impact
            sigma_per_bar=0.005,
            side=1,
        )
        assert result.reward > 0

    def test_high_fees_reduce_reward(self) -> None:
        from tradebot.train.reward import NetAlphaReward
        kwargs = dict(
            price_entry=100.0, price_exit=101.0,
            order_size=100.0, bar_volume=100_000.0,
            sigma_per_bar=0.005, side=1,
        )
        cheap = NetAlphaReward(fee_bps=1.0).compute(**kwargs)
        exp   = NetAlphaReward(fee_bps=50.0).compute(**kwargs)
        assert cheap.reward > exp.reward


# ---------------------------------------------------------------------------
# backtest.evaluation — deflated_sharpe_penalty
# ---------------------------------------------------------------------------

class TestDeflatedSharpePenalty:
    def test_returns_float(self) -> None:
        from tradebot.backtest.evaluation import deflated_sharpe_penalty
        result = deflated_sharpe_penalty(
            hist_sharpes=[1.5, 1.2, 0.8, 1.0],
            n_optuna_trials=50,
        )
        assert isinstance(result, float)
        assert math.isfinite(result)

    def test_more_trials_more_penalty(self) -> None:
        from tradebot.backtest.evaluation import deflated_sharpe_penalty
        sharpes = [1.5, 1.2, 0.9, 0.7]
        p_few  = deflated_sharpe_penalty(hist_sharpes=sharpes, n_optuna_trials=1)
        p_many = deflated_sharpe_penalty(hist_sharpes=sharpes, n_optuna_trials=500)
        # More trials → larger DSR penalty value (penalty to be subtracted from SR)
        assert p_many >= p_few

    def test_cap_factor_limits_penalty(self) -> None:
        from tradebot.backtest.evaluation import deflated_sharpe_penalty
        result = deflated_sharpe_penalty(
            hist_sharpes=[2.0, 1.8, 1.6],
            n_optuna_trials=10_000,
            cap_factor=2.0,
        )
        assert result >= 0.0  # DSR should not go negative


# ---------------------------------------------------------------------------
# backtest.evaluation — resolve_long_short_collision
# ---------------------------------------------------------------------------

class TestResolveLongShortCollision:
    def _call(self, long_active, short_active, prob_long=0.7, prob_short=0.6):
        from tradebot.backtest.evaluation import resolve_long_short_collision
        return resolve_long_short_collision(
            long_active=long_active,
            short_active=short_active,
            prob_long=prob_long,
            prob_short=prob_short,
            threshold_long=0.55,
            threshold_short=0.55,
            horizon_long=12,
            horizon_short=12,
            ret_long=0.005,
            ret_short=0.004,
        )

    def test_long_only_side_is_long(self) -> None:
        result = self._call(long_active=True, short_active=False)
        assert result.side in ("long", 1, "LONG")

    def test_short_only_side_is_short(self) -> None:
        result = self._call(long_active=False, short_active=True, prob_long=0.3, prob_short=0.7)
        assert result.side in ("short", -1, "SHORT")

    def test_both_active_nets_out(self) -> None:
        result = self._call(long_active=True, short_active=True, prob_long=0.7, prob_short=0.7)
        # net_weight should be near zero when evenly matched
        assert abs(result.net_weight) < 1.0

    def test_neither_active_flat(self) -> None:
        result = self._call(long_active=False, short_active=False, prob_long=0.4, prob_short=0.4)
        assert result.net_weight == pytest.approx(0.0, abs=1e-9)

    def test_result_has_expected_fields(self) -> None:
        result = self._call(long_active=True, short_active=False)
        assert hasattr(result, "side")
        assert hasattr(result, "net_weight")
        assert hasattr(result, "horizon")
        assert hasattr(result, "expected_return")


# ---------------------------------------------------------------------------
# backtest.evaluation — gap_risk_kelly_size
# ---------------------------------------------------------------------------

class TestGapRiskKellySize:
    def _call(self, target_risk=0.02, gap_atr_multiple=2.0):
        from tradebot.backtest.evaluation import gap_risk_kelly_size
        return gap_risk_kelly_size(
            target_risk=target_risk,
            atr_at_entry=0.015,
            price_at_entry=100.0,
            sl_width=0.02,
            max_leverage=3.0,
            gap_atr_multiple=gap_atr_multiple,
        )

    def test_returns_kellysizingresult(self) -> None:
        from tradebot.backtest.evaluation import KellySizingResult
        result = self._call()
        assert isinstance(result, KellySizingResult)

    def test_leverage_non_negative(self) -> None:
        result = self._call()
        assert result.leverage >= 0.0

    def test_larger_gap_multiple_reduces_leverage(self) -> None:
        small = self._call(gap_atr_multiple=1.0)
        large = self._call(gap_atr_multiple=10.0)
        assert small.leverage >= large.leverage

    def test_result_has_expected_fields(self) -> None:
        result = self._call()
        assert hasattr(result, "leverage")
        assert hasattr(result, "expected_risk")
        assert hasattr(result, "gap_risk_factor")
        assert hasattr(result, "cvar_factor")
        assert hasattr(result, "capped")


# ---------------------------------------------------------------------------
# features.blocks — split_feature_blocks
# ---------------------------------------------------------------------------

class TestSplitFeatureBlocks:
    def _make_df(self):
        rng = np.random.default_rng(99)
        cols = (
            ["feat_rsi", "feat_macd", "feat_bb"]          # micro
            + ["feat_vol_meso", "feat_trend_meso"]         # meso
            + ["feat_macro_vix", "feat_macro_funding"]     # macro
        )
        return pd.DataFrame(rng.normal(size=(50, len(cols))), columns=cols)

    def _make_feat_map(self):
        return {
            "micro": ["feat_rsi", "feat_macd", "feat_bb"],
            "meso":  ["feat_vol_meso", "feat_trend_meso"],
            "macro": ["feat_macro_vix", "feat_macro_funding"],
        }

    def test_shapes_correct(self) -> None:
        from tradebot.features.blocks import split_feature_blocks
        X_micro, X_meso, X_macro = split_feature_blocks(self._make_df(), self._make_feat_map())
        assert X_micro.shape == (50, 3)
        assert X_meso.shape  == (50, 2)
        assert X_macro.shape == (50, 2)

    def test_empty_tier_yields_zero_columns(self) -> None:
        from tradebot.features.blocks import split_feature_blocks
        X_micro, X_meso, X_macro = split_feature_blocks(
            self._make_df(), {"micro": ["feat_rsi"], "meso": [], "macro": []}
        )
        assert X_micro.shape[1] == 1
        assert X_meso.shape[1]  == 0
        assert X_macro.shape[1] == 0

    def test_dtype_is_float64(self) -> None:
        from tradebot.features.blocks import split_feature_blocks
        for block in split_feature_blocks(self._make_df(), self._make_feat_map()):
            assert block.dtype == np.float64


# ---------------------------------------------------------------------------
# features.pipeline — derive_feature_map
# ---------------------------------------------------------------------------

class TestDeriveFeatureMap:
    def test_buckets_are_disjoint(self) -> None:
        from tradebot.features.pipeline import derive_feature_map
        df = pd.DataFrame(columns=[
            "feat_rsi", "feat_macd",
            "feat_vol_meso",
            "feat_macro_vix", "feat_btc_macro",
            "close", "volume",
        ])
        fm = derive_feature_map(df)
        micro, meso, macro = set(fm["micro"]), set(fm["meso"]), set(fm["macro"])
        assert micro.isdisjoint(meso) and micro.isdisjoint(macro) and meso.isdisjoint(macro)

    def test_non_feat_columns_excluded(self) -> None:
        from tradebot.features.pipeline import derive_feature_map
        fm = derive_feature_map(pd.DataFrame(columns=["close", "volume", "feat_rsi"]))
        all_feats = fm["micro"] + fm["meso"] + fm["macro"]
        assert "close" not in all_feats and "volume" not in all_feats

    def test_meso_suffix_routing(self) -> None:
        from tradebot.features.pipeline import derive_feature_map
        fm = derive_feature_map(pd.DataFrame(columns=["feat_ema_meso", "feat_rsi"]))
        assert "feat_ema_meso" in fm["meso"]
        assert "feat_rsi"      in fm["micro"]

    def test_macro_prefix_and_suffix_routing(self) -> None:
        from tradebot.features.pipeline import derive_feature_map
        fm = derive_feature_map(pd.DataFrame(columns=[
            "feat_macro_vix", "feat_btc_macro", "feat_rsi"
        ]))
        assert "feat_macro_vix" in fm["macro"]
        assert "feat_btc_macro" in fm["macro"]
        assert "feat_rsi"       in fm["micro"]
