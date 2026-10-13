# tests/unit/test_train_ensemble.py
"""`train/ensemble.py`: `ContextualBanditEnsemble` (LinUCB / Thompson over de modellen van een ensemble).

De agents zijn stubs met een vaste `prob_win`; de bandit-wiskunde draait echt.
Wat getoetst wordt is het contract: kernel, matrix-onderhoud, armgewichten, het
afleiden van de macro-context, de asymmetrische counterfactuals en de
proxy-updates die op trade-close weer worden teruggedraaid.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from tradebot.features.orthogonalize import FeatureOrthogonalizer
from tradebot.train import ensemble as ens_mod
from tradebot.train._scalers import RollingRobustScaler
from tradebot.train.catboost import RegimeCatAgent
from tradebot.train.ensemble import ContextualBanditEnsemble
from tradebot.train.schema_guard import SchemaMismatchError
from tradebot.train.stack import QuantArchitectStack

MAP = {"micro": ["m1", "m2"], "meso": ["s1"], "macro": ["g1", "g2", "g3"]}


def _member(prob: float = 0.6, min_conf_long: float = 0.35) -> RegimeCatAgent:
    agent = RegimeCatAgent({}, model_params={"min_conf_long": min_conf_long})
    agent.feature_order = dict(MAP)
    agent.predict_binary = lambda *a, **kw: {"prob_win": prob, "prob_loss": 1 - prob}   # type: ignore[method-assign]
    return agent


def _ens(n_arms: int = 3, context_dim: int = 3, **kw) -> ContextualBanditEnsemble:
    models = kw.pop("models", [_member(0.3 + 0.2 * k) for k in range(n_arms)])
    kw.setdefault("side", "LONG")
    return ContextualBanditEnsemble(models, meta_feature_map=kw.pop("meta_feature_map", MAP),
                                    context_dim=context_dim, **kw)


@pytest.fixture(autouse=True)
def _seed_global_numpy():
    """`_sample_theta_raw` trekt uit de globale numpy-generator; elke test begint op dezelfde stand."""
    state = np.random.get_state()
    np.random.seed(1234)
    yield
    np.random.set_state(state)


# --------------------------------------------------------------------------- #
# construction, kernel and macro orthogonalisation
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_defaults_and_matrix_shapes(self) -> None:
        ens = _ens()
        assert ens.n_arms == 3 and ens._effective_dim == 3 and ens.stack is None
        assert [m.shape for m in ens.B] == [(3, 3)] * 3 and all(np.array_equal(b, np.eye(3)) for b in ens.B_inv)
        assert ens._lwts is ens._entropy_gate is ens._net_alpha is ens._schema_guard is None

    def test_without_models_there_is_one_arm_and_without_context_no_matrices(self) -> None:
        ens = ContextualBanditEnsemble([], context_dim=0)
        assert ens.n_arms == 1 and ens.B == [] and ens._effective_dim == 0

    def test_a_stack_wires_in_the_blueprint_components(self) -> None:
        stack = QuantArchitectStack.default()
        ens = _ens(stack=stack)
        assert ens._lwts is stack.lw_thompson and ens._entropy_gate is stack.entropy_gate
        assert ens._net_alpha is stack.net_alpha and ens._schema_guard is stack.schema_guard

    def test_the_proxy_decay_is_clipped_to_the_unit_interval(self) -> None:
        assert _ens(proxy_decay=7.0).proxy_decay == 1.0 and _ens(proxy_decay=-1.0).proxy_decay == 0.0


class TestKernel:
    def test_the_linear_kernel_is_the_identity(self) -> None:
        np.testing.assert_array_equal(_ens()._apply_kernel(np.array([[1.0, 2.0, 3.0]])), [1.0, 2.0, 3.0])

    def test_the_rbf_kernel_maps_into_the_rff_space_and_builds_the_map_lazily(self) -> None:
        ens = _ens(kernel="rbf", rff_dim=16)
        assert ens._effective_dim == 16 and ens.B[0].shape == (16, 16)
        feat = ens._apply_kernel(np.ones(3))
        assert feat.shape == (16,)
        first = ens._rff
        ens._apply_kernel(np.ones(3))
        assert ens._rff is first                                      # hergebruikt
        ens.rff_dim = 8
        ens._apply_kernel(np.ones(3))
        assert ens._rff is not first and ens._rff.out_dim == 8        # nieuwe vorm: opnieuw opgebouwd

    def test_without_a_known_context_dimension_the_rbf_kernel_returns_the_raw_vector(self) -> None:
        ens = ContextualBanditEnsemble([_member()], context_dim=0, kernel="rbf")
        np.testing.assert_array_equal(ens._apply_kernel(np.array([1.0, 2.0])), [1.0, 2.0])
        assert ens._rff is None


class TestMacroOrthogonalisation:
    def _fitted(self) -> FeatureOrthogonalizer:
        X = np.random.default_rng(0).normal(size=(200, 3)) * np.array([1.0, 2.0, 3.0])
        orth = FeatureOrthogonalizer(spearman_prefilter=False, n_components=0.99)
        orth.fit_on_train_indices(X, np.arange(120), ["g1", "g2", "g3"])
        return orth

    def test_it_is_the_identity_unless_enabled_with_a_fitted_orthogonalizer(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        orth = self._fitted()
        np.testing.assert_array_equal(_ens(orthogonalizer_macro=orth)._orth_macro(v), v)        # niet aangezet
        np.testing.assert_array_equal(_ens(apply_macro_orth=True)._orth_macro(v), v)             # geen orthogonalizer
        np.testing.assert_array_equal(
            _ens(apply_macro_orth=True, orthogonalizer_macro=FeatureOrthogonalizer())._orth_macro(v), v)  # niet gefit
        assert _ens(apply_macro_orth=True, orthogonalizer_macro=orth)._orth_macro(np.array([])).size == 0

    def test_a_fitted_orthogonalizer_projects_the_context(self) -> None:
        orth = self._fitted()
        v = np.array([0.5, -1.0, 2.0])
        out = _ens(apply_macro_orth=True, orthogonalizer_macro=orth)._orth_macro(v)
        np.testing.assert_allclose(out, orth.transform(v.reshape(1, -1)).flatten())


class TestInitMatrices:
    def test_without_a_dimension_nothing_is_allocated(self) -> None:
        ens = ContextualBanditEnsemble([_member()], context_dim=0)
        ens._init_matrices()
        assert ens.B == []

    def test_the_reward_clipper_follows_the_arm_count(self) -> None:
        ens = _ens(n_arms=4)
        assert ens._reward_clipper.n_arms == 4 and ens._sm_steps_since_recompute == 0


# --------------------------------------------------------------------------- #
# scalers, reset, thompson, gate, net alpha, schema
# --------------------------------------------------------------------------- #
class TestScalers:
    def _scalers(self) -> dict:
        rng = np.random.default_rng(0)
        return {"micro": RollingRobustScaler(window=40).fit(rng.normal(size=(40, 2))),
                "macro": RollingRobustScaler(window=40).fit(rng.normal(size=(40, 3)))}

    def test_updates_need_models_scalers_and_a_feature_map(self) -> None:
        ContextualBanditEnsemble([], meta_feature_map=MAP, meta_scalers=self._scalers()).update_live_scalers(np.ones(2))
        ContextualBanditEnsemble([_member()], meta_feature_map=MAP).update_live_scalers(np.ones(2))
        ContextualBanditEnsemble([_member()], meta_scalers=self._scalers()).update_live_scalers(np.ones(2))

    def test_live_rows_are_aligned_and_fed_to_the_scalers_that_can_update(self) -> None:
        scalers = self._scalers()
        scalers["meso"] = object()
        ens = ContextualBanditEnsemble([_member()], meta_feature_map=MAP, meta_scalers=scalers)
        before = scalers["micro"]._bars_since_calib
        ens.update_live_scalers(np.ones(2), np.ones((1, 1)), np.empty((0, 3)))         # 1D-rij, scaler zonder update, leeg blok
        assert scalers["micro"]._bars_since_calib == before + 1
        assert scalers["macro"]._bars_since_calib == 0

    def test_recalibration_reaches_scalers_that_support_it(self) -> None:
        ContextualBanditEnsemble([_member()]).recalibrate_all_scalers()
        scalers = self._scalers()
        scalers["meso"] = object()
        scalers["micro"].center_ = None
        ContextualBanditEnsemble([_member()], meta_scalers=scalers).recalibrate_all_scalers()
        assert scalers["micro"].center_ is not None


class TestResetProxyState:
    def test_the_ema_and_the_in_flight_shifts_are_cleared(self) -> None:
        ens = _ens()
        ens.update_bandit_proxy(np.array([1.0, 0.5, -0.5]), [0.7, 0.3, 0.6], 0.02, -0.01)
        assert ens._proxy_r_n > 0 and any(s.any() for s in ens._y_proxy_shift)
        ens.reset_proxy_state()
        assert ens._proxy_r_n == 0 and ens._proxy_r_mu == 0.0 and ens._proxy_r_var == 1e-6
        assert not any(s.any() for s in ens._y_proxy_shift)


class TestRawThompson:
    def test_draws_centre_on_theta_hat(self) -> None:
        ens = _ens(v=0.1)
        draws = np.array([ens._sample_theta_raw(np.array([1.0, -1.0, 0.5]), np.eye(3) * 0.01) for _ in range(400)])
        np.testing.assert_allclose(draws.mean(axis=0), [1.0, -1.0, 0.5], atol=0.02)

    def test_a_numerically_singular_covariance_falls_back_to_theta_hat_and_warns_every_tenth_time(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def boom(*a, **kw):
            raise np.linalg.LinAlgError("SVD did not converge")
        monkeypatch.setattr(ens_mod.np.random, "multivariate_normal", boom)
        ens = _ens()
        theta = np.array([0.3, 0.2, 0.1])
        with caplog.at_level(logging.WARNING, logger=ens_mod.logger.name):
            for _ in range(10):
                np.testing.assert_array_equal(ens._sample_theta_raw(theta, np.eye(3)), theta)
        assert ens._singular_cov_count == 10 and caplog.text.count("singular-cov fallbacks") == 1


class TestEntropyGate:
    def test_without_a_stack_every_signal_passes(self) -> None:
        assert _ens().gate_signal(0.5) is True

    def test_with_a_stack_an_undecided_model_is_blocked_and_a_decided_one_passes(self) -> None:
        stack = QuantArchitectStack.default()
        stack.entropy_gate.absolute_floor = 0.3                      # eis: duidelijk besliste modellen
        ens = _ens(stack=stack)
        assert ens.gate_signal(0.99) is True
        assert ens.gate_signal(0.5) is False
        assert ens.gate_signal(7.0) is True                           # kans wordt op 1 geklemd


class TestNetAlpha:
    def test_without_a_stack_there_is_no_reward_to_compute(self) -> None:
        assert _ens().compute_net_alpha_reward(100.0, 101.0, 1.0, 10_000.0, 0.01) is None

    def test_with_a_stack_the_kalman_tracked_eta_is_used(self) -> None:
        stack = QuantArchitectStack.default()
        stack.eta_observer.eta = 0.5
        ens = _ens(stack=stack)
        res = ens.compute_net_alpha_reward(100.0, 101.0, 100.0, 10_000.0, 0.01, side=-1)
        assert stack.net_alpha.eta == 0.5 and res.log_return < 0 and res.slippage > 0


class TestSchemaValidation:
    def test_without_a_stack_it_is_a_no_op(self) -> None:
        _ens().validate_feature_schema(["whatever"])

    def test_with_a_stamped_stack_a_changed_schema_stops_trading(self) -> None:
        stack = QuantArchitectStack.default()
        stack.schema_guard.stamp(["a", "b"])
        ens = _ens(stack=stack)
        ens.validate_feature_schema(["a", "b"])
        with pytest.raises(SchemaMismatchError):
            ens.validate_feature_schema(["b", "a"])


# --------------------------------------------------------------------------- #
# arm weights
# --------------------------------------------------------------------------- #
class TestBanditWeights:
    @pytest.mark.parametrize("context", [None, np.array([]), np.ones(3)])
    def test_without_usable_context_the_weights_are_uniform(self, context) -> None:
        ens = _ens(context_dim=0)
        np.testing.assert_allclose(ens.get_bandit_weights(context), np.ones(3) / 3)

    def test_uniform_when_the_matrices_do_not_match_the_arm_count(self) -> None:
        ens = _ens()
        ens.B_inv = ens.B_inv[:2]
        np.testing.assert_allclose(ens.get_bandit_weights(np.ones(3)), np.ones(3) / 3)

    def test_uniform_when_the_kernel_returns_an_empty_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ens = _ens()
        monkeypatch.setattr(ens, "_apply_kernel", lambda c: np.array([]))
        np.testing.assert_allclose(ens.get_bandit_weights(np.ones(3)), np.ones(3) / 3)

    def test_ucb_prefers_the_arm_with_the_higher_estimate_and_the_more_uncertain_arm(self) -> None:
        ens = _ens(selection="ucb", ucb_alpha=0.5)
        ens.theta_hat[2] = np.array([2.0, 0.0, 0.0])
        w = ens.get_bandit_weights(np.array([1.0, 0.0, 0.0]))
        assert w.argmax() == 2 and w.sum() == pytest.approx(1.0)
        ens.theta_hat[2] = np.zeros(3)
        ens.B_inv[1] = np.eye(3) * 9.0                                # veel onzekerheid -> grote bonus
        assert ens.get_bandit_weights(np.array([1.0, 0.0, 0.0])).argmax() == 1

    def test_thompson_with_and_without_the_ledoit_wolf_sampler_gives_proper_weights(self) -> None:
        for stack in (None, QuantArchitectStack.default()):
            ens = _ens(stack=stack)
            w = ens.get_bandit_weights(np.array([1.0, 0.5, -0.5]))
            assert w.shape == (3,) and w.sum() == pytest.approx(1.0) and (w > 0).all()

    def test_non_finite_scores_are_neutralised(self) -> None:
        ens = _ens(selection="ucb")
        ens.theta_hat[0] = np.array([np.nan, 0.0, 0.0])
        w = ens.get_bandit_weights(np.array([1.0, 0.0, 0.0]))
        assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# central pre-scaling and the full prediction
# --------------------------------------------------------------------------- #
def _scalers_for_map() -> dict:
    rng = np.random.default_rng(0)
    return {"micro": RollingRobustScaler(window=60).fit(rng.normal(size=(60, 2))),
            "meso": RollingRobustScaler(window=60).fit(rng.normal(size=(60, 1))),
            "macro": RollingRobustScaler(window=60).fit(rng.normal(size=(60, 3)))}


class TestPreScaleOnce:
    def test_without_models_scalers_or_map_the_input_is_returned_untouched(self) -> None:
        x = (np.ones((1, 2)), None, None)
        assert _ens(models=[], meta_feature_map=MAP)._pre_scale_once(*x)[0] is x[0]
        assert ContextualBanditEnsemble([_member()], meta_feature_map=MAP)._pre_scale_once(*x)[0] is x[0]
        assert ContextualBanditEnsemble([_member()], meta_scalers={})._pre_scale_once(*x)[0] is x[0]

    def test_each_block_is_aligned_and_scaled_once(self) -> None:
        scalers = _scalers_for_map()
        ens = _ens(meta_scalers=scalers)
        micro = np.array([[0.3, -0.2]])
        out_micro, out_meso, out_macro = ens._pre_scale_once(micro, None, np.ones(3))
        np.testing.assert_allclose(out_micro, scalers["micro"].transform(micro))
        assert out_meso is None and out_macro.shape == (1, 3)

    def test_degenerate_blocks_pass_through_in_the_documented_way(self) -> None:
        scalers = _scalers_for_map()
        scalers["micro"] = object()                                   # scaler zonder transform
        ens = _ens(meta_scalers=scalers, meta_feature_map={"micro": MAP["micro"], "meso": []})
        empty = np.empty((0, 2))
        window = np.stack([np.ones((2, 2)), 2 * np.ones((2, 2))], axis=1)               # (2, 2, 2) -> laatste tijdstap
        out = ens._pre_scale_once(window, np.ones((1, 1)), empty)
        np.testing.assert_array_equal(out[0], 2 * np.ones((2, 2)))      # geen transform: wel uitgelijnd
        np.testing.assert_array_equal(out[1], np.ones((1, 1)))          # geen doelfeatures: de matrix zelf
        assert out[2] is empty                                          # lege invoer blijft zoals hij is

    def test_a_shape_mismatch_is_left_to_the_agent_which_raises_it_itself(self) -> None:
        ens = _ens(meta_scalers=_scalers_for_map())
        wrong = np.ones((1, 5))
        assert ens._pre_scale_once(wrong, None, None)[0] is wrong
        with pytest.raises(RuntimeError, match="Shape mismatch"):
            _member().prepare_inference_data(wrong, None, None, base_feature_map=MAP)


class TestPredictGreybox:
    def test_the_macro_block_becomes_the_bandit_context_and_weights_the_models(self) -> None:
        ens = _ens(meta_scalers=_scalers_for_map())
        out = ens.predict_greybox_strategy(np.ones((1, 2)), np.ones((1, 1)), np.array([0.5, -0.5, 1.0]))
        assert len(out["macro_context_used"]) == 3 and len(out["raw_model_probs"]) == 3
        assert out["prob_win"] == pytest.approx(np.dot(ens.get_bandit_weights(np.zeros(3)), [0.3, 0.5, 0.7]), abs=0.3)
        assert out["side_detected"] == "LONG" and out["entropy_gate_passed"] is True

    def test_pre_scaled_input_skips_the_central_scaling_and_the_macro_scaler(self) -> None:
        ens = _ens(meta_scalers=_scalers_for_map())
        macro = np.array([0.5, -0.5, 1.0])
        out = ens.predict_greybox_strategy(np.ones((1, 2)), None, macro, is_pre_scaled=True)
        np.testing.assert_allclose(out["macro_context_used"], macro)

    def test_without_a_macro_scaler_the_aligned_macro_block_is_the_context(self) -> None:
        ens = _ens(meta_scalers=None)
        out = ens.predict_greybox_strategy(np.ones((1, 2)), None, np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(out["macro_context_used"], [1.0, 2.0, 3.0])
        ens2 = _ens(meta_scalers={"micro": object()})                  # scalers, maar geen macro-scaler
        out2 = ens2.predict_greybox_strategy(np.ones((1, 2)), None, np.array([[1.0, 2.0, 3.0]]), is_pre_scaled=False)
        np.testing.assert_allclose(out2["macro_context_used"], [1.0, 2.0, 3.0])

    def test_a_macro_vector_without_target_features_is_used_when_the_length_matches(self) -> None:
        ens = _ens(meta_feature_map={"micro": MAP["micro"]})
        out = ens.predict_greybox_strategy(np.ones((1, 2)), None, np.array([1.0, 2.0, 3.0]))
        assert out["macro_context_used"] == [1.0, 2.0, 3.0]
        short = ens.predict_greybox_strategy(np.ones((1, 2)), None, np.array([1.0, 2.0]))
        assert short["macro_context_used"] == []

    def test_a_missing_or_empty_macro_block_gives_no_context_and_a_plain_average(self) -> None:
        ens = _ens()
        for macro in (None, np.empty((0, 3))):
            out = ens.predict_greybox_strategy(np.ones((1, 2)), None, macro)
            assert out["macro_context_used"] == [] and out["prob_win"] == pytest.approx(0.5)

    def test_the_first_valid_context_initialises_the_matrices(self) -> None:
        ens = ContextualBanditEnsemble([_member(0.4), _member(0.6)], meta_feature_map=MAP, context_dim=0, side="LONG")
        assert ens.B == []
        ens.predict_greybox_strategy(np.ones((1, 2)), None, np.array([1.0, 2.0, 3.0]), is_pre_scaled=True)
        assert ens.context_dim == 3 and len(ens.B) == 2

    def test_the_macro_block_is_orthogonalised_when_enabled(self) -> None:
        X = np.random.default_rng(0).normal(size=(200, 3)) * np.array([1.0, 2.0, 3.0])
        orth = FeatureOrthogonalizer(spearman_prefilter=False, n_components=0.99)
        orth.fit_on_train_indices(X, np.arange(120), ["g1", "g2", "g3"])
        ens = _ens(context_dim=3, apply_macro_orth=True, orthogonalizer_macro=orth)
        out = ens.predict_greybox_strategy(np.ones((1, 2)), None, np.array([[1.0, 2.0, 3.0]]), is_pre_scaled=True)
        np.testing.assert_allclose(out["macro_context_used"], orth.transform(np.array([[1.0, 2.0, 3.0]])).flatten())

    def test_a_short_ensemble_is_detected_by_a_confidence_floor_of_one(self) -> None:
        out = ContextualBanditEnsemble([_member(0.6, min_conf_long=1.0)]).predict_greybox_strategy(
            np.ones((1, 2)), None, None)
        assert out["side_detected"] == "SHORT"

    def test_the_signal_needs_enough_confidence_and_agreement(self) -> None:
        agree = ContextualBanditEnsemble([_member(0.7), _member(0.7)]).predict_greybox_strategy(np.ones((1, 2)), None, None)
        split = ContextualBanditEnsemble([_member(0.1), _member(0.95)]).predict_greybox_strategy(np.ones((1, 2)), None, None)
        weak = ContextualBanditEnsemble([_member(0.2)]).predict_greybox_strategy(np.ones((1, 2)), None, None)
        assert agree["signal"] == 1 and split["signal"] == 0 and weak["signal"] == 0

    def test_the_entropy_gate_can_veto_a_signal_that_passed_the_thresholds(self) -> None:
        ens = ContextualBanditEnsemble([_member(0.6)], stack=QuantArchitectStack.default(), side="LONG")
        ens._entropy_gate.absolute_floor = 0.1                        # drempel: alleen heel besliste modellen
        out = ens.predict_greybox_strategy(np.ones((1, 2)), None, None, min_confidence=0.3)
        assert out["signal"] == 0 and out["entropy_gate_passed"] is False

    def test_after_fifty_observations_the_threshold_is_a_rolling_percentile(self) -> None:
        ens = ContextualBanditEnsemble([_member(0.5)], dynamic_threshold_pct=80.0)
        for _ in range(51):
            out = ens.predict_greybox_strategy(np.ones((1, 2)), None, None)
        assert out["threshold_used"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# updates
# --------------------------------------------------------------------------- #
CTX = np.array([1.0, 0.5, -0.5])


class TestUpdateBandit:
    def test_all_arms_learn_from_the_asymmetric_counterfactuals(self) -> None:
        ens = _ens(n_arms=2, models=[_member(0.9), _member(0.1)])
        ens.update_bandit(CTX, [0.9, 0.1], reward_long=2.0, reward_short=-1.0)
        assert ens.trade_count == 1 and ens._sm_steps_since_recompute == 1
        # bullish arm krijgt de long-uitkomst, bearish arm de short-uitkomst
        assert ens.theta_hat[0] @ CTX > 0 > ens.theta_hat[1] @ CTX
        assert np.allclose(ens.B_inv[0], ens.B_inv[0].T)

    def test_a_short_ensemble_assigns_the_outcomes_the_other_way_round(self) -> None:
        ens = _ens(n_arms=2, models=[_member(0.9), _member(0.1)], side="SHORT")
        ens.update_bandit(CTX, [0.9, 0.1], reward_long=2.0, reward_short=-1.0)
        assert ens.theta_hat[0] @ CTX < 0 < ens.theta_hat[1] @ CTX

    def test_the_first_context_initialises_the_matrices(self, caplog: pytest.LogCaptureFixture) -> None:
        ens = ContextualBanditEnsemble([_member(), _member()], side="LONG")
        with caplog.at_level(logging.INFO, logger=ens_mod.logger.name):
            ens.update_bandit(CTX, [0.6, 0.4], 1.0, -1.0)
        assert ens.context_dim == 3 and "initialiseer matrices" in caplog.text and ens.trade_count == 1

    def test_a_context_of_the_wrong_length_skips_the_update(self, caplog: pytest.LogCaptureFixture) -> None:
        ens = _ens()
        with caplog.at_level(logging.WARNING, logger=ens_mod.logger.name):
            ens.update_bandit(np.ones(5), [0.5] * 3, 1.0, -1.0)
            ContextualBanditEnsemble([_member()]).update_bandit(np.array([]), [0.5], 1.0, -1.0)
        assert ens.trade_count == 0 and caplog.text.count("context dim mismatch") == 2

    def test_missing_arm_probabilities_default_to_neutral(self, caplog: pytest.LogCaptureFixture) -> None:
        ens = _ens()
        with caplog.at_level(logging.WARNING, logger=ens_mod.logger.name):
            ens.update_bandit(CTX, [0.9], 1.0, -1.0)
            ens.update_bandit(CTX, [0.9, 0.1, 0.5, 0.5, 0.5], 1.0, -1.0)
        assert caplog.text.count("probs ontvangen") == 2 and ens.trade_count == 2

    def test_an_empty_kernel_vector_skips_the_update(self, monkeypatch: pytest.MonkeyPatch,
                                                      caplog: pytest.LogCaptureFixture) -> None:
        ens = _ens()
        monkeypatch.setattr(ens, "_apply_kernel", lambda c: np.array([]))
        with caplog.at_level(logging.WARNING, logger=ens_mod.logger.name):
            ens.update_bandit(CTX, [0.5] * 3, 1.0, -1.0)
        assert "lege feature-vector" in caplog.text and ens.trade_count == 0

    @pytest.mark.parametrize(("taken", "r_long", "r_short"), [
        ("LONG", 1.0, 1.0), ("LONG", 1.0, -1.0), ("SHORT", 1.0, 1.0), ("SHORT", -1.0, 1.0), (None, 1.0, 1.0),
    ])
    def test_slippage_is_taken_from_the_side_that_was_not_traded(self, taken, r_long, r_short) -> None:
        ens = _ens(n_arms=1, models=[_member(0.9)], slippage_bps=100.0)         # 1 % slippage
        ens.update_bandit(CTX, [0.9], r_long, r_short, taken_side=taken)
        assert ens.trade_count == 1 and np.isfinite(ens.theta_hat[0]).all()

    def test_the_slippage_adjustment_changes_the_learned_reward_of_the_counterfactual_arm(self) -> None:
        plain, slipped = (_ens(n_arms=1, models=[_member(0.1)], slippage_bps=bps) for bps in (0.0, 200.0))
        for ens in (plain, slipped):
            ens.update_bandit(CTX, [0.1], reward_long=1.0, reward_short=0.5, taken_side="LONG")   # arm kiest short-uitkomst
        assert slipped.theta_hat[0] @ CTX < plain.theta_hat[0] @ CTX

    def test_an_in_flight_proxy_shift_is_removed_before_the_realised_update(self) -> None:
        ens = _ens(n_arms=2, models=[_member(0.9), _member(0.1)])
        ens.update_bandit_proxy(CTX, [0.9, 0.1], 0.03, -0.02)
        assert any(s.any() for s in ens._y_proxy_shift)
        ens.update_bandit(CTX, [0.9, 0.1], 1.0, -1.0)
        assert not any(s.any() for s in ens._y_proxy_shift)

    def test_a_proxy_shift_of_the_wrong_shape_is_reset(self) -> None:
        ens = _ens()
        ens._y_proxy_shift = []
        ens.update_bandit(CTX, [0.5] * 3, 1.0, -1.0)
        assert len(ens._y_proxy_shift) == 3

    def test_without_an_explicit_side_the_confidence_floor_decides_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ens = ContextualBanditEnsemble([_member(0.9, min_conf_long=1.0)], context_dim=3)
        with caplog.at_level(logging.WARNING, logger=ens_mod.logger.name):
            ens.update_bandit(CTX, [0.9], reward_long=2.0, reward_short=-1.0)
        assert "side niet gezet" in caplog.text and ens.theta_hat[0] @ CTX < 0       # short-model: short-uitkomst
        ens = ContextualBanditEnsemble([], context_dim=3)
        ens.n_arms = 1
        ens.B, ens.B_inv, ens.theta_hat, ens.y = [np.eye(3)], [np.eye(3)], [np.zeros(3)], [np.zeros(3)]
        ens.update_bandit(CTX, [0.9], 1.0, -1.0)
        assert ens.trade_count == 1                                                # zonder modellen: long


class TestUpdateArm:
    def test_an_invalid_arm_index_is_logged_and_ignored(self, caplog: pytest.LogCaptureFixture) -> None:
        ens = _ens()
        with caplog.at_level(logging.ERROR, logger=ens_mod.logger.name):
            ens._update_arm(9, CTX, 1.0)
            ens._update_arm(-1, CTX, 1.0)
        assert caplog.text.count("Invalid arm_index") == 2

    def test_a_vanishing_denominator_is_floored(self) -> None:
        ens = _ens(n_arms=1, models=[_member()])
        ens.B_inv[0] = -np.eye(3) * ens.gamma                          # 1 + x' A_inv x ~ 0 voor |x|^2 = 1
        ens._update_arm(0, np.array([1.0, 0.0, 0.0]), 1.0)
        assert np.isfinite(ens.B_inv[0]).all()
        ens.B_inv[0] = -np.eye(3) * ens.gamma * (1.0 + 2e-13)
        ens._update_arm(0, np.array([1.0, 0.0, 0.0]), 1.0)
        assert np.isfinite(ens.B_inv[0]).all()

    def test_the_sherman_morrison_update_matches_the_exact_inverse(self) -> None:
        ens = _ens(n_arms=1, models=[_member()])
        for x in np.random.default_rng(0).normal(size=(30, 3)):
            ens._update_arm(0, x, 0.1)
        np.testing.assert_allclose(ens.B_inv[0], np.linalg.inv(ens.B[0]), rtol=1e-6, atol=1e-8)


class TestWeightedArmUpdate:
    def test_a_unit_weight_equals_the_plain_update(self) -> None:
        a, b = (_ens(n_arms=1, models=[_member()]) for _ in range(2))
        a._update_arm(0, CTX, 0.7)
        b._update_arm_weighted(0, CTX, 0.7, weight=1.0)
        np.testing.assert_allclose(a.B_inv[0], b.B_inv[0])
        np.testing.assert_allclose(a.theta_hat[0], b.theta_hat[0])

    def test_a_lower_weight_moves_the_arm_less(self) -> None:
        full, half = (_ens(n_arms=1, models=[_member()]) for _ in range(2))
        full._update_arm_weighted(0, CTX, 1.0, 1.0)
        half._update_arm_weighted(0, CTX, 1.0, 0.5)
        assert np.linalg.norm(half.theta_hat[0]) < np.linalg.norm(full.theta_hat[0])

    def test_invalid_arms_and_zero_weights_are_ignored(self) -> None:
        ens = _ens(n_arms=1, models=[_member()])
        before = ens.B_inv[0].copy()
        ens._update_arm_weighted(5, CTX, 1.0)
        ens._update_arm_weighted(0, CTX, 1.0, weight=0.0)
        np.testing.assert_array_equal(ens.B_inv[0], before)

    def test_a_vanishing_denominator_is_floored(self) -> None:
        ens = _ens(n_arms=1, models=[_member()])
        ens.B_inv[0] = -np.eye(3) * ens.gamma
        ens._update_arm_weighted(0, np.array([1.0, 0.0, 0.0]), 1.0, 1.0)
        assert np.isfinite(ens.B_inv[0]).all()
        ens.B_inv[0] = -np.eye(3) * ens.gamma * (1.0 + 2e-13)
        ens._update_arm_weighted(0, np.array([1.0, 0.0, 0.0]), 1.0, 1.0)
        assert np.isfinite(ens.B_inv[0]).all()


# --------------------------------------------------------------------------- #
# numerical maintenance
# --------------------------------------------------------------------------- #
class TestResymmetrise:
    def test_it_only_runs_at_the_reset_interval(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], reset_interval=2, cholesky_interval=0)
        ens.update_bandit(CTX, [0.5], 1.0, -1.0)
        drifted = ens.B_inv[0].copy()
        np.testing.assert_allclose(drifted, np.linalg.inv(ens.B[0]), rtol=1e-6)
        ens.B_inv[0] = ens.B_inv[0] + 1e-3                              # kunstmatige drift
        ens.update_bandit(CTX, [0.5], 1.0, -1.0)                        # trade_count 2: exacte herberekening
        np.testing.assert_allclose(ens.B_inv[0], np.linalg.inv(ens.B[0]), rtol=1e-8)

    def test_a_disabled_interval_or_unknown_dimension_does_nothing(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], reset_interval=0)
        before = ens.B_inv[0].copy()
        ens._maybe_resymmetrize()
        np.testing.assert_array_equal(ens.B_inv[0], before)
        empty = ContextualBanditEnsemble([_member()], context_dim=0, reset_interval=1)
        empty._maybe_resymmetrize()                                      # d == 0: geen matrices

    def test_a_singular_matrix_is_regularised_before_inversion(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], reset_interval=1, cholesky_interval=0)
        ens.B[0] = np.zeros((3, 3))
        ens._maybe_resymmetrize()
        np.testing.assert_allclose(ens.B[0], 1e-4 * np.eye(3))
        assert np.isfinite(ens.B_inv[0]).all()


class TestCholeskyRecompute:
    def test_it_waits_for_the_interval_and_can_be_disabled(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], cholesky_interval=0, reset_interval=0)
        for _ in range(3):
            ens.update_bandit(CTX, [0.5], 1.0, -1.0)
        assert ens._sm_steps_since_recompute == 3
        waiting = _ens(n_arms=1, models=[_member()], cholesky_interval=5, reset_interval=0)
        waiting.update_bandit(CTX, [0.5], 1.0, -1.0)
        assert waiting._sm_steps_since_recompute == 1

    def test_at_the_interval_the_inverse_is_recomputed_exactly_and_the_counter_resets(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], cholesky_interval=2, reset_interval=0)
        ens.update_bandit(CTX, [0.5], 1.0, -1.0)
        ens.B_inv[0] = ens.B_inv[0] + 1e-2
        ens.update_bandit(CTX, [0.5], 1.0, -1.0)
        assert ens._sm_steps_since_recompute == 0
        np.testing.assert_allclose(ens.B_inv[0], np.linalg.inv(ens.B[0]), rtol=1e-6, atol=1e-8)

    def test_without_a_dimension_only_the_counter_resets(self) -> None:
        ens = ContextualBanditEnsemble([_member()], context_dim=0, cholesky_interval=1)
        ens._sm_steps_since_recompute = 1
        ens._maybe_cholesky_recompute()
        assert ens._sm_steps_since_recompute == 0

    def test_a_matrix_that_is_not_positive_definite_falls_back_to_the_pseudo_inverse(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], cholesky_interval=1, reset_interval=0)
        ens.B[0] = -np.eye(3)
        ens._sm_steps_since_recompute = 1
        ens._maybe_cholesky_recompute()
        np.testing.assert_allclose(ens.B_inv[0], np.linalg.pinv(-np.eye(3) + 1e-3 * np.eye(3)))

    def test_a_slightly_indefinite_matrix_is_repaired_by_a_larger_jitter(self) -> None:
        ens = _ens(n_arms=1, models=[_member()], cholesky_interval=1, reset_interval=0)
        ens.B[0] = np.diag([1.0, 1.0, -5e-8])                           # jitter 1e-8 is te weinig, 1e-7 helpt
        ens._sm_steps_since_recompute = 1
        ens._maybe_cholesky_recompute()
        assert np.isfinite(ens.B_inv[0]).all() and ens.B_inv[0][0, 0] == pytest.approx(1.0, rel=1e-3)


class TestDynamicGamma:
    def test_gamma_runs_from_max_in_calm_markets_to_min_in_turbulent_ones(self) -> None:
        ens = _ens(gamma_min=0.95, gamma_max=0.999)
        assert ens.set_gamma_dynamic(0.0) == pytest.approx(0.999)
        assert ens.set_gamma_dynamic(1.0) == pytest.approx(0.95)
        assert ens.set_gamma_dynamic(0.5) == pytest.approx(0.9745)
        assert ens.set_gamma_dynamic(5.0) == pytest.approx(0.95) and ens.gamma == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# proxy updates
# --------------------------------------------------------------------------- #
class TestProxyUpdate:
    def test_a_zero_decay_disables_the_proxy(self) -> None:
        ens = _ens(proxy_decay=0.0)
        ens.update_bandit_proxy(CTX, [0.5] * 3, 0.05, -0.05)
        assert ens._proxy_r_n == 0

    def test_only_y_moves_while_b_and_its_inverse_stay_fixed_until_the_trade_closes(self) -> None:
        ens = _ens(n_arms=2, models=[_member(0.9), _member(0.1)])
        ens.update_bandit(CTX, [0.9, 0.1], 0.5, -0.5)                      # eerst wat echte data
        b_before = [b.copy() for b in ens.B]
        inv_before = [b.copy() for b in ens.B_inv]
        theta_before = [t.copy() for t in ens.theta_hat]
        ens.update_bandit_proxy(CTX, [0.9, 0.1], 0.04, -0.03)
        for k in range(2):
            np.testing.assert_array_equal(ens.B[k], b_before[k])
            np.testing.assert_array_equal(ens.B_inv[k], inv_before[k])
            assert not np.allclose(ens.theta_hat[k], theta_before[k])
        assert ens._proxy_r_n == 2 and ens._proxy_r_mu == pytest.approx(0.04, abs=0.1)

    def test_the_ema_bootstraps_on_the_first_observation_and_ignores_non_finite_ones(self) -> None:
        ens = _ens()
        ens.update_bandit_proxy(CTX, [0.5] * 3, 0.02, float("nan"))
        assert ens._proxy_r_n == 1 and ens._proxy_r_mu == 0.02 and ens._proxy_r_var == 1e-6
        ens.update_bandit_proxy(CTX, [0.5] * 3, 0.04, 0.0)
        assert ens._proxy_r_n == 3 and ens._proxy_r_var > 1e-6

    def test_the_in_flight_shift_is_capped_at_a_fifth_of_the_target_norm(self) -> None:
        ens = _ens(n_arms=1, models=[_member(0.9)], proxy_decay=1.0)
        ens.update_bandit(CTX, [0.9], 0.01, 0.01)                          # kleine y
        for _ in range(30):
            ens.update_bandit_proxy(CTX, [0.9], 5.0, 5.0)
        shift = np.linalg.norm(ens._y_proxy_shift[0])
        assert shift <= 0.2 * np.linalg.norm(ens.y[0]) + 1e-9

    def test_a_short_ensemble_picks_the_other_side_of_the_mark_to_market(self) -> None:
        long_, short = (_ens(n_arms=1, models=[_member(0.9)], side=s) for s in ("LONG", "SHORT"))
        for ens in (long_, short):
            ens.update_bandit_proxy(CTX, [0.9], 0.05, -0.05)
        assert long_.theta_hat[0] @ CTX != pytest.approx(short.theta_hat[0] @ CTX)

    def test_dimension_problems_and_probability_padding_are_handled(self) -> None:
        ens = _ens()
        ens.update_bandit_proxy(np.ones(7), [0.5] * 3, 0.01, 0.01)          # verkeerde lengte
        ens.update_bandit_proxy(np.array([]), [0.5] * 3, 0.01, 0.01)
        assert ens._proxy_r_n == 0
        ens.update_bandit_proxy(CTX, [0.9], 0.01, 0.01)                     # te weinig kansen: aangevuld
        assert ens._proxy_r_n == 2

    def test_the_first_context_initialises_the_matrices(self) -> None:
        ens = ContextualBanditEnsemble([_member(), _member()], side="LONG")
        ens.update_bandit_proxy(CTX, [0.6, 0.4], 0.01, -0.01)
        assert ens.context_dim == 3 and ens._proxy_r_n == 2

    def test_an_empty_kernel_vector_skips_the_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ens = _ens()
        monkeypatch.setattr(ens, "_apply_kernel", lambda c: np.array([]))
        ens.update_bandit_proxy(CTX, [0.5] * 3, 0.01, 0.01)
        assert ens._proxy_r_n == 0

    def test_without_an_explicit_side_the_floor_decides_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        ens = ContextualBanditEnsemble([_member(0.9, min_conf_long=1.0)], context_dim=3)
        with caplog.at_level(logging.WARNING, logger=ens_mod.logger.name):
            ens.update_bandit_proxy(CTX, [0.9], 0.05, -0.05)
        assert "side niet gezet" in caplog.text
        bare = ContextualBanditEnsemble([], context_dim=3)
        bare.n_arms = 1
        bare.B, bare.B_inv, bare.theta_hat, bare.y = [np.eye(3)], [np.eye(3)], [np.zeros(3)], [np.zeros(3)]
        bare.update_bandit_proxy(CTX, [0.9], 0.05, -0.05)
        assert bare._proxy_r_n == 2

    def test_a_shift_buffer_of_the_wrong_shape_is_rebuilt(self) -> None:
        ens = _ens()
        ens._y_proxy_shift = [np.zeros(5) for _ in range(3)]
        ens.update_bandit_proxy(CTX, [0.5] * 3, 0.01, -0.01)
        assert all(s.shape == (3,) for s in ens._y_proxy_shift)

