# tests/unit/test_train_catboost.py
"""`train/catboost.py`: `RegimeCatAgent` (laden, schema-controle, uitlijning, inferentie, training) en `RegimeEnsembleCat`.

CatBoost draait echt, op een klein model (30 iteraties, diepte 3, één thread).
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from tradebot.train import catboost as cbmod
from tradebot.train._scalers import RollingRobustScaler
from tradebot.train.catboost import RegimeCatAgent, RegimeEnsembleCat
from tradebot.train.schema_guard import FeatureSchemaGuard

PARAMS = {"iterations": 30, "depth": 3, "learning_rate": 0.1, "thread_count": 1, "random_seed": 0}
ORDER = {"micro": ["m1", "m2", "m3", "m4"], "meso": ["s1", "s2"], "macro": ["g1"]}


def _data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    micro, meso, macro = rng.normal(size=(n, 4)), rng.normal(size=(n, 2)), rng.normal(size=(n, 1))
    y = ((micro[:, 0] + 0.5 * meso[:, 0] + rng.normal(0, 0.7, n)) > 0).astype(int)
    return micro, meso, macro, y


def _agent(**kw) -> RegimeCatAgent:
    agent = RegimeCatAgent({}, model_params={**PARAMS, **kw.pop("params", {})}, **kw)
    agent.feature_order = dict(ORDER)
    return agent


# --------------------------------------------------------------------------- #
# construction and resource loading
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_model_params_override_the_defaults_and_the_confidence_keys_are_split_off(self) -> None:
        agent = RegimeCatAgent({}, model_params={**PARAMS, "min_conf_long": 0.4, "min_conf_short": 0.6,
                                                 "min_conf": 9, "horizon": 12})
        assert agent.params["iterations"] == 30 and agent.params["auto_class_weights"] == "Balanced"
        assert (agent.min_conf_long, agent.min_conf_short) == (0.4, 0.6)
        assert "min_conf" not in agent.params and "horizon" not in agent.params
        assert agent.model is None and not agent.is_trained and len(agent.git_hash) >= 7

    def test_without_model_params_the_defaults_stay(self) -> None:
        agent = RegimeCatAgent({}, symbol="BTCUSDT", side="LONG")
        assert agent.params["iterations"] == 1000 and agent.min_conf_long == 0.35
        assert agent.symbol == "BTCUSDT" and agent.side == "LONG" and agent.feature_schema == "raw"


class TestLoadResources:
    def _dump(self, path: Path, obj) -> None:
        joblib.dump(obj, path)

    def test_the_saved_models_dir_is_read_on_construction_and_prod_scalers_win(self, tmp_path: Path) -> None:
        std, prod = RollingRobustScaler(window=10), RollingRobustScaler(window=20)
        self._dump(tmp_path / "feature_order_BTC.pkl", ORDER)
        self._dump(tmp_path / "scaler_micro_BTC.pkl", std)
        self._dump(tmp_path / "scaler_micro_BTC_prod.pkl", {"scaler": prod})        # dict-verpakt
        self._dump(tmp_path / "scaler_meso_BTC.pkl", std)                             # alleen standaard
        agent = RegimeCatAgent({"machine": {"saved_models_dir": str(tmp_path)}}, symbol="BTC")
        assert agent.feature_order == ORDER
        assert agent.scaler_micro.window == 20 and agent.scaler_meso.window == 10
        assert agent.scaler_macro is None

    def test_without_a_symbol_the_unsuffixed_files_are_used(self, tmp_path: Path) -> None:
        self._dump(tmp_path / "scaler_macro.pkl", RollingRobustScaler(window=5))
        agent = RegimeCatAgent({"machine": {"saved_models_dir": str(tmp_path)}})
        assert agent.scaler_macro.window == 5 and agent.feature_order is None


# --------------------------------------------------------------------------- #
# schema guard
# --------------------------------------------------------------------------- #
class TestSchema:
    def test_stamping_needs_trained_feature_names_and_a_stack_with_a_guard(self) -> None:
        stack = SimpleNamespace(schema_guard=FeatureSchemaGuard())
        _agent().stamp_schema_into_stack(stack)                       # geen namen: no-op
        assert stack.schema_guard.stamped is None
        _agent(feature_names_trained=["a"]).stamp_schema_into_stack(SimpleNamespace())     # geen guard: no-op
        _agent(feature_names_trained=["a", "b"]).stamp_schema_into_stack(stack)
        assert stack.schema_guard.stamped.feature_count == 2
        stack.schema_guard.check(["a", "b"])

    def test_a_schema_mismatch_names_the_remedy(self) -> None:
        with pytest.raises(ValueError, match="use_pca_for_catboost=True"):
            _agent(feature_schema="raw").validate_schema("pca")
        with pytest.raises(ValueError, match="use_pca_for_catboost=False"):
            _agent(feature_schema="pca").validate_schema("raw")

    def test_feature_names_must_match_in_content_and_order(self) -> None:
        agent = _agent(feature_names_trained=["a", "b", "c"])
        agent.validate_schema("raw", ["a", "b", "c"])
        agent.validate_schema("raw")                                  # geen runtime-namen: niets te vergelijken
        with pytest.raises(ValueError, match="FeatureNameMismatch"):
            agent.validate_schema("raw", ["a", "c", "b"])
        long = _agent(feature_names_trained=[f"o{i}" for i in range(8)])
        with pytest.raises(ValueError, match=r"\.\.\."):
            long.validate_schema("raw", [f"n{i}" for i in range(8)])


# --------------------------------------------------------------------------- #
# alignment and inference data
# --------------------------------------------------------------------------- #
class TestAlignment:
    def test_empty_input_gives_a_zero_matrix_of_the_target_width(self) -> None:
        agent = _agent()
        out = agent._align_input_custom(np.empty((3, 0)), ["a", "b"], "micro")
        assert out.shape == (3, 2) and out.dtype == np.float32
        assert agent._align_input_custom(None, ["a"], "micro").shape == (1, 1)           # type: ignore[arg-type]
        assert agent._align_input_custom(None, [], "micro").shape == (1, 0)              # type: ignore[arg-type]

    def test_no_target_features_gives_zero_columns(self) -> None:
        assert _agent()._align_input_custom(np.ones((4, 3)), [], "micro").shape == (4, 0)

    def test_a_matching_width_passes_through_and_a_mismatch_is_an_error(self) -> None:
        agent = _agent()
        x = np.ones((2, 2))
        assert agent._align_input_custom(x, ["a", "b"], "micro") is x
        with pytest.raises(RuntimeError, match="Shape mismatch in \\[meso\\]"):
            agent._align_input_custom(x, ["a", "b", "c"], "meso")


class TestPrepareInferenceData:
    def test_the_three_blocks_are_stacked_in_feature_order(self) -> None:
        micro, meso, macro, _ = _data(5)
        out = _agent().prepare_inference_data(micro, meso, macro)
        assert out.shape == (5, 7)
        np.testing.assert_array_equal(out[:, :4], micro)

    def test_the_local_feature_order_selects_and_reorders_columns(self) -> None:
        micro, *_ = _data(5)
        agent = _agent()
        agent.feature_order = {"micro": ["m3", "m1"]}
        out = agent.prepare_inference_data(micro, None, None, base_feature_map={"micro": ORDER["micro"]})
        np.testing.assert_array_equal(out, micro[:, [2, 0]])
        agent.feature_order = {"micro": ["unknown"]}
        assert agent.prepare_inference_data(micro, None, None, base_feature_map={"micro": ORDER["micro"]}).shape == (5, 0)

    def test_without_a_local_order_the_aligned_block_is_returned_as_is(self) -> None:
        micro, *_ = _data(5)
        agent = _agent()
        agent.feature_order = None
        out = agent.prepare_inference_data(micro, None, None, base_feature_map={"micro": ORDER["micro"]})
        np.testing.assert_array_equal(out, micro)

    def test_data_frames_and_three_dimensional_windows_are_accepted(self) -> None:
        micro, *_ = _data(6)
        agent = _agent()
        frame = pd.DataFrame(micro, columns=ORDER["micro"])
        np.testing.assert_array_equal(agent.prepare_inference_data(frame, None, None), micro)
        window = np.stack([micro, micro + 1.0], axis=1)                # (6, 2, 4): laatste tijdstap telt
        np.testing.assert_array_equal(agent.prepare_inference_data(window, None, None), micro + 1.0)

    def test_blocks_without_target_features_or_data_are_skipped(self) -> None:
        micro, *_ = _data(4)
        agent = _agent()
        agent.feature_order = {"micro": ORDER["micro"]}                # meso/macro hebben geen doelfeatures
        out = agent.prepare_inference_data(micro, np.ones((4, 2)), np.ones((4, 1)))
        assert out.shape == (4, 4)
        assert _agent().prepare_inference_data(None, None, None).shape == (1, 0)
        agent.feature_order = {}
        assert agent.prepare_inference_data(micro, None, None).shape == (4, 0)

    def test_a_block_without_rows_gives_an_empty_matrix(self) -> None:
        assert _agent().prepare_inference_data(np.empty((0, 4)), None, None).shape == (0, 0)

    def test_the_scaler_is_applied_unless_the_data_is_pre_scaled(self) -> None:
        micro, *_ = _data(50)
        agent = _agent()
        agent.scaler_micro = RollingRobustScaler(window=50).fit(micro)
        scaled = agent.prepare_inference_data(micro, None, None)
        assert not np.allclose(scaled, micro)
        np.testing.assert_array_equal(agent.prepare_inference_data(micro, None, None, is_pre_scaled=True), micro)


# --------------------------------------------------------------------------- #
# predict_binary
# --------------------------------------------------------------------------- #
class TestPredictBinary:
    def test_without_a_model_the_answer_is_a_certain_loss(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=cbmod.logger.name):
            assert _agent().predict_binary(np.ones((1, 4))) == {"prob_win": 0.0, "prob_loss": 1.0}
        assert "model is None" in caplog.text

    def test_input_without_usable_columns_is_a_certain_loss(self) -> None:
        agent = _agent()
        agent.model = SimpleNamespace(predict_proba=lambda X: np.array([[0.3, 0.7]]))
        agent.feature_order = {}
        assert agent.predict_binary(np.ones((1, 4)))["prob_win"] == 0.0

    def test_two_class_and_single_class_probabilities_are_both_complete(self) -> None:
        agent = _agent()
        agent.model = SimpleNamespace(predict_proba=lambda X: np.array([[0.3, 0.7]]))
        assert agent.predict_binary(np.ones((1, 4))) == {"prob_win": 0.7, "prob_loss": 0.3}
        agent.model = SimpleNamespace(predict_proba=lambda X: np.array([[0.8]]))
        out = agent.predict_binary(np.ones((1, 4)))
        assert out["prob_win"] == 0.8 and out["prob_loss"] == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _calib(n: int, seed: int = 5):
    micro, meso, macro, y = _data(n, seed)
    return {"calib_x_micro": micro, "calib_x_meso": meso, "calib_x_macro": macro, "calib_y": y}


class TestTrain:
    def test_a_single_class_target_leaves_the_agent_untrained(self) -> None:
        micro, meso, macro, _ = _data(100)
        agent = _agent().train(micro, np.ones(100), meso, macro)
        assert agent.model is None and not agent.is_trained

    def test_without_calibration_data_a_balanced_model_is_fitted(self) -> None:
        micro, meso, macro, y = _data()
        agent = _agent().train(micro, y, meso, macro)
        assert agent.is_trained and agent.model.get_params()["auto_class_weights"] == "Balanced"
        out = agent.predict_binary(micro[:1], meso[:1], macro[:1])
        assert 0.0 <= out["prob_win"] <= 1.0 and out["prob_win"] + out["prob_loss"] == pytest.approx(1.0)

    def test_sample_weights_reach_the_model(self) -> None:
        micro, meso, macro, y = _data()
        weights = np.where(y == 1, 5.0, 1.0)
        agent = _agent().train(micro, y, meso, macro, sample_weight=weights)
        assert agent.is_trained

    def test_enough_calibration_data_gives_a_platt_calibrated_model_on_the_natural_prior(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        micro, meso, macro, y = _data()
        with caplog.at_level(logging.INFO, logger=cbmod.logger.name):
            agent = _agent().train(micro, y, meso, macro, **_calib(300))
        assert type(agent.model).__name__ == "CalibratedClassifierCV" and "Platt scaling succesvol" in caplog.text
        assert "auto_class_weights" not in agent.model.estimator.estimator.get_params() or \
            agent.model.estimator.estimator.get_params()["auto_class_weights"] is None
        proba = agent.model.predict_proba(agent.prepare_inference_data(micro[:20], meso[:20], macro[:20]))
        assert proba.shape == (20, 2) and np.allclose(proba.sum(axis=1), 1.0)

    def test_calibration_weights_are_used_for_early_stopping(self) -> None:
        micro, meso, macro, y = _data()
        calib = _calib(300)
        agent = _agent().train(micro, y, meso, macro, calib_w=np.ones(300), **calib)
        assert agent.is_trained

    def test_a_calibration_set_that_is_too_small_falls_back_to_early_stopping_without_platt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        micro, meso, macro, y = _data()
        with caplog.at_level(logging.WARNING, logger=cbmod.logger.name):
            agent = _agent().train(micro, y, meso, macro, calib_w=np.ones(30), **_calib(30))
        assert type(agent.model).__name__ == "CatBoostClassifier" and "Calib-set te klein" in caplog.text
        assert agent.is_trained

    def test_a_single_class_calibration_set_is_ignored(self) -> None:
        micro, meso, macro, y = _data()
        calib = _calib(100)
        calib["calib_y"] = np.zeros(100, dtype=int)
        agent = _agent().train(micro, y, meso, macro, **calib)
        assert agent.model.get_params()["auto_class_weights"] == "Balanced"

    def test_pre_scaled_blocks_are_stacked_as_given(self) -> None:
        micro, meso, macro, y = _data()
        agent = _agent()
        agent.train(micro, y, meso, macro[:, 0], is_pre_scaled=True)       # 1D-blok wordt een kolom
        assert agent.is_trained and agent.model.feature_names_ is not None and len(agent.model.feature_names_) == 7
        only_micro = _agent().train(micro, y, None, None, is_pre_scaled=True, **{
            k: v for k, v in _calib(300).items() if k in {"calib_x_micro", "calib_y"}})
        assert only_micro.is_trained

    def test_pre_scaled_input_without_any_block_gives_an_empty_matrix_and_a_catboost_error(self) -> None:
        agent = _agent()
        with pytest.raises(Exception):  # noqa: B017 - CatBoost weigert een matrix zonder kolommen
            agent.train(np.empty((50, 0)), np.tile([0, 1], 25), None, None, is_pre_scaled=True)


# --------------------------------------------------------------------------- #
# RegimeEnsembleCat
# --------------------------------------------------------------------------- #
def _member(prob: float, min_conf_long: float = 0.35) -> RegimeCatAgent:
    agent = RegimeCatAgent({}, model_params={"min_conf_long": min_conf_long})
    agent.feature_order = dict(ORDER)
    agent.predict_binary = lambda *a, **kw: {"prob_win": prob, "prob_loss": 1 - prob}     # type: ignore[method-assign]
    return agent


def _scalers() -> dict:
    micro, meso, macro, _ = _data(60)
    return {"micro": RollingRobustScaler(window=60).fit(micro),
            "meso": RollingRobustScaler(window=60).fit(meso),
            "macro": RollingRobustScaler(window=60).fit(macro)}


class TestEnsembleScalers:
    def test_updates_need_models_scalers_and_a_feature_map(self) -> None:
        micro, *_ = _data(3)
        RegimeEnsembleCat([], meta_feature_map=ORDER, meta_scalers=_scalers()).update_live_scalers(micro)
        RegimeEnsembleCat([_member(0.5)], meta_feature_map=ORDER).update_live_scalers(micro)
        RegimeEnsembleCat([_member(0.5)], meta_scalers=_scalers()).update_live_scalers(micro)

    def test_live_rows_are_aligned_and_fed_to_each_scaler(self) -> None:
        scalers = _scalers()
        before = scalers["micro"]._bars_since_calib
        ens = RegimeEnsembleCat([_member(0.5)], meta_feature_map=ORDER, meta_scalers=scalers)
        ens.update_live_scalers(np.ones(4), np.ones((2, 2)), None)                   # 1D-rij, twee rijen, ontbrekend blok
        assert scalers["micro"]._bars_since_calib == before + 1
        assert scalers["meso"]._bars_since_calib == 2 and scalers["macro"]._bars_since_calib == 0

    def test_empty_blocks_and_scalers_without_update_are_skipped(self) -> None:
        scalers = _scalers()
        scalers["macro"] = object()                                                   # geen update-methode
        ens = RegimeEnsembleCat([_member(0.5)], meta_feature_map=ORDER, meta_scalers=scalers)
        ens.update_live_scalers(np.empty((0, 4)), None, np.ones((1, 1)))
        assert scalers["micro"]._bars_since_calib == 0

    def test_recalibration_reaches_every_scaler_that_supports_it(self) -> None:
        RegimeEnsembleCat([_member(0.5)]).recalibrate_all_scalers()                   # geen scalers: no-op
        scalers = _scalers()
        scalers["meso"] = object()
        scalers["micro"].center_ = None
        RegimeEnsembleCat([_member(0.5)], meta_scalers=scalers).recalibrate_all_scalers()
        assert scalers["micro"].center_ is not None


class TestEnsemblePrediction:
    def test_probabilities_are_averaged_and_the_spread_is_the_uncertainty(self) -> None:
        ens = RegimeEnsembleCat([_member(0.6), _member(0.8)])
        out = ens.predict_greybox_strategy(np.ones((1, 4)), None, None, is_pre_scaled=True)
        assert out["prob_win"] == pytest.approx(0.7) and out["uncertainty"] == pytest.approx(0.1)
        assert out["signal"] == 1 and out["threshold_used"] == 0.33 and out["side_detected"] == "LONG"

    def test_a_short_model_is_recognised_by_a_confidence_floor_of_one(self) -> None:
        out = RegimeEnsembleCat([_member(0.6, min_conf_long=1.0)]).predict_greybox_strategy(
            np.ones((1, 4)), None, None, is_pre_scaled=True)
        assert out["side_detected"] == "SHORT"

    def test_too_uncertain_or_too_weak_a_vote_gives_no_signal(self) -> None:
        split = RegimeEnsembleCat([_member(0.1), _member(0.9)]).predict_greybox_strategy(
            np.ones((1, 4)), None, None, is_pre_scaled=True)
        weak = RegimeEnsembleCat([_member(0.2)]).predict_greybox_strategy(
            np.ones((1, 4)), None, None, is_pre_scaled=True)
        assert split["signal"] == 0 and weak["signal"] == 0
        assert RegimeEnsembleCat([_member(0.2)]).predict_greybox_strategy(
            np.ones((1, 4)), None, None, is_pre_scaled=True, min_confidence=0.1)["signal"] == 1

    def test_after_fifty_observations_the_threshold_is_a_rolling_percentile_of_the_own_history(self) -> None:
        ens = RegimeEnsembleCat([_member(0.5)], dynamic_threshold_pct=80.0)
        for _ in range(50):
            ens.predict_greybox_strategy(np.ones((1, 4)), None, None, is_pre_scaled=True)
        out = ens.predict_greybox_strategy(np.ones((1, 4)), None, None, is_pre_scaled=True)
        assert out["threshold_used"] == pytest.approx(0.5) and out["signal"] == 1
        stricter = RegimeEnsembleCat([_member(0.5)])
        stricter._prob_history.extend([0.9] * 60)
        assert stricter.predict_greybox_strategy(np.ones((1, 4)), None, None, is_pre_scaled=True)["signal"] == 0

    def test_raw_input_is_scaled_once_centrally_and_handed_over_pre_scaled(self) -> None:
        micro, meso, macro, _ = _data(60)
        seen = {}

        def spy(*args, **kwargs):
            seen["pre_scaled"] = kwargs["is_pre_scaled"]
            seen["micro"] = args[0]
            return {"prob_win": 0.6, "prob_loss": 0.4}

        member = _member(0.6)
        member.predict_binary = spy                                                  # type: ignore[method-assign]
        scalers = _scalers()
        ens = RegimeEnsembleCat([member], meta_feature_map=ORDER, meta_scalers=scalers)
        ens.predict_greybox_strategy(pd.DataFrame(micro[:2], columns=ORDER["micro"]),
                                     np.stack([meso[:2], meso[:2]], axis=1), macro[0])
        assert seen["pre_scaled"] is True
        np.testing.assert_allclose(seen["micro"], scalers["micro"].transform(micro[:2]), rtol=1e-5)

    def test_degenerate_blocks_are_handled_while_scaling(self) -> None:
        member = _member(0.6)
        calls = []
        member.predict_binary = lambda *a, **kw: calls.append(a) or {"prob_win": 0.6, "prob_loss": 0.4}   # type: ignore[method-assign]
        ens = RegimeEnsembleCat([member], meta_feature_map={"micro": ORDER["micro"], "meso": []},
                                meta_scalers={"micro": object()})
        ens.predict_greybox_strategy(np.ones((1, 4)), np.ones((1, 2)), np.empty((0, 1)))
        micro_arg, meso_arg, macro_arg = calls[0]
        np.testing.assert_array_equal(micro_arg, np.ones((1, 4)))       # scaler zonder transform: ongeschaald
        np.testing.assert_array_equal(meso_arg, np.ones((1, 2)))        # geen doelfeatures: de matrix zelf
        assert macro_arg.size == 0                                      # lege invoer blijft leeg
        calls.clear()
        ens.predict_greybox_strategy(None, None, None)
        assert calls[0] == (None, None, None)
