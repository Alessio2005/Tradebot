# tests/unit/test_tune_objective.py
"""`optuna_objective_binary`: elke tak van de CPCV-trial, met een stub-CV.

De objective krijgt zijn folds van `cv.split(...)`. Een stub levert hier de
indices zelf, zodat elke vroege uitstap (te weinig ES-data, te weinig train na
de purge) en elk afwijkend pad (labels buiten {0,1}, geen kalibratieset, een
mislukte pad-Platt) met opzet bereikbaar is. De modellen zijn echte CatBoosts op
400 rijen: dat kost ongeveer een halve seconde per fold.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tradebot.tune import objective as obj_mod
from tradebot.tune.objective import (
    TunableObjective,
    _f64c,
    _i32c,
    optuna_objective_binary,
)
from tradebot.utils.failfast import DataContractError, TradebotContractError

optuna.logging.set_verbosity(optuna.logging.WARNING)

N = 400
HORIZON = 6


# --------------------------------------------------------------------------- #
# Opzet
# --------------------------------------------------------------------------- #
class _StubCV:
    def __init__(self, folds: list[tuple], n_groups: int = 4) -> None:
        self.folds = folds
        self.n_groups = n_groups

    def split(self, _index: Any, _t1: Any):
        yield from self.folds


class _StubOrthogonalizer:
    """Geeft de eerste twee ruwe kolommen terug en noteert de aanroep."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def fit_on_train_indices(self, raw, train_idx, _cols, prefix):
        self.calls.append((prefix, len(train_idx)))
        return np.asarray(raw)[:, :2].copy(), None


class _StubFFD:
    def __init__(self) -> None:
        self.cleared = 0
        self.fitted: list[tuple[int, str]] = []

    def clear_d_cache(self) -> None:
        self.cleared += 1

    def fit_d_on_train_indices(self, _df, idx, timeframe: str) -> None:
        self.fitted.append((len(idx), timeframe))


@dataclass
class Inputs:
    X1: np.ndarray
    X4: np.ndarray | None
    Xd: np.ndarray | None
    target_store: dict[int, Any]
    timestamps: pd.Series
    cv: _StubCV
    extra: dict[str, Any] = field(default_factory=dict)


def _two_folds() -> list[tuple]:
    return [
        (np.arange(0, 300), np.arange(300, 400), (0, 1)),
        (np.arange(100, 400), np.arange(0, 100), (2, 3)),
    ]


def _inputs(*, y: np.ndarray | None = None, t1: np.ndarray | None = None,
            folds: list[tuple] | None = None, n_groups: int = 4) -> Inputs:
    rng = np.random.default_rng(7)
    y = (rng.random(N) < 0.5).astype(np.int64) if y is None else y
    t1 = np.minimum(np.arange(N) + 3, N - 1) if t1 is None else t1
    returns = rng.normal(0.0, 0.01, N)
    return Inputs(
        X1=rng.normal(size=(N, 3)),
        X4=rng.normal(size=(N, 2)),
        Xd=rng.normal(size=(N, 2)),
        target_store={HORIZON: (y, np.ones(N), returns, t1)},
        timestamps=pd.Series(pd.date_range("2024-01-01", periods=N, freq="h")),
        cv=_StubCV(_two_folds() if folds is None else folds, n_groups),
    )


def _cfg(training: dict | None = None, **top: Any):
    base = {"bootstrap_n": 20, "bootstrap_block_size": 5.0, "optuna_trials": 10,
            "prior_experiments_fallback": 0}
    base.update(training or {})
    return OmegaConf.create({"training": base, **top})


_PARAMS = dict(
    horizon=HORIZON, min_conf=0.35, bandit_gamma=0.99, iterations=600,
    learning_rate=0.02, depth=3, l2_leaf_reg=3.0, random_strength=1.0,
    bagging_temperature=0.5, min_data_in_leaf=50,
)


def _trial(study: optuna.Study | None = None, **overrides: Any) -> optuna.Trial:
    study = study or optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.enqueue_trial({**_PARAMS, **overrides})
    return study.ask()


def _run(inp: Inputs, *, side: str = "LONG", cfg: Any = None, trial: optuna.Trial | None = None,
         **kwargs: Any) -> tuple[float, optuna.Trial]:
    trial = trial or _trial()
    score = optuna_objective_binary(
        trial, cfg or _cfg(), inp.X1, inp.X4, inp.Xd, inp.target_store,
        inp.timestamps, inp.cv, side, **{**inp.extra, **kwargs},
    )
    return score, trial


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_the_numba_helpers_force_dtype_and_contiguity() -> None:
    a = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    f = _f64c(a)
    assert f.dtype == np.float64 and f.flags.c_contiguous
    i = _i32c(np.arange(5, dtype=np.int64)[::2])
    assert i.dtype == np.int32 and i.flags.c_contiguous


# --------------------------------------------------------------------------- #
# Het volledige pad
# --------------------------------------------------------------------------- #
class TestFullTrial:
    def test_a_long_trial_scores_and_stamps_its_attributes(self) -> None:
        score, trial = _run(_inputs())
        assert np.isfinite(score)
        for key in ("sharpe", "log_loss", "trades", "wins", "mt_penalty", "bootstrap_std",
                    "bootstrap_lower_5", "bootstrap_upper_95", "effective_dof",
                    "n_prior_experiments", "best_iteration", "path_platt_n_paths",
                    "path_platt_global_pos_rate"):
            assert key in trial.user_attrs, key
        assert trial.user_attrs["trades"] > 0
        assert 0 <= trial.user_attrs["wins"] <= trial.user_attrs["trades"]

    def test_a_short_trial_runs_the_other_side_of_the_simulation(self) -> None:
        score, trial = _run(_inputs(), side="SHORT")
        assert np.isfinite(score)
        assert trial.user_attrs["trades"] > 0

    def test_the_same_inputs_give_the_same_score(self) -> None:
        a, _ = _run(_inputs())
        b, _ = _run(_inputs())
        assert a == b

    def test_a_threshold_nothing_clears_gives_the_smooth_no_trade_penalty(self) -> None:
        # Een drempel boven elke kans, via een verruimde zoekgrens.
        score, trial = _run(_inputs(), trial=_trial(min_conf=1.5), min_conf_search_high=2.0)
        assert trial.user_attrs["trades"] == 0
        assert trial.user_attrs["sharpe"] == 0.0
        # Zonder trades blijft alleen de gladde frequentiestraf over.
        assert score < -14.0

    def test_a_second_trial_deflates_against_the_first(self) -> None:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.RandomSampler(seed=0))
        inp = _inputs()
        study.enqueue_trial(_PARAMS)
        study.enqueue_trial({**_PARAMS, "min_conf": 0.4})
        study.optimize(
            lambda t: optuna_objective_binary(
                t, _cfg(), inp.X1, inp.X4, inp.Xd, inp.target_store,
                inp.timestamps, inp.cv, "LONG"),
            n_trials=2,
        )
        first, second = study.trials
        assert "sharpe" in first.user_attrs
        assert "mt_penalty" in second.user_attrs

    def test_the_prior_experiment_count_reads_the_configured_repository(self, tmp_path) -> None:
        git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git, "init", "-q"], check=True)
        for msg in ("a", "b", "c"):
            subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", msg], check=True)
        _, trial = _run(_inputs(), cfg=_cfg({"repo_path": str(tmp_path)}))
        assert trial.user_attrs["n_prior_experiments"] == 3

    def test_a_repo_path_that_is_not_a_repository_is_refused_not_defaulted(self, tmp_path) -> None:
        with pytest.raises(DataContractError, match="GEEN fallback"):
            _run(_inputs(), cfg=_cfg({"repo_path": str(tmp_path),
                                      "prior_experiments_fallback": 17}))

    def test_the_portfolio_shrink_penalty_is_off_by_default_and_convex_when_on(self) -> None:
        _, off = _run(_inputs())
        assert "pf_occupancy" not in off.user_attrs
        score_on, on = _run(_inputs(), cfg=_cfg({"optuna_pf_shrink_penalty_coef": 0.5}))
        score_off, _ = _run(_inputs())
        assert on.user_attrs["pf_occupancy"] > 0
        assert on.user_attrs["pf_shrink_penalty"] == pytest.approx(
            0.5 * on.user_attrs["pf_occupancy"] ** 2
        )
        assert score_on == pytest.approx(score_off - on.user_attrs["pf_shrink_penalty"])

    def test_the_trial_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=obj_mod.logger.name):
            _run(_inputs())
        assert "Deflated Obj" in caplog.text
        assert "PathSpecificPlattCalibrator gefit" in caplog.text


# --------------------------------------------------------------------------- #
# Vroege uitstappen
# --------------------------------------------------------------------------- #
class TestEarlyExits:
    def test_too_little_early_stopping_data_ends_the_trial_with_the_penalty(self) -> None:
        inp = _inputs(folds=[(np.arange(0, 40), np.arange(300, 400), (0, 1))])
        score, _ = _run(inp)
        assert score == -1000.0

    def test_a_purge_that_empties_the_training_set_ends_the_trial(self) -> None:
        # Elk event eindigt op de laatste bar, dus na de start van de ES-set.
        inp = _inputs(t1=np.full(N, N - 1))
        score, _ = _run(inp)
        assert score == -1000.0

    def test_no_folds_at_all_falls_back_to_zero_sharpe(self) -> None:
        score, trial = _run(_inputs(folds=[]))
        assert trial.user_attrs["trades"] == 0
        assert trial.user_attrs["sharpe"] == 0.0
        assert "path_platt_n_paths" not in trial.user_attrs
        assert np.isfinite(score)


# --------------------------------------------------------------------------- #
# Klassen, kalibratie en HP-segment
# --------------------------------------------------------------------------- #
class TestLabelsAndCalibration:
    def test_labels_other_than_zero_and_one_train_without_class_weights(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.choice([0, 2], N)
        seen: list[dict] = []
        real = obj_mod.cb.CatBoostClassifier

        def spy(**params: Any):
            seen.append(params)
            return real(**params)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(obj_mod.cb, "CatBoostClassifier", spy)
            score, _ = _run(_inputs(y=y))
        assert np.isfinite(score)
        assert seen and all("class_weights" not in p for p in seen)

    def test_two_classes_in_zero_one_get_inverse_frequency_weights(self) -> None:
        seen: list[dict] = []
        real = obj_mod.cb.CatBoostClassifier

        def spy(**params: Any):
            seen.append(params)
            return real(**params)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(obj_mod.cb, "CatBoostClassifier", spy)
            _run(_inputs())
        for p in seen:
            w = p["class_weights"]
            assert set(w) == {0, 1} and all(v > 0 for v in w.values())

    def test_a_small_platt_set_falls_back_to_the_uncalibrated_model(self) -> None:
        # 100 trainingsrijen: 20 voor ES, daarvan blijven 5 over voor Platt (< 10).
        inp = _inputs(folds=[(np.arange(0, 100), np.arange(300, 400), (0, 1)),
                             (np.arange(100, 200), np.arange(300, 400), (2, 3))])
        score, trial = _run(inp)
        assert np.isfinite(score)
        assert trial.user_attrs["trades"] > 0

    def test_a_single_class_hp_segment_gets_the_bad_logloss_not_a_val_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Train 0..99: ES = 80..99, vroeg 80..89, HP 90..94, Platt 95..99.
        y = (np.random.default_rng(1).random(N) < 0.5).astype(np.int64)
        y[90:95] = 1
        inp = _inputs(y=y, folds=[(np.arange(0, 100), np.arange(300, 400), (0, 1)),
                                  (np.arange(0, 100), np.arange(200, 300), (2, 3))])
        with caplog.at_level(logging.WARNING, logger=obj_mod.logger.name):
            _, trial = _run(inp)
        assert "HP-segment te klein" in caplog.text
        assert trial.user_attrs["log_loss"] == 1000.0

    def test_a_failing_prediction_aborts_the_trial_instead_of_inventing_a_score(self) -> None:
        class _Broken:
            tree_count_ = 1

            def __init__(self, **_p: Any) -> None: ...

            def fit(self, *_a: Any, **_k: Any) -> None: ...

            def predict_proba(self, _X: Any) -> np.ndarray:
                raise RuntimeError("boom")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(obj_mod.cb, "CatBoostClassifier", _Broken)
            with pytest.raises(TradebotContractError, match="Predictie faalde in fold 0"):
                _run(_inputs(folds=[(np.arange(0, 100), np.arange(300, 400), (0, 1))]))

    def test_a_one_column_probability_matrix_scores_as_no_conviction(self) -> None:
        class _OneColumn:
            tree_count_ = 0

            def __init__(self, **_p: Any) -> None: ...

            def fit(self, *_a: Any, **_k: Any) -> None: ...

            def predict_proba(self, X: Any) -> np.ndarray:
                return np.full((len(X), 1), 0.5)

        inp = _inputs(folds=[(np.arange(0, 100), np.arange(300, 400), (0, 1))])
        inp.cv.n_groups = 2
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(obj_mod.cb, "CatBoostClassifier", _OneColumn)
            mp.setattr(obj_mod, "log_loss", lambda *_a, **_k: 0.69)
            score, trial = _run(inp)
        assert trial.user_attrs["trades"] == 0
        assert np.isfinite(score)


# --------------------------------------------------------------------------- #
# Pad-Platt
# --------------------------------------------------------------------------- #
class TestPathPlatt:
    def test_too_few_validation_scores_skip_the_path_calibrator(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        inp = _inputs(folds=[(np.arange(0, 300), np.arange(300, 306), (0, 1))], n_groups=2)
        with caplog.at_level(logging.WARNING, logger=obj_mod.logger.name):
            _, trial = _run(inp)
        assert "PathSpecificPlattCalibrator overgeslagen" in caplog.text
        assert "path_platt_n_paths" not in trial.user_attrs

    def test_a_failing_path_calibrator_aborts_the_trial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Failing:
            def __init__(self, **_k: Any) -> None: ...

            def fit_per_path(self, **_k: Any) -> None:
                raise ValueError("singular")

        monkeypatch.setattr(obj_mod, "PathSpecificPlattCalibrator", _Failing)
        with pytest.raises(TradebotContractError, match="niet stilzwijgend"):
            _run(_inputs())

    def test_a_constant_return_fold_gets_unit_sigma(self) -> None:
        inp = _inputs()
        y, w, _ret, t1 = inp.target_store[HORIZON]
        inp.target_store[HORIZON] = (y, w, np.zeros(N), t1)
        score, trial = _run(inp)
        assert np.isfinite(score)
        assert "path_platt_n_paths" in trial.user_attrs

    def test_a_single_row_validation_fold_gets_unit_sigma(self) -> None:
        inp = _inputs(folds=[(np.arange(0, 300), np.arange(300, 301), (0, 1))], n_groups=2)
        score, _ = _run(inp)
        assert np.isfinite(score)


# --------------------------------------------------------------------------- #
# Feature-blokken
# --------------------------------------------------------------------------- #
class TestFeatureBlocks:
    def test_per_fold_orthogonalisers_see_only_the_training_rows(self) -> None:
        inp = _inputs()
        rng = np.random.default_rng(5)
        orths = {k: _StubOrthogonalizer() for k in ("micro", "meso", "macro")}
        score, _ = _run(
            inp,
            X1_raw=rng.normal(size=(N, 3)), X4_raw=rng.normal(size=(N, 3)),
            Xd_raw=rng.normal(size=(N, 3)),
            orth_micro=orths["micro"], orth_meso=orths["meso"], orth_macro=orths["macro"],
            cols_micro=["a", "b", "c"], cols_meso=["d", "e", "f"], cols_macro=["g", "h", "i"],
        )
        assert np.isfinite(score)
        for name, prefix in (("micro", "micro_pc"), ("meso", "meso_pc"), ("macro", "macro_pc")):
            assert [c[0] for c in orths[name].calls] == [prefix, prefix]
            # De purge laat 5 % van de 240 trainingsrijen weg: nooit meer dan 240.
            assert all(0 < c[1] <= 240 for c in orths[name].calls)

    def test_raw_blocks_with_a_single_column_skip_the_orthogonaliser(self) -> None:
        orth = _StubOrthogonalizer()
        rng = np.random.default_rng(5)
        _run(_inputs(), X1_raw=rng.normal(size=(N, 1)), orth_micro=orth)
        assert orth.calls == []

    def test_empty_meso_and_macro_blocks_are_dropped(self) -> None:
        inp = _inputs()
        inp.X4 = None
        inp.Xd = np.empty((0, 2))
        score, _ = _run(inp)
        assert np.isfinite(score)

    def test_the_ffd_cache_is_reset_per_fold(self) -> None:
        ffd = _StubFFD()
        _run(_inputs(), ffd_engineer=ffd, df_bars=pd.DataFrame({"close": np.ones(N)}))
        assert ffd.cleared == 2
        assert [t for _n, t in ffd.fitted] == ["micro", "micro"]

    def test_catboost_can_be_fed_the_raw_features_while_the_bandit_keeps_the_pca(self) -> None:
        cfg = _cfg(feature_pipeline={"use_pca_for_catboost": False})
        rng = np.random.default_rng(5)
        widths: list[int] = []
        real = obj_mod.cb.Pool

        def spy(data, *a: Any, **k: Any):
            widths.append(np.asarray(data).shape[1])
            return real(data, *a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(obj_mod.cb, "Pool", spy)
            score, _ = _run(
                _inputs(), cfg=cfg,
                X1_raw=rng.normal(size=(N, 5)), X4_raw=rng.normal(size=(N, 4)),
                Xd_raw=np.empty((N, 0)),
            )
        assert np.isfinite(score)
        # micro 5 + meso 4 (macro is leeg en valt weg); train- en eval-pool.
        assert set(widths) == {9}

    def test_without_raw_blocks_the_scaled_blocks_are_used_even_when_raw_is_requested(self) -> None:
        cfg = _cfg(feature_pipeline={"use_pca_for_catboost": False})
        widths: list[int] = []
        real = obj_mod.cb.Pool

        def spy(data, *a: Any, **k: Any):
            widths.append(np.asarray(data).shape[1])
            return real(data, *a, **k)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(obj_mod.cb, "Pool", spy)
            _run(_inputs(), cfg=cfg)
        assert set(widths) == {7}  # 3 + 2 + 2


# --------------------------------------------------------------------------- #
# TunableObjective
# --------------------------------------------------------------------------- #
class TestTunableObjective:
    def test_it_hands_the_trial_to_the_objective_with_everything_else_fixed(self) -> None:
        inp = _inputs()
        tunable = TunableObjective(_cfg(), inp.X1, inp.X4, inp.Xd, inp.target_store,
                                   inp.timestamps, inp.cv, "LONG", spread=0.0005)
        trial = _trial()
        assert tunable(trial) == pytest.approx(_run(inp, trial=_trial(), spread=0.0005)[0])
        assert trial.user_attrs["trades"] > 0

    def test_it_can_be_given_to_study_optimize(self) -> None:
        inp = _inputs()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.RandomSampler(seed=0))
        study.enqueue_trial(_PARAMS)
        study.optimize(
            TunableObjective(_cfg(), inp.X1, inp.X4, inp.Xd, inp.target_store,
                             inp.timestamps, inp.cv, "SHORT"),
            n_trials=1,
        )
        assert study.best_trial.state == optuna.trial.TrialState.COMPLETE
