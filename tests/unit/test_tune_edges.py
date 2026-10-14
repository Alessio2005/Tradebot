# tests/unit/test_tune_edges.py
"""De kleine tune-modules: pruner, sampler-fabriek, zoekruimte en storage."""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from tradebot.tune import pruning, samplers, search_space, storage
from tradebot.utils.failfast import ConfigContractError

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --------------------------------------------------------------------------- #
# pruning
# --------------------------------------------------------------------------- #
class TestMedianPruner:
    def test_the_defaults_are_the_audited_values(self) -> None:
        pruner = pruning.create_median_pruner()
        assert isinstance(pruner, optuna.pruners.MedianPruner)
        assert pruner._n_startup_trials == 20
        assert pruner._n_warmup_steps == 10
        assert pruner._interval_steps == 1

    def test_the_arguments_are_passed_through_and_the_alias_is_the_same_function(self) -> None:
        pruner = pruning.make_pruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=4)
        assert (pruner._n_startup_trials, pruner._n_warmup_steps, pruner._interval_steps) == (3, 2, 4)
        assert pruning.make_pruner is pruning.create_median_pruner


# --------------------------------------------------------------------------- #
# samplers
# --------------------------------------------------------------------------- #
class TestMakeSampler:
    def test_tpe_is_the_default_and_takes_its_configuration(self) -> None:
        sampler = samplers.make_sampler(n_startup_trials=7, multivariate=False)
        assert isinstance(sampler, optuna.samplers.TPESampler)
        assert sampler._n_startup_trials == 7
        assert sampler._multivariate is False

    @pytest.mark.parametrize(
        ("name", "cls"),
        [
            ("gp", optuna.samplers.GPSampler),
            ("qmc", optuna.samplers.QMCSampler),
            ("nsgaii", optuna.samplers.NSGAIISampler),
            ("random", optuna.samplers.RandomSampler),
        ],
    )
    def test_each_name_builds_its_own_sampler(self, name: str, cls: type) -> None:
        assert type(samplers.make_sampler(name)) is cls  # type: ignore[arg-type]

    def test_an_unknown_name_falls_through_to_random_search(self) -> None:
        assert type(samplers.make_sampler("nonsense")) is optuna.samplers.RandomSampler  # type: ignore[arg-type]

    @pytest.mark.parametrize(("name", "missing"), [("gp", "GPSampler"), ("qmc", "QMCSampler")])
    def test_a_missing_sampler_is_refused_not_replaced_by_tpe(
        self, monkeypatch: pytest.MonkeyPatch, name: str, missing: str
    ) -> None:
        stripped = SimpleNamespace(
            **{k: v for k, v in vars(optuna.samplers).items() if k != missing}
        )
        monkeypatch.setattr(optuna, "samplers", stripped)
        with pytest.raises(ConfigContractError, match="NIET"):
            samplers.make_sampler(name)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# search_space
# --------------------------------------------------------------------------- #
class TestSearchSpace:
    def test_the_sampled_configuration_stays_inside_its_bounds(self) -> None:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        seen = []

        def objective(trial: optuna.Trial) -> float:
            h, conf, gamma, cb = search_space.build_search_space(
                trial, [12, 24, 48], min_conf_high=0.42
            )
            seen.append((h, conf, gamma, cb))
            return 0.0

        study.optimize(objective, n_trials=25)
        for h, conf, gamma, cb in seen:
            assert h in (12, 24, 48)
            assert 0.35 <= conf <= 0.42
            assert 0.970 <= gamma <= 0.9995
            assert cb["bandit_gamma"] == gamma
            assert 600 <= cb["iterations"] <= 1500
            assert 3 <= cb["depth"] <= 8
            assert cb["loss_function"] == "Logloss"
            assert "auto_class_weights" not in cb

    def test_the_same_seed_gives_the_same_parameters(self) -> None:
        def run() -> list[tuple]:
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=9))
            out: list[tuple] = []
            study.optimize(
                lambda t: out.append(search_space.sample_catboost_params(t, [1, 2])[:3]) or 0.0,
                n_trials=5,
            )
            return out

        assert run() == run()


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #
class TestMakeStorage:
    def test_an_explicit_url_wins_over_everything(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("OPTUNA_STORAGE", "sqlite:///env.db")
        url = storage.make_storage("postgresql://h/db", sqlite_path=tmp_path / "x.db")
        assert url == "postgresql://h/db"

    def test_the_environment_beats_a_sqlite_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OPTUNA_STORAGE", "sqlite:///env.db")
        with caplog.at_level(logging.INFO, logger=storage.logger.name):
            assert storage.make_storage(sqlite_path=tmp_path / "x.db") == "sqlite:///env.db"
        assert "OPTUNA_STORAGE" in caplog.text

    def test_a_sqlite_path_creates_its_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("OPTUNA_STORAGE", raising=False)
        target = tmp_path / "nested" / "study.db"
        url = storage.make_storage(sqlite_path=target)
        assert url == f"sqlite:///{target.as_posix()}"
        assert target.parent.is_dir()

    def test_without_any_configuration_the_study_is_in_memory_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("OPTUNA_STORAGE", raising=False)
        with caplog.at_level(logging.WARNING, logger=storage.logger.name):
            assert storage.make_storage() is None
        assert "in-memory" in caplog.text
