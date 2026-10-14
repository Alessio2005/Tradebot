# tests/unit/test_train_small_edges.py
"""De kleine `train/`-modules: bandit-helpers, Thompson, seeding, stack, checkpoints, schema- en entropiepoort."""
from __future__ import annotations

import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tradebot.train import seeded
from tradebot.train._bandit_helpers import _RandomFourierFeatures, _RewardClipper
from tradebot.train.checkpoints import (
    FoldCheckpoint,
    list_checkpoints,
    load_fold_checkpoint,
    save_fold_checkpoint,
)
from tradebot.train.schema_guard import (
    EntropyGate,
    FeatureSchemaGuard,
    SchemaMismatchError,
)
from tradebot.train.seeded import SeedConfig, derive_fold_seed, seed_everything
from tradebot.train.stack import QuantArchitectStack
from tradebot.train.thompson import LedoitWolfThompsonSampler
from tradebot.utils.failfast import DataContractError

# --------------------------------------------------------------------------- #
# _bandit_helpers
# --------------------------------------------------------------------------- #


class TestRandomFourierFeatures:
    def test_the_inner_product_approximates_the_rbf_kernel(self) -> None:
        rff = _RandomFourierFeatures(in_dim=3, out_dim=8192, sigma=1.5, seed=1)
        x, y = np.array([0.2, -0.4, 0.9]), np.array([0.5, 0.1, 0.4])
        approx = float(rff.transform(x) @ rff.transform(y))
        exact = float(np.exp(-np.sum((x - y) ** 2) / (2 * 1.5**2)))
        assert approx == pytest.approx(exact, abs=0.05)
        assert rff.transform(x).shape == (8192,)

    def test_a_wrongly_sized_input_is_padded_or_truncated_instead_of_crashing(self) -> None:
        rff = _RandomFourierFeatures(in_dim=3, out_dim=16, seed=2)
        np.testing.assert_allclose(rff.transform([1.0, 2.0]), rff.transform([1.0, 2.0, 0.0]))
        np.testing.assert_allclose(rff.transform([1.0, 2.0, 3.0, 99.0]), rff.transform([1.0, 2.0, 3.0]))

    def test_a_non_positive_bandwidth_falls_back_to_one(self) -> None:
        assert _RandomFourierFeatures(2, 4, sigma=-3.0).sigma == 1.0

    def test_the_same_seed_gives_the_same_map(self) -> None:
        a, b = _RandomFourierFeatures(2, 8, seed=5), _RandomFourierFeatures(2, 8, seed=5)
        np.testing.assert_array_equal(a.W, b.W)


class TestRewardClipper:
    def test_during_warmup_only_a_hard_bound_applies(self) -> None:
        clip = _RewardClipper(n_arms=2)
        assert clip.clip(0, 500.0) == 10.0 and clip.clip(0, -500.0) == -10.0
        assert clip.clip(0, 0.5) == 0.5

    def test_after_warmup_an_outlier_is_cut_at_mean_plus_k_sigma(self) -> None:
        clip = _RewardClipper(n_arms=1, sigma_clip=3.0)
        rng = np.random.default_rng(0)
        for r in rng.normal(0.0, 1.0, 60):
            clip.clip(0, float(r))
        out = clip.clip(0, 1_000.0)
        assert out == pytest.approx(clip.mean[0] + 3.0 * np.sqrt(clip.var[0])) and out < 1_000.0

    def test_non_finite_and_unknown_arm_rewards_are_handled(self) -> None:
        clip = _RewardClipper(n_arms=2)
        assert clip.clip(0, float("nan")) == 0.0 and clip.clip(1, float("inf")) == 0.0
        assert clip.clip(7, 123.0) == 123.0 and clip.clip(-1, 5.0) == 5.0

    def test_resize_resets_the_state_only_when_the_arm_count_changes(self) -> None:
        clip = _RewardClipper(n_arms=2)
        clip.clip(0, 1.0)
        clip.resize(2)
        assert clip.warmup[0] == 1
        clip.resize(0)                                          # minimaal één arm
        assert clip.n_arms == 1 and clip.warmup.tolist() == [0]


# --------------------------------------------------------------------------- #
# Thompson
# --------------------------------------------------------------------------- #
class TestLedoitWolfThompson:
    THETA = np.array([0.1, -0.2, 0.3])
    B_INV = np.diag([0.5, 1.0, 2.0])

    def test_draws_centre_on_theta_hat_and_report_a_shrinkage_in_the_unit_interval(self) -> None:
        sampler = LedoitWolfThompsonSampler(v=1.0, rng=np.random.default_rng(3))
        draws, deltas = zip(*(sampler.sample_theta(self.THETA, self.B_INV * 0.01) for _ in range(300)),
                            strict=True)
        np.testing.assert_allclose(np.mean(draws, axis=0), self.THETA, atol=0.03)
        assert all(0.0 <= d <= 1.0 for d in deltas)

    def test_the_same_seed_gives_the_same_draw(self) -> None:
        a = LedoitWolfThompsonSampler(rng=np.random.default_rng(7)).sample_theta(self.THETA, self.B_INV)
        b = LedoitWolfThompsonSampler(rng=np.random.default_rng(7)).sample_theta(self.THETA, self.B_INV)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1] == b[1]

    def test_an_empty_parameter_vector_gives_an_empty_draw(self) -> None:
        theta, delta = LedoitWolfThompsonSampler().sample_theta(np.array([]), np.zeros((0, 0)))
        assert theta.size == 0 and delta == 0.0

    def test_an_indefinite_covariance_is_repaired_by_eigenvalue_clipping(self) -> None:
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])          # eigenwaarden 3 en -1
        theta, delta = LedoitWolfThompsonSampler(rng=np.random.default_rng(1)).sample_theta(
            np.zeros(2), indefinite)
        assert theta.shape == (2,) and np.isfinite(theta).all() and 0.0 <= delta <= 1.0

    def test_an_unrepairable_covariance_is_an_error_and_never_a_greedy_fallback(self) -> None:
        with pytest.raises(DataContractError, match="singulier"):
            LedoitWolfThompsonSampler().sample_theta(np.zeros(2), np.full((2, 2), np.nan))

    def test_a_failing_mvn_draw_is_an_error_and_never_a_greedy_fallback(self) -> None:
        sampler = LedoitWolfThompsonSampler(rng=np.random.default_rng(0))

        class _BrokenRng:
            standard_normal = staticmethod(np.random.default_rng(0).standard_normal)

            @staticmethod
            def multivariate_normal(*a, **kw):
                raise np.linalg.LinAlgError("svd did not converge")

        sampler._rng = _BrokenRng()                              # type: ignore[assignment]
        with pytest.raises(DataContractError, match="MVN-trekking"):
            sampler.sample_theta(self.THETA, self.B_INV)


# --------------------------------------------------------------------------- #
# seeded
# --------------------------------------------------------------------------- #
class TestSeeding:
    def test_the_seed_config_refuses_negative_seeds_and_zero_folds(self) -> None:
        with pytest.raises(ValueError, match="global_seed"):
            SeedConfig(global_seed=-1)
        with pytest.raises(ValueError, match="n_folds"):
            SeedConfig(n_folds=0)
        assert SeedConfig(extra_seeds={"optuna": 42}).extra_seeds == {"optuna": 42}

    def test_seed_everything_makes_python_and_numpy_reproducible(self) -> None:
        seed_everything(11)
        a = (random.random(), float(np.random.random()))
        seed_everything(11)
        assert a == (random.random(), float(np.random.random()))

    def test_a_negative_seed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            seed_everything(-5)

    def test_installed_gpu_libraries_are_seeded_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, int]] = []
        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = lambda s: calls.append(("torch", s))               # type: ignore[attr-defined]
        fake_torch.cuda = types.SimpleNamespace(                                      # type: ignore[attr-defined]
            is_available=lambda: True, manual_seed_all=lambda s: calls.append(("cuda", s)))
        fake_torch.backends = types.SimpleNamespace(                                  # type: ignore[attr-defined]
            cudnn=types.SimpleNamespace(deterministic=False, benchmark=True))
        fake_cupy = types.ModuleType("cupy")
        fake_cupy.random = types.SimpleNamespace(seed=lambda s: calls.append(("cupy", s)))   # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
        monkeypatch.setattr(seeded, "has_module", lambda name: name in {"torch", "cupy"})
        seed_everything(7)
        assert calls == [("torch", 7), ("cuda", 7), ("cupy", 7)]
        assert fake_torch.backends.cudnn.deterministic is True and fake_torch.backends.cudnn.benchmark is False

    def test_cpu_only_torch_skips_the_cuda_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        fake_torch = types.ModuleType("torch")
        fake_torch.manual_seed = lambda s: calls.append("torch")                      # type: ignore[attr-defined]
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)           # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(seeded, "has_module", lambda name: name == "torch")
        seed_everything(1)
        assert calls == ["torch"]

    def test_fold_seeds_are_deterministic_distinct_and_bounded(self) -> None:
        seeds = [derive_fold_seed(42, k) for k in range(10)]
        assert seeds == [42_000 + k for k in range(10)] and len(set(seeds)) == 10
        assert derive_fold_seed(10**9, 5) < 2**31


# --------------------------------------------------------------------------- #
# stack
# --------------------------------------------------------------------------- #
class TestStack:
    def test_the_bare_stack_leaves_the_late_bound_components_empty(self) -> None:
        stack = QuantArchitectStack()
        assert stack.lw_thompson is None and stack.net_alpha is None

    def test_default_builds_every_component(self) -> None:
        stack = QuantArchitectStack.default(v_thompson=2.0, eta_init=0.3, fee_bps=7.0)
        assert stack.lw_thompson is not None and stack.lw_thompson.v == 2.0
        assert stack.net_alpha is not None and stack.net_alpha.eta == 0.3 and stack.net_alpha.fee_bps == 7.0


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
class TestCheckpoints:
    def _ckpt(self, fold: int = 3, side: str = "LONG") -> FoldCheckpoint:
        return FoldCheckpoint(fold_id=fold, side=side, model={"w": [1, 2, 3]},
                              feature_names=["a", "b"], oos_logloss=0.61, n_train=100, n_test=20)

    def test_the_feature_hash_is_derived_from_the_names_when_not_given(self) -> None:
        ckpt = self._ckpt()
        assert len(ckpt.feature_hash) == 16
        assert FoldCheckpoint(0, "LONG", None).feature_hash == ""
        ckpt.validate_features(["a", "b"])
        with pytest.raises(ValueError, match="Feature hash mismatch for fold 3"):
            ckpt.validate_features(["b", "a"])

    def test_a_checkpoint_round_trips_through_disk(self, tmp_path: Path) -> None:
        path = save_fold_checkpoint(self._ckpt(), tmp_path, study_name="s1")
        assert path == tmp_path / "s1" / "fold_0003_LONG.joblib" and path.is_file()
        loaded = load_fold_checkpoint(3, "long", tmp_path, study_name="s1")
        assert loaded.model == {"w": [1, 2, 3]} and loaded.oos_logloss == 0.61
        assert loaded.feature_hash == self._ckpt().feature_hash

    def test_loading_a_missing_checkpoint_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="fold=9"):
            load_fold_checkpoint(9, "SHORT", tmp_path)

    def test_listing_is_sorted_and_can_filter_on_side(self, tmp_path: Path) -> None:
        assert list_checkpoints(tmp_path / "nothing", "x") == []
        for fold, side in ((2, "LONG"), (0, "SHORT"), (1, "LONG")):
            save_fold_checkpoint(self._ckpt(fold, side), tmp_path, study_name="s")
        everything = [p.name for p in list_checkpoints(tmp_path, "s")]
        assert everything == ["fold_0000_SHORT.joblib", "fold_0001_LONG.joblib", "fold_0002_LONG.joblib"]
        assert [p.name for p in list_checkpoints(tmp_path, "s", side="long")] == [
            "fold_0001_LONG.joblib", "fold_0002_LONG.joblib"]


# --------------------------------------------------------------------------- #
# schema guard and entropy gate
# --------------------------------------------------------------------------- #
class TestFeatureSchemaGuard:
    def test_checking_before_stamping_is_refused(self) -> None:
        with pytest.raises(SchemaMismatchError, match="no schema stamped"):
            FeatureSchemaGuard().check(["a"])

    def test_a_matching_schema_passes_and_the_stamp_is_exposed(self) -> None:
        guard = FeatureSchemaGuard()
        assert guard.stamped is None
        fp = guard.stamp(["a", "b", "c"])
        guard.check(["a", "b", "c"])
        assert guard.stamped is fp and fp.feature_count == 3

    def test_a_reordered_or_changed_schema_names_what_differs(self) -> None:
        guard = FeatureSchemaGuard()
        guard.stamp(["a", "b", "c"])
        with pytest.raises(SchemaMismatchError):
            guard.check(["c", "b", "a"])                         # alleen de volgorde wijkt af
        with pytest.raises(SchemaMismatchError, match=r"missing \(in live\): \['b'\]") as info:
            guard.check(["a", "c", "z"])
        assert "extra   (in live): ['z']" in str(info.value)

    def test_long_difference_lists_are_truncated(self) -> None:
        guard = FeatureSchemaGuard()
        guard.stamp([f"old{i}" for i in range(8)])
        with pytest.raises(SchemaMismatchError, match="…"):
            guard.check([f"new{i}" for i in range(8)])


class TestEntropyGate:
    @pytest.mark.parametrize("tau", [49.9, 99.6])
    def test_the_percentile_must_stay_in_range(self, tau: float) -> None:
        with pytest.raises(ValueError, match="tau_percentile"):
            EntropyGate(tau_percentile=tau)

    def test_shannon_entropy_handles_unnormalised_and_empty_input(self) -> None:
        assert EntropyGate.shannon_entropy([0.5, 0.5]) == pytest.approx(np.log(2))
        assert EntropyGate.shannon_entropy([2.0, 2.0]) == pytest.approx(np.log(2))
        assert EntropyGate.shannon_entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
        assert EntropyGate.shannon_entropy([0.0, 0.0]) == 0.0
        assert EntropyGate.shannon_entropy([-1.0, -2.0]) == 0.0

    def test_with_a_short_history_the_threshold_is_the_floor_or_binary_maximum(self) -> None:
        assert EntropyGate().evaluate([0.5, 0.5]).threshold == pytest.approx(np.log(2))
        assert EntropyGate(absolute_floor=0.3).evaluate([0.5, 0.5]).threshold == 0.3

    def test_an_undecided_model_is_forced_to_neutral(self) -> None:
        gate = EntropyGate(absolute_floor=0.3)
        sure, unsure = gate.evaluate([0.99, 0.01]), gate.evaluate([0.5, 0.5])
        assert sure.pass_through and sure.action == "PASS"
        assert not unsure.pass_through and unsure.action == "NEUTRAL" and unsure.entropy > unsure.threshold

    def test_with_enough_history_the_threshold_is_a_rolling_percentile(self) -> None:
        gate = EntropyGate(history_size=50, tau_percentile=80.0)
        for _ in range(40):
            gate.evaluate([0.9, 0.1])
        decision = gate.evaluate([0.6, 0.4])
        assert decision.threshold == pytest.approx(EntropyGate.shannon_entropy([0.9, 0.1]))
        assert not decision.pass_through                          # 0,6/0,4 is onzekerder dan de eigen historie

    def test_the_floor_lifts_a_rolling_threshold_that_is_too_low(self) -> None:
        gate = EntropyGate(tau_percentile=80.0, absolute_floor=0.6)
        for _ in range(40):
            gate.evaluate([0.99, 0.01])
        assert gate.evaluate([0.99, 0.01]).threshold == 0.6

    def test_the_history_is_capped(self) -> None:
        gate = EntropyGate(history_size=10)
        for _ in range(25):
            gate.evaluate([0.7, 0.3])
        assert len(gate._history) == 10
