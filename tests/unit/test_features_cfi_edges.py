# tests/unit/test_features_cfi_edges.py
"""`features/cfi.py`: clustering, scoring-metrieken, structurele-stabiliteitsfilter en `filter_feature_map`.

Het model is een stub die alleen kolom 0 leest, zodat de verwachte importance
exact te beredeneren is: kolom 0 (en zijn bijna-kopie kolom 1) vormen een
cluster met positieve importance, de ruiskolommen krijgen precies nul.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.features.cfi import (
    clustered_feature_importance,
    compute_clustered_feature_importance,
    filter_feature_map,
)
from tradebot.utils.failfast import DataContractError

NAMES = ["a", "b", "c", "d"]


class _Logit:
    """Leest alleen kolom 0."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-2.5 * X[:, 0]))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] > 0).astype(int)


class _PredictOnly:
    """Geen `predict_proba`: de score valt terug op accuracy."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] > 0).astype(int)


def _data(seed: int = 0, n: int = 1200, informative_from: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Kolom 1 is een bijna-kopie van kolom 0; y hangt vanaf `informative_from` aan kolom 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    X[:, 1] = X[:, 0] + rng.normal(0.0, 0.01, n)
    y = rng.integers(0, 2, n)
    y[informative_from:] = ((X[informative_from:, 0] + rng.normal(0.0, 0.5, n - informative_from)) > 0).astype(int)
    return X, y


def _run(model, X, y, **kw):
    kw = {"n_regimes": 4, "purge_bars": 10, "n_perms": 3, "rng_seed": 1, **kw}
    return compute_clustered_feature_importance(model, X, y, NAMES, **kw)


class TestImportance:
    def test_the_informative_cluster_scores_and_the_noise_scores_exactly_zero(self) -> None:
        X, y = _data()
        out, link = _run(_Logit(), X, y)
        assert out.loc["a", "cluster_id"] == out.loc["b", "cluster_id"] != out.loc["c", "cluster_id"]
        assert out.loc["a", "n_members"] == 2 and out.loc["a", "stability_ratio"] == 1.0
        assert out.loc["a", "importance"] > 0.5
        assert (out.loc[["c", "d"], "importance"] == 0.0).all()
        assert list(out.index[:2]) == ["a", "b"]                 # aflopend gesorteerd
        assert link.shape == (3, 4)

    def test_it_is_deterministic_for_a_fixed_seed(self) -> None:
        X, y = _data()
        first, _ = _run(_Logit(), X, y)
        second, _ = _run(_Logit(), X, y)
        pd.testing.assert_frame_equal(first, second)

    def test_a_single_permutation_has_no_replicate_error_estimate(self) -> None:
        X, y = _data()
        out, _ = _run(_Logit(), X, y, n_perms=1)
        assert out["replicate_se"].isna().all() and (out["n_perms"] == 1).all()

    def test_a_feature_useful_in_only_half_the_regimes_is_zeroed_and_the_drop_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        X, y = _data(seed=0, informative_from=0)
        y[300:] = np.random.default_rng(5).integers(0, 2, len(y) - 300)   # alleen regime 0 is informatief
        with caplog.at_level(logging.INFO, logger="tradebot.features.cfi"):
            out, _ = _run(_Logit(), X, y)
        assert out.loc["a", "raw_mean_imp"] > 0 and out.loc["a", "stability_ratio"] <= 0.5
        assert out.loc["a", "importance"] == 0.0
        assert "Structural Break Filter: destroyed" in caplog.text

    def test_the_public_alias_is_the_same_function(self) -> None:
        assert clustered_feature_importance is compute_clustered_feature_importance


class TestScoringAndInputs:
    def test_accuracy_works_with_and_without_predict_proba(self) -> None:
        X, y = _data()
        with_proba, _ = _run(_Logit(), X, y, scoring_metric="accuracy")
        predict_only, _ = _run(_PredictOnly(), X, y, scoring_metric="log_loss")
        assert with_proba.loc["a", "importance"] > 0
        assert predict_only.loc["a", "importance"] > 0              # accuracy-tekenregel, ook bij log_loss gevraagd
        assert (with_proba.loc[["c", "d"], "importance"] == 0.0).all()

    def test_a_failing_log_loss_is_an_error_and_never_a_silent_metric_switch(self) -> None:
        X, y = _data()
        y[:] = 0                                                 # een klasse: log_loss weigert
        with pytest.raises(DataContractError, match="log_loss faalde"):
            _run(_Logit(), X, y)

    def test_wrong_length_feature_names_are_replaced_by_dummies_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        X, y = _data()
        with caplog.at_level(logging.WARNING, logger="tradebot.features.cfi"):
            out, _ = compute_clustered_feature_importance(
                _Logit(), X, y, ["only_one"], n_regimes=4, purge_bars=10, rng_seed=1)
        assert list(out.index.sort_values()) == ["feat_0", "feat_1", "feat_2", "feat_3"]
        assert "using dummies" in caplog.text

    def test_clustering_uses_only_the_train_rows_and_falls_back_when_there_are_too_few(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        X, y = _data()
        out, _ = _run(_Logit(), X, y, train_indices=np.arange(0, 600))
        assert out.loc["a", "cluster_id"] == out.loc["b", "cluster_id"]
        with caplog.at_level(logging.WARNING, logger="tradebot.features.cfi"):
            fallback, _ = _run(_Logit(), X, y, train_indices=np.arange(3))
        assert "falling back to full-X clustering" in caplog.text
        assert fallback.loc["a", "cluster_id"] == fallback.loc["b", "cluster_id"]


class TestFilterFeatureMap:
    @pytest.fixture
    def cfi_df(self) -> pd.DataFrame:
        return pd.DataFrame({"importance": [0.4, 0.3, 0.1, 0.1, 0.05, 0.05, 0.0]},
                            index=list("abcdefg"))

    @pytest.fixture
    def feat_map(self) -> dict[str, list[str]]:
        return {"micro": ["a", "f", "g"], "meso": ["b", "d"], "macro": ["c", "e"]}

    def test_top_n_keeps_the_n_best_features_per_tier(self, cfi_df, feat_map) -> None:
        assert filter_feature_map(cfi_df, feat_map, mode="top_n", param=2) == {
            "micro": ["a"], "meso": ["b"], "macro": []}

    def test_cumulative_keeps_at_least_three_features(self, cfi_df, feat_map) -> None:
        out = filter_feature_map(cfi_df, feat_map, mode="cumulative", param=0.5)
        assert out == {"micro": ["a"], "meso": ["b"], "macro": ["c"]}      # 0.5 -> 1 feature, minimum 3

    def test_cumulative_stops_at_the_requested_share_of_the_total_importance(self, cfi_df, feat_map) -> None:
        out = filter_feature_map(cfi_df, feat_map, mode="cumulative", param=0.92)
        assert out == {"micro": ["a"], "meso": ["b", "d"], "macro": ["c"]}  # a, b, c, d = 90 %; zero-importance g never

    def test_zero_total_importance_returns_the_original_map(self, feat_map, caplog) -> None:
        zero = pd.DataFrame({"importance": [0.0, 0.0]}, index=["a", "b"])
        with caplog.at_level(logging.WARNING, logger="tradebot.features.cfi"):
            assert filter_feature_map(zero, feat_map) is feat_map
        assert "returning all features" in caplog.text

    def test_an_unknown_mode_is_refused(self, cfi_df, feat_map) -> None:
        with pytest.raises(ValueError, match="Unknown filter mode 'median'"):
            filter_feature_map(cfi_df, feat_map, mode="median")
