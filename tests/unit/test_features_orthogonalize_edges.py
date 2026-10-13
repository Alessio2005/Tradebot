# tests/unit/test_features_orthogonalize_edges.py
"""`features/orthogonalize.py`: asset-drempels, Spearman-dedup, de no-op-uitgangen en de componentenkeuze.

Beide fit-paden (`fit_transform` en het lekvrije `fit_on_train_indices`) hebben
dezelfde vertakkingen en worden daarom met dezelfde scenario's getest.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from tradebot.features.orthogonalize import FeatureOrthogonalizer

NAMES = ["a", "b", "c", "d"]


def _independent(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 4)) * np.array([1.0, 2.0, 3.0, 4.0])


def _fit(orth: FeatureOrthogonalizer, how: str, X: np.ndarray, names: list[str] = NAMES):
    """Beide fit-paden met dezelfde aanroep; het legacy-pad waarschuwt (DeprecationWarning)."""
    if how == "legacy":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return orth.fit_transform(X, names, prefix="pc")
    return orth.fit_on_train_indices(X, np.arange(int(len(X) * 0.6)), names, prefix="pc")


BOTH = pytest.mark.parametrize("how", ["legacy", "train_indices"])


class TestAssetThreshold:
    @pytest.mark.parametrize(("symbol", "expected"), [
        ("BTCUSDT", 0.10),      # exacte treffer
        ("BTCPERP", 0.10),      # prefix BTC
        ("ETHUSDT", 0.15),
        ("SOLBUSD", 0.20),
        ("XRPUSDT", 0.25),      # onbekend: de default
    ])
    def test_the_symbol_selects_its_spearman_threshold(self, symbol: str, expected: float) -> None:
        assert FeatureOrthogonalizer(symbol=symbol).spearman_cluster_threshold == expected

    def test_an_explicit_threshold_wins_over_the_symbol(self) -> None:
        assert FeatureOrthogonalizer(symbol="BTCUSDT", spearman_cluster_threshold=0.4).spearman_cluster_threshold == 0.4


class TestSpearmanDedup:
    def test_two_or_fewer_features_are_never_deduplicated(self) -> None:
        mask, names = FeatureOrthogonalizer._spearman_dedup_mask(_independent()[:, :2], ["a", "b"], 0.25)
        assert mask.all() and names == ["a", "b"]

    def test_a_near_copy_is_dropped_in_favour_of_the_wider_feature(self) -> None:
        X = _independent()
        X[:, 3] = X[:, 1] * 3.0 + np.random.default_rng(1).normal(0, 0.01, len(X))   # b ~ d, d heeft de grootste std
        mask, names = FeatureOrthogonalizer._spearman_dedup_mask(X, NAMES, 0.25)
        assert mask.tolist() == [True, False, True, True] and names == ["a", "c", "d"]

    def test_independent_features_all_survive(self) -> None:
        mask, names = FeatureOrthogonalizer._spearman_dedup_mask(_independent(), NAMES, 0.25)
        assert mask.all() and names == NAMES

    @BOTH
    def test_the_default_prefilter_removes_the_duplicate_before_the_pca(self, how: str) -> None:
        X = _independent()
        X[:, 3] = X[:, 1] * 3.0 + np.random.default_rng(1).normal(0, 0.01, len(X))
        out, names = _fit(FeatureOrthogonalizer(n_components=0.99), how, X)
        assert out.shape[1] == 3 and names == ["pc0", "pc1", "pc2"]


class TestNoOpExits:
    @BOTH
    def test_a_single_feature_passes_through_as_float32(self, how: str) -> None:
        X = _independent()[:, :1]
        orth = FeatureOrthogonalizer()
        out, names = _fit(orth, how, X, ["a"])
        assert out.dtype == np.float32 and names == ["a"] and orth.is_fitted
        assert orth.output_names == ["a"]

    @BOTH
    def test_all_constant_features_pass_through(self, how: str) -> None:
        orth = FeatureOrthogonalizer()
        out, names = _fit(orth, how, np.ones((200, 4)))
        assert out.shape == (200, 4) and names == NAMES and orth.is_fitted

    def test_fewer_than_two_train_rows_pass_through(self) -> None:
        orth = FeatureOrthogonalizer()
        out, names = orth.fit_on_train_indices(_independent(), np.array([3]), NAMES)
        assert out.shape == (400, 4) and names == NAMES and orth.is_fitted


class TestComponentChoice:
    @BOTH
    def test_without_the_prefilter_every_independent_feature_gets_a_component(self, how: str) -> None:
        orth = FeatureOrthogonalizer(spearman_prefilter=False, n_components=0.99)
        out, names = _fit(orth, how, _independent())
        assert out.shape == (400, 4) and names == ["pc0", "pc1", "pc2", "pc3"]
        assert "fitted=True" in repr(orth) and "output_dims=4" in repr(orth)

    @BOTH
    def test_a_variance_target_that_one_component_meets_is_raised_to_the_minimum(self, how: str) -> None:
        rng = np.random.default_rng(2)
        base = rng.normal(size=(400, 1)) * 5.0
        X = base + rng.normal(0, 0.1, size=(400, 4))            # sterk gecorreleerd: 1 component haalt 95 %
        orth = FeatureOrthogonalizer(spearman_prefilter=False, n_components=0.9, min_components=2)
        out, _ = _fit(orth, how, X)
        assert out.shape[1] == 2                                  # min_components wint

    @BOTH
    def test_a_block_no_wider_than_the_minimum_keeps_exactly_the_minimum(self, how: str) -> None:
        orth = FeatureOrthogonalizer(spearman_prefilter=False, min_components=2)
        out, names = _fit(orth, how, _independent()[:, :2], ["a", "b"])
        assert out.shape == (400, 2) and names == ["pc0", "pc1"]

    @BOTH
    def test_constant_columns_are_dropped_before_the_pca(self, how: str) -> None:
        X = _independent()
        X[:, 1] = 7.0
        orth = FeatureOrthogonalizer(spearman_prefilter=False, n_components=0.99)
        out, _ = _fit(orth, how, X)
        assert out.shape[1] == 3
        assert orth.transform(X).shape == (400, 3)                # dezelfde maskers bij inferentie


class TestTransform:
    def test_transform_before_any_fit_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="vóór fit_transform"):
            FeatureOrthogonalizer().transform(_independent())

    @BOTH
    def test_transform_reproduces_the_fitted_projection(self, how: str) -> None:
        X = _independent()
        orth = FeatureOrthogonalizer(spearman_prefilter=False, n_components=0.99)
        fitted, _ = _fit(orth, how, X)
        np.testing.assert_allclose(orth.transform(X), fitted, rtol=1e-5, atol=1e-5)
        assert orth.transform(X).dtype == np.float32
