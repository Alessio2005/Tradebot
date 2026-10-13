# tests/unit/test_features_scaling_edges.py
"""`features/scaling.py`: oscillator-herkenning op naam en de causale SymQS-overlay."""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.features.scaling import (
    _apply_symqs_overlay,
    _build_oscillator_masks,
    apply_symqs_overlay,
    build_oscillator_masks,
)
from tradebot.utils.failfast import CausalityViolationError

NAMES = ["feat_rsi14", "feat_atr", "feat_is_weekend", "feat_macd_signal", "feat_regime_id"]


def _matrix(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 3.0, size=(n, len(NAMES)))
    X[:, 2] = rng.integers(0, 2, n)                      # binair
    X[:, 4] = rng.integers(0, 3, n)                      # discreet regime
    return X


class TestMasks:
    def test_no_features_give_empty_masks(self) -> None:
        osc, passthrough = build_oscillator_masks([])
        assert osc.size == 0 and passthrough.size == 0

    def test_names_are_classified_case_insensitively_and_passthrough_wins(self) -> None:
        osc, passthrough = build_oscillator_masks(NAMES + ["FEAT_RSI_IS_FLAG"])
        assert osc.tolist() == [True, False, False, True, False, False]
        assert passthrough.tolist() == [False, False, True, False, True, True]   # 'is_' en 'regime_' sluiten rsi uit

    def test_the_underscore_aliases_are_the_same_functions(self) -> None:
        assert _build_oscillator_masks is build_oscillator_masks and _apply_symqs_overlay is apply_symqs_overlay


class TestSymQsOverlay:
    def test_a_matrix_without_columns_is_returned_as_is(self) -> None:
        empty = np.zeros((5, 0))
        assert apply_symqs_overlay(empty, empty, []) is empty

    def test_without_any_oscillator_the_scaled_matrix_is_returned_untouched(self) -> None:
        X = _matrix()[:, [1, 2]]
        assert apply_symqs_overlay(X, X, ["feat_atr", "feat_is_weekend"], train_indices=np.arange(100)) is X

    def test_a_raw_matrix_without_rows_gives_a_copy(self) -> None:
        scaled = _matrix(10)
        out = apply_symqs_overlay(scaled, np.empty((0, len(NAMES))), NAMES, train_indices=np.arange(0))
        assert out is not scaled and np.array_equal(out, scaled)

    def test_the_overlay_without_train_indices_is_a_causality_violation(self) -> None:
        X = _matrix()
        with pytest.raises(CausalityViolationError, match="train_indices"):
            apply_symqs_overlay(X, X, NAMES)

    def test_only_oscillator_columns_are_rescaled_and_they_keep_the_zero_line(self) -> None:
        raw = _matrix()
        scaled = raw.copy()
        out = apply_symqs_overlay(scaled, raw, NAMES, train_indices=np.arange(200))
        for j in (1, 2, 4):                                             # geen oscillator: ongewijzigd
            np.testing.assert_array_equal(out[:, j], scaled[:, j])
        for j in (0, 3):
            assert out[:, j].min() >= -1.0 and out[:, j].max() <= 1.0
            assert not np.array_equal(out[:, j], scaled[:, j])
            assert (np.sign(out[:, j][np.abs(raw[:, j]) > 1e-6]) == np.sign(raw[:, j][np.abs(raw[:, j]) > 1e-6])).all()
        assert np.array_equal(scaled, raw)                              # de invoer is niet gemuteerd

    def test_the_fit_uses_the_train_rows_only(self) -> None:
        raw = _matrix()
        shifted = raw.copy()
        shifted[200:, 0] *= 100.0                                        # alleen buiten de train-rijen
        a = apply_symqs_overlay(raw.copy(), raw, NAMES, train_indices=np.arange(200))
        b = apply_symqs_overlay(shifted.copy(), shifted, NAMES, train_indices=np.arange(200))
        np.testing.assert_array_equal(a[:200, 0], b[:200, 0])
