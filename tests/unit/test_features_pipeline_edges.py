# tests/unit/test_features_pipeline_edges.py
"""`features/pipeline.py`: de legacy-adapter `build_features` en de L1-containers.

`FeaturePipeline.transform` is voor de adapter gestubd: de adapter pakt alleen
het 4-tupel uit, leidt de feature-map af en bewaart de gefitte orthogonalizers.
De pipeline zelf wordt in `test_features_regime_edges.py` op minuutdata getest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.features import regime as rg
from tradebot.features.orthogonalize import FeatureOrthogonalizer
from tradebot.features.pipeline import (
    PanelFeature,
    PanelPipeline,
    TransformStep,
    build_features,
    derive_feature_map,
    load_orthogonalizers,
)


def _merged() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    cols = ["feat_atr", "feat_rsi", "feat_atr_meso", "feat_trend_macro", "feat_macro_vix", "close"]
    return pd.DataFrame(np.arange(30.0).reshape(5, 6), index=idx, columns=cols)


def _fitted_orthogonalizer() -> FeatureOrthogonalizer:
    X = np.random.default_rng(0).normal(size=(200, 3)) * np.array([1.0, 2.0, 3.0])
    orth = FeatureOrthogonalizer(spearman_prefilter=False)
    orth.fit_on_train_indices(X, np.arange(120), ["a", "b", "c"], prefix="micro_pc")
    return orth


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch, result, **orthogonalizers) -> None:
    class _Stub:
        def __init__(self, cfg, sym) -> None:
            self.sym = sym
            for name, orth in orthogonalizers.items():
                setattr(self, name, orth)

        def transform(self, df):
            return result

    monkeypatch.setattr(rg, "FeaturePipeline", _Stub)


class TestDeriveFeatureMap:
    def test_columns_are_bucketed_by_suffix_and_prefix_and_sorted(self) -> None:
        assert derive_feature_map(_merged()) == {
            "micro": ["feat_atr", "feat_rsi"],
            "meso": ["feat_atr_meso"],
            "macro": ["feat_macro_vix", "feat_trend_macro"],
        }


class TestBuildFeatures:
    def test_it_returns_the_merged_frame_with_its_feature_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        merged = _merged()
        _stub_pipeline(monkeypatch, (np.empty(0), np.empty(0), np.empty(0), merged))
        df, feat_map = build_features(object(), "BTCUSDT", pd.DataFrame())     # type: ignore[arg-type]
        assert df is merged and feat_map["meso"] == ["feat_atr_meso"]

    @pytest.mark.parametrize("bad", [None, (1, 2, 3), [1, 2, 3, 4], 7])
    def test_a_result_that_is_not_a_four_tuple_is_refused(self, monkeypatch: pytest.MonkeyPatch, bad) -> None:
        _stub_pipeline(monkeypatch, bad)
        with pytest.raises(RuntimeError, match="expected 4-tuple"):
            build_features(object(), "BTCUSDT", pd.DataFrame())                 # type: ignore[arg-type]

    @pytest.mark.parametrize("empty", [None, pd.DataFrame()])
    def test_an_empty_merged_frame_is_refused(self, monkeypatch: pytest.MonkeyPatch, empty) -> None:
        _stub_pipeline(monkeypatch, (None, None, None, empty))
        with pytest.raises(ValueError, match="empty df_merged"):
            build_features(object(), "BTCUSDT", pd.DataFrame())                 # type: ignore[arg-type]

    def test_only_fitted_orthogonalizers_are_persisted_and_they_load_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fitted = _fitted_orthogonalizer()
        _stub_pipeline(monkeypatch, (None, None, None, _merged()),
                       orthogonalizer_micro=fitted,
                       orthogonalizer_meso=FeatureOrthogonalizer(),         # niet gefit: overgeslagen
                       orthogonalizer_macro=None)                            # ontbreekt: overgeslagen
        target = tmp_path / "orth"
        build_features(object(), "BTCUSDT", pd.DataFrame(), orthogonalizer_save_dir=target)   # type: ignore[arg-type]
        assert sorted(p.name for p in target.iterdir()) == ["orthogonalizer_micro_BTCUSDT.pkl"]

        loaded = load_orthogonalizers("BTCUSDT", target)
        assert loaded["meso"] is None and loaded["macro"] is None
        X = np.random.default_rng(1).normal(size=(10, 3))
        np.testing.assert_allclose(loaded["micro"].transform(X), fitted.transform(X))

    def test_loading_from_an_empty_directory_gives_three_missing_tiers(self, tmp_path: Path) -> None:
        assert load_orthogonalizers("ETHUSDT", tmp_path) == {"micro": None, "meso": None, "macro": None}


class TestPanelContainers:
    def test_the_pipeline_exposes_its_steps_and_a_panel_feature_its_symbols(self) -> None:
        step = TransformStep("rolling_log_return", {"window": 3, "skip_bars": 1})
        pipe = PanelPipeline([step])
        assert pipe.steps == (step,) and pipe.burn_in_period == 4
        values = pd.DataFrame({"A": np.arange(8.0), "B": np.arange(8.0)},
                              index=pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"))
        feature = PanelFeature(values=values, data_hashes=(("ohlcv", "h"),), steps=pipe.steps,
                               burn_in_period=pipe.burn_in_period, panel_hash="x")
        assert feature.symbols == ("A", "B")
        assert len(feature.drop_burn_in()) == 4 and feature.drop_burn_in().index[0] == values.index[4]
