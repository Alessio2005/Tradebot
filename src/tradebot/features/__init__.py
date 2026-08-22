"""Feature engineering suite."""
from .blocks import split_feature_blocks
from .cfi import clustered_feature_importance
from .orthogonalize import FeatureOrthogonalizer
from .pipeline import build_features, derive_feature_map, load_orthogonalizers
from .regime import (
    FeaturePipeline,
    HurstComputer,
    RegimeRouter,
    StationarityComputer,
    StructuralBreakRegime,
)
from .regime_features import REGIME_FEATURE_COLS, add_regime_features
from .scaling import RollingRobustScaler
from .ta import FeatureEngineer

# Backward-compat alias (was build_feature_pipeline → now build_features)
build_feature_pipeline = build_features

__all__ = [
    "REGIME_FEATURE_COLS",
    "FeatureEngineer",
    "FeatureOrthogonalizer",
    "FeaturePipeline",
    "HurstComputer",
    "RegimeRouter",
    "RollingRobustScaler",
    "StationarityComputer",
    "StructuralBreakRegime",
    "add_regime_features",
    "build_feature_pipeline",
    "build_features",
    "clustered_feature_importance",
    "derive_feature_map",
    "load_orthogonalizers",
    "split_feature_blocks",
]
