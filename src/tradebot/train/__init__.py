# src/tradebot/train/__init__.py
"""Training sub-package — CatBoost agents, bandit ensemble, calibration, checkpoints.

Public surface
--------------
catboost
    RegimeCatAgent, RegimeEnsembleCat, RollingRobustScaler
ensemble
    ContextualBanditEnsemble
calibration
    PathSpecificPlattCalibrator, PathPlattParams
quant_arch
    SymmetricQuantileScalerState, SymmetricQuantileScaler,
    _p2_update_njit, P2OnlineQuantile, dynamic_embargo_bars
thompson
    LedoitWolfThompsonSampler
reward
    NetAlphaResult, NetAlphaReward,
    ImpactObservationRecord, KalmanImpactObserver
schema_guard
    SchemaFingerprint, FeatureSchemaGuard, SchemaMismatchError,
    EntropyGateDecision, EntropyGate
stack
    QuantArchitectStack
checkpoints
    FoldCheckpoint, save_fold_checkpoint, load_fold_checkpoint
seeded
    SeedConfig, seed_everything, derive_fold_seed
"""
from __future__ import annotations

from .calibration import PathPlattParams, PathSpecificPlattCalibrator
from .catboost import (
    RegimeCatAgent,
    RegimeEnsembleCat,
    RollingRobustScaler,
    numba_rolling_robust_scale,
)
from .checkpoints import FoldCheckpoint, load_fold_checkpoint, save_fold_checkpoint
from .ensemble import ContextualBanditEnsemble
from .quant_arch import (
    P2OnlineQuantile,
    SymmetricQuantileScaler,
    SymmetricQuantileScalerState,
    _p2_update_njit,
    dynamic_embargo_bars,
)
from .reward import (
    ImpactObservationRecord,
    KalmanImpactObserver,
    NetAlphaResult,
    NetAlphaReward,
)
from .schema_guard import (
    EntropyGate,
    EntropyGateDecision,
    FeatureSchemaGuard,
    SchemaFingerprint,
    SchemaMismatchError,
)
from .seeded import SeedConfig, derive_fold_seed, seed_everything
from .stack import QuantArchitectStack
from .meta_train import build_judge_features, make_judge_labels, train_judge
from .thompson import LedoitWolfThompsonSampler

__all__ = [
    # catboost
    "RegimeCatAgent",
    "RegimeEnsembleCat",
    "RollingRobustScaler",
    "numba_rolling_robust_scale",
    # ensemble
    "ContextualBanditEnsemble",
    # calibration
    "PathPlattParams",
    "PathSpecificPlattCalibrator",
    # quant_arch
    "SymmetricQuantileScalerState",
    "SymmetricQuantileScaler",
    "_p2_update_njit",
    "P2OnlineQuantile",
    "dynamic_embargo_bars",
    # thompson
    "LedoitWolfThompsonSampler",
    # reward
    "NetAlphaResult",
    "NetAlphaReward",
    "ImpactObservationRecord",
    "KalmanImpactObserver",
    # schema_guard
    "SchemaFingerprint",
    "FeatureSchemaGuard",
    "SchemaMismatchError",
    "EntropyGateDecision",
    "EntropyGate",
    # stack
    "QuantArchitectStack",
    # checkpoints
    "FoldCheckpoint",
    "save_fold_checkpoint",
    "load_fold_checkpoint",
    # seeded
    "SeedConfig",
    "seed_everything",
    "derive_fold_seed",
    # meta_train
    "train_judge",
    "build_judge_features",
    "make_judge_labels",
]
