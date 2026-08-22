"""selection — Feature selectie modules (v3 T0.4).

Bevat: SFI (Single Feature Importance), Causal MDA (Mean Decrease Accuracy),
en wrapper voor Clustered Feature Importance (CFI uit features/cfi.py).
"""
from .mda import causal_mda, filter_by_mda
from .sfi import rank_features_by_sfi, single_feature_importance

__all__ = [
    "causal_mda",
    "filter_by_mda",
    "rank_features_by_sfi",
    "single_feature_importance",
]
