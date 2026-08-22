"""selection — Feature selectie modules (v3 T0.4).

Bevat: SFI (Single Feature Importance), Causal MDA (Mean Decrease Accuracy),
en wrapper voor Clustered Feature Importance (CFI uit features/cfi.py).
"""
from .sfi import single_feature_importance, rank_features_by_sfi
from .mda import causal_mda, filter_by_mda
