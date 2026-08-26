"""L3 Regime Engines — M0 Causal Vol-Buckets (baseline) en de Markov-familie.

De hiërarchie is bindend (audit §10.1):

    M0  Causal Vol-Buckets      BASELINE PRODUCTIE   -- `buckets.py`
    M1  Markov Chain            Research Track
    M2  Gaussian/Student-t HMM  Restricted Research Only
    M3  Markov-Switching GARCH  Exclusief Theoretisch -- NIET implementeren

Alleen filtered probabilities `P(S_t | F_t)` mogen een backtest bereiken; de
smoothed variant `P(S_t | F_T)` gebruikt de volledige dataset en is in
backtests STRENG VERBODEN (§10.2).
"""
from .buckets import (
    UNDEFINED_BUCKET,
    BucketAssignment,
    VolBucket,
    causal_atr,
    causal_vol_zscore,
    classify_vol_buckets,
)

__all__ = [
    "UNDEFINED_BUCKET",
    "BucketAssignment",
    "VolBucket",
    "causal_atr",
    "causal_vol_zscore",
    "classify_vol_buckets",
]
