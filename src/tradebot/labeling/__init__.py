"""AFML labeling methods."""
from .cusum import cusum_filter, dual_cusum_events, get_cusum_events_for_short
from .fixed_horizon import fixed_horizon_labels
from .meta import (
    MetaLabelFilter,
    MetaLabelingEngine,
    PrimaryScout,
    assert_training_scout_is_causal,
)
from .trend_scanning import TrendScanningLabeler
from .triple_barrier import TripleBarrierLabeler

__all__ = [
    "MetaLabelFilter",
    "MetaLabelingEngine",
    "PrimaryScout",
    "TrendScanningLabeler",
    "TripleBarrierLabeler",
    "assert_training_scout_is_causal",
    "cusum_filter",
    "dual_cusum_events",
    "fixed_horizon_labels",
    # SHORT-model verbetering (Agent C — 2026-05-22):
    # Asymmetrische CUSUM met lagere drempel voor DOWN events.
    "get_cusum_events_for_short",
]
