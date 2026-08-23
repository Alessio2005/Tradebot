"""L0 validatielaag — Phase 1, deliverable 7.

Vier validators, één regel: **elke validator raiset bij schending en retourneert
géén boolean.** Een aanroeper mag de uitkomst niet kunnen negeren.

    schema.py      kolomtypen, verplichte velden, monotone timestamps,
                   OHLC-ordening
    gaps.py        ontbrekende bars -> gap-ledger; nooit interpolatie
    outliers.py    prijssprongen, nul-volume bars, geïnverteerde ranges
    continuity.py  futures rolls, delistings, renames -> adjustment ledger
"""
from __future__ import annotations

from .continuity import (
    AdjustmentFactor,
    AdjustmentLedger,
    SymbolLifecycle,
    apply_adjustments,
    validate_continuity,
)
from .gaps import GapLedger, GapRecord, detect_gaps, enforce_gap_policy
from .outliers import OutlierReport, detect_outliers, enforce_outlier_policy
from .schema import (
    FUNDING_SPEC,
    OHLCV_SPEC,
    OPEN_INTEREST_SPEC,
    SeriesSpec,
    validate_schema,
)

__all__ = [
    "FUNDING_SPEC",
    "OHLCV_SPEC",
    "OPEN_INTEREST_SPEC",
    "AdjustmentFactor",
    "AdjustmentLedger",
    "GapLedger",
    "GapRecord",
    "OutlierReport",
    "SeriesSpec",
    "SymbolLifecycle",
    "apply_adjustments",
    "detect_gaps",
    "detect_outliers",
    "enforce_gap_policy",
    "enforce_outlier_policy",
    "validate_continuity",
    "validate_schema",
]
