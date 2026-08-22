# src/tradebot/data/sources/__init__.py
"""Point-in-time multi-market data sources (Wave 20, step 0.2).

Every source implements the same contract (Mandate v3 §7):

    fetch -> Pandera validation -> ``asof_ts`` column = AVAILABILITY moment
    (publication lag explicit) -> parquet under market_data_parquet/<market>/

``asof_ts`` is the only timestamp downstream joins may condition on (R-1).
Each source registers its metadata in docs/DATA_REGISTER.md (G8) and has a
lookahead test in tests/lookahead/ (G6).
"""
from __future__ import annotations

from .base import SourceMeta, http_get_text, stamp_asof, validate_pit, write_market_parquet

__all__ = [
    "SourceMeta",
    "http_get_text",
    "stamp_asof",
    "validate_pit",
    "write_market_parquet",
]
