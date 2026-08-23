"""L0 ingestion — Phase 1, deliverable 2.

Het generieke contract is `fetch -> validate -> normalise -> hash -> persist`
(zie `contract.py`). Geen enkele bron mag die volgorde overslaan; de volgorde
zit in `run_ingestion` en niet in de bronnen.

    contract.py        het contract en `run_ingestion`
    bybit.py           publieke Bybit V5 REST-client (fail-fast, gepagineerd)
    crypto_sources.py  OHLCV, funding rates, open interest
    legacy.py          het pre-PIT `ingest_raw`-pad, ongewijzigd behouden voor
                       apps/build_features.py tot Phase 5 consolideert
"""
from __future__ import annotations

from .bybit import BybitV5Client
from .contract import (
    IngestionResult,
    IngestionSource,
    IngestionSpec,
    run_ingestion,
)
from .crypto_sources import FundingSource, OhlcvSource, OpenInterestSource
from .legacy import ingest_raw

__all__ = [
    "BybitV5Client",
    "FundingSource",
    "IngestionResult",
    "IngestionSource",
    "IngestionSpec",
    "OhlcvSource",
    "OpenInterestSource",
    "ingest_raw",
    "run_ingestion",
]
