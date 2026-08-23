# src/tradebot/utils/hashing.py
"""Deterministic hashing utilities for config and content addressing.

Used by the audit log (R-8) and FeatureSchemaGuard to produce stable,
short identifiers for feature lists, model configs, and Parquet batches.

Phase 1 voegt `dataframe_content_hash` toe: de `data_hash` uit sectie 7.2, die
over de INHOUD van een dataset gaat en niet over bestandsmetadata.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

#: Lengte van de `data_hash` in hexadecimale tekens (128 bit).
#: Dit is een INVARIANT, geen beleidsknop: hem wijzigen invalideert elke
#: historische hash in het data-register en in de ledger. Hij hoort daarom
#: bewust NIET in `conf/` thuis.
DATA_HASH_LENGTH = 32

#: Lengte van de korte fingerprints voor configs, features en artefacten.
FINGERPRINT_LENGTH = 16

__all__ = [
    "DATA_HASH_LENGTH",
    "FINGERPRINT_LENGTH",
    "dataframe_content_hash",
    "hash_config",
    "hash_content",
    "hash_feature_names",
    "hash_file",
]


def hash_config(cfg: dict[str, Any], length: int = FINGERPRINT_LENGTH) -> str:
    """Return a deterministic hex digest of a JSON-serialisable config dict.

    Keys are sorted before serialisation so insertion order does not affect
    the hash.  Non-JSON types are converted via ``str()``.
    """
    serialised = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:length]


def hash_content(data: bytes, length: int = FINGERPRINT_LENGTH) -> str:
    """Return SHA-256 hex digest (first ``length`` chars) of raw bytes."""
    return hashlib.sha256(data).hexdigest()[:length]


def hash_file(path: Path | str, length: int = FINGERPRINT_LENGTH) -> str:
    """Return SHA-256 hex digest of a file's contents.

    Reads in 64 KiB chunks to handle large artefacts without loading them
    fully into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def hash_feature_names(names: list[str], length: int = FINGERPRINT_LENGTH) -> str:
    """Return a stable hash of an ordered list of feature names."""
    joined = "\n".join(names).encode()
    return hashlib.sha256(joined).hexdigest()[:length]


def dataframe_content_hash(
    df: pd.DataFrame, length: int = DATA_HASH_LENGTH
) -> str:
    """Deterministische `data_hash` over de INHOUD van een DataFrame.

    Phase 1, deliverable 8. Bewust NIET over bestandsmetadata: mtime, pad,
    compressieniveau en pyarrow-versie zijn machine-afhankelijk, waardoor twee
    identieke datasets verschillende hashes zouden krijgen en het hele
    provenance-contract betekenisloos wordt.

    Determinisme wordt afgedwongen door:
      * kolommen alfabetisch te sorteren (schrijfvolgorde is irrelevant);
      * rijen te sorteren op de tijdkolommen wanneer die bestaan;
      * de index weg te gooien (positioneel, geen inhoud);
      * per kolom de dtype-naam mee te hashen, zodat int64 en float64 met
        dezelfde waarden NIET dezelfde hash krijgen;
      * numeriek te serialiseren via het canonieke numpy-bytesformaat in plaats
        van via ``repr()``.

    Twee onafhankelijke ingestion-runs op dezelfde bronperiode leveren hiermee
    dezelfde hash op. Is dat niet zo, dan zit er non-determinisme in de
    pipeline en moet dat eerst worden opgelost (Phase 1, stap 9).
    """
    import numpy as np
    import pandas as pd

    if df.empty:
        return hashlib.sha256(b"__EMPTY__").hexdigest()[:length]

    work = df.reset_index(drop=True)
    sort_cols = [c for c in ("event_ts_ns", "asof_ts_ns", "symbol")
                 if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    h = hashlib.sha256()
    h.update(str(len(work)).encode())
    for col in sorted(work.columns):
        s = work[col]
        h.update(b"|COL|")
        h.update(str(col).encode())
        h.update(b"|DTYPE|")
        h.update(str(s.dtype).encode())
        h.update(b"|VALUES|")
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            h.update(np.ascontiguousarray(s.to_numpy()).tobytes())
        else:
            h.update("\x1f".join(map(str, s.tolist())).encode("utf-8"))
    return h.hexdigest()[:length]
