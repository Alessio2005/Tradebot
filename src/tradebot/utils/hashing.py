# src/tradebot/utils/hashing.py
"""Deterministic hashing utilities for config and content addressing.

Used by the audit log (R-8) and FeatureSchemaGuard to produce stable,
short identifiers for feature lists, model configs, and Parquet batches.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["hash_config", "hash_content", "hash_file", "hash_feature_names"]


def hash_config(cfg: dict[str, Any], length: int = 16) -> str:
    """Return a deterministic hex digest of a JSON-serialisable config dict.

    Keys are sorted before serialisation so insertion order does not affect
    the hash.  Non-JSON types are converted via ``str()``.
    """
    serialised = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()[:length]


def hash_content(data: bytes, length: int = 16) -> str:
    """Return SHA-256 hex digest (first ``length`` chars) of raw bytes."""
    return hashlib.sha256(data).hexdigest()[:length]


def hash_file(path: Path | str, length: int = 16) -> str:
    """Return SHA-256 hex digest of a file's contents.

    Reads in 64 KiB chunks to handle large artefacts without loading them
    fully into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def hash_feature_names(names: list[str], length: int = 16) -> str:
    """Return a stable hash of an ordered list of feature names."""
    joined = "\n".join(names).encode()
    return hashlib.sha256(joined).hexdigest()[:length]
