# src/tradebot/registry/catalog.py
"""JSONL on-disk model catalog — record per (symbol, side, version).

Every trained model is registered with:
  - git_sha       : short commit hash at training time
  - dvc_hash      : DVC md5 of the output artifact
  - metrics       : oos_logloss, sharpe, etc.
  - feature_hash  : SHA-256[:16] of feature names for lineage

The catalog is append-only (one JSONL line per registration).  Reads
scan the file and return the latest entry matching the query.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ModelCatalog", "ModelRecord"]

_CATALOG_FILENAME = "model_catalog.jsonl"


@dataclass
class ModelRecord:
    """One model registration entry.

    Attributes
    ----------
    symbol : trading symbol (e.g. "BTCUSDT").
    side : "LONG" or "SHORT".
    version : int version counter within (symbol, side).
    stage : "research" | "staging" | "prod".
    artifact_path : relative path to the saved model artifact.
    git_sha : short git commit hash.
    dvc_hash : DVC md5 of the artifact file.
    feature_hash : SHA-256[:16] of the ordered feature list.
    metrics : dict of scalar metrics (oos_logloss, sharpe, etc.).
    registered_at : ISO-8601 UTC timestamp.
    extra : arbitrary extra metadata.
    """

    symbol: str
    side: str
    version: int
    stage: str = "research"
    artifact_path: str = ""
    git_sha: str = ""
    dvc_hash: str = ""
    feature_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


class ModelCatalog:
    """Append-only JSONL model catalog.

    Parameters
    ----------
    catalog_dir : directory where ``model_catalog.jsonl`` is stored.
    """

    def __init__(self, catalog_dir: str | Path) -> None:
        self._dir = Path(catalog_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / _CATALOG_FILENAME

    def register(self, record: ModelRecord) -> ModelRecord:
        """Append a record to the catalog.  Returns the record."""
        line = json.dumps(record.to_dict(), default=str)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info("Registered model: %s/%s v%d stage=%s", record.symbol, record.side, record.version, record.stage)
        return record

    def query(
        self,
        symbol: str | None = None,
        side: str | None = None,
        stage: str | None = None,
        latest: bool = True,
    ) -> list[ModelRecord]:
        """Query the catalog.

        Parameters
        ----------
        symbol : filter by symbol (None = all).
        side : filter by side (None = all).
        stage : filter by stage (None = all).
        latest : if True, return only the last matching record per (symbol, side).

        Returns
        -------
        list of ModelRecord.
        """
        if not self._path.exists():
            return []

        records: list[ModelRecord] = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()  # noqa: PLW2901
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = ModelRecord.from_dict(d)
                except (json.JSONDecodeError, TypeError):
                    continue
                if symbol and rec.symbol != symbol:
                    continue
                if side and rec.side != side:
                    continue
                if stage and rec.stage != stage:
                    continue
                records.append(rec)

        if latest and records:
            seen: dict[tuple, ModelRecord] = {}
            for rec in records:
                key = (rec.symbol, rec.side)
                seen[key] = rec  # last wins (append-only ordering)
            return list(seen.values())

        return records

    def next_version(self, symbol: str, side: str) -> int:
        """Return the next version integer for a (symbol, side) pair."""
        existing = self.query(symbol=symbol, side=side, latest=False)
        if not existing:
            return 1
        return max(r.version for r in existing) + 1
