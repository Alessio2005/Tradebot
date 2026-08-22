"""parquet_io.py — schema-validated parquet writer/reader for DAG stage outputs.

Design contract:
  • Every stage-boundary artefact is written via write_validated_parquet().
  • Every stage-boundary artefact is read via read_validated_parquet().
  • The schema is embedded in the Parquet metadata so that readers can
    reconstruct dtypes without re-running the schema check from scratch.
  • PyArrow schema is derived from the Pandera schema's dtype annotations
    to prevent silent dtype degradation on round-trip (the leading cause
    of _f64c calls in train_regime.py).

Blueprint §3.3:
  Float32 on disk for feat_* columns → halves footprint.
  Float64 in Numba kernels → cast happens once in numba_array().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ── Writer ────────────────────────────────────────────────────────────────────

def write_validated_parquet(
    df: pd.DataFrame,
    path: Path | str,
    schema,           # pandera DataFrameModel subclass
    *,
    compression: str = "zstd",
    lazy_validate: bool = True,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Validate df against schema, then write to Parquet with schema metadata.

    Parameters
    ----------
    df:             DataFrame to write.
    path:           Output file path (.parquet).
    schema:         Pandera DataFrameModel.  Validation runs BEFORE write.
    compression:    PyArrow compression codec (default: zstd — best ratio/speed).
    lazy_validate:  Collect all schema errors before raising (default True).
    metadata:       Optional dict of string key/value pairs embedded in the
                    Parquet file metadata (e.g. {"symbol": "BTCUSDT"}).

    Returns
    -------
    Path
        Resolved absolute path of the written file.

    Raises
    ------
    pandera.errors.SchemaErrors
        If schema validation fails.  File is NOT written on failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Validate first — never write a schema-invalid artefact
    validated_df = schema.validate(df, lazy=lazy_validate)

    # 2. Convert to PyArrow table (preserves dtypes from Pandera validation)
    table = pa.Table.from_pandas(validated_df, preserve_index=True)

    # 3. Embed custom metadata
    existing_meta = table.schema.metadata or {}
    extra = {
        b"tradebot_schema": schema.__name__.encode(),
        b"tradebot_version": b"0.1.0",
    }
    if metadata:
        extra.update({k.encode(): str(v).encode() for k, v in metadata.items()})
    table = table.replace_schema_metadata({**existing_meta, **extra})

    # 4. Write
    pq.write_table(table, path, compression=compression)
    logger.info(
        "Wrote %s rows -> %s (schema=%s, %.1f MB)",
        len(validated_df),
        path,
        schema.__name__,
        path.stat().st_size / 1e6,
    )
    return path.resolve()


# ── Reader ────────────────────────────────────────────────────────────────────

def read_validated_parquet(
    path: Path | str,
    schema,           # pandera DataFrameModel subclass
    *,
    lazy_validate: bool = True,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a Parquet file and validate against schema.

    Parameters
    ----------
    path:           Input file path.
    schema:         Pandera DataFrameModel.  Validation runs AFTER read.
    lazy_validate:  Collect all schema errors before raising.
    columns:        Optional column subset to read (index always included).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with correct dtypes.

    Raises
    ------
    pandera.errors.SchemaErrors
        If the on-disk artefact violates the schema.  Indicates either
        upstream write bug or schema evolution without migration.
    FileNotFoundError
        If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Artefact not found: {path}\n"
            "Run the upstream DAG stage first or check DVC cache."
        )

    table = pq.read_table(path, columns=columns)
    df = table.to_pandas()

    # Validate on read — catches schema drift between writer and reader
    return schema.validate(df, lazy=lazy_validate)


# ── Utility: parquet metadata inspector ──────────────────────────────────────

def read_parquet_metadata(path: Path | str) -> dict[str, str]:
    """Return the custom tradebot metadata embedded in a parquet file."""
    path = Path(path)
    pf = pq.ParquetFile(path)
    raw = pf.schema_arrow.metadata or {}
    return {
        k.decode(errors="replace"): v.decode(errors="replace")
        for k, v in raw.items()
    }
