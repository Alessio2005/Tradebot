"""Optuna storage factory — SQLite (default) or PostgreSQL.

SQLite is used for single-machine runs.
PostgreSQL is required for parallel workers (async I/O safety).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def make_storage(
    storage_url: str | None = None,
    sqlite_path: Path | None = None,
    study_name: str = "tradebot",
) -> str | None:
    """Return an Optuna storage URL or None (in-memory).

    Priority:
      1. Explicit storage_url (e.g. postgresql://... or sqlite:///...)
      2. sqlite_path argument
      3. OPTUNA_STORAGE env var
      4. None → in-memory (lost on crash)

    In-memory studies are acceptable for CI smoke tests but NOT for
    production tuning runs (all trial history lost on crash).
    """
    if storage_url:
        return storage_url

    env_url = os.getenv("OPTUNA_STORAGE")
    if env_url:
        logger.info("tune/storage: using OPTUNA_STORAGE env var.")
        return env_url

    if sqlite_path is not None:
        sqlite_path = Path(sqlite_path)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{sqlite_path.as_posix()}"
        logger.info("tune/storage: SQLite at %s", url)
        return url

    logger.warning(
        "tune/storage: no storage configured — study is in-memory "
        "and will be lost on crash. Set OPTUNA_STORAGE or pass sqlite_path."
    )
    return None
