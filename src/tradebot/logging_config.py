"""Logging bootstrap for tradebot.

Call setup_logging() once at application entry (apps/*.py).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback as tb
from typing import Literal

_LEVEL_MAP = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = tb.format_exception(*record.exc_info)[-1].strip()
        return json.dumps(payload)


def setup_logging(
    mode: Literal["dev", "ci", "prod"] = "dev",
    level: str | None = None,
) -> None:
    """Configure root logger. Idempotent — safe to call multiple times."""
    global _configured
    if _configured:
        return
    _configured = True

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = _LEVEL_MAP.get(level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    if mode == "dev":
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    else:
        handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    for noisy in ("numba", "catboost", "optuna"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
