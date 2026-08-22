# src/tradebot/registry/__init__.py
"""Registry sub-package — model catalog, lineage, promotion."""
from __future__ import annotations

from .catalog import ModelCatalog, ModelRecord
from .hypothesis_ledger import DEFAULT_LEDGER_PATH, HypothesisLedger, LedgerEntry
from .lineage import get_file_hash, get_git_sha, verify_lineage
from .promotion import PromotionGates, promote

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "HypothesisLedger",
    "LedgerEntry",
    "ModelCatalog",
    "ModelRecord",
    "PromotionGates",
    "get_file_hash",
    "get_git_sha",
    "promote",
    "verify_lineage",
]
