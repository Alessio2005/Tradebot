# src/tradebot/oms/audit_log.py
"""Append-only JSONL order audit trail (R-8).

Every order-fill event is recorded as a single JSONL line containing all
14 required fields.  The log is append-only; no records are ever mutated.

Integrity: a SHA-256 batch hash is written as a separate ``.hash`` file
alongside each log file so post-trade reconstruction can detect truncation.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .order import Fill, Order

logger = logging.getLogger(__name__)

__all__ = ["AuditLog", "AuditRecord"]

_REQUIRED_FIELDS = frozenset(
    {
        "event_ts", "order_id", "symbol", "side", "qty_base",
        "notional_usdt", "order_type", "signal_prob", "kelly_fraction",
        "model_version", "git_sha", "feature_hash", "portfolio_weight",
        "fill_price",
    }
)


def _build_record(order: Order, fill: Fill) -> dict:
    return {
        "event_ts": fill.fill_ts.isoformat(),
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side.value,
        "qty_base": fill.fill_qty,
        "notional_usdt": fill.notional_usdt,
        "order_type": order.order_type.value,
        "signal_prob": order.signal_prob,
        "kelly_fraction": order.kelly_fraction,
        "model_version": order.model_version,
        "git_sha": order.git_sha,
        "feature_hash": order.feature_hash,
        "portfolio_weight": order.portfolio_weight,
        "pre_trade_cost_bps": order.pre_trade_cost_bps,
        "fill_price": fill.fill_price,
        "fill_ts": fill.fill_ts.isoformat(),
        "post_trade_cost_bps": fill.post_trade_cost_bps,
        "circuit_breaker_state": fill.circuit_breaker_state,
        "exchange_order_id": fill.exchange_order_id,
    }


# Re-exported for external use
AuditRecord = dict


class AuditLog:
    """Append-only JSONL order-audit trail.

    Parameters
    ----------
    log_path : Path to the JSONL audit file.  Created if absent.
    """

    def __init__(self, log_path: Path | str) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records_written: int = 0

        # Wave 15 P1-24 — SHA-256 hash chain initialisation
        # Genesis hash: 64 zeros; overwritten if log already exists with entries.
        self._prev_hash: str = "0" * 64
        self._prev_hash = self._load_last_hash()

    def record(self, order: Order, fill: Fill) -> None:
        """Append one fill event to the audit log.

        Raises ValueError if any of the 14 required R-8 fields is absent.
        """
        rec = _build_record(order, fill)
        missing = _REQUIRED_FIELDS - rec.keys()
        if missing:
            raise ValueError(f"AuditLog: missing required fields: {missing}")

        # Wave 15 P1-24 — SHA-256 hash chain: add prev_hash and hash fields
        rec["prev_hash"] = self._prev_hash
        rec["hash"] = self._compute_hash(
            {k: v for k, v in rec.items() if k not in ("prev_hash", "hash")},
            self._prev_hash,
        )
        self._prev_hash = rec["hash"]

        line = json.dumps(rec, default=str)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        self._records_written += 1
        logger.debug("AuditLog: recorded order_id=%s fill=%.2f", order.order_id, fill.fill_price)

    def read_as_dataframe(self) -> pd.DataFrame:
        """Load the full audit log as a DataFrame (for TCA / reporting)."""
        if not self._path.exists():
            return pd.DataFrame()
        records = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("AuditLog: skipped malformed line.")
        return pd.DataFrame(records)

    def write_integrity_hash(self) -> str:
        """Compute SHA-256 of the current log file and write a ``.hash`` sidecar."""
        if not self._path.exists():
            return ""
        h = hashlib.sha256()
        with open(self._path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        digest = h.hexdigest()
        hash_path = self._path.with_suffix(".hash")
        hash_path.write_text(digest + "\n")
        return digest

    # ------------------------------------------------------------------
    # Wave 15 P1-24 — SHA-256 hash chain helpers
    # ------------------------------------------------------------------

    def _compute_hash(self, entry: dict, prev_hash: str) -> str:
        """Compute SHA-256 hash of entry + previous hash (chain link)."""
        payload = json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)
        chain_input = prev_hash + payload
        return hashlib.sha256(chain_input.encode("utf-8")).hexdigest()

    def _load_last_hash(self) -> str:
        """Read the last hash from the existing log file to continue the chain.

        Returns the genesis hash (64 zeros) if the file is absent or empty.
        """
        if not self._path.exists():
            return "0" * 64
        last_hash: str = "0" * 64
        try:
            with open(self._path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if "hash" in entry:
                                last_hash = entry["hash"]
                        except json.JSONDecodeError:
                            continue
        except OSError:
            pass
        return last_hash

    @property
    def records_written(self) -> int:
        return self._records_written
