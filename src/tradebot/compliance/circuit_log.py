# src/tradebot/compliance/circuit_log.py
"""Log elke circuit-breaker trigger met volledige context.

Append-only JSONL log; elke record bevat de volledige systeemtoestand op het
moment van trippen zodat post-incident reconstructie mogelijk is.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["CircuitLogEntry", "CircuitLog"]


@dataclass
class CircuitLogEntry:
    """One circuit-breaker trigger event.

    Attributes
    ----------
    event_ts :
        UTC timestamp of the trigger.
    halt_reason :
        HaltReason constant (e.g. "MAX_DRAWDOWN").
    equity :
        Equity at trigger time.
    equity_peak :
        All-time equity peak at trigger time.
    current_drawdown :
        Drawdown fraction at trigger time.
    daily_pnl_pct :
        Today's PnL fraction at trigger time.
    open_positions :
        snapshot of symbol → qty at trigger time.
    last_feed_ts :
        Timestamp of last received bar.
    model_version :
        Model version string.
    git_sha :
        Git commit of the running code.
    extra :
        Any additional context (e.g. feature_hash, order_id).
    """

    event_ts: str
    halt_reason: str
    equity: float
    equity_peak: float
    current_drawdown: float
    daily_pnl_pct: float
    open_positions: dict[str, float]
    last_feed_ts: str
    model_version: str
    git_sha: str
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CircuitLog:
    """Append-only JSONL circuit-breaker event log.

    Parameters
    ----------
    log_path :
        Path to the JSONL log file.
    """

    def __init__(self, log_path: Path | str) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        halt_reason: str,
        equity: float,
        equity_peak: float,
        current_drawdown: float,
        daily_pnl_pct: float,
        open_positions: dict[str, float],
        last_feed_ts: pd.Timestamp | None = None,
        model_version: str = "",
        git_sha: str = "",
        extra: dict[str, Any] | None = None,
    ) -> CircuitLogEntry:
        """Append one circuit-breaker event to the log."""
        entry = CircuitLogEntry(
            event_ts=pd.Timestamp.now(tz="UTC").isoformat(),
            halt_reason=halt_reason,
            equity=equity,
            equity_peak=equity_peak,
            current_drawdown=current_drawdown,
            daily_pnl_pct=daily_pnl_pct,
            open_positions=dict(open_positions),
            last_feed_ts=last_feed_ts.isoformat() if last_feed_ts else "",
            model_version=model_version,
            git_sha=git_sha,
            extra=extra or {},
        )
        line = json.dumps(entry.to_dict(), default=str)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.critical(
            "CircuitLog: halt_reason=%s equity=%.2f dd=%.4f",
            halt_reason, equity, current_drawdown,
        )
        return entry

    def read_all(self) -> list[CircuitLogEntry]:
        """Return all logged circuit-breaker events."""
        if not self._path.exists():
            return []
        entries: list[CircuitLogEntry] = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append(CircuitLogEntry(**d))
                except (json.JSONDecodeError, TypeError):
                    logger.warning("CircuitLog: skipped malformed line.")
        return entries

    def as_dataframe(self) -> pd.DataFrame:
        entries = self.read_all()
        if not entries:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in entries])
