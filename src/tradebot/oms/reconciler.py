# src/tradebot/oms/reconciler.py
"""Exchange-position vs. internal-book reconciliation.

In live mode, periodically queries the exchange REST API for actual
positions and compares them against the PositionTracker's internal book.
Discrepancies above a threshold trigger a WARNING and (optionally) a
circuit-breaker notification.

In paper mode, the reconciler is a no-op (internal book IS the ground truth).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .position_tracker import PositionRecord, PositionTracker

logger = logging.getLogger(__name__)

__all__ = ["ReconcilerConfig", "ReconciliationResult", "Reconciler"]


@dataclass
class ReconcilerConfig:
    """Configuration for the reconciler.

    Attributes
    ----------
    qty_tolerance :
        Maximum absolute quantity discrepancy (base-asset units) before
        raising a warning.
    notional_tolerance_pct :
        Maximum relative notional discrepancy (0.01 = 1%) before warning.
    paper_mode :
        If True, skip all exchange calls — always returns no discrepancies.
    fill_grace_period_sec :
        CHIEF AUDIT-FIX (Sim-to-Reality #15): seconds since the last fill
        on a symbol during which a discrepancy is tagged as ``PENDING``
        rather than ``DISCREPANT``.  Bybit REST snapshots lag the live
        order stream by 1-3 seconds; without this grace window every
        just-filled order produces a false positive that drowns out real
        reconciliation events.  Default 5.0 covers the observed P99 lag.
    require_two_strikes :
        When True a discrepancy must persist across two consecutive
        reconciliation passes before being flagged.  Cheap defence
        against intermittent REST hiccups; default True.
    """

    qty_tolerance: float = 1e-6
    notional_tolerance_pct: float = 0.005
    paper_mode: bool = True
    fill_grace_period_sec: float = 5.0
    # CHIEF AUDIT-FIX #15: opt-in two-strikes (default False = original
    # behaviour for backward-compat); production callers should set True.
    require_two_strikes: bool = False


@dataclass
class ReconciliationResult:
    """Result of one reconciliation pass."""

    symbol: str
    internal_qty: float
    exchange_qty: float
    qty_diff: float
    is_discrepant: bool
    message: str = ""


class Reconciler:
    """Reconciles internal position book against exchange state.

    In live mode, ``exchange_positions`` must be populated by the caller
    (typically from a Bybit REST ``GET /v5/position/list`` response).
    In paper mode, all reconciliation passes return ``is_discrepant=False``.

    Parameters
    ----------
    tracker :
        The PositionTracker holding the internal book.
    config :
        Reconciler configuration.
    """

    def __init__(self, tracker: PositionTracker, config: Optional[ReconcilerConfig] = None) -> None:
        self._tracker = tracker
        self._cfg = config or ReconcilerConfig()
        # CHIEF AUDIT-FIX #15 — recent-fill tracking for the grace window
        self._last_fill_ts: dict[str, float] = {}
        # CHIEF AUDIT-FIX #15 — two-strikes state: symbol → consecutive bad passes
        self._strike_count: dict[str, int] = {}

    def note_fill(self, symbol: str, ts: Optional[float] = None) -> None:
        """Record the timestamp of the most recent fill on ``symbol``.

        Called by the router whenever an order is filled.  The reconciler
        uses this to grant a grace period before flagging discrepancies.
        """
        self._last_fill_ts[symbol] = float(ts if ts is not None else time.time())

    def reconcile(
        self,
        exchange_positions: Optional[Dict[str, float]] = None,
        now: Optional[float] = None,
    ) -> list[ReconciliationResult]:
        """Run one reconciliation pass.

        Parameters
        ----------
        exchange_positions :
            Mapping of symbol → net qty from the exchange.  None in paper mode.
        now :
            Wall-clock time (UNIX seconds) for grace-period evaluation.
            Defaults to ``time.time()``.

        Returns
        -------
        List of ReconciliationResult, one per symbol in the internal book.
        """
        if self._cfg.paper_mode or exchange_positions is None:
            return []

        now_ts = float(now if now is not None else time.time())
        results: list[ReconciliationResult] = []
        for symbol, pos in self._tracker.get_all_positions().items():
            exch_qty = exchange_positions.get(symbol, 0.0)
            diff = abs(pos.qty - exch_qty)
            raw_discrepant = diff > self._cfg.qty_tolerance

            # CHIEF AUDIT-FIX #15: grace window for just-filled orders.
            recent_fill = (
                symbol in self._last_fill_ts
                and (now_ts - self._last_fill_ts[symbol]) < self._cfg.fill_grace_period_sec
            )

            # CHIEF AUDIT-FIX #15: two-strikes — only flag after persistent fail.
            if raw_discrepant and not recent_fill:
                self._strike_count[symbol] = self._strike_count.get(symbol, 0) + 1
            else:
                self._strike_count[symbol] = 0

            discrepant = (
                raw_discrepant
                and not recent_fill
                and (
                    not self._cfg.require_two_strikes
                    or self._strike_count[symbol] >= 2
                )
            )

            if discrepant:
                msg = (
                    f"RECONCILIATION DISCREPANCY: {symbol} internal={pos.qty:.6f} "
                    f"exchange={exch_qty:.6f} diff={diff:.6f}"
                )
                logger.warning(msg)
            elif raw_discrepant and recent_fill:
                msg = (
                    f"PENDING (grace): {symbol} diff={diff:.6f} — recent fill "
                    f"{now_ts - self._last_fill_ts[symbol]:.2f}s ago"
                )
                logger.debug(msg)
            elif raw_discrepant:
                msg = (
                    f"PENDING (strike {self._strike_count[symbol]}): "
                    f"{symbol} diff={diff:.6f}"
                )
                logger.debug(msg)
            else:
                msg = f"OK: {symbol} qty={pos.qty:.6f}"

            results.append(
                ReconciliationResult(
                    symbol=symbol,
                    internal_qty=pos.qty,
                    exchange_qty=exch_qty,
                    qty_diff=diff,
                    is_discrepant=discrepant,
                    message=msg,
                )
            )
        return results
