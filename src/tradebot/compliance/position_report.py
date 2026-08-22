# src/tradebot/compliance/position_report.py
"""Dagelijkse positie-snapshot voor compliance rapportage.

Genereert een gestructureerd rapport van alle open posities, PnL, leverage
en risicometrieken aan het einde van elke handelsdag (UTC midnight).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["PositionSnapshot", "DailyPositionReport", "generate_position_report"]


@dataclass
class PositionSnapshot:
    """Snapshot of one position at report time."""

    symbol: str
    qty: float
    avg_entry_price: float
    mark_price: float
    notional_usdt: float
    unrealised_pnl: float
    realised_pnl_today: float
    position_age_h: float


@dataclass
class DailyPositionReport:
    """Daily position report for compliance.

    Attributes
    ----------
    report_date :
        UTC date of the report.
    equity :
        Mark-to-market equity at report time.
    gross_leverage :
        Sum of |notional| / equity.
    net_imbalance_pct :
        (Long notional - Short notional) / equity.
    daily_pnl :
        Total PnL for the day.
    positions :
        List of PositionSnapshot (one per open position).
    model_version :
        Model version that generated today's trades.
    git_sha :
        Git commit of the running code.
    """

    report_date: str
    equity: float
    gross_leverage: float
    net_imbalance_pct: float
    daily_pnl: float
    positions: list[PositionSnapshot] = field(default_factory=list)
    model_version: str = ""
    git_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_date": self.report_date,
            "equity": self.equity,
            "gross_leverage": self.gross_leverage,
            "net_imbalance_pct": self.net_imbalance_pct,
            "daily_pnl": self.daily_pnl,
            "model_version": self.model_version,
            "git_sha": self.git_sha,
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "mark_price": p.mark_price,
                    "notional_usdt": p.notional_usdt,
                    "unrealised_pnl": p.unrealised_pnl,
                    "realised_pnl_today": p.realised_pnl_today,
                    "position_age_h": p.position_age_h,
                }
                for p in self.positions
            ],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return positions as a DataFrame."""
        return pd.DataFrame([p.__dict__ for p in self.positions])


def generate_position_report(
    equity: float,
    positions: dict[str, Any],  # symbol → PositionRecord
    daily_pnl: float,
    model_version: str = "",
    git_sha: str = "",
    report_date: str | None = None,
    realised_pnl_today_by_sym: dict[str, float] | None = None,
    position_age_h_by_sym: dict[str, float] | None = None,
) -> DailyPositionReport:
    """Build a DailyPositionReport from live PositionTracker state.

    Parameters
    ----------
    positions :
        Dict of symbol → PositionRecord (from PositionTracker).
    realised_pnl_today_by_sym :
        CHIEF AUDIT 2026-05-23 (H10): per-symbol realised PnL since midnight
        UTC.  Caller MOET dit aanleveren voor compliance-grade rapportage —
        anders blijft het veld 0.0 en wordt een warning gelogd wanneer er
        actieve posities zijn.  Typische bron: ``PositionTracker.realised_pnl_today``
        of een dagelijkse-reset accumulator.
    position_age_h_by_sym :
        Per-symbol open-time leeftijd in uren (now_utc - open_ts).  Idem
        compliance-grade aanlevering vereist; anders 0.0 + warning.
    """
    if report_date is None:
        report_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    snapshots: list[PositionSnapshot] = []
    total_notional = 0.0
    long_notional = 0.0
    short_notional = 0.0

    _rpnl_map = realised_pnl_today_by_sym or {}
    _age_map = position_age_h_by_sym or {}

    for symbol, pos in positions.items():
        if abs(pos.qty) < 1e-12:
            continue
        notional = abs(pos.qty) * pos.last_mark_price
        total_notional += notional
        if pos.qty > 0:
            long_notional += notional
        else:
            short_notional += notional

        realised_pnl_today = float(_rpnl_map.get(symbol, 0.0))
        position_age_h = float(_age_map.get(symbol, 0.0))

        snapshots.append(
            PositionSnapshot(
                symbol=symbol,
                qty=pos.qty,
                avg_entry_price=pos.avg_entry_price,
                mark_price=pos.last_mark_price,
                notional_usdt=notional,
                unrealised_pnl=pos.unrealised_pnl,
                realised_pnl_today=realised_pnl_today,
                position_age_h=position_age_h,
            )
        )

    # CHIEF AUDIT 2026-05-23 (H10): runtime warning bij ontbrekende intraday
    # tracking. Open posities zonder realised_pnl_today / position_age_h zijn
    # niet compliance-grade.
    if snapshots and (realised_pnl_today_by_sym is None and position_age_h_by_sym is None):
        logger.warning(
            "PositionSnapshot: realised_pnl_today and position_age_h are 0.0 — "
            "compliance reports may be incomplete. Pass "
            "realised_pnl_today_by_sym / position_age_h_by_sym from the live "
            "PositionTracker.",
        )

    gross_leverage = total_notional / equity if equity > 1e-12 else 0.0
    net_imbalance = (long_notional - short_notional) / equity if equity > 1e-12 else 0.0

    return DailyPositionReport(
        report_date=report_date,
        equity=equity,
        gross_leverage=gross_leverage,
        net_imbalance_pct=net_imbalance,
        daily_pnl=daily_pnl,
        positions=snapshots,
        model_version=model_version,
        git_sha=git_sha,
    )
