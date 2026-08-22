# src/tradebot/live/state.py
"""SystemState dataclass — single source of truth for the live engine.

All components of the LiveEngine read from and write to this shared state
object.  It is NOT thread-safe by design: asyncio's single-threaded event
loop provides the necessary serialisation guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

__all__ = ["EngineMode", "SystemState"]


class EngineMode(str, Enum):
    PAPER = "paper"
    SHADOW = "shadow"   # real Bybit WS feed + PaperOMS (no real orders)
    LIVE = "live"
    STOPPED = "stopped"


@dataclass
class SystemState:
    """Mutable engine state shared across all live components.

    Attributes
    ----------
    mode :
        Current operating mode.
    equity :
        Latest mark-to-market equity in USDT.
    equity_peak :
        All-time equity peak (used for drawdown calculation).
    positions :
        symbol → net_qty mapping (current open positions).
    target_weights :
        symbol → target portfolio weight from last optimisation.
    last_bar :
        symbol → last received bar as a single-row DataFrame.
    last_bar_ts :
        symbol → timestamp of the last received bar.
    last_feed_ts :
        Timestamp of the last feed event received (any symbol).
    circuit_breaker_active :
        True when any halt condition is triggered.
    daily_pnl :
        Realised + unrealised PnL since last midnight UTC.
    daily_pnl_open :
        Equity at the start of the current UTC day.
    errors :
        List of error messages accumulated during the session.
    """

    mode: EngineMode = EngineMode.PAPER
    equity: float = 100_000.0
    equity_peak: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] = field(default_factory=dict)
    last_bar: dict[str, pd.DataFrame] = field(default_factory=dict)
    last_bar_ts: dict[str, pd.Timestamp] = field(default_factory=dict)
    last_feed_ts: pd.Timestamp | None = None
    circuit_breaker_active: bool = False
    daily_pnl: float = 0.0
    daily_pnl_open: float = 100_000.0
    errors: list[str] = field(default_factory=list)
    # CHIEF-4 (2026-05-28): historic *max* drawdown (worst peak-to-trough fraction
    # observed since start).  ``current_drawdown`` is INSTANTANEOUS — it returns 0
    # the moment equity recovers to a new peak, which made the dashboard "Max DD"
    # KPI read 0.00% even after a -1.35 % intraday dip.  We track the historic
    # max here so the dashboard and audit can read the real worst case.
    peak_drawdown: float = 0.0
    # CHIEF-4: per-UTC-session worst drawdown.  Reset every midnight by the
    # CircuitBreaker's session roll.  Useful for ops to see "worst dip today".
    intraday_peak_equity: float = 100_000.0
    intraday_peak_drawdown: float = 0.0

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def current_drawdown(self) -> float:
        """Drawdown from equity_peak as a fraction in [0, 1]."""
        if self.equity_peak <= 1e-12:
            return 0.0
        return max(0.0, (self.equity_peak - self.equity) / self.equity_peak)

    @property
    def daily_pnl_pct(self) -> float:
        """Today's PnL as a fraction of daily_pnl_open."""
        if self.daily_pnl_open <= 1e-12:
            return 0.0
        return self.daily_pnl / self.daily_pnl_open

    def update_equity(self, new_equity: float) -> None:
        """Update equity and track the all-time peak.

        CHIEF-4 (2026-05-28): also tracks ``peak_drawdown`` (historic worst
        peak-to-trough) and ``intraday_peak_drawdown`` (session worst), so the
        dashboard "Max Drawdown" KPI reflects the actual worst dip rather than
        the current (post-recovery) drawdown.
        """
        self.equity = new_equity
        self.equity_peak = max(self.equity_peak, new_equity)
        self.intraday_peak_equity = max(self.intraday_peak_equity, new_equity)
        # Lifetime peak DD
        if self.equity_peak > 1e-12:
            cur_dd = max(0.0, (self.equity_peak - new_equity) / self.equity_peak)
            self.peak_drawdown = max(self.peak_drawdown, cur_dd)
        # Intraday peak DD
        if self.intraday_peak_equity > 1e-12:
            intra_dd = max(0.0, (self.intraday_peak_equity - new_equity) / self.intraday_peak_equity)
            self.intraday_peak_drawdown = max(self.intraday_peak_drawdown, intra_dd)
        self.daily_pnl = new_equity - self.daily_pnl_open

    def reset_daily(self) -> None:
        """Called at UTC midnight to reset daily PnL tracking.

        CHIEF-4: also resets the intraday DD tracking so ops sees "worst dip
        today" rather than "worst dip ever".
        """
        self.daily_pnl_open = self.equity
        self.daily_pnl = 0.0
        self.intraday_peak_equity = self.equity
        self.intraday_peak_drawdown = 0.0
