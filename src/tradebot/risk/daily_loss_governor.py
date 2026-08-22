"""risk/daily_loss_governor.py — Propfirm compliance governor.

The propfirm parallel-deployment audit (artefacts/PROPFIRM_PARALLEL_AUDIT.md)
identified two HARD account-killer constraints that the existing peak-based
``DrawdownBreaker`` does NOT cover:

  1. **Daily loss limit** (~5% of the day-START balance) — an INTRADAY breach
     terminates the account.  The DrawdownBreaker measures peak-to-trough, not
     calendar-day-vs-start, so it cannot enforce this.
  2. **Static max drawdown** (~10% of the INITIAL balance, fixed floor) — NOT a
     rolling/high-water-mark drawdown.  Measured against the starting balance.

This module enforces BOTH against fixed reference balances, with buffered
internal thresholds (halt well before the firm's lines, leaving room for
slippage/latency on the final fills), plus a CHALLENGE/FUNDED regime that
scales the vol target and locks profit once funded.

Design mirrors ``risk.drawdown.DrawdownBreaker`` (dataclass config + enum
state + ``update`` returning a sizing decision) so it slots into the live
engine the same way.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

__all__ = [
    "GovernorAction",
    "PropfirmLimits",
    "GovernorDecision",
    "PropfirmGovernor",
    "Regime",
    "RegimeConfig",
    "target_gross_multiplier",
]


class GovernorAction(str, Enum):
    """Severity-ordered governor verdicts (later = more severe)."""

    OK = "OK"                      # normal trading
    SOFT_STOP = "SOFT_STOP"        # no new risk-INCREASING orders; reduce-only
    HARD_FLATTEN = "HARD_FLATTEN"  # flatten all; halt until daily reset
    FAIL = "FAIL"                  # firm hard line breached → account dead


_SEVERITY = {
    GovernorAction.OK: 0,
    GovernorAction.SOFT_STOP: 1,
    GovernorAction.HARD_FLATTEN: 2,
    GovernorAction.FAIL: 3,
}


@dataclass(frozen=True)
class PropfirmLimits:
    """Propfirm rule profile + buffered internal governor thresholds.

    Fractions are positive loss magnitudes (0.05 = 5 % loss).  ``firm_*`` are
    the EXCHANGE/firm hard lines (breach = account death); the internal
    thresholds sit strictly inside them so the governor halts with headroom.

    Defaults match the audit's canonical static profile: firm 10 % max-DD /
    5 % daily, internal halt at 6 % DD (resume 3 %), daily soft 3.5 % / hard
    4.5 %.
    """

    initial_balance: float
    firm_daily_loss: float = 0.05
    firm_max_dd: float = 0.10
    daily_soft: float = 0.035
    daily_hard: float = 0.045
    dd_trigger: float = 0.06
    dd_resume: float = 0.03

    def __post_init__(self) -> None:
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0.")
        if not (0 < self.daily_soft < self.daily_hard < self.firm_daily_loss):
            raise ValueError(
                "require 0 < daily_soft < daily_hard < firm_daily_loss "
                f"(got {self.daily_soft}, {self.daily_hard}, {self.firm_daily_loss})."
            )
        if not (0 < self.dd_resume < self.dd_trigger < self.firm_max_dd):
            raise ValueError(
                "require 0 < dd_resume < dd_trigger < firm_max_dd "
                f"(got {self.dd_resume}, {self.dd_trigger}, {self.firm_max_dd})."
            )


@dataclass(frozen=True)
class GovernorDecision:
    """Result of a governor update for one bar."""

    action: GovernorAction
    sizing_mult: float          # multiplier for NEW risk (0.0 when halted)
    allow_risk_increase: bool   # False on SOFT_STOP/HARD_FLATTEN/FAIL
    daily_loss: float           # positive fraction vs day-start
    static_dd: float            # positive fraction vs initial balance
    reason: str


class PropfirmGovernor:
    """Stateful daily-loss + static-max-DD governor with hysteresis.

    Usage
    -----
    >>> gov = PropfirmGovernor(PropfirmLimits(initial_balance=100_000))
    >>> gov.start_day(100_000)
    >>> d = gov.update(98_000)        # -2% on the day → still OK
    >>> d.action
    <GovernorAction.OK: 'OK'>
    """

    def __init__(self, limits: PropfirmLimits) -> None:
        self.limits = limits
        self._day_start: float = limits.initial_balance
        self._dd_tripped: bool = False
        self._failed: bool = False
        self._day_halted: bool = False  # latched until reset_day

    # ------------------------------------------------------------------
    @property
    def day_start_balance(self) -> float:
        return self._day_start

    @property
    def failed(self) -> bool:
        return self._failed

    def start_day(self, equity: float) -> None:
        """Set the day-start reference balance (call at the firm's daily reset)."""
        if equity <= 0:
            raise ValueError("equity must be > 0.")
        self._day_start = float(equity)
        self._day_halted = False

    # Alias matching SystemState.reset_daily semantics.
    reset_day = start_day

    def update(self, equity: float) -> GovernorDecision:
        """Evaluate ``equity`` against both constraints and return a decision."""
        lim = self.limits
        eq = float(equity)

        daily_loss = max(0.0, (self._day_start - eq) / self._day_start)
        static_dd = max(0.0, (lim.initial_balance - eq) / lim.initial_balance)

        # ── FAIL: a firm hard line is breached (should be unreachable if the
        #    internal thresholds did their job; latched permanently). ───────
        if self._failed or static_dd >= lim.firm_max_dd or daily_loss >= lim.firm_daily_loss:
            self._failed = True
            return self._decision(
                GovernorAction.FAIL, 0.0, False, daily_loss, static_dd,
                f"FIRM LINE BREACHED daily={daily_loss:.4f} static_dd={static_dd:.4f}",
            )

        action = GovernorAction.OK
        reason = "ok"

        # ── Static max-DD breaker with hysteresis (vs INITIAL balance) ──────
        if self._dd_tripped:
            if static_dd <= lim.dd_resume:
                self._dd_tripped = False
                logger.info("Governor: static-DD resumed (dd=%.4f <= %.4f).",
                            static_dd, lim.dd_resume)
            else:
                action = _max(action, GovernorAction.HARD_FLATTEN)
                reason = f"static_dd halt {static_dd:.4f} > resume {lim.dd_resume:.4f}"
        if not self._dd_tripped and static_dd >= lim.dd_trigger:
            self._dd_tripped = True
            action = _max(action, GovernorAction.HARD_FLATTEN)
            reason = f"static_dd TRIP {static_dd:.4f} >= {lim.dd_trigger:.4f}"
            logger.warning("Governor: static-DD TRIPPED dd=%.4f >= %.4f.",
                           static_dd, lim.dd_trigger)

        # ── Daily-loss governor (vs DAY-START balance) ──────────────────────
        if self._day_halted or daily_loss >= lim.daily_hard:
            self._day_halted = True  # latched until reset_day
            action = _max(action, GovernorAction.HARD_FLATTEN)
            reason = f"daily HARD {daily_loss:.4f} >= {lim.daily_hard:.4f}"
        elif daily_loss >= lim.daily_soft:
            action = _max(action, GovernorAction.SOFT_STOP)
            reason = f"daily SOFT {daily_loss:.4f} >= {lim.daily_soft:.4f}"

        allow_inc = action == GovernorAction.OK
        sizing = 1.0 if action == GovernorAction.OK else 0.0
        return self._decision(action, sizing, allow_inc, daily_loss, static_dd, reason)

    def _decision(self, action, sizing, allow_inc, daily_loss, static_dd, reason):
        return GovernorDecision(
            action=action, sizing_mult=sizing, allow_risk_increase=allow_inc,
            daily_loss=daily_loss, static_dd=static_dd, reason=reason,
        )


def _max(a: GovernorAction, b: GovernorAction) -> GovernorAction:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


# =============================================================================
# CHALLENGE / FUNDED regime + vol-target sizing
# =============================================================================
class Regime(str, Enum):
    CHALLENGE = "challenge"  # push to hit the profit target (higher vol)
    FUNDED = "funded"        # capital preservation + profit lock


@dataclass(frozen=True)
class RegimeConfig:
    """Per-regime sizing/behaviour.

    vol_target_mult : scales the funded vol target.  Audit: ~2× in challenge,
        1× when funded (more leverage hits the +8% target faster but lowers the
        pass-probability for a Sharpe~1.1 book — see audit §1).
    profit_lock_frac : when funded and cumulative profit >= this fraction, a
        soft-trailing stop is armed to bank a payout (0 disables).
    """

    regime: Regime = Regime.FUNDED
    vol_target_mult: float = 1.0
    profit_lock_frac: float = 0.06

    @staticmethod
    def challenge() -> "RegimeConfig":
        return RegimeConfig(regime=Regime.CHALLENGE, vol_target_mult=2.0, profit_lock_frac=0.0)

    @staticmethod
    def funded() -> "RegimeConfig":
        return RegimeConfig(regime=Regime.FUNDED, vol_target_mult=1.0, profit_lock_frac=0.06)


def target_gross_multiplier(
    target_vol_ann: float,
    realised_vol_ann: float,
    regime: RegimeConfig | None = None,
    cap: float = 3.0,
    floor: float = 0.0,
) -> float:
    """Gross-exposure scalar to hit a vol target given realised vol.

    Couples position sizing to the propfirm daily/DD budget (audit §3.3/§4):
    size so the book runs at ``target_vol_ann`` (× the regime multiplier).
    Returns ``(target × mult) / realised`` clamped to ``[floor, cap]``.

    A degenerate realised vol (≈0) returns ``floor`` (no information → no size).
    """
    mult = regime.vol_target_mult if regime is not None else 1.0
    if not (realised_vol_ann > 1e-9) or target_vol_ann <= 0:
        return floor
    raw = (target_vol_ann * mult) / realised_vol_ann
    return float(min(max(raw, floor), cap))
