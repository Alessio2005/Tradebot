# src/tradebot/live/circuit_breaker.py
"""Dead-man's switch for the live trading engine (R-7).

Halt conditions (any one sufficient):
  1. max_drawdown_pct : equity drawdown from peak > threshold
  2. max_daily_loss_pct : today's loss > threshold
  3. max_var_breach : realised VaR > 1.5× limit (placeholder — requires VaR model)
  4. feed_timeout_sec : no bar received for > N seconds
  5. model_hash_mismatch : feature hash at predict time != training-time hash
  6. max_position_age_h : position open > 48h without a closing signal

On HALT:
  1. All open positions closed (OMS.close_all())
  2. Engine stopped (engine.request_stop())
  3. CRITICAL alert sent
  4. CircuitLog written
  5. NEVER auto-resume — human intervention required
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .state import SystemState

logger = logging.getLogger(__name__)

__all__ = ["CircuitBreakerConfig", "CircuitBreaker", "HaltReason"]

# Wave 15 P0-5.5 — append-only circuit-breaker trip log
_CB_LOG_PATH = pathlib.Path("artefacts/circuit_breaker.log")


class HaltReason(str):
    """Halt reason constants."""
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    DAILY_LOSS = "DAILY_LOSS"
    VAR_BREACH = "VAR_BREACH"
    FEED_TIMEOUT = "FEED_TIMEOUT"
    HASH_MISMATCH = "HASH_MISMATCH"
    POSITION_AGE = "POSITION_AGE"


@dataclass
class CircuitBreakerConfig:
    """Parameters for the dead-man's switch.

    Attributes
    ----------
    max_drawdown_pct :
        Stop when equity drawdown from peak exceeds this fraction.
    max_intraday_drawdown_pct :
        Stop when drawdown from the *session* (intraday) peak exceeds this
        fraction.  CHIEF AUDIT-FIX (Sim-to-Reality #14): previously only
        lifetime peak was tracked, so a normal recovery-then-relapse pattern
        on a single day could re-trip CB once equity hovered near the
        lifetime threshold for months.  Intraday tracking gives an
        operationally meaningful per-session view.  Set to 0.0 to disable.
    max_daily_loss_pct :
        Stop when today's loss exceeds this fraction of opening equity.
    max_var_breach :
        Multiplier on the VaR limit (placeholder, future use).
    feed_timeout_sec :
        Stop when no bar is received for this many seconds.
    model_hash_mismatch :
        Stop on feature-hash mismatch at predict time.
    max_position_age_h :
        Force-close positions older than this many hours.
    cb_log_path :
        Path to the append-only circuit-breaker trip log (Wave 15 P0-5.5).
        Set to a tmp path in tests to avoid cross-test contamination.
        ``None`` uses the module-level default ``artefacts/circuit_breaker.log``.
    """

    max_drawdown_pct: float = 0.08
    max_intraday_drawdown_pct: float = 0.05
    max_daily_loss_pct: float = 0.03
    max_var_breach: float = 1.5
    feed_timeout_sec: int = 30
    model_hash_mismatch: bool = True
    max_position_age_h: int = 48
    cb_log_path: Optional[pathlib.Path] = None


class CircuitBreaker:
    """Stateful circuit-breaker checked after every bar.

    Parameters
    ----------
    config :
        Halt thresholds.
    state :
        Shared engine state.
    """

    def __init__(self, config: CircuitBreakerConfig, state: SystemState) -> None:
        self._cfg = config
        self._state = state
        self._halt_reason: Optional[str] = None
        self._halt_ts: Optional[pd.Timestamp] = None
        self._checks: int = 0

        # CHIEF AUDIT-FIX (Sim-to-Reality #14): session peak tracking for
        # intraday drawdown.  Reset every UTC midnight by ``_maybe_roll_session``.
        self._session_peak_equity: float = float(self._state.equity)
        self._session_date: Optional[pd.Timestamp] = None

        # Wave 15 P0-5.5 — resolve effective log path (config overrides module default)
        self._cb_log_path: pathlib.Path = (
            config.cb_log_path if config.cb_log_path is not None else _CB_LOG_PATH
        )

        # Wave 15 P0-5.5 — refuse start if unacknowledged trip < 24h ago
        self._check_prior_trips()

    @property
    def is_active(self) -> bool:
        return self._state.circuit_breaker_active

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    # ------------------------------------------------------------------
    # Per-bar check
    # ------------------------------------------------------------------

    def check(
        self,
        now: pd.Timestamp,
        feature_hash: Optional[str] = None,
        expected_hash: Optional[str] = None,
        position_open_ts: Optional[pd.Timestamp] = None,
    ) -> Optional[str]:
        """Run all halt conditions.  Returns the halt reason string or None.

        Side-effect: sets ``state.circuit_breaker_active = True`` on first trip.
        Subsequent calls after a trip return the original halt reason immediately
        (no re-evaluation; human intervention required to reset).
        """
        self._checks += 1

        if self._state.circuit_breaker_active:
            return self._halt_reason

        reason = self._evaluate(now, feature_hash, expected_hash, position_open_ts)
        if reason:
            self._trip(reason, now)
        return reason

    def _maybe_roll_session(self, now: pd.Timestamp) -> None:
        """Reset the intraday session peak at each new UTC date."""
        cur_date = pd.Timestamp(now).normalize()
        if self._session_date is None or cur_date != self._session_date:
            self._session_date = cur_date
            self._session_peak_equity = float(self._state.equity)
        else:
            self._session_peak_equity = max(
                self._session_peak_equity, float(self._state.equity)
            )

    def _intraday_drawdown(self) -> float:
        if self._session_peak_equity <= 0.0:
            return 0.0
        return max(
            0.0,
            (self._session_peak_equity - float(self._state.equity))
            / self._session_peak_equity,
        )

    def _evaluate(
        self,
        now: pd.Timestamp,
        feature_hash: Optional[str],
        expected_hash: Optional[str],
        position_open_ts: Optional[pd.Timestamp],
    ) -> Optional[str]:
        # CHIEF AUDIT-FIX (Sim-to-Reality #14): session tracking precedes checks.
        self._maybe_roll_session(now)

        # 1. Max drawdown (lifetime — catastrophic)
        if self._state.current_drawdown >= self._cfg.max_drawdown_pct:
            return HaltReason.MAX_DRAWDOWN

        # 1b. Intraday drawdown (operational — single session)
        if (
            self._cfg.max_intraday_drawdown_pct > 0.0
            and self._intraday_drawdown() >= self._cfg.max_intraday_drawdown_pct
        ):
            return HaltReason.MAX_DRAWDOWN

        # 2. Daily loss
        if -self._state.daily_pnl_pct >= self._cfg.max_daily_loss_pct:
            return HaltReason.DAILY_LOSS

        # 3. Feed timeout
        if self._state.last_feed_ts is not None:
            elapsed = (now - self._state.last_feed_ts).total_seconds()
            if elapsed > self._cfg.feed_timeout_sec:
                return HaltReason.FEED_TIMEOUT

        # 4. Model hash mismatch
        if (
            self._cfg.model_hash_mismatch
            and feature_hash is not None
            and expected_hash is not None
            and feature_hash != expected_hash
        ):
            return HaltReason.HASH_MISMATCH

        # 5. Position age
        if position_open_ts is not None:
            age_h = (now - position_open_ts).total_seconds() / 3600.0
            if age_h > self._cfg.max_position_age_h:
                return HaltReason.POSITION_AGE

        return None

    def _trip(self, reason: str, ts: pd.Timestamp) -> None:
        self._halt_reason = reason
        self._halt_ts = ts
        self._state.circuit_breaker_active = True
        logger.critical(
            "CIRCUIT BREAKER TRIPPED: reason=%s ts=%s "
            "equity=%.2f dd=%.4f daily_pnl_pct=%.4f",
            reason, ts.isoformat(),
            self._state.equity,
            self._state.current_drawdown,
            self._state.daily_pnl_pct,
        )
        # Wave 15 P0-5.5 — persist trip to append-only log
        self._persist_trip(reason)
        # Fire Slack / stdout alert (routed via TRADEBOT_SLACK_WEBHOOK if set)
        try:
            from ..monitoring.alerts import send_alert, AlertSeverity
            send_alert(
                title="CIRCUIT BREAKER TRIPPED",
                message=(
                    f"reason={reason} ts={ts.isoformat()} "
                    f"equity={self._state.equity:.2f} "
                    f"dd={self._state.current_drawdown:.4f} "
                    f"daily_pnl_pct={self._state.daily_pnl_pct:.4f}"
                ),
                severity=AlertSeverity.CRITICAL,
                metadata={
                    "reason": reason,
                    "equity": f"{self._state.equity:.2f}",
                    "drawdown": f"{self._state.current_drawdown:.4f}",
                },
            )
        except Exception as alert_exc:
            logger.error("CircuitBreaker: alert delivery failed: %s", alert_exc)

    def reset(self) -> None:
        """Manual override reset (requires human intervention)."""
        self._state.circuit_breaker_active = False
        self._halt_reason = None
        self._halt_ts = None
        logger.warning("CircuitBreaker MANUALLY RESET by operator.")

    # ------------------------------------------------------------------
    # Wave 15 P0-5.5 — Circuit breaker state persistence
    # ------------------------------------------------------------------

    def _persist_trip(self, reason: str) -> None:
        """Write CB trip to append-only log (Wave 15 P0-5.5)."""
        # Fall back to module-level default if _cb_log_path not yet initialised
        # (e.g. when called via CircuitBreaker.__new__ in tests)
        log_path = getattr(self, "_cb_log_path", _CB_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "reason": reason,
            "acknowledged": False,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _check_prior_trips(self) -> None:
        """Refuse start if unacknowledged trip < 24h ago (Wave 15 P0-5.5)."""
        if not self._cb_log_path.exists():
            return
        cutoff_ts = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
        with open(self._cb_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if not entry.get("acknowledged", True):
                        trip_ts = pd.Timestamp(entry["ts"])
                        if trip_ts > cutoff_ts:
                            raise RuntimeError(
                                f"CircuitBreaker: Unacknowledged trip at {trip_ts} "
                                f"(reason: {entry['reason']}). Acknowledge in "
                                f"{self._cb_log_path} before restarting."
                            )
                except (json.JSONDecodeError, KeyError):
                    continue
