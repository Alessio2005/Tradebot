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
from typing import Any

import pandas as pd

from ..risk.kill_switches import HaltRecord, HaltStore
from ..schemas.config import RiskConfig
from .state import SystemState

logger = logging.getLogger(__name__)

__all__ = ["CircuitBreakerConfig", "CircuitBreaker", "HaltReason"]


@dataclass(frozen=True)
class _Breach:
    """Een geschonden halteervoorwaarde, met de cijfers erbij.

    Bestaat zodat het halt-journaal kan vastleggen HOEVER de drempel werd
    overschreden en welke configuratiesleutel hem stelde. Een reden zonder
    getallen maakt een post-mortem tot giswerk.
    """

    reason: str
    measured: float
    threshold: float
    config_key: str

#: De wortel van de repo, afgeleid van dit bestand en NIET van de
#: werkdirectory. Beide paden hieronder waren `Path("artefacts/...")` --
#: relatief aan `os.getcwd()`. Dat is voor een noodrem onhoudbaar: start het
#: live-proces vanuit een andere map (een systemd-unit zonder
#: `WorkingDirectory`, een cron-job, een submap) en `_refuse_start_when_halted`
#: kijkt in een pad dat niet bestaat, vindt dus geen halt, en begint te handelen
#: terwijl het boek gesloten hoort te zijn -- waarna `_trip` de halt in weer een
#: ander bestand schrijft. De rest van de boom leidt zijn soevereine paden al
#: zo af (`live/engine.py`, `schemas/config.py`, `apps/freeze_monitoring.py`).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Wave 15 P0-5.5 — append-only circuit-breaker trip log
_CB_LOG_PATH = _REPO_ROOT / "artefacts" / "circuit_breaker.log"

#: De SOEVEREINE halt-state. Dit is dezelfde store die `risk/engine.py`
#: gebruikt; de live-breaker heeft er geen eigen variant meer naast.
_HALT_STORE_PATH = _REPO_ROOT / "artefacts" / "risk" / "halt_state.json"


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
    cb_log_path: pathlib.Path | None = None
    #: Pad naar de SOEVEREINE halt-state (`risk/kill_switches.py::HaltStore`).
    #: `None` gebruikt het pad uit de module-default.
    halt_store_path: pathlib.Path | None = None

    @classmethod
    def from_risk_config(
        cls,
        risk: RiskConfig,
        *,
        halt_store_path: pathlib.Path | None = None,
        **overrides: Any,
    ) -> CircuitBreakerConfig:
        """Bouw de drempels uit de soevereine policy in plaats van uit code.

        STAGE D, C1. Tot deze wijziging droeg deze dataclass haar eigen
        `max_drawdown_pct = 0.08` en `max_daily_loss_pct = 0.03`. Die getallen
        waren GELIJK aan `conf/risk/default.yaml` -- en er was niets dat ze
        gelijk hield. Twee onafhankelijke kopieen die vandaag toevallig
        overeenkomen, zijn geen limietstelsel; zie
        `reports/phase7_divergence_map.md` §4.

        Wat NIET uit `RiskConfig` komt, komt er ook niet in: `feed_timeout_sec`
        en `model_hash_mismatch` zijn operationele condities van de live-loop en
        hebben in de backtest geen tegenhanger. Ze staan hier expliciet als
        live-eigen, niet als stilzwijgend verschil.
        """
        return cls(
            max_drawdown_pct=float(risk.max_drawdown_pct),
            max_daily_loss_pct=float(risk.daily_loss_limit),
            max_position_age_h=int(risk.max_position_age_h),
            halt_store_path=halt_store_path,
            **overrides,
        )


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
        self._halt_reason: str | None = None
        self._halt_ts: pd.Timestamp | None = None
        self._checks: int = 0

        # CHIEF AUDIT-FIX (Sim-to-Reality #14): session peak tracking for
        # intraday drawdown.  Reset every UTC midnight by ``_maybe_roll_session``.
        self._session_peak_equity: float = float(self._state.equity)
        self._session_date: pd.Timestamp | None = None

        # Wave 15 P0-5.5 — resolve effective log path (config overrides module default)
        self._cb_log_path: pathlib.Path = (
            config.cb_log_path if config.cb_log_path is not None else _CB_LOG_PATH
        )

        # STAGE D, C2 — de halt-toestand staat in de SOEVEREINE store. Het
        # append-only CB-log blijft bestaan als operationeel spoor, maar het is
        # niet langer de bron van waarheid over "zijn we gehalt".
        self._halt_store = HaltStore(
            config.halt_store_path if config.halt_store_path is not None
            else _HALT_STORE_PATH
        )
        self._refuse_start_when_halted()

    @property
    def is_active(self) -> bool:
        return self._state.circuit_breaker_active

    @property
    def halt_reason(self) -> str | None:
        return self._halt_reason

    # ------------------------------------------------------------------
    # Per-bar check
    # ------------------------------------------------------------------

    def check(
        self,
        now: pd.Timestamp,
        feature_hash: str | None = None,
        expected_hash: str | None = None,
        position_open_ts: pd.Timestamp | None = None,
    ) -> str | None:
        """Run all halt conditions.  Returns the halt reason string or None.

        Side-effect: sets ``state.circuit_breaker_active = True`` on first trip.
        Subsequent calls after a trip return the original halt reason immediately
        (no re-evaluation; human intervention required to reset).
        """
        self._checks += 1

        if self._state.circuit_breaker_active:
            return self._halt_reason

        breach = self._evaluate(now, feature_hash, expected_hash,
                                position_open_ts)
        if breach is None:
            return None
        self._trip(breach, now)
        return breach.reason

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
        feature_hash: str | None,
        expected_hash: str | None,
        position_open_ts: pd.Timestamp | None,
    ) -> _Breach | None:
        """De halteervoorwaarden, elk met de GEMETEN waarde en zijn drempel.

        Tot Stage D gaf deze functie alleen een reden terug. Voor het
        halt-journaal is dat te weinig: een post-mortem wil weten hoe ver de
        drempel werd overschreden en welke configuratiesleutel hem stelde. Zie
        `HaltRecord`.
        """
        # CHIEF AUDIT-FIX (Sim-to-Reality #14): session tracking precedes checks.
        self._maybe_roll_session(now)

        # 1. Max drawdown (lifetime — catastrophic)
        if self._state.current_drawdown >= self._cfg.max_drawdown_pct:
            return _Breach(
                HaltReason.MAX_DRAWDOWN, float(self._state.current_drawdown),
                float(self._cfg.max_drawdown_pct), "risk.max_drawdown_pct")

        # 1b. Intraday drawdown (operational — single session)
        if (
            self._cfg.max_intraday_drawdown_pct > 0.0
            and self._intraday_drawdown() >= self._cfg.max_intraday_drawdown_pct
        ):
            return _Breach(
                HaltReason.MAX_DRAWDOWN, float(self._intraday_drawdown()),
                float(self._cfg.max_intraday_drawdown_pct),
                "live.max_intraday_drawdown_pct")

        # 2. Daily loss
        if -self._state.daily_pnl_pct >= self._cfg.max_daily_loss_pct:
            return _Breach(
                HaltReason.DAILY_LOSS, float(-self._state.daily_pnl_pct),
                float(self._cfg.max_daily_loss_pct), "risk.daily_loss_limit")

        # 3. Feed timeout
        if self._state.last_feed_ts is not None:
            elapsed = (now - self._state.last_feed_ts).total_seconds()
            if elapsed > self._cfg.feed_timeout_sec:
                return _Breach(
                    HaltReason.FEED_TIMEOUT, float(elapsed),
                    float(self._cfg.feed_timeout_sec), "live.feed_timeout_sec")

        # 4. Model hash mismatch
        if (
            self._cfg.model_hash_mismatch
            and feature_hash is not None
            and expected_hash is not None
            and feature_hash != expected_hash
        ):
            return _Breach(
                HaltReason.HASH_MISMATCH, 1.0, 0.0, "live.model_hash_mismatch")

        # 5. Position age
        if position_open_ts is not None:
            age_h = (now - position_open_ts).total_seconds() / 3600.0
            if age_h > self._cfg.max_position_age_h:
                return _Breach(
                    HaltReason.POSITION_AGE, float(age_h),
                    float(self._cfg.max_position_age_h),
                    "risk.max_position_age_h")

        return None

    def trip(
        self,
        reason: str,
        *,
        measured: float = 1.0,
        threshold: float = 0.0,
        config_key: str = "live.external_monitor",
        ts: pd.Timestamp | None = None,
    ) -> None:
        """Halteer op grond van een EXTERNE waarnemer.

        Bestaat omdat niet elke halteervoorwaarde uit `_evaluate` komt: de
        exchange-statusmonitor en de live-Sharpe-monitor meten iets dat deze
        klasse zelf niet ziet. Zij riepen `_trip` aan met een STRING, terwijl
        die sinds Stage D een `_Breach` verwacht -- de halt kwam er dus niet en
        de monitor stierf aan een `AttributeError` op het moment dat hij nodig
        was. Dit is de publieke, getypeerde ingang; `_Breach` blijft privé.
        """
        self._trip(
            _Breach(reason, float(measured), float(threshold), config_key),
            pd.Timestamp.utcnow() if ts is None else ts,
        )

    def _trip(self, breach: _Breach, ts: pd.Timestamp) -> None:
        reason = breach.reason
        self._halt_store.engage(HaltRecord(
            kind="circuit_breaker", reason=reason, measured=breach.measured,
            threshold=breach.threshold, config_key=breach.config_key,
            halted_at=ts.isoformat(),
        ))
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
        from ..monitoring.alerts import AlertSeverity, send_alert
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

    def reset(self, *, operator: str, justification: str) -> None:
        """De ENIGE weg terug: handmatig, met naam en motivering.

        STAGE D, C2. Deze methode had geen argumenten en zette simpelweg de
        vlaggen terug. Nu gaat zij door `HaltStore.release`, die een operator en
        een motivering EIST en beide in het journaal schrijft. Er is geen
        tijdgebaseerde en geen herstel-gebaseerde ontgrendeling; audit sectie 14
        laat geen bypass toe.
        """
        self._halt_store.release(operator=operator, justification=justification)
        self._state.circuit_breaker_active = False
        self._halt_reason = None
        self._halt_ts = None
        logger.warning(
            "CircuitBreaker HANDMATIG VRIJGEGEVEN door %s: %s",
            operator, justification,
        )

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

    @classmethod
    def release_halt(
        cls,
        *,
        operator: str,
        justification: str,
        halt_store_path: pathlib.Path | None = None,
    ) -> HaltRecord:
        """Hef een staande halt op ZONDER een breaker te hoeven bouwen.

        `reset()` is een instantiemethode, en `__init__` weigert te construeren
        zolang de halt staat -- de enige reachable weg terug na een herstart
        liep dus buiten deze klasse om. Deze classmethod is diezelfde weg, met
        dezelfde eisen: `HaltStore.release` verlangt een operator en een
        motivering en journaliseert beide.
        """
        store = HaltStore(
            halt_store_path if halt_store_path is not None else _HALT_STORE_PATH
        )
        record = store.release(operator=operator, justification=justification)
        logger.warning(
            "CircuitBreaker HANDMATIG VRIJGEGEVEN door %s: %s (halt was: %s)",
            operator, justification, record.reason,
        )
        return record

    def _refuse_start_when_halted(self) -> None:
        """Weiger te starten zolang de soevereine halt actief is.

        STAGE D, C2. De voorganger (`_check_prior_trips`, Wave 15 P0-5.5) keek
        naar trips van de laatste **24 uur**:

            cutoff_ts = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)

        Een halt die 25 uur oud was, blokkeerde niets meer -- terwijl niemand
        hem had geaccordeerd. Dat is een kill switch met een klok eraan, en
        precies wat `HaltStore.release` verbiedt. Er is hier geen venster: de
        halt blijft staan tot een operator hem met motivering opheft.
        """
        active = self._halt_store.load()
        if active is None:
            return
        raise RuntimeError(
            f"CircuitBreaker: het boek staat op HALT sinds {active.halted_at} "
            f"(reden: {active.reason}; gemeten {active.measured:.4f} tegen "
            f"drempel {active.threshold:.4f} op {active.config_key}). Deze "
            f"toestand verloopt NIET met de tijd. Hef hem op met "
            f"CircuitBreaker.release_halt(operator=..., justification=..., "
            f"halt_store_path={self._halt_store.path!s}), en leg vast waarom. "
            f"NIET met `CircuitBreaker.reset(...)`: die vereist een instantie, "
            f"en zolang deze halt staat komt geen enkele constructor voorbij "
            f"deze regel."
        )
