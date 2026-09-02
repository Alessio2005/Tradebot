# src/tradebot/monitoring/alerts.py
"""Alert routing — Slack webhook / email / PagerDuty stub.

Default mode (dev): stdout-only.  Production: set TRADEBOT_SLACK_WEBHOOK
environment variable.  All alert calls are fire-and-forget (non-blocking).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib import request as urllib_request

import pandas as pd

from ..risk.kill_switches import HaltRecord, HaltStore
from ..utils.failfast import ConfigContractError, require

logger = logging.getLogger(__name__)

__all__ = ["Alert", "AlertRouter", "AlertSeverity", "configure_default_router",
           "send_alert"]


class AlertSeverity(Enum):
    """Drie berichtniveaus en een handeling.

    INFO, WARNING en CRITICAL verschillen alleen in dringendheid: het zijn
    berichten aan een mens. `HALT` is geen bericht maar een HANDELING -- hij
    sluit het boek via de soevereine `HaltStore` en vraagt geen toestemming.
    Zie `AlertRouter.send`; fase-opdracht Stage D-3, exit-criterium D4.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    HALT = "HALT"


@dataclass
class Alert:
    """A single alert message."""
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.WARNING
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def to_slack_payload(self) -> dict[str, Any]:
        """Format for Slack incoming webhook."""
        emoji = {"INFO": ":information_source:", "WARNING": ":warning:",
                 "CRITICAL": ":rotating_light:", "HALT": ":octagonal_sign:"}
        sev = self.severity.value
        return {
            "text": f"{emoji.get(sev, '')} *[{sev}] {self.title}*\n{self.message}",
            "attachments": [{"fields": [{"title": k, "value": str(v), "short": True}
                                        for k, v in self.metadata.items()]}],
        }


class AlertRouter:
    """Routes alerts to configured backends.

    Backends checked in priority order:
    1. TRADEBOT_SLACK_WEBHOOK env var → Slack POST
    2. stdout (always)
    """

    def __init__(
        self,
        slack_webhook: str | None = None,
        halt_store: HaltStore | None = None,
    ) -> None:
        self._slack_url = slack_webhook or os.environ.get("TRADEBOT_SLACK_WEBHOOK")
        #: De soevereine halt-state. Verplicht zodra er een `HALT` langskomt;
        #: zie `_engage_halt`.
        self._halt_store = halt_store

    def send(self, alert: Alert) -> None:
        """Dispatch an alert to all configured backends (non-blocking).

        Een `HALT` wordt EERST uitgevoerd en daarna pas verstuurd. Zou de
        volgorde omgekeerd zijn, dan bepaalt de bereikbaarheid van Slack hoe
        snel het boek dichtgaat.
        """
        if alert.severity is AlertSeverity.HALT:
            self._engage_halt(alert)

        # Always log
        log_fn = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
            AlertSeverity.HALT: logger.critical,
        }.get(alert.severity, logger.warning)
        log_fn("[%s] %s: %s", alert.severity.value, alert.title, alert.message)

        if self._slack_url:
            self._send_slack(alert)

    def _engage_halt(self, alert: Alert) -> None:
        """Sluit het boek. Geen bevestiging, geen venster, geen uitzondering.

        Zonder store crasht dit met opzet: een router die `HALT` accepteert maar
        niet kan halteren, zou een alarm afgeven dat belooft wat het niet doet.
        Dat is gevaarlijker dan geen alarm, want er wordt op vertrouwd.
        """
        if self._halt_store is None:
            # De melding gaat EERST het log in en pas daarna de uitzondering.
            # Andersom -- zoals het was -- verdween het dringendste alarm dat
            # dit systeem kent volledig: `_engage_halt` draait vóór de logregel
            # en vóór Slack, dus een router zonder store liet niets achter
            # behalve een traceback bij wie hem toevallig opving.
            logger.critical(
                "[HALT] %s: %s -- deze AlertRouter heeft GEEN HaltStore; het "
                "boek is NIET gesloten.", alert.title, alert.message,
            )
        require(
            self._halt_store is not None,
            "Een HALT-alert vereist een HaltStore. Deze router is zonder er "
            "een geconfigureerd; hij kan het boek dus niet sluiten en zou een "
            "belofte doen die hij niet waarmaakt.",
            ConfigContractError, title=alert.title,
        )
        assert self._halt_store is not None  # door require() gegarandeerd
        metadata = alert.metadata or {}
        self._halt_store.engage(HaltRecord(
            kind="alert",
            reason=f"{alert.title}: {alert.message}",
            measured=float(metadata.get("measured", 1.0)),
            threshold=float(metadata.get("threshold", 0.0)),
            config_key=str(metadata.get("config_key", "monitoring.alert")),
            halted_at=pd.Timestamp.utcnow().isoformat(),
        ))

    def _send_slack(self, alert: Alert) -> None:
        payload = json.dumps(alert.to_slack_payload()).encode("utf-8")
        req = urllib_request.Request(
            self._slack_url,  # type: ignore[arg-type]
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 204):
                logger.warning("Slack alert failed: HTTP %d", resp.status)


# Module-level default router — configure once at startup.
_default_router: AlertRouter | None = None


def configure_default_router(router: AlertRouter) -> None:
    """Zet de router die `send_alert` gebruikt. Eén keer, bij het opstarten.

    Bestaat omdat `AlertSeverity.HALT` anders onbereikbaar is: `send_alert`
    bouwde bij de eerste aanroep een `AlertRouter()` ZONDER `HaltStore`, en
    `_engage_halt` weigert dan terecht. Zonder deze functie was er geen enkele
    manier om die lazy router een store te geven -- het enige alarmniveau dat
    een handeling is, kon dus alleen crashen.
    """
    global _default_router
    _default_router = router


def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Send an alert via the module-level router (creates stdout-only router if not configured)."""
    global _default_router
    if _default_router is None:
        _default_router = AlertRouter()
    _default_router.send(Alert(title=title, message=message, severity=severity, metadata=metadata or {}))
