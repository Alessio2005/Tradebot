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

logger = logging.getLogger(__name__)

__all__ = ["Alert", "AlertRouter", "AlertSeverity", "send_alert"]


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


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
        emoji = {"INFO": ":information_source:", "WARNING": ":warning:", "CRITICAL": ":rotating_light:"}
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

    def __init__(self, slack_webhook: str | None = None) -> None:
        self._slack_url = slack_webhook or os.environ.get("TRADEBOT_SLACK_WEBHOOK")

    def send(self, alert: Alert) -> None:
        """Dispatch an alert to all configured backends (non-blocking)."""
        # Always log
        log_fn = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.warning)
        log_fn("[%s] %s: %s", alert.severity.value, alert.title, alert.message)

        if self._slack_url:
            self._send_slack(alert)

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
