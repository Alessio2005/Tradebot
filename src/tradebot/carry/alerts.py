"""Alarmen van het Kraken-boek: naar een webhook en/of Telegram, en altijd naar het log.

Een alarm dat niet aankomt, mag het boek niet laten vallen: `Alerter.send` gooit NOOIT een
fout. Hij geeft terug of minstens één kanaal het bericht aannam, en schrijft elke poging
naar het log (stdout van de dienst, dus `docker logs` / `journalctl`).

Ontdubbeling: hetzelfde alarm (zelfde `key`) gaat hoogstens eens per `min_interval_s`;
een kritiek alarm eens per kwart daarvan. Een herhaling binnen die tijd wordt alleen gelogd.

Kanalen uit de omgeving (`Alerter.from_env`):
* `ALERT_WEBHOOK_URL`: een POST met `{"text": ...}` (Slack, Mattermost; Discord via
  `.../slack` achter de webhook-URL);
* `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`: de Bot API (`sendMessage`).
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = ["LEVELS", "Alerter"]

LEVELS = ("info", "warning", "critical")
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


def _post(url: str, payload: Mapping[str, Any]) -> None:
    r = requests.post(url, json=dict(payload), timeout=10)
    r.raise_for_status()


@dataclass
class Alerter:
    webhook_url: str | None = None
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    min_interval_s: float = 3600.0
    prefix: str = "tradebot/u1"
    post: Callable[[str, Mapping[str, Any]], None] = _post
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = print
    _last: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_env(env: Mapping[str, str], **kw: Any) -> Alerter:
        return Alerter(webhook_url=env.get("ALERT_WEBHOOK_URL") or None,
                       telegram_token=env.get("TELEGRAM_BOT_TOKEN") or None,
                       telegram_chat_id=env.get("TELEGRAM_CHAT_ID") or None, **kw)

    @property
    def channels(self) -> list[str]:
        out = []
        if self.webhook_url:
            out.append("webhook")
        if self.telegram_token and self.telegram_chat_id:
            out.append("telegram")
        return out

    def send(self, level: str, key: str, text: str) -> bool:
        """Verstuur (ontdubbeld). Nooit een fout; True als minstens één kanaal het aannam."""
        level = level if level in LEVELS else "warning"
        now = self.clock()
        gap = self.min_interval_s / 4.0 if level == "critical" else self.min_interval_s
        last = self._last.get(key)
        msg = f"[{self.prefix}] {level.upper()}: {text}"
        if last is not None and now - last < gap:
            self.log(f"alarm (onderdrukt, herhaling): {msg}")
            return False
        self._last[key] = now
        self.log(f"alarm: {msg}")
        ok = False
        if self.webhook_url:
            ok |= self._try(self.webhook_url, {"text": msg}, "webhook")
        if self.telegram_token and self.telegram_chat_id:
            ok |= self._try(TELEGRAM_URL.format(token=self.telegram_token),
                            {"chat_id": self.telegram_chat_id, "text": msg}, "telegram")
        return ok

    def _try(self, url: str, payload: Mapping[str, Any], name: str) -> bool:
        try:
            self.post(url, payload)
            return True
        except Exception as e:  # een alarmkanaal dat faalt, mag de dienst niet stoppen
            self.log(f"alarmkanaal {name} faalde: {type(e).__name__}: "
                     f"{self._redact(str(e))[:200]}")
            return False

    def _redact(self, text: str) -> str:
        """Een HTTP-fout noemt de URL, en daarin staan de token en het webhookgeheim."""
        for secret in (self.telegram_token, self.webhook_url):
            if secret:
                text = text.replace(secret, "***")
        return text
