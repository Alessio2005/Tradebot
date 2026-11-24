"""De live-schakelaar (detailontwerp 13): gebouwd, getest en uit.

Live handelen kan alleen als ALLE vijf voorwaarden tegelijk waar zijn. Elke andere
combinatie (31 van de 32) weigert, met de reden per ontbrekende voorwaarde.

1. `requested`: de dienst is gestart met mode `live` (geen config in deze repository doet
   dat; een test bewaakt het).
2. `confirmed`: de omgevingsvariabele `TRADEBOT_OPTIONS_LIVE_CONFIRM` is exact
   `LIVE_CONFIRM_PHRASE`.
3. `approved_file`: in de run-map staat `LIVE_APPROVED`, met daarin de account-hash.
4. `real_account`: het IBKR-account is geen paper-account (paper-accounts beginnen met `DU`).
5. `paper_record`: de killgate van de paper-periode staat op `pass` in `paper_verdict.json`.

Zolang live uit staat, weigert `require_paper` elk account dat niet met `DU` begint: de
paper-dienst kan dus nooit per ongeluk op een echt account handelen.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..utils.failfast import TradebotContractError

__all__ = ["CONDITIONS", "LIVE_CONFIRM_PHRASE", "LiveChecks", "LiveRefused", "account_hash",
           "gather", "is_paper_account", "live_decision", "require_paper"]

LIVE_CONFIRM_PHRASE = "IK BEGRIJP DAT DIT ECHT GELD IS"
PAPER_PREFIX = "DU"
CONDITIONS = ("requested", "confirmed", "approved_file", "real_account", "paper_record")


class LiveRefused(TradebotContractError):
    """Live (of een niet-paper-account) is geweigerd."""


def account_hash(account_id: str) -> str:
    """Een korte hash: het account-id zelf komt nooit in staat of journal."""
    return hashlib.sha256(account_id.encode()).hexdigest()[:16]


def is_paper_account(account_id: str) -> bool:
    return account_id.startswith(PAPER_PREFIX)


@dataclass(frozen=True, slots=True)
class LiveChecks:
    requested: bool
    confirmed: bool
    approved_file: bool
    real_account: bool
    paper_record: bool


def live_decision(c: LiveChecks) -> tuple[bool, tuple[str, ...]]:
    """(toegestaan, ontbrekende voorwaarden)."""
    missing = tuple(n for n in CONDITIONS if not getattr(c, n))
    return not missing, missing


def gather(mode: str, account_id: str, run_dir: Path,
           env: dict[str, str] | None = None) -> LiveChecks:
    env = dict(os.environ) if env is None else env
    approved = run_dir / "LIVE_APPROVED"
    verdict = run_dir / "paper_verdict.json"
    record = False
    if verdict.is_file():
        record = json.loads(verdict.read_text(encoding="utf-8")).get("verdict") == "pass"
    return LiveChecks(
        requested=mode == "live",
        confirmed=env.get("TRADEBOT_OPTIONS_LIVE_CONFIRM") == LIVE_CONFIRM_PHRASE,
        approved_file=approved.is_file()
        and approved.read_text(encoding="utf-8").strip() == account_hash(account_id),
        real_account=not is_paper_account(account_id),
        paper_record=record)


def require_paper(account_id: str) -> None:
    """De paper-dienst: alleen een paper-account, anders `LiveRefused`."""
    if not is_paper_account(account_id):
        raise LiveRefused("dit is geen IBKR-paper-account (verwacht een id dat met "
                          f"{PAPER_PREFIX!r} begint); live staat uit")
