"""Ruis van `ib_async` die voor de options-dienst niets betekent, uit de log.

* IBKR-meldingen die bij deze opzet altijd komen en niets vragen:
  - 200 "No security definition": de feed vraagt het hele strikeraster van een naam en niet
    elke strike bestaat bij elke expiratie; die contracten vallen gewoon af.
  - 10090 en 10167: vertraagde in plaats van realtime data (paper, open vraag A).
* "API connection failed": `IBKRBroker.connect` probeert opnieuw en logt dat zelf.
* Tick 90 (vertraagd "halted") kent `ib_async` 2.0.1 niet; elke keer een `AssertionError`
  met traceback. Hij krijgt dezelfde betekenis als tick 49 ("halted").

Alle andere meldingen blijven staan.
"""
from __future__ import annotations

import logging

from ib_async import wrapper as ib_wrapper

__all__ = ["IRRELEVANT_CODES", "DropIrrelevant", "quiet_ib"]

IRRELEVANT_CODES = frozenset({200, 10090, 10167})
_PREFIXES = tuple(f"{kind} {c}, " for c in sorted(IRRELEVANT_CODES)
                  for kind in ("Error", "Warning")) + ("API connection failed",)
_LOGGERS = ("ib_async.wrapper", "ib_async.client")
_DELAYED_HALTED, _HALTED = 90, 49


class DropIrrelevant(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_PREFIXES)


def quiet_ib() -> None:
    """Zet het filter en de tick-90-vertaling; twee keer aanroepen doet niets extra."""
    ib_wrapper.GENERIC_TICK_MAP.setdefault(_DELAYED_HALTED,
                                           ib_wrapper.GENERIC_TICK_MAP[_HALTED])
    for name in _LOGGERS:
        lg = logging.getLogger(name)
        if not any(isinstance(f, DropIrrelevant) for f in lg.filters):
            lg.addFilter(DropIrrelevant())
