# src/tradebot/reporting/__init__.py
"""Reporting sub-package — markdown-rendering van meetresultaten.

Elke module hier neemt een reeds GEMETEN payload en maakt er een rapport van.
Meten en opmaken staan bewust in verschillende modules: een module die allebei
doet, verleidt tot het aanpassen van de meting omdat de tabel er anders beter
uitziet.

GEVONDEN IN PHASE 6, STAP 3
----------------------------
Dit bestand deed tot `f87bac1` een re-export van `.tearsheet`, dat in
`8071dc5` ("remove the legacy engines after the parity proof") is verwijderd.
`import tradebot.reporting` crashte daardoor sinds Phase 5 met
`ModuleNotFoundError`. Niemand merkte het, omdat er in de hele repository geen
enkele aanroepsite van dit pakket meer over was — de re-export hield een
pakket in leven dat leeg was.

Dat is geen cosmetisch detail: een `__init__.py` die een verwijderde module
importeert, maakt het hele pakket onbereikbaar, ook voor modules die er wél
zijn. De les die dit oplevert staat in `reports/phase6_exit_report.md`: bij het
verwijderen van een module hoort het opruimen van elke re-export ervan, en een
importtest die het pakket daadwerkelijk importeert vangt dat.
"""
from __future__ import annotations

__all__: list[str] = []
