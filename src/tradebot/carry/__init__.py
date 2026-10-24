"""Het uitvoeringspad van het carryboek (robuust boek v8-T2): van besluit naar orders.

Audit 2026-10-24, E-1: de enige overgebleven kandidaat (spot long, perp short, een
gefinancierde tranche) had geen uitvoeringspad. De live-engine in `tradebot.live` is een
ML-engine op één been voor Bybit en kent geen spotbeen, geen lening en geen uniMMR.

Dit pakket draait het boek met DEZELFDE mechaniek als de backtest:
`systematic.leverage.LeveredBook.step` (paper/shadow) of `.decide_live` (een echte beurs).

* `orders` vertaalt een besluit naar orders per been, in een margeveilige volgorde.
* `state` bewaart het boek tussen twee runs.
* `venue` is het contract met een beurs, met een paper-venue (het model is de waarheid).
* `runner` is de dagelijkse stap; `python -I -m tradebot.carry` is de CLI.

Een echte beursadapter (Binance spot, USDⓈ-M en portfolio margin) staat hier bewust NIET
in. Zie `venue.Venue` voor wat hij moet leveren.
"""
