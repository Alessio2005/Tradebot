"""Het uitvoeringspad van het carryboek (robuust boek v8-T2): van besluit naar orders.

Audit 2026-10-24, E-1: de enige overgebleven kandidaat (spot long, perp short, een
gefinancierde tranche) had geen uitvoeringspad. De live-engine in `tradebot.live` is een
ML-engine op één been voor Bybit en kent geen spotbeen, geen lening en geen uniMMR.

Dit pakket draait het boek met DEZELFDE mechaniek als de backtest:
`systematic.leverage.LeveredBook.step` (paper/shadow) of `.decide_live` (een echte beurs).

* `orders` vertaalt een besluit naar orders per been, in een margeveilige volgorde.
* `state` bewaart het boek tussen twee runs.
* `venue` is het contract met een beurs, met een paper-venue (het model is de waarheid).
* `runner` is de dagelijkse stap van v8-T2; `python -I -m tradebot.carry` is de CLI.

Sinds de beurswissel van 2026-10-25 (Binance is niet beschikbaar in Nederland):
* `venues` legt per beurs vast wat een boek er kan en wat het kost; v8-T2 kan op Kraken
  EU niet (geen unified margin, alts geen onderpand).
* `kraken` is de adapter voor Kraken EU (spot + futures), standaard dry-run.
* `u1` is het boek dat op Kraken EU wel kan (v9-U1, de regels van v5-H3 op BTC en ETH),
  met een shadow op de publieke Kraken-data en een live-stap:
  `python -I -m tradebot.carry.u1 {shadow|step|status|unwind}`.
"""
