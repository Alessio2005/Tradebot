"""De options-setup: regelgebaseerde strategieën op US-genoteerde aandelen- en ETF-opties.

Sinds 2026-11 de actieve setup van Tradebot; de crypto-stack staat in standby
(`docs/crypto-standby.md`). Ontwerp: `docs/research/2026-11-24-options-pivot-design.md` en
`docs/research/2026-11-24-options-pivot-detailontwerp.md`.

Prijzen, implied volatility en Greeks komen uit QuantLib, met Amerikaanse uitoefening en
dividenden (`tradebot.options.pricing`). Er wordt geen model getraind: strategieën zijn vaste
regels uit `conf/options/default.yaml`. Alles draait met fictief geld (replay of paper).
"""
