"""Regelgebaseerde options-strategieën (detailontwerp 7).

Elke strategie is een pure functie van een bevroren `MarketState` en het `Book`: `scan` geeft
kandidaten, `manage` acties op een open positie. Geen klok, geen I/O, geen broker, geen
training: elke regel is een vaste drempel uit `conf/options/default.yaml`.
"""
