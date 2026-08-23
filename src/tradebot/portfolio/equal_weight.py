# src/tradebot/portfolio/equal_weight.py
"""L8 - 1/N Equal Weight. De absolute referentie-baseline.

Phase 3, deliverable 7. Audit sectie 13.1: 1/N is de BASELINE waartegen elke
allocator zich moet verantwoorden; Inverse Volatility (Naive Risk Parity) is de
PRODUCTION DEFAULT.

Waarom 1/N er hoort te staan, ook al is hij triviaal: DeMiguel, Garlappi en
Uppal (2009) laten zien dat 1/N in out-of-sample vergelijkingen een groot deel
van de geoptimaliseerde allocators verslaat, omdat die laatste hun schattingsfout
in de gewichten stoppen. Een portefeuille-model dat 1/N niet verslaat na kosten,
heeft geen bestaansrecht - en zonder de referentie in de codebase is die vraag
niet te stellen.

Deze module bevat GEEN risicologica. Er is geen leverage, geen positielimiet en
geen drawdown-check: die wonen in L7 en zijn soeverein. Wat hier gebeurt is
uitsluitend het verdelen van een gegeven gross exposure over de assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = [
    "equal_weight_long_only",
    "normalise_to_gross",
    "sized_by_equal_weight",
]


def normalise_to_gross(weights: pd.DataFrame, *, gross_target: float) -> pd.DataFrame:
    """Schaal elke rij zodat `sum(|w|) == gross_target`.

    Rijen zonder enige positie blijven op nul staan; er wordt niet gedeeld door
    nul en er wordt al helemaal geen gelijk gewicht opgelegd om de rij "vol" te
    maken. Geen view betekent geen positie.

    `gross_target` is een NORMALISATIECONVENTIE, geen risicolimiet. De limiet op
    bruto-exposure hoort in L7 (`conf/risk/`) en wordt daar toegepast NA deze
    laag.
    """
    require(
        gross_target > 0.0,
        "Een niet-positieve gross_target levert geen portefeuille op.",
        DataContractError,
        gross_target=gross_target,
    )
    values = weights.to_numpy(dtype="float64")
    gross = np.nansum(np.abs(values), axis=1)
    scale = np.where(gross > 0.0, gross_target / np.where(gross > 0.0, gross, 1.0), 0.0)
    return pd.DataFrame(
        values * scale[:, None], index=weights.index, columns=weights.columns
    )


def equal_weight_long_only(
    available: pd.DataFrame, *, gross_target: float
) -> pd.DataFrame:
    """1/N over de assets die op `t` bestaan. De long-only referentie.

    `available` is een panel waarin NaN betekent *dit instrument bestond nog
    niet*. Het aantal `N` is daarmee per rij het aantal daadwerkelijk
    verhandelbare namen, en niet het aantal kolommen - dat laatste zou een
    positie toekennen aan een instrument dat nog niet was gelanceerd.
    """
    require(
        len(available.columns) > 0,
        "Leeg universum; er valt niets te verdelen.",
        DataContractError,
    )
    present = available.notna().to_numpy()
    n = present.sum(axis=1)
    raw = np.where(present, 1.0, 0.0)
    weights = pd.DataFrame(
        np.divide(raw, np.where(n > 0, n, 1)[:, None], where=(n > 0)[:, None],
                  out=np.zeros_like(raw)),
        index=available.index,
        columns=available.columns,
    )
    return normalise_to_gross(weights, gross_target=gross_target)


def sized_by_equal_weight(
    exposures: pd.DataFrame, *, gross_target: float
) -> pd.DataFrame:
    """Verdeel de gevraagde exposure zonder risico-informatie: `w_i ~ a_i`.

    Dit is de sizing-tegenhanger van Naive Risk Parity: hij gebruikt de RICHTING
    en de STERKTE van de alpha maar weet niets van volatiliteit. Het verschil
    tussen deze twee, gemeten op dezelfde `a_t`, is precies de bijdrage van de
    vol-schatting - en dat is wat het benchmarkrapport wil isoleren.

    Ontbrekende exposure (NaN) is geen positie en telt niet mee in de
    normalisatie. Er wordt NIET stilzwijgend nul of gelijk gewicht ingevuld.
    """
    require(
        len(exposures.columns) > 0,
        "Leeg universum; er valt niets te verdelen.",
        DataContractError,
    )
    raw = pd.DataFrame(
        np.nan_to_num(exposures.to_numpy(dtype="float64"), nan=0.0),
        index=exposures.index,
        columns=exposures.columns,
    )
    return normalise_to_gross(raw, gross_target=gross_target)
