# src/tradebot/validation/breadth.py
"""Breedte — de noemer van de fundamentele wet. Fase 11, breedte en tijdschaal.

TWEE GROOTHEDEN, TWEE NAMEN (AD-30)
===================================
In deze repository heetten twee verschillende grootheden allebei "N_eff":

* **het ontwerpeffect** van een gepoold, gelijkgewogen gemiddelde (Kish):
  `N / (1 + (N-1) rho_bar)`. Het zegt hoeveel onafhankelijke waarnemingen het
  gemiddelde van N gecorreleerde reeksen waard is. Het woont in
  `validation/inference.py::effective_breadth` en hoort daar: `clustered_mean`
  defleert er een gepoolde t mee.
* **het aantal onafhankelijke weddenschappen** uit de fundamentele wet
  (`SR ≈ IC · sqrt(BR)`): de participatieratio van de eigenwaarden van de
  correlatiematrix, `(sum lambda)^2 / sum lambda^2`. Dat getal is per
  constructie begrensd door de rang van de matrix.

Het verschil is geen afronding. Op het dollar-neutrale residu van de zes
gecertificeerde namen (`W_DEV`) geeft het ontwerpeffect **123,58** en de
participatieratio **4,353**: het residu sommeert over de namen tot nul, dus de
noemer van de rho-formule gaat naar nul, terwijl de rang vijf is
(`Prompts-fases/fase_11_breedte_en_tijdschaal.md` §2.1).

WAAROM HIER GEEN IMPLEMENTATIE STAAT
====================================
De participatieratio bestaat al: `portfolio/covariance.py::effective_n_assets`.
Dat bestand blijft door DI-10 bewust ongewijzigd, omdat
`portfolio/legacy_sizing.py` ervan afhangt en de Phase 3-baseline herrekenbaar
moet blijven. Een tweede berekening hier zou een tweede implementatie van
dezelfde grootheid zijn (R-3). Deze module geeft haar dus alleen de naam
waaronder de validatielaag haar gebruikt, en bewaakt de invoer.
"""
from __future__ import annotations

import numpy as np

from ..portfolio.covariance import effective_n_assets
from ..utils.failfast import DataContractError, require

__all__ = ["independent_bets"]


def independent_bets(correlation_matrix: np.ndarray) -> float:
    """Het aantal onafhankelijke weddenschappen in een correlatiematrix.

    De participatieratio van de eigenwaarden, begrensd door de rang. De
    berekening staat in `portfolio/covariance.py::effective_n_assets` en blijft
    daar (DI-10, AD-30); deze functie weigert alleen invoer die geen
    correlatiematrix is, want `effective_n_assets` symmetriseert stilzwijgend en
    zou een scheve of onvolledige matrix een getal geven.
    """
    c = np.asarray(correlation_matrix, dtype=np.float64)
    require(
        c.ndim == 2 and c.shape[0] == c.shape[1] and c.shape[0] >= 2,
        "Onafhankelijke weddenschappen vragen een vierkante correlatiematrix met "
        "minstens twee namen; één naam is geen cross-sectie.",
        DataContractError,
        shape=tuple(c.shape),
    )
    require(
        bool(np.isfinite(c).all()),
        "De correlatiematrix bevat een niet-eindige waarde. Een ontbrekende "
        "correlatie is geen nul: zij wordt hier niet ingevuld.",
        DataContractError,
    )
    require(
        bool(np.allclose(c, c.T)),
        "De correlatiematrix is niet symmetrisch. Symmetriseren zou een getal "
        "opleveren voor een matrix die geen correlatiematrix is.",
        DataContractError,
    )
    return float(effective_n_assets(c))
