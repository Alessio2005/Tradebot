"""Het VolState-contract -- de discrete toestand die een BESLUIT conditioneert.

WAAROM DIT BESTAAT
==================
Volatiliteitsmodellen in deze repository werden op twee manieren gebruikt en
beide zijn de verkeerde: als puntvoorspeller van sigma^2 beoordeeld met QLIKE
(H1, gesloten op de meetbasis), en als schaalvermenigvuldiger op de exposure
(AD-15). Dat tweede kan per constructie niet werken -- AD-16 meet dat L7 elke
cross-sectioneel uniforme schaal er weer uitdeelt.

Een volatiliteitsmodel bepaalt de volatiliteit en kent op grond daarvan een
DISCRETE TOESTAND toe. Die toestand conditioneert een besluit, niet een schaal.
Zie REGEL V in `Prompts-fases/fase_10_herstart_dagbars.md` §4.3.

DE LAG IS NIET COSMETISCH
=========================
De toestand van bar t is een functie van sigma-dak op t-1 en van kwantielen
over bars <= t-1. Beide helften zijn nodig:

* sigma-dak op t bevat r_t. Een toestand die daarop is gebaseerd, kent het
  rendement dat hij zou moeten voorspellen.
* Een expanding kwantiel dat bar t bevat, laat de DREMPEL van de observatie
  afhangen die hij classificeert.

Truncatie-invariantie ziet geen van beide fouten; `tests/lookahead/
test_vol_state_causality.py` wel.

DE ORDENING IS OP VARIANTIE, EN DAT IS EEN BESLUIT
===================================================
Een k-toestands-HMM schat per toestand zowel een gemiddelde als een variantie
(`regime/markov.py`, `means: (k, d)`). De verleiding is om te ordenen op het
GEMIDDELDE -- dat levert 0 = bearish, 1 = flat, 2 = bullish.

Op dit paneel wordt die lezing niet gedragen. Gemeten met de bestaande M0
vol-buckets over 10.400 symbool-bars, met de gepoolde t naast de t die op
datum is geclusterd en met N_eff = 1,271 is gedefleerd:

    toestand   n      ann. rendement   ann. vol   t_pooled   t_eff
    LAAG       1720   + 49,6 %          62,6 %      1,72      0,79
    NORMAAL    8181   +  9,5 %          79,2 %      0,57      0,26
    HOOG        499   +136,4 %         137,0 %      1,16      0,53

De volatiliteit scheidt met een factor 2,2 en dat is betrouwbaar. Het rendement
scheidt niet: geen enkele gedefleerde t haalt 0,8, en de tekens spreken elkaar
per symbool tegen (DOTUSDT +131,6 bp/dag in HOOG, SOLUSDT -45,1 bp).

Daarom: identificatie op VARIANTIE -- de grootheid die betrouwbaar wordt
geschat -- en de semantiek van de toestand wordt GEMETEN en gerapporteerd
(stap 6), nooit aangenomen.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from ..utils.time import assert_utc_index

__all__ = ["StateAssignment", "VolState", "assign_by_variance"]


class VolState(IntEnum):
    """De drie toestanden, geordend op volatiliteitsniveau."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(frozen=True)
class StateAssignment:
    """Een toestandstoewijzing plus de herkomst die haar auditbaar maakt."""

    states: pd.DataFrame
    source: str
    ordering: str
    params: Mapping[str, float]

    def occupancy(self) -> pd.DataFrame:
        """Aandeel bars per toestand per symbool."""
        rows = {}
        for column in self.states.columns:
            series = self.states[column].dropna()
            total = max(len(series), 1)
            rows[column] = {
                state.name: float((series == int(state)).sum()) / total
                for state in VolState
            }
        return pd.DataFrame(rows).T

    def episodes(self) -> pd.DataFrame:
        """Aantal aaneengesloten EPISODES per toestand per symbool.

        Dit is de grootheid die de bezettingspoort in stap 9 nodig heeft. Een
        toestand die 499 bars beslaat in twaalf episodes, levert twaalf
        onafhankelijke observaties en geen 499: de persistentie zit in de
        toestand zelf.
        """
        rows = {}
        for column in self.states.columns:
            series = self.states[column]
            changed = series.ne(series.shift(1)) & series.notna()
            rows[column] = {
                state.name: int((changed & (series == int(state))).sum())
                for state in VolState
            }
        return pd.DataFrame(rows).T

    def mean_duration(self) -> pd.DataFrame:
        """Gemiddelde episodelengte in bars -- occupancy x n_bars / episodes."""
        occ = self.occupancy()
        epi = self.episodes().replace(0, np.nan)
        n_bars = self.states.notna().sum()
        return occ.mul(n_bars, axis=0).div(epi)


def assign_by_variance(
    sigma: pd.DataFrame,
    *,
    low_q: float,
    high_q: float,
    min_periods: int,
    source: str = "unspecified",
    lag: int = 1,
) -> StateAssignment:
    """Wijs elke bar een VolState toe op EXPANDING kwantielen van gelagde sigma.

    Expanding en niet rolling, en al helemaal niet over de volledige sample:
    dat laatste is DI-2 en het zou de toestand van bar `t` laten afhangen van
    bars die op `t` nog niet bestonden.

    `lag=1` is geen instelling maar het contract. Hij staat in de handtekening
    zodat een test hem op 0 kan zetten en kan AANTONEN dat de causaliteitstest
    dan faalt -- een poort die niet rood kan worden, bewijst niets.
    """
    assert_utc_index(sigma, name="assign_by_variance")
    require(
        0.0 < low_q < high_q < 1.0,
        "De kwantielen moeten oplopen en binnen (0, 1) liggen; anders kan een "
        "bar tegelijk LOW en HIGH zijn en bepaalt de volgorde van twee if-takken "
        "de uitkomst.",
        DataContractError, low_q=low_q, high_q=high_q,
    )
    require(
        min_periods > 1,
        "Een kwantiel over minder dan twee observaties bestaat niet.",
        DataContractError, min_periods=min_periods,
    )
    require(
        lag >= 0,
        "Een negatieve lag is een lookahead met een vriendelijke naam.",
        DataContractError, lag=lag,
    )

    known = sigma.shift(lag).astype("float64")
    states = pd.DataFrame(
        np.nan, index=sigma.index, columns=sigma.columns, dtype="float64"
    )
    for column in sigma.columns:
        series = known[column]
        # Het kwantiel loopt over dezelfde gelagde reeks, dus de drempel op bar
        # t is een functie van {sigma_s : s <= t-lag}. Beide helften van Q1.
        lo = series.expanding(min_periods=min_periods).quantile(low_q)
        hi = series.expanding(min_periods=min_periods).quantile(high_q)
        assigned = pd.Series(np.nan, index=series.index, dtype="float64")
        valid = series.notna() & lo.notna() & hi.notna()
        assigned[valid] = float(VolState.NORMAL)
        assigned[valid & (series <= lo)] = float(VolState.LOW)
        assigned[valid & (series >= hi)] = float(VolState.HIGH)
        states[column] = assigned

    return StateAssignment(
        states=states,
        source=source,
        ordering="variance",
        params={"low_q": float(low_q), "high_q": float(high_q),
                "min_periods": float(min_periods), "lag": float(lag)},
    )
