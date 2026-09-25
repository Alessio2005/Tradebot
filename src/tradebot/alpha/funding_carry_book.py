# src/tradebot/alpha/funding_carry_book.py
"""Het carryboek van H-11.1: cross-sectioneel, dollar-neutraal, parameterloos.

Fase 11, stap 7. De constructie is vastgelegd in de bevroren pre-registratie
`7b602047ebffbab5f6ea155303d69f98` en wordt hier letterlijk uitgevoerd:

1. **De bekende carry.** Op bar `t` is bekend: de som van de fundingafrekeningen
   in dagbar `t - lag` (`data/funding_panel.py::daily_funding_panel`,
   verschoven). Met `lag = 1` zijn dat uitsluitend afrekeningen van vóór het
   sluitmoment van bar `t - 1`, dus van vóór het besluit (R-1).
2. **De rangschikking.** Op elke herbalanceringsbar aflopend op de bekende
   carry; gelijke waarden op symboolnaam, zodat de kolomvolgorde niets beslist.
   De bovenste helft gaat short (`a = -1`), de onderste helft long (`a = +1`).
   De marktrichting valt eruit per constructie; wat overblijft is het VERSCHIL
   in carry tussen de namen.
3. **Het vasthouden.** Elke `h` bars een besluit, vanaf de eerste bar waarop
   elke naam een bekende carry heeft; daartussen verandert de gewenste
   exposure niet. De grootte bepaalt de soevereine laag (vol-target).

Geen drempel, geen venster, geen z-score: elke drempel is een parameter en elke
parameter een trial (R-2). De onconditionele vorm is de goedkoopste toets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = ["carry_weights", "known_carry", "permuted_known_carry"]


def known_carry(funding_panel: pd.DataFrame, *, lag_bars: int) -> pd.DataFrame:
    """De carry die op elke bar bekend is: het dagpaneel, `lag_bars` bars later.

    `lag_bars = 0` levert de afrekeningen van de EIGEN bar; dat is precies de
    negatieve controle van de causaliteitstoets en geen geldige keuze voor het
    boek (de ontwerpconfig eist `>= 1`).
    """
    require(
        lag_bars >= 0,
        "Een negatieve lag leest afrekeningen uit de toekomst.",
        DataContractError, lag_bars=lag_bars,
    )
    return funding_panel.shift(lag_bars)


def carry_weights(known: pd.DataFrame, *, holding_period: int) -> pd.DataFrame:
    """De gewenste exposure per bar: short de hoogste bekende carry, long de laagste."""
    require(
        holding_period >= 1,
        "Een houdduur van minder dan één bar is geen houdduur.",
        DataContractError, holding_period=holding_period,
    )
    n_names = known.shape[1]
    require(
        n_names >= 2 and n_names % 2 == 0,
        "Een dollar-neutraal boek met gelijke gewichten vraagt een even aantal "
        "namen: dan zijn er evenveel shorts als longs.",
        DataContractError, n_names=n_names,
    )
    complete = known.notna().all(axis=1).to_numpy()
    require(
        bool(complete.any()),
        "Op geen enkele bar heeft elke naam een bekende carry; er valt niets "
        "te rangschikken.",
        DataContractError,
    )
    start = int(np.argmax(complete))
    names = sorted(str(c) for c in known.columns)
    values = known[names].to_numpy(dtype="float64")
    out = np.zeros_like(values)
    book = np.zeros(n_names)
    for i in range(start, len(known)):
        if (i - start) % holding_period == 0:
            require(
                bool(np.isfinite(values[i]).all()),
                "Op een herbalanceringsbar ontbreekt de bekende carry van een naam.",
                DataContractError, bar=str(known.index[i]),
            )
            order = sorted(range(n_names), key=lambda j: (-values[i, j], names[j]))
            book = np.ones(n_names)
            book[order[: n_names // 2]] = -1.0
        out[i] = book
    return pd.DataFrame(out, index=known.index, columns=names)[list(known.columns)]


def permuted_known_carry(known: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Negatieve controle 2: de bekende carry per symbool geschud in de tijd.

    Per naam (op naam gesorteerd, uit één generator) worden de GELDIGE waarden
    gepermuteerd; de posities zonder waarde blijven leeg. Wat de rangschikking
    daarna nog aan carry oogst, komt niet uit de persistentie van funding.
    """
    rng = np.random.default_rng(seed)
    out = known.copy()
    for name in sorted(str(c) for c in known.columns):
        col = out[name].to_numpy(dtype="float64", copy=True)
        valid = np.isfinite(col)
        col[valid] = rng.permutation(col[valid])
        out[name] = col
    return out
