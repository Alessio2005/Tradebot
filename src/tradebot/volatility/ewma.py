# src/tradebot/volatility/ewma.py
"""L2 - EWMA / RiskMetrics volatiliteitsestimator. De Level-1 BASELINE.

Phase 3, deliverable 1. Sluit DI-12 voor dit bestand.

    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r_t^2

WAT ER MIS WAS
--------------
De vorige implementatie luidde in de kern:

    ewma_var = log_ret.ewm(halflife=20, min_periods=5).var()
    return np.sqrt(ewma_var.clip(lower=0.0)).ffill().fillna(0.0)

Drie problemen, in oplopende ernst:

1. `halflife=20` als DEFAULT-ARGUMENT. De bindende conventie is lambda = 0.94
   (RiskMetrics, audit sectie 9.1) en die hoort in `conf/model/volatility.yaml`,
   niet als getal in een functiehandtekening.
2. `ffill()` over de opstartfase draagt de eerste bruikbare schatting
   achterwaarts noch voorwaarts correct - hij verlengt hem stilzwijgend.
3. **`fillna(0.0)` is de gevaarlijke.** Het levert een impliciete volatiliteit
   van NUL op voor elke bar in de burn-in. In Naive Risk Parity is het gewicht
   omgekeerd evenredig aan de vol, dus nul vol betekent een ONEINDIG gewicht.
   Dat is geen afrondingsprobleem maar een positiegrootte die door niets meer
   wordt begrensd, en hij ontstaat precies daar waar de schatter het minst weet.

DE INITIALISATIECONVENTIE, EXPLICIET
------------------------------------
Een EWMA moet ergens beginnen. De verleiding is te seeden met de variantie over
de volledige sample; dat is exact DI-2, alleen beter verstopt. Hier geldt:

* `sigma^2` wordt geseed op positie `burn_in_bars` met het EXPANDING gemiddelde
  van `r^2` over uitsluitend de eerste `burn_in_bars` returns;
* vanaf daar loopt de recursie voorwaarts;
* **voor die positie bestaat er geen schatting en is de uitkomst NaN.**

Er is geen `fillna`, geen `ffill` en geen constante. Een consument die een
vol-schatting nodig heeft en NaN krijgt, moet crashen - niet raden. Dat is de
opdracht van `portfolio/risk_parity.py`.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 9, 9.1, 13.1, 19 (L2), 22, 26.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, TradebotContractError, require

__all__ = [
    "ewma_variance_causal",
    "ewma_volatility",
    "ewma_volatility_panel",
    "get_ewma_volatility",
]


def ewma_variance_causal(
    returns: pd.Series, *, lam: float, burn_in_bars: int
) -> pd.Series:
    """Causale RiskMetrics-variantie. De canonieke recursie van het platform.

    De waarde op `t` hangt uitsluitend af van `r_0..r_t`. Voor positie
    `burn_in_bars` is de uitkomst NaN: daar bestaat nog geen schatting.

    Parameters
    ----------
    returns : log-returns; leidende NaN's tellen niet mee als observatie.
    lam : de RiskMetrics-decay. Komt uit `conf/model/volatility.yaml`; er is
        bewust geen default in deze handtekening.
    burn_in_bars : aantal returns dat de seed voedt via een expanding gemiddelde.
    """
    require(
        0.0 < lam < 1.0,
        "De EWMA-decay lambda moet strikt tussen 0 en 1 liggen "
        "(RiskMetrics-conventie: 0.94 voor daily bars).",
        DataContractError,
        lam=lam,
    )
    require(
        burn_in_bars > 1,
        "De seed van de EWMA-variantie vereist minstens twee returns.",
        DataContractError,
        burn_in_bars=burn_in_bars,
    )
    arr = returns.to_numpy(dtype="float64")
    n = len(arr)
    var = np.full(n, np.nan, dtype="float64")
    if n > burn_in_bars:
        # Seed: gemiddelde van r^2 over r_1..r_burn_in. Positie 0 draagt geen
        # return (die is NaN na een diff) en telt dus niet mee.
        squared = arr**2.0
        seed = float(np.mean(squared[1 : burn_in_bars + 1]))
        tail = squared[burn_in_bars:].copy()
        tail[0] = seed
        # `ewm(adjust=False)` levert exact sigma^2_t = lam*sigma^2_{t-1}
        # + (1-lam)*r_t^2, met sigma^2 op de seed-positie gelijk aan de seed.
        var[burn_in_bars:] = (
            pd.Series(tail)
            .ewm(alpha=1.0 - lam, adjust=False)
            .mean()
            .to_numpy(dtype="float64")
        )
    return pd.Series(var, index=returns.index, name="ewma_var")


def ewma_volatility(
    close: pd.Series,
    *,
    lam: float,
    burn_in_bars: int,
    annualisation_factor: float,
) -> pd.Series:
    """Geannualiseerde causale EWMA-volatiliteit uit een closereeks.

    Tijdens de burn-in is de uitkomst NaN en dat blijft hij. Er is GEEN
    `fillna`, geen `ffill` en geen impliciete constante volatiliteit.
    """
    require(
        annualisation_factor > 0.0,
        "annualisation_factor moet positief zijn.",
        DataContractError,
        annualisation_factor=annualisation_factor,
    )
    values = close.to_numpy(dtype="float64")
    finite = np.isfinite(values)
    require(
        bool((values[finite] > 0.0).all()),
        "Niet-positieve prijs; ln(P) is dan niet gedefinieerd en levert -inf op "
        "die stilzwijgend door elke vol-schatting propageert.",
        DataContractError,
        n_non_positive=int((values[finite] <= 0.0).sum()),
    )
    log_price = pd.Series(np.log(values), index=close.index)
    returns = log_price.diff()
    var = ewma_variance_causal(returns, lam=lam, burn_in_bars=burn_in_bars)
    vol: pd.Series = pd.Series(
        np.sqrt(var.to_numpy(dtype="float64")) * math.sqrt(annualisation_factor),
        index=close.index,
        name="ewma_vol",
    )
    return vol


def ewma_volatility_panel(
    panel: pd.DataFrame,
    *,
    lam: float,
    burn_in_bars: int,
    annualisation_factor: float,
) -> pd.DataFrame:
    """`ewma_volatility` per kolom van een breed close-panel.

    Elke kolom wordt onafhankelijk geschat. Er wordt NIET over symbolen heen
    gepoold: een gepoolde schatting zou de vol van een rustig instrument
    verhogen met die van een onrustig instrument, en dat is een cross-sectionele
    besmetting die in de gewichten terechtkomt.
    """
    require(
        len(panel.columns) > 0,
        "Leeg panel; er valt niets te schatten.",
        DataContractError,
    )
    out = {
        str(col): ewma_volatility(
            panel[col], lam=lam, burn_in_bars=burn_in_bars,
            annualisation_factor=annualisation_factor,
        )
        for col in panel.columns
    }
    return pd.DataFrame(out, index=panel.index).astype("float64")


def get_ewma_volatility(*args: object, **kwargs: object) -> pd.Series:
    """GEDEPRECIEERD sinds Phase 3. Crasht onvoorwaardelijk.

    De oude implementatie vulde de burn-in met `fillna(0.0)` - een impliciete
    volatiliteit van nul, die in Naive Risk Parity een oneindig gewicht
    oplevert. Er is bewust GEEN doorgeefpad naar de nieuwe functie: een
    aanroeper die `halflife` doorgaf, verwacht een andere parametrisatie dan
    `lambda`, en die stil vertalen zou precies de soort onopgemerkte
    gedragswijziging zijn die dit platform uitsluit.

    Gebruik `ewma_volatility(close, lam=..., burn_in_bars=...,
    annualisation_factor=...)` met de waarden uit `conf/model/volatility.yaml`.
    """
    raise TradebotContractError(
        "get_ewma_volatility() is verwijderd in Phase 3 (DI-12). Hij vulde de "
        "burn-in met fillna(0.0), oftewel een impliciete volatiliteit van NUL; "
        "in Naive Risk Parity levert dat een oneindig gewicht op. Gebruik "
        "volatility.ewma.ewma_volatility(close, lam=..., burn_in_bars=..., "
        "annualisation_factor=...) met de waarden uit conf/model/volatility.yaml."
    )
