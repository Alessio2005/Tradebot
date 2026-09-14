# src/tradebot/portfolio/weights.py
"""Het onevenwichtige paneel — een cross-sectie die per bar van breedte mag zijn.

WAAROM DIT BESTAAT. De portefeuillelaag eiste een GEBALANCEERD paneel: een bar
telde pas mee wanneer élk symbool er een koers had. Op dit universum betekent dat
wachten op het laatst begonnen symbool. Gemeten op het gecertificeerde paneel:

    BTCUSDT   vanaf 2020-03-26   LINKUSDT  vanaf 2020-10-22
    ETHUSDT   vanaf 2021-03-16   DOTUSDT   vanaf 2021-03-20
    AVAXUSDT  vanaf 2021-09-16   SOLUSDT   vanaf 2021-10-16

Het gebalanceerde paneel begint dus op 2021-10-16 en gooit alles weg wat BTC en
LINK in de anderhalf jaar daarvoor hebben gedaan. Dat is geen meetbeslissing maar
een implementatiebeperking, en zij kost precies waar deze fase het meest gebrek
aan heeft: lengte van de steekproef.

WAT DIT NIET IS. Dit is geen alfaclaim en het kan er geen worden. Een langere
steekproef verlaagt de t = 2-drempel; zij verhoogt geen enkele Sharpe. De winst
zit volledig in MEETBAARHEID, en dat is ook de reden dat deze stap één trial kost
in plaats van nul: `min_symbols_per_bar` is een keuze en elke keuze is een trial
(R-2).

DE VIER EIGENSCHAPPEN, en de derde is de subtiele.

(1) Gewichten normaliseren per bar over de BESCHIKBARE namen. Een bar met twee
    namen draagt hetzelfde bruto als een bar met zes; de breedte verandert de
    samenstelling en niet de omvang van het boek.

(2) EEN NaN IS AFWEZIGHEID EN GEEN NUL. Dit is het hele verschil tussen een
    onevenwichtig paneel en een paneel met gaten. `NaN` betekent "dit instrument
    bestond niet"; `0.0` betekent "het bestond en ik neem geen positie". Wie de
    twee gelijkstelt, laat de allocator over niet-bestaande instrumenten
    beslissen en verandert stilzwijgend de noemer van elke cross-sectionele
    statistiek. Daarom blijft een afwezige cel hier ONDER ALLE OMSTANDIGHEDEN
    `NaN`, ook op een bar die vlak wordt gezet.

(3) ONDER DE MINIMUMBREEDTE IS DE BAR VLAK, NIET GEDEELTELIJK GEVULD. Een bar
    met één naam is geen cross-sectie: er is niets om tussen te kiezen, en een
    "cross-sectioneel" signaal op één naam is een directionele gok met een
    cross-sectioneel label. Zulke bars worden op nul gezet en niet op hun enige
    beschikbare naam geconcentreerd. Dat is het verschil tussen een breedte-eis
    en een dekkingsgat.

(4) De covariantieschatting krijgt alleen de beschikbare namen. Dat hoort niet
    hier maar bij de schatter zelf; deze module levert het paneel waarop dat
    mogelijk is, met `NaN` intact zodat de schatter afwezigheid kan zien.

R-3: ER KOMT HIER GEEN TWEEDE NORMALISATIE BIJ. Het schalen naar bruto staat al
in `equal_weight.py::normalise_to_gross`, inclusief de NaN-behandeling
(`np.nansum`) en de weigering om door nul te delen. Deze functie voegt UITSLUITEND
de breedtepoort toe en geeft het paneel daarna aan die bestaande implementatie
door. Een eigen kopie van de schaalregel zou hetzelfde getal geven tot het moment
dat een van de twee wordt aangepast.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .equal_weight import normalise_to_gross

__all__ = ["normalise_weights"]

#: De normalisatieCONVENTIE, geen risicolimiet: `sum(|w|) == 1` per bar. De
#: limiet op bruto-exposure hoort in L7 (`conf/risk/`) en wordt daar NA deze laag
#: toegepast — zie de docstring van `normalise_to_gross`, die hetzelfde zegt.
#: Hij staat hier met een naam omdat een kale `1.0` in een signatuur niet laat
#: zien of het een conventie of een gekalibreerde grens is.
_GROSS_CONVENTION: Final[float] = 1.0


def normalise_weights(
    raw: pd.DataFrame,
    *,
    min_symbols_per_bar: int,
    gross_target: float = _GROSS_CONVENTION,
) -> pd.DataFrame:
    """Normaliseer een RAGGED gewichtspaneel, met een poort op de breedte.

    Parameters
    ----------
    raw
        Ruwe gewichten of exposures, bars x symbolen. `NaN` betekent dat het
        instrument op die bar niet bestond — niet dat er geen positie is.
    min_symbols_per_bar
        De minimale cross-sectionele breedte. Bars met minder beschikbare namen
        worden VLAK gezet. Dit getal is een vooraf geregistreerde keuze en kost
        een trial; een tweede waarde proberen kost een tweede trial en maakt de
        eerste er retroactief ook een (R-2).
    gross_target
        De normalisatieconventie, standaard `sum(|w|) == 1` per bar.

    Returns
    -------
    pd.DataFrame
        Zelfde as en kolommen als `raw`. Afwezige cellen blijven `NaN`;
        beschikbare cellen op een te smalle bar worden `0.0`.

    De uitzondering op eigenschap (1), expliciet. Een bar die breed genoeg is maar
    waarvan élke beschikbare naam exact nul is, heeft geen richting om te
    normaliseren. Die bar blijft nul en haalt het bruto-doel dus niet. Dat is
    `normalise_to_gross`' bestaande gedrag ("Geen view betekent geen positie") en
    het wordt hier niet overschreven: een nulvector opblazen naar bruto 1 zou een
    positie verzinnen die de allocator niet heeft gekozen.
    """
    require(
        isinstance(raw, pd.DataFrame),
        "normalise_weights verwacht een DataFrame-paneel (bars x symbolen). Een "
        "Series draagt geen kolomidentiteit, en dan is niet vast te stellen "
        "WELKE namen op een bar beschikbaar waren — precies de vraag die de "
        "breedtepoort stelt.",
        DataContractError,
        raw_type=type(raw).__name__,
    )
    require(
        isinstance(min_symbols_per_bar, (int, np.integer))
        and not isinstance(min_symbols_per_bar, bool),
        "min_symbols_per_bar moet een geheel getal zijn. Een halve naam bestaat "
        "niet, en een boolean zou als 0 of 1 doorglippen terwijl de aanroeper een "
        "breedte bedoelde.",
        DataContractError,
        min_symbols_per_bar=repr(min_symbols_per_bar),
    )
    require(
        min_symbols_per_bar >= 1,
        "min_symbols_per_bar moet ten minste 1 zijn. Bij 0 is er geen poort en "
        "zou een bar zonder enige beschikbare naam alsnog als cross-sectie "
        "worden geteld.",
        DataContractError,
        min_symbols_per_bar=int(min_symbols_per_bar),
    )
    require(
        min_symbols_per_bar <= raw.shape[1],
        "min_symbols_per_bar is groter dan het aantal kolommen in het paneel. "
        "Dan is ELKE bar per constructie te smal en levert deze functie een "
        "leeg boek op — een stilzwijgende no-op in plaats van een breedte-eis.",
        DataContractError,
        min_symbols_per_bar=int(min_symbols_per_bar),
        n_columns=int(raw.shape[1]),
    )

    n_available = raw.notna().sum(axis=1)
    wide_enough = n_available >= int(min_symbols_per_bar)
    # Een rij-multiplicator in plaats van `where(..., other=0.0)`: die laatste zou
    # ook de AFWEZIGE cellen op 0.0 zetten en daarmee eigenschap (2) breken.
    # `NaN * 0.0` blijft `NaN`, `waarde * 0.0` wordt `0.0` — exact het onderscheid
    # dat deze module moet bewaren.
    gated = raw.mul(np.where(wide_enough, 1.0, 0.0), axis=0)
    return normalise_to_gross(gated, gross_target=gross_target)
