# src/tradebot/portfolio/decision_frequency.py
"""De beslisfrequentie — H-10.1, één besluit per k bars in plaats van per bar.

WAAROM DIT BESTAAT
==================
De haltketen zet 1.559 van de 1.743 bars vlak (`tests/unit/
test_layer_transitions.py`) omdat zij per dag een besluit EIST. H-10.1 vraagt
niet om een nieuwe voorspelling en niet om een nieuwe feature, maar of het
bestaande systeem op de goede tijdschaal opereert: wordt het besluit met
frequentie k genomen en tussentijds VASTGEHOUDEN, dan dalen de omzet en het
aantal haltes zonder dat er iets aan het model verandert.

Dat maakt dit bestand klein en zijn contract groot. De hele hypothese hangt aan
de vraag waar de blokgrenzen liggen, en dat is precies waar de lookahead zit.
Zie `Prompts-fases/fase_10_herstart_dagbars.md` stap 11 (H-10.1).

VASTHOUDEN IS NIET RESAMPLEN
============================
De naieve implementatie is `exposures.resample("5D").first().ffill()`. Die is op
dit paneel fout, om twee onafhankelijke redenen:

* `resample` legt zijn rooster op een KALENDERgrens — het begin van de week, van
  de maand, of de 5-daagse bin gerekend vanaf een origin die de aanroeper niet
  heeft gekozen. Welk blok bar t krijgt, hangt dan af van de DATUM van bar t in
  plaats van van zijn positie in de steekproef. Twee panelen die een dag na
  elkaar beginnen, krijgen andere blokgrenzen en dus een ander resultaat, en de
  begindatum van de sample is geen onderzoeksbeslissing die iemand bewust heeft
  genomen.
* Met `origin="end"`, of met elke variant die de blokken vanaf de LAATSTE bar
  terugrekent, hangen de blokGRENZEN af van waar de sample EINDIGT. Dat is de
  toekomst. Het bedrieglijke eraan is dat elke bar nog steeds netjes de waarde
  van het begin van zijn eigen blok draagt: per bar ziet de operatie er
  volstrekt causaal uit en is zij het niet. `tests/lookahead/
  test_decision_frequency_causality.py::
  test_the_truncation_guard_goes_red_on_an_end_anchored_hold` is exact die
  negatieve controle, en zonder haar zou de causaliteitstest hierboven niets
  onderscheiden.

DE BLOKKEN LIGGEN POSITIONEEL, VANAF DE EERSTE BAR VAN HET PANEEL
=================================================================
Blok b beslaat de posities `[b*k, (b+1)*k)` en de bar op positie t draagt de
waarde van positie `(t // k) * k`. Truncatie-invariantie is daarmee geen
eigenschap die met een test wordt afgedwongen maar een gevolg van de
constructie: `(t // k) * k` is een functie van t alleen, dus bars NA t — of hun
afwezigheid — kunnen de waarde op t niet bewegen. Het laatste blok mag korter
zijn dan k; die prijs wordt vooraan betaald in plaats van achteraan
weggemoffeld met een anker dat de toekomst kent.

DE HALTKETEN LOOPT NÁ HET VASTHOUDEN
====================================
Deze module bevat, modelleert en noemt geen enkele haltlogica, en dat is een
ORDENING en geen omissie. Een halt is een risicobesluit en een risicobesluit
wordt niet vastgehouden. Zou de keten vóór het vasthouden lopen, dan draagt bar
t+1 een halt die op t gold en op t+1 misschien al is opgeheven — en dan
overschrijft een besluitfrequentie uit L8 de soevereine risicolaag. Eerst
vasthouden, daarna `risk/kill_switches.py` in de vastgelegde
`constraint_order`.

WAT HIER NIET WOONT
===================
De omzetmeting. R-3 laat één implementatie per statistische grootheid toe en
`portfolio/constraints.py::compute_turnover` is die ene plek. Wie het effect van
k op de omzet wil meten, meet het daar.
"""
from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = ["breakeven_cost_bps", "hold_decision"]

#: De enige toegestane ankering. Geen instelling met één geldige waarde bij
#: gebrek aan fantasie, maar een parameter die de aanroepplek dwingt op te
#: schrijven WELKE ankering hij bedoelt — zie de docstring van `hold_decision`.
_ANCHOR_FIRST_BAR: Final[str] = "first_bar"

#: Basispunten per eenheid. Geen drempel en geen gekalibreerd getal, maar de
#: definitie van "basispunt" — 1 bp = 1e-4. Hij staat hier met een naam omdat
#: `* 1e4` in een returnregel niet zichtbaar maakt WELKE eenheid eruit komt.
_BPS_PER_UNIT: Final[float] = 1e4


def hold_decision(
    exposures: pd.DataFrame,
    *,
    k: int,
    anchor: str = _ANCHOR_FIRST_BAR,
) -> pd.DataFrame:
    """Houd het exposurebesluit k bars vast; elke bar draagt zijn blokbegin.

    Blok b beslaat de POSITIES `[b*k, (b+1)*k)` van het paneel, geteld vanaf de
    eerste bar, en de bar op positie t draagt de waarde van positie
    `(t // k) * k`. Het laatste blok mag korter zijn dan k.

    `k=1` is exact de identiteit: dezelfde waarden, dtypes, kolomorde en index
    inclusief zijn `freq`. Dat is geen aparte tak in de code maar het gevolg van
    dezelfde berekening, want `(t // 1) * 1 == t`. Een tweede codepad voor k=1
    zou een tweede implementatie van dezelfde operatie zijn en daarmee de kans
    introduceren dat de twee paden uiteenlopen zonder dat iemand het merkt.

    `anchor` bestaat om de ankering bij de AANROEP zichtbaar te maken en niet om
    een keuze aan te bieden: `"first_bar"` is de enige toegestane waarde en er
    komt er geen tweede. Een eindankering (`resample(origin="end")`) laat de
    blokgrenzen afhangen van waar de steekproef eindigt en dat is de toekomst;
    een kalenderankering laat ze afhangen van op welke datum de steekproef
    begint, wat een onderzoeksbeslissing is die niemand heeft genomen. Wie hier
    een tweede anker toevoegt, voegt een lookahead toe.

    ORDENING — de haltketen loopt NA deze functie. Een halt is een risicobesluit
    en wordt niet vastgehouden: vasthouden ná de keten zou bar t+1 een halt
    laten dragen die op t gold en op t+1 opgeheven kan zijn. Deze functie kent
    daarom geen haltlogica, en die afwezigheid is contractueel.

    Parameters
    ----------
    exposures : paneel van exposures, bars x symbolen, met een monotoon
        oplopende `DatetimeIndex` zonder dubbele tijdstempels.
    k : bloklengte in bars, `k >= 1`; `int` of `np.integer`. Een `float` en een
        `bool` worden geweigerd.
    anchor : uitsluitend `"first_bar"`.

    Returns
    -------
    pd.DataFrame
        Nieuwe frame met dezelfde index, kolommen en dtypes als `exposures`. De
        invoer wordt niet gemuteerd.

    Raises
    ------
    DataContractError
        Bij elke schending van het bovenstaande contract.

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
    >>> frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    >>> hold_decision(frame, k=2)["A"].tolist()
    [1.0, 1.0, 3.0, 3.0, 5.0]
    """
    require(
        isinstance(exposures, pd.DataFrame),
        "hold_decision verwacht een DataFrame-paneel (bars x symbolen). Een "
        "Series of een array draagt geen kolomidentiteit, en dan is niet vast "
        "te stellen welk symbool welk besluit vasthoudt.",
        DataContractError, got_type=type(exposures).__name__,
    )
    index = exposures.index
    require(
        isinstance(index, pd.DatetimeIndex),
        "De index van het exposurepaneel moet een DatetimeIndex zijn. De "
        "blokindeling is alleen als een BESLISFREQUENTIE te lezen wanneer de "
        "rijen bars in de tijd zijn; op een willekeurige index is k een "
        "rijgroepering zonder betekenis.",
        DataContractError, got_index_type=type(index).__name__,
    )
    require(
        index.is_monotonic_increasing,
        "De DatetimeIndex moet monotoon oplopend zijn. De blokgrenzen zijn "
        "POSITIONEEL, dus op een niet-gesorteerde index zou een blok bars "
        "omvatten die niet aaneengesloten in de tijd liggen en zou bar t "
        "stilzwijgend een besluit vasthouden dat ná t is genomen — een "
        "lookahead die geen enkele foutmelding oplevert.",
        DataContractError, n_rows=len(index),
    )
    require(
        not index.has_duplicates,
        "De DatetimeIndex mag geen dubbele tijdstempels bevatten. Omdat de "
        "blokken op POSITIE worden gelegd, verschuift elke dubbele bar alle "
        "latere blokgrenzen: de vastgehouden waarde zou dan afhangen van hoe "
        "vaak een bar per ongeluk in het paneel staat.",
        DataContractError, n_duplicates=int(index.duplicated().sum()),
    )
    # `bool` is in Python een subklasse van `int`, dus `isinstance(True, int)`
    # is waar. True zou hier als k=1 doorglippen en dan meet het experiment de
    # identiteit terwijl de aanroeper een bloklengte bedoelde.
    require(
        isinstance(k, (int, np.integer)) and not isinstance(k, bool),
        "k is de bloklengte in BARS en moet een geheel getal zijn (int of "
        "np.integer). Een float heeft geen bloklengte-interpretatie — er is "
        "geen half besluit — en een bool is een aanroepfout die als k=1 door "
        "de poort zou glippen en de bedoelde frequentie nooit zou meten.",
        DataContractError, k_type=type(k).__name__,
    )
    require(
        int(k) >= 1,
        "k moet ten minste 1 zijn. k=1 betekent 'elk bar een nieuw besluit' en "
        "beneden dat bestaat geen frequentie: k=0 is een deling door nul in de "
        "blokindeling en een negatieve k zou de blokken achterwaarts leggen.",
        DataContractError, k=int(k),
    )
    require(
        anchor == _ANCHOR_FIRST_BAR,
        f"anchor={_ANCHOR_FIRST_BAR!r} is de enige ondersteunde ankering. De "
        "parameter maakt de ankering bij de aanroep expliciet; hij biedt GEEN "
        "eind- of kalendergeankerd alternatief, want die laten de blokgrenzen "
        "afhangen van waar de steekproef eindigt respectievelijk begint, en "
        "waar zij eindigt is de toekomst.",
        DataContractError, anchor=anchor,
    )

    block_length = int(k)
    # De positie van het blokBEGIN van elke bar. Uitsluitend een functie van de
    # eigen positie: daarom is deze operatie causaal per constructie en niet
    # bij gratie van een test.
    block_start = (
        np.arange(len(exposures), dtype=np.int64) // block_length
    ) * block_length
    # `.iloc[...]` bewaart de dtype per kolom en de kolomorde; `set_axis` zet de
    # ORIGINELE index terug (inclusief zijn `freq`, die positionele indexering
    # laat vallen zodra posities zich herhalen). Zonder die tweede stap is k=1
    # "bijna de identiteit" — een frame met dezelfde getallen op een index die
    # stilletjes zijn frequentie is kwijtgeraakt.
    return exposures.iloc[block_start].set_axis(index, axis=0)


def breakeven_cost_bps(
    delta_sharpe_gross: float,
    turnover_delta: float,
    *,
    volatility_per_bar: float,
    bars_per_year: float,
) -> float:
    """De kosten per eenheid omzet waarbij het netto Sharpe-verschil nul is.

    WAAROM DEZE GROOTHEID BESTAAT. H-10.1 is een KOSTENhypothese: vasthouden
    verlaagt de omzet en dus de kostendrag, en de vraag is of dat het bruto
    verlies goedmaakt. Het antwoord hangt daarmee aan een kostenaanname, en de
    kostenaanname van deze repository is niet gekalibreerd -- `eta = 2.991922`
    draagt `status: IMPACT_UNCALIBRATED` (AD-2) en is een BOVENGRENS uit de
    dagrange, geen schatting. Een conclusie van de vorm "k = 5 is beter" leunt
    dan op een getal dat niet is gemeten. De conclusie die dat overleeft is
    "k = 5 is beter zodra de werkelijke kosten boven X bp liggen", en X is wat
    hier wordt uitgerekend. Het omgekeerde leest even hard: ligt X boven wat
    deze repository aan kosten beweert, dan is de winst een gevolg van de
    kostenaanname en geen eigenschap van het systeem.

    DE REKENING, EN HAAR AANNAME. Met kosten `c` per eenheid omzet is het
    netto bar-rendement `r_net = r_gross - c * turnover` (dat is letterlijk
    `backtest/vectorized.py:150`). De geannualiseerde Sharpe is dan
    `SR(c) = (mu_gross - c * tau) * sqrt(B) / sigma` met `tau` de gemiddelde
    omzet per bar, `B` de bars per jaar en `sigma` de standaardafwijking per
    bar. Voor het VERSCHIL tussen twee sporen geldt dan, ONDER DE AANNAME dat
    beide sporen dezelfde `sigma` hebben:

        delta_SR(c) = delta_SR_gross + c * turnover_delta * sqrt(B) / sigma

    met `turnover_delta = tau(k=1) - tau(k)`, dus positief wanneer vasthouden
    de omzet VERLAAGT. Nul stellen geeft

        c* = - delta_SR_gross * sigma / (turnover_delta * sqrt(B))

    en `c*` maal 1e4 is het antwoord in basispunten.

    DE AANNAME IS EEN LINEARISATIE EN WORDT NIET VERZWEGEN. Twee sporen met
    een verschillende beslisfrequentie hebben niet exact dezelfde `sigma`:
    vasthouden verandert ook de variantie van het spoor, niet alleen zijn
    gemiddelde. Het artefact van stap 11 rapporteert daarom de `sigma` van
    beide sporen naast elkaar, zodat de lezer ziet hoe ver de aanname afstaat
    van de meting in plaats van haar op gezag te moeten aannemen. Wijken de
    twee sterk af, dan is DAT de bevinding en niet dit getal (R-8).

    `eta` KOMT HIER NIET IN VOOR, en dat is opzet. De impactterm treedt
    uitsluitend op `L3_execution` op, via de router, waar hij onscheidbaar is
    van spread, fees en funding. De breakeven-as is daarom de LINEAIRE
    kostenparameter `cost_per_side` waarmee L0/L1/L2 rekenen -- de enige as
    waarop "kosten per eenheid omzet" een enkel getal is.

    Parameters
    ----------
    delta_sharpe_gross
        Geannualiseerd Sharpe-verschil VOOR kosten, `SR(k) - SR(k=1)`.
    turnover_delta
        Daling in gemiddelde omzet per bar ten opzichte van k=1, dus
        `tau(k=1) - tau(k)`. Positief wanneer vasthouden de omzet verlaagt.
    volatility_per_bar
        De standaardafwijking van het bar-rendement, niet geannualiseerd.
    bars_per_year
        VERPLICHT en zonder default, om dezelfde reden als in
        `validation/inference.py::sharpe_with_se`: een stilzwijgende default
        heeft daar jarenlang 8760 gelezen op een dagbar.

    Returns
    -------
    float
        Het breakevenniveau in basispunten per eenheid omzet. Een NEGATIEVE
        uitkomst is een geldige uitkomst en geen fout: zij betekent dat het
        vasthouden al bij nul kosten wint en dat de kostenaanname de conclusie
        dus niet draagt.
    """
    require(
        turnover_delta != 0.0,
        "Een omzetverschil van exact nul heeft geen breakevenniveau. Zonder "
        "omzetverschil kan geen kostenniveau het teken van het Sharpe-verschil "
        "omdraaien, en dan is de uitkomst een BEVINDING over de keten van de "
        "hypothese -- vasthouden raakt de omzet niet -- en geen getal. Een "
        "oneindigheid of een NaN teruggeven zou die bevinding wegschrijven als "
        "een rekenkundig detail.",
        DataContractError,
        turnover_delta=turnover_delta,
    )
    require(
        volatility_per_bar > 0.0,
        "De standaardafwijking per bar moet positief zijn. Zij staat in de "
        "NOEMER van de Sharpe en een niet-positieve waarde betekent dat het "
        "spoor geen variatie heeft; dan bestaat er geen Sharpe om te "
        "vergelijken en dus ook geen breakevenniveau.",
        DataContractError,
        volatility_per_bar=volatility_per_bar,
    )
    require(
        bars_per_year > 0.0,
        "bars_per_year moet positief zijn. De annualisatie is de brug tussen "
        "de kosten per bar en het geannualiseerde Sharpe-verschil; zonder haar "
        "is de uitkomst dimensieloos en dus niet in basispunten te lezen.",
        DataContractError,
        bars_per_year=bars_per_year,
    )
    cost = (
        -float(delta_sharpe_gross)
        * float(volatility_per_bar)
        / (float(turnover_delta) * math.sqrt(float(bars_per_year)))
    )
    return cost * _BPS_PER_UNIT
