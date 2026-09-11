"""EWMA tegen GARCH op de TOESTANDSAS -- een telling, geen toets.

WAAROM DIT BESTAAT
==================
H1 vroeg welk model sigma^2 beter voorspelt. Die vraag heeft een
variantieproxy nodig, en de hele meetbasis van H1 hing daarna aan de keuze
ervan (`validation/vol_campaign.py`, REFERENCE_PROXY). De TOESTANDSvraag heeft
er geen nodig: wijzen de twee modellen een andere toestand toe, en verandert
dat het besluit? Dat is te tellen zonder ooit een realisatie te hoeven
benoemen -- en dus zonder het ene onderdeel waarop de vorige vergelijking kon
kapseizen.

Deze module SELECTEERT NIETS. Er wordt geen estimator, geen kwantiel en geen
parameter uit deze uitkomst gekozen; zij kost daarom nul trials. Zodra iemand
de productie-estimator wisselt OMDAT deze telling er zo uitzag, is dat een
hypothese voor Stage C met de trialkosten die daarbij horen, en geen stille
promotie van "de betere estimator".

DE VERGELIJKING IS INVARIANT ONDER ELKE MONOTONE HERSCHALING (ruling P35)
==========================================================================
`assign_by_variance` leidt zijn drempels af uit de reeks die hij krijgt --
expanding kwantielen van de gelagde sigma-dak zelf. Twee modellen worden dus
elk tegen hun EIGEN verdeling gescoord, en dat is opzet: GARCH tegen de schaal
van EWMA leggen zou een puur NIVEAUverschil tussen de estimators als
TOESTANDSverschil rapporteren, en dat is niet de vraag van deze stap.

Twee gevolgen, en het tweede is een grens op de claim (R-8):

* Een variantieforecast en een volatiliteitspaneel mogen zonder schaalstap
  naast elkaar: sqrt is monotoon, dus de toestanden zijn identiek. Er is hier
  geen schaalafstemming nodig en er wordt er geen gedaan.
* Daarom kan deze meting ook NIET zien dat "GARCH systematisch hoger staat
  dan EWMA". Die onzichtbaarheid is voor de toestandsvraag de juiste
  eigenschap -- AD-16 mat dat L7 een cross-sectioneel uniforme schaal er weer
  uitdeelt -- maar zij is en blijft een grens op wat elk getal hieronder mag
  beweren. `tests/unit/test_state_agreement.py` maakt haar meetbaar in plaats
  van beweerd.

TWEE TELLINGEN VAN "ANDER BESLUIT", EN ZE ZEGGEN NIET HETZELFDE (ruling P36)
=============================================================================
`n_decision_changes` telt de bars waarop de POORT anders uitvalt: minstens
een symbool wisselt van vlak naar niet-vlak of omgekeerd onder
`flat_states`. Dat is het getal dat de vooraf vastgelegde beslisregel van
stap 8.5 leest, want dat is wat "het besluit verandert" betekent zodra de
poort vastligt.

`n_any_state_changes` telt de bars waarop minstens een symbool van TOESTAND
wisselt, welke wissel dan ook. Een poort kan toestanden alleen SAMENVOEGEN,
nooit splitsen, dus dit getal is een BOVENGRENS op de besluitwijziging onder
elke denkbare toestandspoort. Haalt die bovengrens de 5 %-drempel, dan haalt
elke poort hem -- ook een die later anders wordt gekozen. Omgekeerd geldt het
niet: zakt de bovengrens erdoor, dan is daarmee niet aangetoond dat de
uiteindelijk gekozen poort wel kantelt.

De cel-overeenstemming (`fraction_identical`) is een DERDE getal, op
(bar, symbool)-niveau. Clausule 1 van de beslisregel leest die, clausule 2 het
bar-paar. Zij mogen niet in elkaar schuiven: dan staat er een getal in twee
kostuums en toetst de regel de helft van wat zij beweert.

KAPPA IS VERPLICHT EN STAAT NOOIT ALLEEN ACHTER HET RUWE PERCENTAGE
====================================================================
Bij een bezetting van 79 % NORMAAL is 80 % ruwe overeenstemming vrijwel
precies toeval. Een rapport dat alleen dat percentage noemt, leest als sterk
bewijs voor iets wat niets zegt. Cohens kappa corrigeert voor de bezetting en
is daarom de maat die telt; hij wordt hier niet nagebouwd maar uit
`sklearn.metrics` gehaald, net als de 3x3-verwarringsmatrix waaruit een lezer
hem met de hand kan narekenen.

DE GEMEENSCHAPPELIJKE DEKKING IS ZELF EEN RESULTAAT (ruling P37)
=================================================================
`walk_forward_variance_forecasts` laat elke bar buiten een testvenster en elke
niet-geconvergeerde fold op NaN staan, en vult niet. Het GARCH-paneel heeft
dus gaten die het EWMA-paneel niet heeft. Elke breuk hieronder loopt over de
cellen respectievelijk bars die BEIDE toewijzingen kennen, en elke noemer
staat als eigen veld naast zijn breuk, samen met het aantal weggevallen
cellen. Er wordt nergens gevuld, niet voorwaarts gevuld en nergens op EWMA
teruggevallen waar GARCH ontbreekt: dat zou een ander getal opleveren onder de
naam van dit getal.

Een bar telt pas mee in de BAR-noemer wanneer elk symbool in beide
toewijzingen bekend is. Een bar met een onbekende naam kan niet "ongewijzigd"
heten zonder de toestand van die naam te verzinnen, en die noemer is met zes
symbolen strenger dan de cel-noemer.

Die noemer is bovendien een VENSTER en niet alleen een aantal, en daarom
draagt het rapport hem als `bars_compared` en niet uitsluitend als telling.
Een walk-forward-paneel heeft zijn gaten aan BEIDE uiteinden -- voor de eerste
trainperiode en na de laatste volle testperiode -- dus "551 van de 1.389 bars"
laat een lezer raden WELKE 551, en de voor de hand liggende gok (de laatste
551) is de verkeerde. Zonder begin- en einddatum is een dekkingsbevinding niet
te plaatsen tegen de gebeurtenissen in het venster.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from ..utils.failfast import DataContractError, require
from .state import StateAssignment, VolState
from .state_mapping import gate_by_state

__all__ = ["PRE_REGISTERED_FLAT_STATES", "AgreementReport", "agreement"]

#: De poort die deze fase VOORAF heeft vastgelegd. Stap 13.3 van
#: `Prompts-fases/fase_10_herstart_dagbars.md` draait het tweede spoor met
#: `gate_by_state(flat_states={HIGH})`, en stap 13 boekt daar exact een trial
#: voor met de toelichting "een configuratie: flat_states = {HIGH}. Geen
#: zoektocht over flat_states". Hij staat hier als default omdat hij elders is
#: geregistreerd -- niet omdat hij hier is gekozen. Wie er een tweede probeert,
#: draait een zoektocht en betaalt de trial.
PRE_REGISTERED_FLAT_STATES: frozenset[VolState] = frozenset({VolState.HIGH})

#: De volledige toestandsruimte, in de volgorde van `VolState`. Hij wordt aan
#: `confusion_matrix` en `cohen_kappa_score` MEEGEGEVEN en niet uit de data
#: afgeleid: een toestand die op dit venster niet voorkomt, hoort een lege rij
#: te krijgen en de matrix niet te laten krimpen.
_LABELS = [int(state) for state in VolState]


@dataclass(frozen=True)
class AgreementReport:
    """Hoe vaak twee toestandstoewijzers hetzelfde zeggen, en wat dat waard is."""

    #: Aandeel (bar, symbool)-cellen met dezelfde toestand, over
    #: `n_cells_compared` -- nooit over alle cellen van het paneel.
    fraction_identical: float
    #: Cohens kappa over diezelfde cellen. Corrigeert voor de bezetting; zonder
    #: hem leest `fraction_identical` bij een scheve bezetting veel te sterk.
    cohen_kappa: float
    #: symbool -> (3, 3) tellingen, rijen `a`, kolommen `b`, in VolState-volgorde.
    confusion: Mapping[str, np.ndarray]
    #: Bars waarop de POORT anders uitvalt: minstens een symbool wisselt van
    #: vlak naar niet-vlak of omgekeerd onder `flat_states`. Dit is het getal
    #: waarop de beslisregel van stap 8.5 wordt afgerekend.
    n_decision_changes: int
    #: `n_decision_changes / n_bars_compared`.
    fraction_bars_with_decision_change: float
    #: Bars waarop minstens een symbool van toestand wisselt, welke wissel ook.
    #: BOVENGRENS op de besluitwijziging onder elke toestandspoort, want een
    #: poort voegt toestanden samen en splitst ze nooit.
    n_any_state_changes: int
    #: `n_any_state_changes / n_bars_compared`.
    fraction_bars_with_any_state_change: float
    #: Teller van `fraction_identical`.
    n_identical: int
    #: Cellen die BEIDE toewijzingen kennen -- de noemer van elke celbreuk.
    n_cells_compared: int
    #: Cellen die minstens een van beide niet kent. Zij zijn weggelaten, niet
    #: gevuld; hun aantal hoort naast elke celbreuk gelezen te worden.
    n_cells_dropped: int
    #: Bars waarop ELK symbool in beide toewijzingen bekend is -- de noemer van
    #: beide barbreuken, en strenger dan `n_cells_compared`.
    n_bars_compared: int
    #: Bars waarop minstens een symbool bij minstens een van beide ontbreekt.
    n_bars_dropped: int
    #: De bars die `n_bars_compared` telt, als index -- de bar-noemer als
    #: VENSTER. `n_bars_compared` is per constructie `len` hiervan, zodat de
    #: telling en het venster niet uit elkaar kunnen lopen.
    bars_compared: pd.Index
    #: De poort waarop `n_decision_changes` is geteld.
    flat_states: frozenset[VolState]
    source_a: str
    source_b: str


def _flat_mask(
    assignment: StateAssignment, flat_states: frozenset[VolState]
) -> pd.DataFrame:
    """Welke cellen de poort VLAK zet, gelezen uit de poort zelf.

    Bewust niet `states.isin(...)`. Dat zou het predikaat van `gate_by_state`
    een tweede keer implementeren (R-3), en vanaf de eerste wijziging aan die
    poort zou deze diagnose iets anders meten dan de poort doet. Een
    eenheidsexposure is de goedkoopste sonde: de poort zet hem op 0 waar hij
    vlak maakt en laat hem op 1 waar hij de view doorlaat.
    """
    states = assignment.states
    unit = pd.DataFrame(1.0, index=states.index, columns=states.columns)
    return gate_by_state(unit, assignment, flat_states=flat_states) == 0.0


def agreement(
    a: StateAssignment,
    b: StateAssignment,
    *,
    flat_states: frozenset[VolState] = PRE_REGISTERED_FLAT_STATES,
) -> AgreementReport:
    """Tel hoe vaak `a` en `b` dezelfde toestand toekennen, en wat dat kost.

    Parameters
    ----------
    a, b
        Twee toewijzingen op HETZELFDE venster en dezelfde symbolen. Een
        stilzwijgende doorsnede zou een noemer opleveren die in geen enkel veld
        van het resultaat staat, dus een afwijkend venster wordt geweigerd.
    flat_states
        De poort waarop `n_decision_changes` wordt geteld. Zie
        `PRE_REGISTERED_FLAT_STATES` voor waar de default vandaan komt.
    """
    states_a, states_b = a.states, b.states
    require(
        bool(states_a.index.equals(states_b.index)),
        "De twee toewijzingen staan niet op hetzelfde venster. Een "
        "stilzwijgende doorsnede zou een noemer opleveren die in geen enkel "
        "veld van dit rapport staat; twee modellen horen op dezelfde bars te "
        "worden vergeleken of helemaal niet.",
        DataContractError,
        n_bars_a=int(len(states_a)), n_bars_b=int(len(states_b)),
        source_a=a.source, source_b=b.source,
    )
    require(
        tuple(states_a.columns) == tuple(states_b.columns),
        "De twee toewijzingen dragen niet dezelfde symbolen in dezelfde "
        "volgorde; een overeenstemming over verschillende namen is geen "
        "overeenstemming.",
        DataContractError,
        columns_a=[str(c) for c in states_a.columns],
        columns_b=[str(c) for c in states_b.columns],
    )
    require(
        len(flat_states) > 0,
        "Een lege `flat_states` is geen poort: zij maakt geen enkele bar vlak, "
        "dus `n_decision_changes` zou per constructie nul zijn en als "
        "overeenstemming worden gelezen. De poortvrije bovengrens staat al in "
        "`n_any_state_changes`.",
        DataContractError, flat_states=sorted(int(s) for s in flat_states),
    )

    known = states_a.notna() & states_b.notna()
    n_cells_compared = int(known.to_numpy().sum())
    require(
        n_cells_compared > 0,
        "Er is geen enkele cel die beide toewijzingen kennen. Zonder "
        "gemeenschappelijke dekking bestaat er geen overeenstemming om te "
        "tellen -- dat is een bevinding over de dekking, geen nulmeting.",
        DataContractError,
        n_known_a=int(states_a.notna().to_numpy().sum()),
        n_known_b=int(states_b.notna().to_numpy().sum()),
        n_cells=int(states_a.size),
    )
    n_identical = int((known & (states_a == states_b)).to_numpy().sum())

    # De bar-noemer is strenger dan de cel-noemer: elk symbool moet bekend zijn.
    complete = known.all(axis=1)
    bars_compared = complete.index[complete.to_numpy()]
    n_bars_compared = int(len(bars_compared))
    require(
        n_bars_compared > 0,
        "Geen enkele bar is in beide toewijzingen VOLLEDIG bekend. De "
        "bar-tellingen zouden dan over een lege noemer lopen; met zes namen is "
        "die eis strikt genoeg om op een dun GARCH-venster leeg te blijven, en "
        "dat is de dekkingsbevinding zelf.",
        DataContractError,
        n_bars=int(len(complete)), n_cells_compared=n_cells_compared,
    )
    differs = known & (states_a != states_b)
    gate_differs = known & (
        _flat_mask(a, flat_states) != _flat_mask(b, flat_states))
    n_any_state_changes = int((complete & differs.any(axis=1)).sum())
    n_decision_changes = int((complete & gate_differs.any(axis=1)).sum())

    mask = known.to_numpy()
    paired_a = states_a.to_numpy(dtype="float64")[mask].astype(int)
    paired_b = states_b.to_numpy(dtype="float64")[mask].astype(int)
    observed = set(paired_a.tolist()) | set(paired_b.tolist())
    require(
        len(observed) > 1,
        "Beide toewijzingen kennen op de gemeenschappelijke dekking maar EEN "
        f"toestand toe ({VolState(min(observed)).name}). De "
        "toevalsovereenstemming is dan 1 en Cohens kappa is 0/0: er valt geen "
        "overeenstemming BOVEN toeval te meten omdat er geen toeval te "
        "verslaan valt. Een NaN in dat veld zou als meting worden gelezen.",
        DataContractError,
        state=VolState(min(observed)).name, n_cells_compared=n_cells_compared,
    )

    return AgreementReport(
        fraction_identical=n_identical / n_cells_compared,
        cohen_kappa=float(
            cohen_kappa_score(paired_a, paired_b, labels=_LABELS)),
        confusion={
            str(column): confusion_matrix(
                states_a[column].to_numpy(dtype="float64")[
                    known[column].to_numpy()].astype(int),
                states_b[column].to_numpy(dtype="float64")[
                    known[column].to_numpy()].astype(int),
                labels=_LABELS)
            for column in states_a.columns
        },
        n_decision_changes=n_decision_changes,
        fraction_bars_with_decision_change=n_decision_changes / n_bars_compared,
        n_any_state_changes=n_any_state_changes,
        fraction_bars_with_any_state_change=(
            n_any_state_changes / n_bars_compared),
        n_identical=n_identical,
        n_cells_compared=n_cells_compared,
        n_cells_dropped=int(states_a.size) - n_cells_compared,
        n_bars_compared=n_bars_compared,
        n_bars_dropped=int(len(complete)) - n_bars_compared,
        bars_compared=bars_compared,
        flat_states=frozenset(flat_states),
        source_a=a.source,
        source_b=b.source,
    )
