"""De poort die REGEL V respecteert -- stap 7, AD-25.

WAAROM DIT EEN POORT IS EN GEEN FACTOR
=======================================
REGEL V (`Prompts-fases/fase_10_herstart_dagbars.md` §4.3, de precisering van
AD-16): een cross-sectioneel UNIFORME factor is per constructie een lege
operatie, want L7 herschaalt naar sigma-target en deelt elke constante er weer
uit. Een cross-sectioneel GEDIFFERENTIEERDE factor raakt wel de samenstelling
en is dus niet leeg -- maar zij kost `k - 1` vrije parameters per toestand, en
elke daarvan is een trial (R-2). De enige gedifferentieerde afbeelding zonder
vrije parameter is de POORT: `c_t(i) in {0, 1}`, met de toestandsverzameling
die op nul gaat vooraf geregistreerd via `flat_states`:

    s_t in flat_states   ->  0    (geen view)
    s_t elders           ->  a_t  (volle view, ongeschaald)
    s_t onbekend (NaN)   ->  NaN  (geen besluit)

Geen multiplier per toestand -- ook niet als optie met een default. Dat zou
het alternatief zijn dat AD-15 al afwees (een sizing-experiment in plaats van
een compositie-experiment), en het zou hier zonder vooraf geregistreerde
trial-kosten binnensluipen.

CONCENTRATIE IS EEN NIEUW RISICO EN HOORT GEMETEN (Q9)
========================================================
De poort zet namen op nul; de soevereine laag L7 herschaalt de REST naar het
vol-target. Dat verhoogt het gewicht van de overgebleven namen -- een risico
dat de poort zelf introduceert en niet het onderliggende model.
`concentration_report` meet dat per bar op BEIDE boeken (basis en gepoort):
het maximale genormaliseerde gewicht, het effectieve aantal namen
(`1 / Sum(w_i^2)`, de inverse Herfindahl-index) en of het boek zijn
vol-target kan halen (`gross_after_vol_target`). Een volledig platgepoorte bar
kan dat target niet halen -- L7 mag daar niet op reageren met een deling door
nul, zie `test_a_fully_gated_bar_is_flat_and_not_infinitely_levered` -- en
wordt dus expliciet als 0 gerapporteerd, niet als 1 en niet als oneindig.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .state import StateAssignment, VolState

__all__ = ["ConcentrationReport", "concentration_report", "gate_by_state"]


def gate_by_state(
    exposures: pd.DataFrame,
    assignment: StateAssignment,
    *,
    flat_states: frozenset[VolState],
) -> pd.DataFrame:
    """Poort `exposures` op `assignment.states`. Geen multiplier, geen schaal.

    Elke naam in `flat_states` gaat op elke bar waar zij die toestand draagt
    naar exact 0 -- geen demping, geen partiele reductie. Elke andere bekende
    toestand laat de exposure ONGEWIJZIGD (de volle view of geen view, nooit
    iets ertussenin). Een onbekende toestand (NaN, typisch de opstartfase van
    `assign_by_variance`) wordt NaN: geen toestand is geen besluit, en 0 zou
    hier "wij kiezen vlak" betekenen terwijl niemand die keuze heeft gemaakt.
    """
    require(
        bool(exposures.index.equals(assignment.states.index)),
        "Exposures en de toestandstoewijzing staan niet op dezelfde tijdas. "
        "Een stilzwijgende reindex zou een symbool een toestand op een bar "
        "geven die het model daar nooit heeft toegewezen.",
        DataContractError,
        n_exposures=len(exposures), n_states=len(assignment.states),
    )
    require(
        tuple(exposures.columns) == tuple(assignment.states.columns),
        "Exposures en de toestandstoewijzing dragen niet dezelfde symbolen.",
        DataContractError,
        exposures=list(exposures.columns),
        states=list(assignment.states.columns),
    )
    require(
        set(flat_states) != set(VolState),
        "Elke toestand poorten is geen conditioneerder maar een uit-knop: het "
        "boek zou dan op elke bekende bar vlak zijn, en er is niets meer om "
        "te conditioneren.",
        DataContractError,
        flat_states=sorted(int(s) for s in flat_states),
    )

    states = assignment.states
    flat_values = [float(s) for s in flat_states]
    is_flat = states.isin(flat_values)
    is_unknown = states.isna()

    gated = exposures.astype("float64").where(~is_flat, 0.0)
    gated = gated.where(~is_unknown, np.nan)
    return gated


@dataclass(frozen=True)
class ConcentrationReport:
    """Concentratie vóór en na de poort -- Q9.

    De zes scalaire velden zijn de MEDIAAN (p50) over de tijd, voor wie het
    getal zonder kwantiel-navigatie wil lezen. `quantiles` draagt de volledige
    50/95/99-tabel (index = het kwantiel, kolommen = de zes reeksen hieronder,
    elk met de suffix `_base` of `_gated`) voor het rapportartefact.

    BEIDE boeken zijn gemeten over HETZELFDE bar-venster: de bars waarop het
    GEPOORTE boek een bekende toestand draagt op elk symbool (`gated.isna()`
    is nergens True in de rij). Buiten dat venster (typisch de opstartfase
    van `assign_by_variance`) is er niets om te vergelijken -- zie
    `_per_bar_metrics`. Zonder die gedeelde restrictie zou het basisboek over
    meer bars zijn gemeten dan het gepoorte boek en zou een before/after-
    vergelijking van twee verschillende steekproeven een vergelijking van
    dezelfde lijken (fix ronde 1, bevinding P33). `n_bars_compared` en
    `n_bars_total` maken dat venster zichtbaar in het artefact.
    """

    max_weight_base: float
    max_weight_gated: float
    effective_names_base: float
    effective_names_gated: float
    #: Per-constructie 1,0 zodra het boek op die bar niet leeg en niet
    #: onbekend is, en NOOIT iets anders -- er is geen sigma-dak in deze
    #: tweeparametrige handtekening (`exposures`, `gated`) om een echte
    #: bruto-na-targeting-waarde uit te rekenen (die zou `min(max_leverage,
    #: sigma_target / sigma_boek) x bruto` zijn, en dus onder de
    #: hefboomgrens kunnen zakken). Dit veld is daarom een INDICATOR van
    #: vlak/onbekend, geen gemeten grootheid -- fix ronde 1, bevinding P34.
    gross_after_vol_target_base: float
    #: Zie `gross_after_vol_target_base`. Dezelfde constructie, op het
    #: gepoorte boek: 0,0 op een volledig platgepoorte bar (nul namen, dus
    #: geen enkel target haalbaar), NaN op een onbekende bar (uitgesloten
    #: door het gedeelde venster hierboven, dus in de praktijk nooit NaN
    #: binnen dit rapport), anders exact 1,0.
    gross_after_vol_target_gated: float
    #: Aantal bars waarover de zes velden hierboven en `quantiles` zijn
    #: berekend -- het gedeelde venster (zie klassedocstring).
    n_bars_compared: int
    #: Totaal aantal bars in `exposures`/`gated`, vóór de venster-restrictie.
    #: `n_bars_compared < n_bars_total` betekent dat er bars zijn uitgesloten
    #: (typisch opstart-NaN), en dat is zichtbaar in plaats van stilzwijgend.
    n_bars_total: int
    quantiles: pd.DataFrame


def _per_bar_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """`max_weight`, `effective_names` en `gross_after_vol_target`, per bar.

    Genormaliseerd op de bruto exposure -- dezelfde operatie als L7's
    vol-target-herschaling, die de samenstelling ongemoeid laat en alleen de
    schaal raakt (AD-16). Twee ontaarde bars krijgen een expliciete, aparte
    lezing in plaats van een stilzwijgende 0/0:

    * een bar met een ONBEKENDE toestand (NaN in `frame`, typisch de
      opstartfase) is geen "vlak boek" maar een boek waarover niets is
      besloten -- alle drie de grootheden worden NaN;
    * een bar die WEL bekend is maar waarvan de bruto exposure exact 0 is
      (elke gepoorte naam tegelijk), is een echt vlak boek --
      `gross_after_vol_target` is daar 0 en niet 1 (het target wordt niet
      gehaald, en L7 mag dat niet oplossen door door nul te delen);
      `max_weight`/`effective_names` zijn ook daar NaN, want concentratie
      ONDER nul namen is geen grootheid.
    """
    has_unknown = frame.isna().any(axis=1)
    absolute = frame.abs()
    gross = absolute.sum(axis=1)
    is_flat = (gross <= 0.0) & ~has_unknown
    undefined = is_flat | has_unknown

    safe_gross = gross.where(~undefined, 1.0)
    normalised = absolute.div(safe_gross, axis=0)

    # De lokale naam is NIET `max_weight`, en dat is opzet. `max_weight` is in
    # dit platform een SOEVEREINE LIMIET: `portfolio/constraints.py` noemt hem
    # in `sovereign_fields`. Wat hier wordt berekend is het tegenovergestelde
    # -- een METING van het grootste waargenomen gewicht per bar, geen cap die
    # iets begrenst. Met de limietnaam ernaast kan een lezer die twee niet
    # onderscheiden, en `tests/integration/test_sovereign_wiring.py` stond er
    # terecht rood op: zijn AST-scan kan alleen de NAAM zien, niet de
    # bedoeling. De uitvoerkolom blijft `max_weight` -- die naam draagt via
    # `add_suffix` de velden van `ConcentrationReport` en staat in artefacten.
    largest_weight = normalised.max(axis=1).where(~undefined, np.nan)
    effective_names = (1.0 / normalised.pow(2).sum(axis=1)).where(
        ~undefined, np.nan)
    gross_after_vol_target = pd.Series(
        np.select([has_unknown, is_flat], [np.nan, 0.0], default=1.0),
        index=frame.index,
    )

    return pd.DataFrame({
        "max_weight": largest_weight,
        "effective_names": effective_names,
        "gross_after_vol_target": gross_after_vol_target,
    })


def concentration_report(
    exposures: pd.DataFrame, gated: pd.DataFrame,
) -> ConcentrationReport:
    """Meet Q9: hoeveel concentratie introduceert de poort in `gated`?

    Het vergelijkingsvenster is het GEPOORTE boek zijn support: de bars
    waarop `gated` op elk symbool een bekende toestand draagt (geen NaN in de
    rij). Het basisboek wordt op DEZELFDE bars gemeten, niet op zijn eigen
    volledige reeks -- anders vergelijkt het rapport een basisboek over ~400
    bars met een gepoort boek over ~300 en zet het twee verschillende
    steekproeven naast elkaar als was het een before/after (fix ronde 1,
    bevinding P33). Bars buiten dat venster (typisch de opstartfase) zijn
    UITGESLOTEN, niet stilzwijgend als 0 of 1 meegeteld; `n_bars_compared`
    en `n_bars_total` op het resultaat maken zichtbaar hoeveel dat scheelt.
    """
    require(
        bool(exposures.index.equals(gated.index)),
        "Het basisboek en het gepoorte boek staan niet op dezelfde tijdas.",
        DataContractError, n_base=len(exposures), n_gated=len(gated),
    )
    require(
        tuple(exposures.columns) == tuple(gated.columns),
        "Het basisboek en het gepoorte boek dragen niet dezelfde symbolen.",
        DataContractError,
        base=list(exposures.columns), gated=list(gated.columns),
    )

    support = ~gated.isna().any(axis=1)
    require(
        bool(support.any()),
        "Het gepoorte boek heeft geen enkele bar met een bekende toestand op "
        "elk symbool; er is geen enkele bar om de twee boeken op te "
        "vergelijken.",
        DataContractError, n_bars_total=len(gated),
    )

    per_bar = _per_bar_metrics(exposures.loc[support]).add_suffix("_base").join(
        _per_bar_metrics(gated.loc[support]).add_suffix("_gated"))
    quantiles = per_bar.quantile([0.5, 0.95, 0.99])

    def p50(column: str) -> float:
        return float(quantiles.loc[0.5, column])

    return ConcentrationReport(
        max_weight_base=p50("max_weight_base"),
        max_weight_gated=p50("max_weight_gated"),
        effective_names_base=p50("effective_names_base"),
        effective_names_gated=p50("effective_names_gated"),
        gross_after_vol_target_base=p50("gross_after_vol_target_base"),
        gross_after_vol_target_gated=p50("gross_after_vol_target_gated"),
        n_bars_compared=int(support.sum()),
        n_bars_total=len(gated),
        quantiles=quantiles,
    )
