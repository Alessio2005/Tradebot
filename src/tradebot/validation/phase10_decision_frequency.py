# src/tradebot/validation/phase10_decision_frequency.py
"""H-10.1 — de meting achter de beslisfrequentie. Fase 10, stap 11.

WAT HIER WOONT EN WAT NIET
==========================
`portfolio/decision_frequency.py` doet de OPERATIE (vasthouden) en de
breakevenREKENING. Dit bestand doet de MEETCAMPAGNE: het verzamelt per k wat de
ladder heeft opgeleverd, snijdt het op de ontwikkelsample, en zet er de
onzekerheid bij die R-8 eist. Er staat hier geen enkele nieuwe statistische
formule: elke Sharpe komt uit `validation/inference.py::sharpe_with_se`, elk
verschil uit `sharpe_difference_test`, elk interval uit `block_bootstrap_ci`, de
deflatie uit `validation/dsr.py::dsr_gate` en het breakevenniveau uit
`portfolio/decision_frequency.py::breakeven_cost_bps` (R-3).

DE PRIMAIRE CEL EN DE KOSTENAS ZIJN TWEE VERSCHILLENDE LAGEN, EN DAT IS GEMETEN
NOODZAAK
===============================================================================
De pre-registratie benoemt één primaire cel: track `long_only_equal_weight`,
laag `L3_execution`, maat netto Sharpe. Daar wordt de hypothese op getoetst.

De BREAKEVENanalyse kan daar niet wonen, en dat is geen keuze maar een gevolg
van hoe de ladder is gebouwd. `backtest/phase5_baseline.py::run_all_layers` zet
op `L3_execution` `gross_returns=net`: de event-driven engine levert géén
bruto/netto-splitsing, want fees, spread, funding en impact zijn daar in de
routerpad verweven en niet als één "kosten per eenheid omzet" te schrijven. Een
`delta_sharpe_gross` op L3 zou dus letterlijk gelijk zijn aan
`delta_sharpe_net`, en een breakevenniveau daarop berekend zou een bruto reeks
gebruiken die de kosten al draagt.

`breakeven_cost_bps` zegt in zijn eigen docstring waar zijn as ligt: *"de
breakeven-as is daarom de LINEAIRE kostenparameter `cost_per_side` waarmee
L0/L1/L2 rekenen -- de enige as waarop 'kosten per eenheid omzet' een enkel
getal is"*, en `r_net = r_gross - c * turnover` is letterlijk
`backtest/vectorized.py:150`. Het breakevenblok staat daarom op
`L0_vectorized`, draagt dat laagveld expliciet mee, en wordt NIET als een
L3-uitspraak gepresenteerd. Wie het als L3-bewijs citeert, citeert de verkeerde
laag.

DE OMZETDEFINITIE IS DIE VAN DE BACKTEST, ÉÉN KEER, EN NIET OPNIEUW GESCHREVEN
==============================================================================
Deze repository kent twee omzetdefinities en dit bestand voegt er geen derde
aan toe (R-3):

* `backtest/vectorized.py:149` — `(weights - held).abs().sum(axis=1)`,
  TWEEZIJDIG en ONGEHALVEERD, waar `held = weights.shift(1)`. Dit is de reeks
  waarop de kosten in de backtest drukken (`net = gross - turnover *
  cost_per_side`, regel 150) en dus de enige definitie waarop een
  kosten-per-eenheid-omzet-as bestaat. `mean_turnover` in dit bestand is het
  gemiddelde van EXACT die reeks, zoals de ladder hem op `L0_vectorized`
  teruggeeft (`VectorizedResult.turnover`, samengevat in het L0-auditveld
  `mean_turnover`). De campagne controleert die overeenkomst ook: zie
  `_check_turnover_provenance`.
* `portfolio/constraints.py::compute_turnover` — EENZIJDIG en GEHALVEERD
  (`0.5 * sum|w_new - w_old|`). Die definitie dient de
  BEPERKINGSHANDHAVING (een omzetlimiet per bar) en niet de kostenmeting; zij
  geeft op hetzelfde paneel de helft van het getal hierboven. Zij wordt hier
  daarom NIET gebruikt, en dat verschil moet in het rapport staan in plaats van
  stilzwijgend één van beide getallen te noemen.

TWEE VENSTERS IN ÉÉN RECORD, BEIDE BENOEMD
==========================================
Een run-record draagt twee steekproefgroottes en dat is geen slordigheid:

* `n_obs` (en daarmee `t_years`) is het aantal ONTWIKKELbars waarop de Sharpe,
  het bootstrap-interval en de DSR zijn gemeten. Gehalteerde bars staan daar als
  rendement nul in en tellen mee -- dat is wat het spoor heeft gedaan.
* `delta_vs_reference.n_effective` is het aantal GEMEENSCHAPPELIJK ACTIEVE bars
  waarop het gepaarde verschil is gemeten. `align="common_active"` is
  onvoorwaardelijk voor een gepaarde vergelijking (MEASUREMENT_CONTRACT.md §4):
  een bar waarop één van beide sporen vlak staat, draagt geen informatie over
  het VERSCHIL. Op een spoor dat 89 % van zijn bars halteert, scheelt dat een
  orde van grootte, en dat verschil hoort zichtbaar te zijn.

`n_sovereign_halted` en `n_sovereign_clipped` dragen een DERDE venster: de
event-driven engine geeft ze als tellers over de hele run terug en niet als
reeks per bar, dus zij gelden over het VOLLEDIGE bruikbare venster. Het veld
`halt_counter_window` zegt dat, zodat niemand ze naast de ontwikkel-Sharpe legt
alsof ze op hetzelfde venster zijn geteld.

WAAROM DE DIAGNOSTIEKTABEL GEEN ENKEL RENDEMENTSGETAL DRAAGT
============================================================
De pre-registratie staat toe dat de ladder alle vier de tracks doorrekent en
noemt die overige cellen DIAGNOSTIEK waaruit niets mag worden geselecteerd. Dat
verbod is hier CONSTRUCTIEF gemaakt in plaats van aangeraden: `DiagnosticCell`
heeft geen Sharpe-, rendement- of equityveld. Er is dus niets in de
diagnostiektabel wat als prestatie kan worden geciteerd; zij draagt uitsluitend
de mechanische schakels van de causale keten (verandert het besluit? beweegt de
omzet? vuurt de halt?), en dat is precies waarvoor zij bestaat -- vaststellen
WELKE schakel breekt.

Ref: `Prompts-fases/fase_10_herstart_dagbars.md` stap 11;
`conf/experiment/h10_1_decision_frequency.yaml`.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from ..registry.preregistration import StopCriterion
from ..utils.failfast import DataContractError, require
from .dsr import DsrResult
from .inference import (
    BootstrapInterval,
    SharpeDifference,
    SharpeEstimate,
    require_sharpe_triple,
)

__all__ = [
    "Breakeven",
    "DecisionFrequencyCampaign",
    "DecisionFrequencyRun",
    "DiagnosticCell",
    "HoldInertness",
    "LadderRead",
    "measure_hold_inertness",
]

#: De REFERENTIE van deze hypothese, en de enige die er kan zijn.
#: `hold_decision(frame, k=1)` is de identiteit, dus k = 1 is het spoor waarin
#: niets wordt vastgehouden. Elke andere referentie zou zelf een vastgehouden
#: spoor zijn en dan meet het "verschil" twee frequenties tegen elkaar in plaats
#: van vasthouden tegen niet-vasthouden. Dit is daarom geen instelbare keuze:
#: een grid zonder 1 wordt geweigerd.
_IDENTITY_K: Final[int] = 1

#: De uitlijning van elke gepaarde vergelijking. MEASUREMENT_CONTRACT.md §4 is
#: onvoorwaardelijk; zie de moduledocstring.
_ALIGN: Final[str] = "common_active"

#: De statistiek van het bootstrap-interval rond één Sharpe.
_SHARPE: Final[str] = "sharpe"

#: Het venster waarop elke statistiek in dit bestand is gemeten.
_SAMPLE: Final[str] = "development"

#: Het venster waarover de event-driven engine zijn haltteller optelt.
_COUNTER_WINDOW: Final[str] = "full_usable_window_including_gate_bars"

#: Kolomnaam waaronder een reeks door `development_slice` wordt gehaald, die een
#: DataFrame verwacht. Geen betekenis, wel één naam in plaats van vier.
_COL: Final[str] = "value"

#: De omzetdefinitie die dit bestand gebruikt, woordelijk in het artefact.
_TURNOVER_DEFINITION: Final[str] = (
    "backtest/vectorized.py:149 -- (weights - weights.shift(1)).abs().sum(axis=1), "
    "two-sided and NOT halved, as exposed on VectorizedResult.turnover at "
    "L0_vectorized. This is the series the backtest charges costs on (line 150: "
    "net = gross - turnover * cost_per_side) and therefore the only definition on "
    "which a cost-per-unit-turnover axis exists. portfolio/constraints.py::"
    "compute_turnover is the OTHER definition in this repository -- one-sided and "
    "halved -- and serves constraint enforcement, not cost; it returns half of "
    "this number on the same panel and is deliberately not used here."
)


# =========================================================================== #
# De inertheid van de vasthoudoperatie op een echt besluitpaneel
# =========================================================================== #
@dataclass(frozen=True)
class HoldInertness:
    """Hoeveel het vasthouden op DIT besluitpaneel feitelijk heeft bewogen.

    Dit is de meting die een nulresultaat interpreteerbaar maakt. Verandert de
    allocator zijn besluit nooit, dan is vasthouden de identiteit en meet de
    hele hypothese niets -- en dat is een uitspraak over de ALLOCATOR, niet over
    de beslisfrequentie. Zonder dit veld is zo'n nulresultaat niet te
    onderscheiden van een kapotte `hold_decision`, en dat onderscheid is het
    hele verschil tussen een bevinding en een bug.
    """

    k: int
    n_bars: int
    #: Het aantal UNIEKE gewichtsrijen in het besluitpaneel. 1 betekent: een
    #: constante vector, dus geen besluit dat kan worden vastgehouden.
    n_unique_decision_rows: int
    #: Het aantal bars waarop het vastgehouden paneel afwijkt van het besluit.
    n_bars_moved_by_hold: int

    @property
    def hold_is_identity(self) -> bool:
        return self.n_bars_moved_by_hold == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "n_bars": int(self.n_bars),
            "n_unique_decision_rows": int(self.n_unique_decision_rows),
            "n_bars_moved_by_hold": int(self.n_bars_moved_by_hold),
            "hold_is_identity": bool(self.hold_is_identity),
        }


def measure_hold_inertness(
    decision_weights: pd.DataFrame, held_weights: pd.DataFrame, *, k: int
) -> HoldInertness:
    """Tel hoeveel bars het vasthouden heeft verplaatst, en hoe vaak het besluit varieert."""
    require(
        isinstance(decision_weights, pd.DataFrame)
        and isinstance(held_weights, pd.DataFrame),
        "De inertheidsmeting vergelijkt twee gewichtsPANELEN. Op een Series is "
        "niet vast te stellen welk symbool zijn besluit vasthield, en dan zegt "
        "het aantal bewogen bars niets over de operatie die is gemeten.",
        DataContractError,
        decision_type=type(decision_weights).__name__,
        held_type=type(held_weights).__name__,
    )
    require(
        bool(decision_weights.index.equals(held_weights.index))
        and list(decision_weights.columns) == list(held_weights.columns),
        "Besluitpaneel en vastgehouden paneel staan niet op dezelfde as. Het "
        "aantal bewogen bars is een VERGELIJKING per bar en per symbool; op twee "
        "verschillende assen telt hij verschillen die uit de uitlijning komen in "
        "plaats van uit het vasthouden.",
        DataContractError,
        n_decision=len(decision_weights.index),
        n_held=len(held_weights.index),
    )
    return HoldInertness(
        k=int(k),
        n_bars=int(len(decision_weights.index)),
        n_unique_decision_rows=int(len(decision_weights.drop_duplicates())),
        n_bars_moved_by_hold=int(
            (held_weights != decision_weights).any(axis=1).sum()
        ),
    )


# =========================================================================== #
# Wat één ladderrun aan de meting overdraagt
# =========================================================================== #
@dataclass(frozen=True)
class LadderRead:
    """De reeksen en audits van één ladderrun bij één k, op het VOLLE venster.

    Het volle venster en niet de ontwikkelsample, en dat is een ORDENING die het
    verschil maakt tussen twee systemen. De L3-engine is padafhankelijk (equity,
    high-water mark, drawdown, en dus de haltbeslissing); een run die bij de
    split begint, start die toestand opnieuw op en meet een ANDER systeem. De
    ladder draait daarom over het volledige bruikbare venster en de SNEDE naar de
    ontwikkelsample gebeurt hier, op de resultaatreeksen. Blokken liggen
    positioneel vanaf de eerste verhandelbare bar, dus de blokindeling van een
    ontwikkelbar hangt nooit van een latere bar af: de volle-venster-run is
    causaal.
    """

    k: int
    #: De netto rendementsreeks van de PRIMAIRE cel (`L3_execution`).
    primary_net_returns: pd.Series
    #: De BRUTO reeks van de kostenas (`L0_vectorized`), vóór kosten.
    cost_axis_gross_returns: pd.Series
    #: De NETTO reeks van dezelfde as: `gross - turnover * cost_per_side`.
    cost_axis_net_returns: pd.Series
    #: `VectorizedResult.turnover` van dezelfde as, per bar.
    cost_axis_turnover: pd.Series
    #: Het auditwoordenboek van `L3_execution` (haltteller, clipteller).
    execution_audit: Mapping[str, Any]
    #: Het auditwoordenboek van `L0_vectorized` (`mean_turnover` over de run).
    cost_axis_audit: Mapping[str, Any]
    inertness: HoldInertness

    def __post_init__(self) -> None:
        require(
            int(self.k) == self.inertness.k,
            "De k van de ladderlezing en die van de inertheidsmeting lopen "
            "uiteen. Dan hoort de inertheid bij een ANDERE vasthoudoperatie dan "
            "de reeksen ernaast, en wordt een nulresultaat aan de verkeerde "
            "bloklengte toegeschreven.",
            DataContractError,
            k=int(self.k),
            inertness_k=self.inertness.k,
        )
        for name, series in (
            ("cost_axis_gross_returns", self.cost_axis_gross_returns),
            ("cost_axis_net_returns", self.cost_axis_net_returns),
            ("cost_axis_turnover", self.cost_axis_turnover),
        ):
            require(
                bool(series.index.equals(self.primary_net_returns.index)),
                f"{name} staat niet op dezelfde tijdas als de primaire reeks. "
                f"Alle vier komen uit ÉÉN ladderrun; lopen de assen uiteen, dan "
                f"komen zij niet uit dezelfde run en meet het breakevenblok een "
                f"ander venster dan de cel waarvoor het wordt gerapporteerd.",
                DataContractError,
                k=int(self.k),
                n_primary=len(self.primary_net_returns.index),
                n_other=len(series.index),
            )


# =========================================================================== #
# Het breakevenblok
# =========================================================================== #
@dataclass(frozen=True)
class Breakeven:
    """Het kostenniveau waarop het netto Sharpe-verschil nul is, met zijn noemer.

    `breakeven_cost_bps` deelt door `turnover_delta`. Een klein omzetverschil
    geeft dus een groot niveau, en dat grote getal is dan een eigenschap van de
    noemer en niet van het systeem. Daarom staat `turnover_delta` hier NAAST het
    niveau, samen met zijn fractie van de referentie-omzet -- als GEMETEN getal.
    Er wordt geen materialiteitsdrempel verzonnen: een drempel die na de meting
    wordt gekozen, is geen drempel (R-2), en de lezer kan de fractie zelf zien.

    `volatility_per_bar` staat er van BEIDE sporen naast, omdat de rekening
    lineariseert onder de aanname dat zij gelijk zijn. `breakeven_cost_bps`
    belooft in zijn docstring dat het artefact van stap 11 die twee naast elkaar
    zet, "zodat de lezer ziet hoe ver de aanname afstaat van de meting in plaats
    van haar op gezag te moeten aannemen".
    """

    layer: str
    n_obs: int
    bars_per_year: float
    t_years: float
    #: `tau(k=1) - tau(k)`, positief wanneer vasthouden de omzet VERLAAGT.
    turnover_delta: float
    turnover_k: float
    turnover_reference: float
    turnover_delta_fraction_of_reference: float
    delta_sharpe_gross: float
    se_delta_sharpe_gross: float
    delta_sharpe_net: float
    se_delta_sharpe_net: float
    #: `None` wanneer er geen breakevenniveau BESTAAT; zie `note`.
    breakeven_cost_bps: float | None
    volatility_per_bar: float
    volatility_per_bar_reference: float
    volatility_ratio: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "layer": self.layer,
            "sample": _SAMPLE,
            "n_obs": int(self.n_obs),
            "bars_per_year": float(self.bars_per_year),
            "t_years": float(self.t_years),
            "turnover_delta": float(self.turnover_delta),
            "turnover_k": float(self.turnover_k),
            "turnover_reference": float(self.turnover_reference),
            "turnover_delta_fraction_of_reference": float(
                self.turnover_delta_fraction_of_reference
            ),
            "delta_sharpe_gross": float(self.delta_sharpe_gross),
            "se_delta_sharpe_gross": float(self.se_delta_sharpe_gross),
            "delta_sharpe_net": float(self.delta_sharpe_net),
            "se_delta_sharpe_net": float(self.se_delta_sharpe_net),
            "breakeven_cost_bps": (
                None if self.breakeven_cost_bps is None
                else float(self.breakeven_cost_bps)
            ),
            "volatility_per_bar": float(self.volatility_per_bar),
            "volatility_per_bar_reference": float(self.volatility_per_bar_reference),
            "volatility_ratio": float(self.volatility_ratio),
            "turnover_definition": _TURNOVER_DEFINITION,
            "note": self.note,
        }
        require_sharpe_triple(record, where="Breakeven.to_dict")
        return record


# =========================================================================== #
# Eén k
# =========================================================================== #
@dataclass(frozen=True)
class DecisionFrequencyRun:
    """Wat er bij één k op de primaire cel is gemeten, met al zijn onzekerheid."""

    k: int
    is_reference: bool
    track: str
    layer: str
    cost_axis_layer: str
    n_obs: int
    bars_per_year: float
    t_years: float
    n_sovereign_halted: int
    n_sovereign_clipped: int
    mean_turnover: float
    mean_turnover_full_window: float
    sharpe: SharpeEstimate
    sharpe_ci: BootstrapInterval
    #: `None` op de referentie: die wordt NIET tegen zichzelf getoetst.
    difference: SharpeDifference | None
    dsr: DsrResult
    breakeven: Breakeven
    inertness: HoldInertness

    @property
    def delta_sharpe(self) -> float:
        """Het verschil tegen de referentie; op de referentie nul per constructie."""
        return 0.0 if self.difference is None else float(self.difference.delta_sharpe)

    @property
    def ci_high_delta_sharpe(self) -> float:
        return 0.0 if self.difference is None else float(self.difference.ci_high)

    def _delta_block(self) -> dict[str, Any]:
        if self.difference is not None:
            return self.difference.to_dict()
        record = {
            "delta_sharpe": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": None,
            "n_obs": int(self.n_obs),
            "bars_per_year": float(self.bars_per_year),
            "t_years": float(self.t_years),
            "align": _ALIGN,
            "test_was_called": False,
            "note": (
                "k = 1 is de identiteit van hold_decision, dus het verschil tegen "
                "de referentie is nul PER CONSTRUCTIE en niet bij benadering. De "
                "verschiltoets is hier niet aangeroepen: op twee identieke reeksen "
                "valt hij in de schaal-invariante ontaardingstak (commit 812a232) "
                "en die geeft het juiste getal om de verkeerde reden -- nul omdat "
                "de covariantie degenereert, in plaats van nul omdat er niets is "
                "vastgehouden. Er is dus ook geen p-waarde: er is niet getoetst."
            ),
        }
        require_sharpe_triple(record, where="DecisionFrequencyRun.delta_vs_reference")
        return record

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "k": int(self.k),
            "is_reference": bool(self.is_reference),
            "track": self.track,
            "layer": self.layer,
            "cost_axis_layer": self.cost_axis_layer,
            "sample": _SAMPLE,
            "n_obs": int(self.n_obs),
            "bars_per_year": float(self.bars_per_year),
            "t_years": float(self.t_years),
            "n_sovereign_halted": int(self.n_sovereign_halted),
            "n_sovereign_clipped": int(self.n_sovereign_clipped),
            "halt_counter_window": _COUNTER_WINDOW,
            "mean_turnover": float(self.mean_turnover),
            "mean_turnover_full_window": float(self.mean_turnover_full_window),
            "turnover_definition": _TURNOVER_DEFINITION,
            "sharpe": self.sharpe.to_dict(),
            "sharpe_ci_block_bootstrap": self.sharpe_ci.to_dict(),
            "delta_vs_reference": self._delta_block(),
            "dsr": self.dsr.as_dict(),
            "breakeven": self.breakeven.to_dict(),
            "hold_inertness": self.inertness.to_dict(),
        }
        require_sharpe_triple(record, where="DecisionFrequencyRun.to_dict")
        return record


# =========================================================================== #
# De diagnostiekcel — mechaniek zonder prestatie
# =========================================================================== #
@dataclass(frozen=True)
class DiagnosticCell:
    """Eén (track, k) in de mechanische keten. Draagt bewust GEEN rendement.

    De keten van de hypothese is: minder besluiten -> minder omzet -> minder
    kosten -> minder drawdown -> minder haltes. Deze cel meet de eerste, de
    tweede en de laatste schakel, en niets anders. Er is geen Sharpe- of
    rendementsveld, dus er valt niets uit te selecteren -- het verbod van de
    pre-registratie is hier constructie in plaats van belofte.
    """

    track: str
    is_primary_track: bool
    k: int
    n_bars: int
    n_unique_decision_rows: int
    n_bars_moved_by_hold: int
    mean_turnover: float
    turnover_reduction_fraction_vs_reference: float
    n_sovereign_halted: int
    n_sovereign_clipped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "is_primary_track": bool(self.is_primary_track),
            "k": int(self.k),
            "sample": _SAMPLE,
            "n_bars": int(self.n_bars),
            "n_unique_decision_rows": int(self.n_unique_decision_rows),
            "n_bars_moved_by_hold": int(self.n_bars_moved_by_hold),
            "mean_turnover": float(self.mean_turnover),
            "turnover_reduction_fraction_vs_reference": float(
                self.turnover_reduction_fraction_vs_reference
            ),
            "n_sovereign_halted": int(self.n_sovereign_halted),
            "n_sovereign_clipped": int(self.n_sovereign_clipped),
            "halt_counter_window": _COUNTER_WINDOW,
            "carries_no_performance_number": True,
        }


# =========================================================================== #
# De campagne
# =========================================================================== #
@dataclass(frozen=True)
class DecisionFrequencyCampaign:
    """Alle vier de k's op de primaire cel, plus de mechanische diagnostiek."""

    runs: tuple[DecisionFrequencyRun, ...]
    diagnostics: tuple[DiagnosticCell, ...]
    track: str
    layer: str
    cost_axis_layer: str
    anchor: str
    n_obs: int
    bars_per_year: float
    t_years: float
    cost_per_side_bps: float
    stop_criteria: tuple[StopCriterion, ...]

    @property
    def reference_k(self) -> int:
        return _IDENTITY_K

    @property
    def k_grid(self) -> tuple[int, ...]:
        return tuple(run.k for run in self.runs)

    def run(self, k: int) -> DecisionFrequencyRun:
        for candidate in self.runs:
            if candidate.k == k:
                return candidate
        require(
            False,
            f"Geen run voor k = {k} in deze campagne. Een verwijzing naar een k "
            f"die niet is gedraaid, zou een getal uit een andere run citeren.",
            DataContractError,
            k=int(k),
            k_grid=list(self.k_grid),
        )
        raise AssertionError("unreachable")  # pragma: no cover

    @property
    def best_k(self) -> int:
        """De gunstigste KANDIDAAT op de ontwikkelsample; de referentie doet niet mee.

        Het hoogste `delta_sharpe`, bij gelijkspel de KLEINSTE k. De referentie
        is uitgesloten omdat zij per constructie nul is: zou zij meedoen, dan
        zou een campagne waarin geen enkele k iets toevoegt "k = 1" als beste
        aanwijzen en daarmee de vraag ontwijken in plaats van beantwoorden.
        Staan alle kandidaten exact gelijk, dan is de keuze willekeurig EN
        inconsequent: het eerste stop-criterium bindt dan bij elke kandidaat, dus
        geen tiebreak kan de uitkomst verplaatsen.
        """
        candidates = [run for run in self.runs if not run.is_reference]
        require(
            bool(candidates),
            "Een campagne zonder kandidaat naast de referentie meet niets: er is "
            "dan geen k > 1 waarvan het verschil tegen niet-vasthouden kan worden "
            "gelezen, en de hypothese gaat juist over dat verschil.",
            DataContractError,
            k_grid=list(self.k_grid),
        )
        return min(candidates, key=lambda run: (-run.delta_sharpe, run.k)).k

    def stop_criteria_metrics(self) -> dict[str, Any]:
        """De metrieknamen van de pre-registratie, letterlijk, met hun waarde.

        `delta_net_sharpe_gate_best_k` is `None` en blijft dat in dit artefact:
        de poortsample is NIET gelezen (R-7, ten hoogste één lezing per
        hypothese, en die lezing is van de controller). `n_binding_stop_criteria`
        is daarom ook `None` -- een conjunctie waarvan één term onbekend is, is
        onbekend en niet nul. Het aantal bindende criteria dat WEL meetbaar is,
        staat als `n_binding_criteria_measurable` in het campagnerecord.
        """
        best = self.run(self.best_k)
        return {
            "delta_net_sharpe_development_best_k": best.delta_sharpe,
            "dsr_development_best_k": float(best.dsr.dsr),
            "delta_net_sharpe_gate_best_k": None,
            "breakeven_cost_bps_development_best_k": (
                None if best.breakeven.breakeven_cost_bps is None
                else float(best.breakeven.breakeven_cost_bps)
            ),
            "ci_high_delta_net_sharpe_development_best_k": best.ci_high_delta_sharpe,
            "n_binding_stop_criteria": None,
            # MEASUREMENT_CONTRACT.md §10: dit record draagt Sharpe-sleutels en
            # gaat dus niet zonder zijn drietal naar buiten. De consument zoekt
            # een criterium op via zijn `metric`-veld, dus de drie extra sleutels
            # hinderen die lezing niet.
            "n_obs": int(self.n_obs),
            "bars_per_year": float(self.bars_per_year),
            "t_years": float(self.t_years),
        }

    def evaluated_stop_criteria(self) -> list[dict[str, Any]]:
        """Elk pre-geregistreerd criterium tegen zijn gemeten waarde.

        `binds = None` betekent: niet te beoordelen omdat de metriek niet
        bestaat. Dat is geen "gehaald": een criterium dat niet kan worden
        beoordeeld, is niet vrijgegeven.
        """
        metrics = self.stop_criteria_metrics()
        out: list[dict[str, Any]] = []
        for criterion in self.stop_criteria:
            measured = metrics.get(criterion.metric, None)
            record = criterion.as_dict()
            record["measured"] = measured
            record["binds"] = (
                None if measured is None else bool(criterion.binds(float(measured)))
            )
            if measured is None:
                record["unmeasurable_reason"] = (
                    "de metriek bestaat niet in dit artefact: het poortsample is "
                    "niet gelezen (R-7) of het breakevenniveau bestaat niet omdat "
                    "de omzet niet is bewogen. Een criterium dat niet kan worden "
                    "beoordeeld, is niet gehaald."
                )
            out.append(record)
        return out

    @property
    def n_binding_criteria_measurable(self) -> int:
        return sum(
            1 for record in self.evaluated_stop_criteria() if record["binds"] is True
        )

    @property
    def n_measurable_criteria(self) -> int:
        return sum(
            1 for record in self.evaluated_stop_criteria()
            if record["binds"] is not None
        )

    def note(self) -> str:
        """De uitspraak, GEGENEREERD uit de gemeten getallen en niet met de hand.

        Dezelfde reden als in `apps/run_phase5_baseline.py::_verdict`: een met de
        hand geschreven zin kan de getallen naast zich tegenspreken zodra het
        artefact opnieuw wordt gedraaid. Elke bewering hieronder is een
        vertakking op een gemeten grootheid.
        """
        candidates = [run for run in self.runs if not run.is_reference]
        reference = self.run(self.reference_k)
        halts = {run.k: run.n_sovereign_halted for run in self.runs}
        inert = [run for run in candidates if run.inertness.hold_is_identity]
        best = self.run(self.best_k)
        parts = [
            f"Primaire cel: {self.track} / {self.layer}, netto Sharpe, "
            f"{self.n_obs} ontwikkelbars = {self.t_years:.4f} jaar. "
            f"k-grid {list(self.k_grid)}, referentie k = {self.reference_k} "
            f"(de identiteit van hold_decision)."
        ]
        if len(inert) == len(candidates) and candidates:
            parts.append(
                f"HET VASTHOUDEN IS OP DEZE CEL DE IDENTITEIT, GEMETEN: het "
                f"besluitpaneel draagt "
                f"{reference.inertness.n_unique_decision_rows} unieke gewichtsrij"
                f"(en) op {reference.inertness.n_bars} bars, en bij ELKE k > 1 "
                f"beweegt het vasthouden 0 bars. De eerste schakel van de causale "
                f"keten (minder besluiten -> minder omzet) bestaat hier dus niet, "
                f"en de drie kandidaat-k's zijn bit-identiek aan de referentie. "
                f"Dat is een uitspraak over de ALLOCATOR van deze track en niet "
                f"over de beslisfrequentie: een gelijkgewogen boek op een "
                f"universum dat niet van samenstelling verandert, IS een "
                f"constante vector. `hold_decision` is niet inert -- de "
                f"diagnostiektabel meet op de tracks die WEL van besluit "
                f"veranderen dat de omzet met k daalt -- maar op de "
                f"pre-geregistreerde cel valt er niets vast te houden."
            )
        elif inert:
            parts.append(
                f"Het vasthouden is de identiteit bij k = "
                f"{[run.k for run in inert]} en beweegt bars bij de overige k; de "
                f"campagne is daarmee op een deel van het grid leeg."
            )
        if len(set(halts.values())) == 1:
            parts.append(
                f"De haltteller beweegt NIET met k: {sorted(set(halts.values()))[0]} "
                f"haltes bij elke k, geteld over het {_COUNTER_WINDOW}. De "
                f"nulhypothese van de pre-registratie noemt dat expliciet: een "
                f"omzetdaling die niet door de haltes wordt gevolgd, weerlegt de "
                f"keten en bevestigt haar niet."
            )
        else:
            parts.append(f"Haltes per k: {dict(sorted(halts.items()))}.")
        parts.append(
            f"Beste kandidaat op de ontwikkelsample: k = {self.best_k} met "
            f"delta_sharpe = {best.delta_sharpe:+.6f} en 95 %-interval "
            f"[{(0.0 if best.difference is None else best.difference.ci_low):+.6f}, "
            f"{best.ci_high_delta_sharpe:+.6f}]; {best.breakeven.note}"
        )
        parts.append(
            f"{self.n_binding_criteria_measurable} van de "
            f"{self.n_measurable_criteria} beoordeelbare stop-criteria binden; "
            f"{len(self.stop_criteria) - self.n_measurable_criteria} criteria zijn "
            f"NIET beoordeelbaar zolang de poortsample niet is gelezen. Het "
            f"verdict hoort in het rapport van stap 11.9 en staat bewust niet in "
            f"dit artefact."
        )
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "track": self.track,
            "layer": self.layer,
            "cost_axis_layer": self.cost_axis_layer,
            "anchor": self.anchor,
            "reference_k": int(self.reference_k),
            "k_grid": list(self.k_grid),
            "n_trials": len(self.runs),
            "sample": _SAMPLE,
            "n_obs": int(self.n_obs),
            "bars_per_year": float(self.bars_per_year),
            "t_years": float(self.t_years),
            "cost_per_side_bps": float(self.cost_per_side_bps),
            "best_k": int(self.best_k),
            "best_k_selection_rule": (
                "highest delta_sharpe on the development sample among k != 1; ties "
                "broken by the smallest k. The reference k = 1 is not a candidate."
            ),
            "delta_net_sharpe_development_best_k": self.run(self.best_k).delta_sharpe,
            "runs": {str(run.k): run.to_dict() for run in self.runs},
            "diagnostics": [cell.to_dict() for cell in self.diagnostics],
            "stop_criteria": self.evaluated_stop_criteria(),
            "stop_criteria_metrics": self.stop_criteria_metrics(),
            "n_binding_criteria_measurable": self.n_binding_criteria_measurable,
            "n_measurable_criteria": self.n_measurable_criteria,
            "gate_sample": {
                "read": False,
                "note": (
                    "Het poortsample is door deze meting NIET aangeraakt: er is geen "
                    "aanroep van validation/holdout.py::gate_slice in het pad dat dit "
                    "artefact produceert, en holdout_lock.json draagt dus geen lezing "
                    "voor deze hypothese. R-7 staat één lezing toe en die hoort bij "
                    "stap 11.8, voor de k die op de ontwikkelsample het gunstigst is."
                ),
            },
            "note": self.note(),
        }
        require_sharpe_triple(record, where="DecisionFrequencyCampaign.to_dict")
        return record
