# src/tradebot/validation/phase10_decision_frequency_measurement.py
"""De METING van H-10.1, gescheiden van de records die zij oplevert.

WAAROM DIT EEN EIGEN BESTAND IS. `phase10_decision_frequency.py` legt vast WAT
er wordt gemeten: de records, hun velden, hun contracten en hun serialisatie.
Dit bestand legt vast HOE die records tot stand komen: welke reeks uit welke
laag wordt gelezen, waar de ontwikkelsample wordt afgesneden, welke toets op
welk paar wordt losgelaten. Dat zijn twee verantwoordelijkheden, en R-4 vraagt
te splitsen op verantwoordelijkheid en niet op regelaantal.

De splitsing is uitgevoerd toen het samengevoegde bestand op 1.124 regels stond
en dus boven de R-4-grens van 800. Zij is een VERPLAATSING en geen herschrijving:
geen enkele berekening, geen enkele constante en geen enkel veld is gewijzigd,
zodat geen gemeten getal van deze splitsing kan afhangen. Er is bewust GEEN
`CAPS`-regel in `scripts/check_file_size.py` bijgeschreven -- die tabel legt de
omvang vast van bestanden die bij invoering van de ratel al te groot waren, en
haar gebruiken om een NIEUW te groot bestand te vergunnen keert haar doel om.

DE RICHTING VAN DE AFHANKELIJKHEID. Dit bestand importeert uit de recordmodule;
die weet niets van dit bestand. Een record moet te lezen en te serialiseren zijn
zonder dat de ladder draait -- dat is precies wat de tests op de records doen.

DE VIER GEDEELDE CONSTANTEN. `_ALIGN`, `_COL`, `_IDENTITY_K` en `_SHARPE` worden
uit de recordmodule geimporteerd en hier NIET opnieuw gedeclareerd. Zij horen bij
het meetcontract dat beide helften delen; een tweede kopie zou kunnen gaan
afwijken, en dan meet dit bestand iets anders dan het record beweert te dragen.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..portfolio.decision_frequency import breakeven_cost_bps
from ..registry.preregistration import StopCriterion
from ..registry.trial_counter import TrialCount
from ..schemas.config import ValidationConfig
from ..utils.failfast import DataContractError, require
from .dsr import dsr_gate
from .holdout import development_slice
from .inference import (
    block_bootstrap_ci,
    sharpe_difference_test,
    sharpe_with_se,
)
from .phase10_decision_frequency import (
    _ALIGN,
    _COL,
    _IDENTITY_K,
    _SHARPE,
    Breakeven,
    DecisionFrequencyCampaign,
    DecisionFrequencyRun,
    DiagnosticCell,
    LadderRead,
)

__all__ = ["measure_campaign"]



# =========================================================================== #
# De meting
# =========================================================================== #
def _development(series: pd.Series, lock_path: Path) -> pd.Series:
    """De ontwikkelsample van één reeks; lezen hiervan is vrij van R-7."""
    return development_slice(series.to_frame(_COL), lock_path=lock_path)[_COL]


def _check_turnover_provenance(read: LadderRead) -> float:
    """Is de omzetreeks die hier binnenkomt DIE van de ladder op L0?

    De ladder vat `VectorizedResult.turnover` samen in het L0-auditveld
    `mean_turnover`. Deze controle vergelijkt het gemiddelde van de MEEGEGEVEN
    reeks over hetzelfde volle venster met dat auditgetal. Zij is niet
    tautologisch: de aanroeper kan een andere reeks meegeven -- een gehalveerde
    omzet uit `portfolio/constraints.py`, of de omzet van een andere track -- en
    dan zou de kostenas van het breakevenblok niet die van de gemeten ladder
    zijn.
    """
    audited = read.cost_axis_audit.get("mean_turnover")
    require(
        audited is not None,
        "Het L0-auditwoordenboek draagt geen `mean_turnover`. Dat veld is de "
        "herkomstcontrole op de omzetreeks: zonder hem is niet vast te stellen "
        "dat de omzet waarop het breakevenniveau rust, dezelfde is die de ladder "
        "op L0_vectorized heeft gerekend, en R-3 laat geen tweede omzetdefinitie "
        "toe.",
        DataContractError,
        k=int(read.k),
        audit_keys=sorted(read.cost_axis_audit),
    )
    measured = float(read.cost_axis_turnover.mean())
    require(
        bool(np.isclose(measured, float(audited))),
        f"De meegegeven omzetreeks heeft gemiddelde {measured!r} terwijl het "
        f"L0-audit {audited!r} rapporteert. Dan komt de reeks niet uit de "
        f"ladderrun waarvan de rest van dit record komt, en meet het "
        f"breakevenniveau een kostenas die naast de gemeten backtest staat.",
        DataContractError,
        k=int(read.k),
        measured=measured,
        audited=float(audited),
    )
    return measured


def _halt_counter(read: LadderRead, field: str) -> int:
    value = read.execution_audit.get(field)
    require(
        value is not None,
        f"Het L3-auditwoordenboek draagt geen {field!r}. Alleen de event-driven "
        f"laag telt dit; ontbreekt het veld, dan komt dit record niet van "
        f"L3_execution en is de haltvraag van de hypothese niet gemeten maar "
        f"onbeantwoord gelaten.",
        DataContractError,
        k=int(read.k),
        audit_keys=sorted(read.execution_audit),
    )
    return int(value)


def _breakeven(
    read: LadderRead,
    reference: LadderRead,
    *,
    lock_path: Path,
    bars_per_year: float,
    layer: str,
) -> Breakeven:
    """Het breakevenblok van één k op de KOSTENAS (L0). Zie de moduledocstring."""
    turnover_k = float(_development(read.cost_axis_turnover, lock_path).mean())
    turnover_ref = float(_development(reference.cost_axis_turnover, lock_path).mean())
    turnover_delta = turnover_ref - turnover_k
    net_k = _development(read.cost_axis_net_returns, lock_path)
    net_ref = _development(reference.cost_axis_net_returns, lock_path)
    vol_k = float(np.std(net_k.to_numpy(dtype="float64"), ddof=1))
    vol_ref = float(np.std(net_ref.to_numpy(dtype="float64"), ddof=1))
    fraction = turnover_delta / turnover_ref if turnover_ref != 0.0 else float("nan")
    shared = {
        "layer": layer,
        "turnover_delta": turnover_delta,
        "turnover_k": turnover_k,
        "turnover_reference": turnover_ref,
        "turnover_delta_fraction_of_reference": fraction,
        "volatility_per_bar": vol_k,
        "volatility_per_bar_reference": vol_ref,
        "volatility_ratio": vol_k / vol_ref if vol_ref != 0.0 else float("nan"),
        "bars_per_year": float(bars_per_year),
    }

    if read.k == reference.k:
        n_obs = int(len(net_k))
        return Breakeven(
            n_obs=n_obs, t_years=n_obs / float(bars_per_year),
            delta_sharpe_gross=0.0, se_delta_sharpe_gross=0.0,
            delta_sharpe_net=0.0, se_delta_sharpe_net=0.0,
            breakeven_cost_bps=None,
            note=(
                "Geen breakevenniveau: dit IS de referentie. Er bestaat geen "
                "kostenniveau waarop het verschil van een spoor met zichzelf van "
                "teken verandert, en dat is een constructie en geen meting."
            ),
            **shared,
        )

    gross = sharpe_difference_test(
        _development(read.cost_axis_gross_returns, lock_path),
        _development(reference.cost_axis_gross_returns, lock_path),
        bars_per_year=bars_per_year, align=_ALIGN,
    )
    net = sharpe_difference_test(
        net_k, net_ref, bars_per_year=bars_per_year, align=_ALIGN,
    )
    n_obs = int(net.n_effective)
    measured = {
        "n_obs": n_obs,
        "t_years": n_obs / float(bars_per_year),
        "delta_sharpe_gross": float(gross.delta_sharpe),
        "se_delta_sharpe_gross": float(gross.se),
        "delta_sharpe_net": float(net.delta_sharpe),
        "se_delta_sharpe_net": float(net.se),
    }

    # VOORAF CONTROLEREN EN NIET ACHTERAF OPVANGEN. `breakeven_cost_bps` gooit
    # bij `turnover_delta == 0` een `DataContractError`, en dat is opzet: de
    # uitkomst is dan een BEVINDING over de keten van de hypothese en geen
    # getal. Die uitzondering wegvangen zou de bevinding wegschrijven als een
    # rekenkundig detail; haar laten ontsnappen zou de campagne laten crashen op
    # precies het geval dat zij moet kunnen rapporteren.
    if turnover_delta == 0.0:
        return Breakeven(
            breakeven_cost_bps=None,
            note=(
                f"Geen breakevenniveau: het omzetverschil is EXACT nul "
                f"(turnover k = {turnover_k:.9f}, k = 1 = {turnover_ref:.9f}). "
                f"Vasthouden heeft de omzet niet bewogen, dus er bestaat geen "
                f"kostenniveau dat het teken van het Sharpe-verschil omdraait. "
                f"Dit is de uitkomst van de meting en niet een ontbrekend getal: "
                f"de kostenschakel van de causale keten is hier leeg."
            ),
            **shared, **measured,
        )

    level = breakeven_cost_bps(
        measured["delta_sharpe_gross"], turnover_delta,
        volatility_per_bar=vol_k, bars_per_year=bars_per_year,
    )
    return Breakeven(
        breakeven_cost_bps=level,
        note=(
            f"Breakeven bij {level:.4f} bp per eenheid omzet. Lees dat niveau "
            f"NOOIT zonder zijn noemer: het is delta_sharpe_gross gedeeld door "
            f"turnover_delta = {turnover_delta:.9f}, en die daling is "
            f"{fraction:.6f} van de omzet bij k = 1 ({turnover_ref:.9f}). Hoe "
            f"kleiner die fractie, hoe meer het niveau een eigenschap van de "
            f"noemer is en niet van het systeem; er staat hier geen "
            f"materialiteitsdrempel, alleen het gemeten aandeel. De rekening "
            f"lineariseert bovendien op gelijke volatiliteit: gemeten is "
            f"{vol_k:.8f} per bar tegen {vol_ref:.8f} bij k = 1, een verhouding "
            f"van {shared['volatility_ratio']:.6f}."
        ),
        **shared, **measured,
    )


def _run_record(
    read: LadderRead,
    reference: LadderRead,
    *,
    track: str,
    primary_layer: str,
    cost_axis_layer: str,
    lock_path: Path,
    bars_per_year: float,
    trial_count: TrialCount,
    validation: ValidationConfig,
) -> DecisionFrequencyRun:
    """Eén k op de primaire cel, gemeten op de ONTWIKKELsample."""
    primary = _development(read.primary_net_returns, lock_path)
    n_obs = int(len(primary))
    full_window_turnover = _check_turnover_provenance(read)
    is_reference = read.k == reference.k
    difference = None
    if not is_reference:
        difference = sharpe_difference_test(
            primary, _development(reference.primary_net_returns, lock_path),
            bars_per_year=bars_per_year, align=_ALIGN,
        )
    return DecisionFrequencyRun(
        k=int(read.k),
        is_reference=is_reference,
        track=track,
        layer=primary_layer,
        cost_axis_layer=cost_axis_layer,
        n_obs=n_obs,
        bars_per_year=float(bars_per_year),
        t_years=n_obs / float(bars_per_year),
        n_sovereign_halted=_halt_counter(read, "n_sovereign_halted"),
        n_sovereign_clipped=_halt_counter(read, "n_sovereign_clipped"),
        mean_turnover=float(_development(read.cost_axis_turnover, lock_path).mean()),
        mean_turnover_full_window=full_window_turnover,
        sharpe=sharpe_with_se(primary, bars_per_year=bars_per_year),
        sharpe_ci=block_bootstrap_ci(
            primary, statistic=_SHARPE, bars_per_year=bars_per_year),
        difference=difference,
        dsr=dsr_gate(
            primary.to_numpy(dtype="float64"),
            trial_count=trial_count, config=validation),
        breakeven=_breakeven(
            read, reference, lock_path=lock_path, bars_per_year=bars_per_year,
            layer=cost_axis_layer),
        inertness=read.inertness,
    )

def _ordered(reads: Sequence[LadderRead], *, track: str) -> tuple[LadderRead, ...]:
    """De lezingen op k gesorteerd, met de referentie aanwezig en elke k één keer."""
    ks = [int(read.k) for read in reads]
    require(
        bool(ks),
        f"Geen enkele ladderlezing voor track {track!r}. Een campagne zonder "
        f"lezing meet niets en zou een leeg artefact als resultaat presenteren.",
        DataContractError,
        track=track,
    )
    require(
        len(set(ks)) == len(ks),
        f"Dubbele k in de lezingen van {track!r}: {sorted(ks)}. Twee runs onder "
        f"dezelfde k betekent dat één van beide stilzwijgend wordt genegeerd, en "
        f"dan is niet te zeggen welke van de twee in het artefact staat.",
        DataContractError,
        track=track, k_grid=sorted(ks),
    )
    require(
        _IDENTITY_K in ks,
        f"Het k-grid {sorted(ks)} van {track!r} bevat {_IDENTITY_K} niet. k = "
        f"{_IDENTITY_K} is de identiteit van hold_decision en daarmee de ENIGE "
        f"referentie waartegen 'vasthouden' kan worden afgezet; zonder haar zou "
        f"elk verschil twee vastgehouden sporen vergelijken en meet de campagne "
        f"niet wat de hypothese stelt.",
        DataContractError,
        track=track, k_grid=sorted(ks),
    )
    return tuple(sorted(reads, key=lambda read: int(read.k)))


def _diagnostics(
    reads_by_track: Mapping[str, Sequence[LadderRead]], *, primary_track: str,
    lock_path: Path,
) -> tuple[DiagnosticCell, ...]:
    """De mechanische keten per (track, k). Geen rendement; zie `DiagnosticCell`."""
    cells: list[DiagnosticCell] = []
    for track in sorted(reads_by_track):
        ordered = _ordered(reads_by_track[track], track=track)
        reference = next(read for read in ordered if read.k == _IDENTITY_K)
        turnover_ref = float(
            _development(reference.cost_axis_turnover, lock_path).mean())
        for read in ordered:
            turnover = float(_development(read.cost_axis_turnover, lock_path).mean())
            cells.append(DiagnosticCell(
                track=track,
                is_primary_track=track == primary_track,
                k=int(read.k),
                n_bars=read.inertness.n_bars,
                n_unique_decision_rows=read.inertness.n_unique_decision_rows,
                n_bars_moved_by_hold=read.inertness.n_bars_moved_by_hold,
                mean_turnover=turnover,
                turnover_reduction_fraction_vs_reference=(
                    (turnover_ref - turnover) / turnover_ref
                    if turnover_ref != 0.0 else float("nan")
                ),
                n_sovereign_halted=_halt_counter(read, "n_sovereign_halted"),
                n_sovereign_clipped=_halt_counter(read, "n_sovereign_clipped"),
            ))
    return tuple(cells)

def measure_campaign(
    reads_by_track: Mapping[str, Sequence[LadderRead]],
    *,
    primary_track: str,
    primary_layer: str,
    cost_axis_layer: str,
    anchor: str,
    lock_path: Path,
    bars_per_year: float,
    cost_per_side_bps: float,
    trial_count: TrialCount,
    validation: ValidationConfig,
    stop_criteria: Sequence[StopCriterion],
) -> DecisionFrequencyCampaign:
    """De volledige H-10.1-campagne op de ONTWIKKELsample.

    Parameters
    ----------
    reads_by_track
        Per track de ladderlezingen van elke k, op het VOLLE bruikbare venster.
        De primaire track levert de vier run-records; alle tracks leveren de
        mechanische diagnostiek.
    primary_track, primary_layer
        De pre-geregistreerde cel. Er wordt niets uit een andere cel geselecteerd.
    cost_axis_layer
        De laag waarop de breakevenas bestaat (`L0_vectorized`); zie de
        moduledocstring voor waarom dat niet de primaire laag kan zijn.
    anchor
        De ankering waarmee `hold_decision` is aangeroepen, meegeschreven zodat
        het artefact vastlegt WELKE blokindeling is gemeten.
    lock_path
        Het bevroren `holdout_lock.json`. Uitsluitend `development_slice` wordt
        hierop aangeroepen: geen enkel pad in dit bestand raakt het poortsample.
    trial_count
        De bevroren `M` (AD-24: 25), uit `registry.ledger_reset`. `dsr_gate`
        weigert een live `M`.

    Raises
    ------
    DataContractError
        Bij een grid zonder k = 1, bij een dubbele k, bij een ontbrekende
        primaire track, en bij een omzetreeks die niet van de gemeten ladder komt.
    """
    require(
        primary_track in reads_by_track,
        f"De primaire track {primary_track!r} staat niet in de lezingen "
        f"({sorted(reads_by_track)}). De pre-registratie benoemt één cel vooraf; "
        f"een campagne die hem niet heeft gedraaid, zou de hypothese op een "
        f"andere cel moeten toetsen en dat is precies de vrijheid die de "
        f"pre-registratie wegneemt.",
        DataContractError,
        primary_track=primary_track,
        tracks=sorted(reads_by_track),
    )
    grids = {
        track: tuple(sorted(int(read.k) for read in reads))
        for track, reads in reads_by_track.items()
    }
    require(
        len(set(grids.values())) == 1,
        f"De tracks zijn op verschillende k-grids gedraaid: {grids}. De "
        f"diagnostiektabel legt de tracks per k naast elkaar; op verschillende "
        f"grids vergelijkt zij ongelijke kolommen en suggereert zij een verschil "
        f"dat uit het grid komt.",
        DataContractError,
        grids={track: list(grid) for track, grid in grids.items()},
    )
    ordered = _ordered(reads_by_track[primary_track], track=primary_track)
    reference = next(read for read in ordered if read.k == _IDENTITY_K)
    runs = tuple(
        _run_record(
            read, reference, track=primary_track, primary_layer=primary_layer,
            cost_axis_layer=cost_axis_layer, lock_path=lock_path,
            bars_per_year=bars_per_year, trial_count=trial_count,
            validation=validation,
        )
        for read in ordered
    )
    n_obs = runs[0].n_obs
    return DecisionFrequencyCampaign(
        runs=runs,
        diagnostics=_diagnostics(
            reads_by_track, primary_track=primary_track, lock_path=lock_path),
        track=primary_track,
        layer=primary_layer,
        cost_axis_layer=cost_axis_layer,
        anchor=anchor,
        n_obs=n_obs,
        bars_per_year=float(bars_per_year),
        t_years=n_obs / float(bars_per_year),
        cost_per_side_bps=float(cost_per_side_bps),
        stop_criteria=tuple(stop_criteria),
    )
