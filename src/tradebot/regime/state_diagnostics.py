"""De toestandsdiagnose -- wat DRAAGT de toestand, gemeten op twee assen.

WAAROM DIT BESTAAT
==================
`regime/state.py` kent een discrete toestand toe en ORDENT die op variantie.
Dat is een besluit dat alleen te verdedigen is als de semantiek van de toestand
wordt GEMETEN in plaats van aangenomen. Deze module is die meting.

Zij meet twee assen naast elkaar en houdt ze strikt uit elkaar:

  * de VOLATILITEITSAS -- de gerealiseerde volatiliteit per toestand. Dit is de
    as waarop de toestand per constructie zou moeten scheiden, want de
    toewijzing loopt over kwantielen van sigma-dak.
  * de RICHTINGSAS -- het conditionele gemiddelde rendement per toestand. Dit
    is de as waarop de nulmeting van revisie 1 GEEN scheiding vond zodra de
    afhankelijkheid in het paneel werd meegeteld.

DEZE MODULE SELECTEERT NIETS
============================
Er wordt geen toestandsindeling, geen kwantiel en geen parameter uit deze
uitkomst gekozen. Daarom kost hij nul trials. Zodra iemand `low_q` of `high_q`
verschuift OMDAT deze tabel er anders uitzag, is dat een trial en is de eerste
keuze er retroactief ook een.

DRIE T-LEZINGEN, EN WAAROM ER GEEN TWEE ZIJN
============================================
R-8: een getal zonder onzekerheid is geen bevinding. Op PANEELDATA is dat
scherper dan het klinkt. Zes namen met een gemiddelde correlatie van 0,74
leveren `N_eff = 1,271`; ~10.400 symbool-bars die als onafhankelijk worden
geteld, produceren dan een t die met een factor ~2,2 te groot is. De nulmeting
van revisie 1 laat precies dat zien:

    toestand   t_pooled   t_clustered/gedefleerd
    LAAG        1,72       0,79
    NORMAAL     0,57       0,26
    HOOG        1,16       0,53

Vandaar dat elke cel op de richtingsas DRIE lezingen draagt -- gepoold,
geclusterd op datum, en N_eff-gedefleerd -- en dat de gepoolde nooit alleen
staat. Alle drie komen uit `validation/inference.py::clustered_mean`; deze
module rekent geen enkele standaardfout zelf uit (R-3).

WAAROM DE VOLATILITEITSAS DEZELFDE MACHINERIE KRIJGT (een BESLUIT)
==================================================================
De gerealiseerde volatiliteit van een toestand is het GEMIDDELDE van de
gecentreerde kwadratische afwijking -- een conditioneel gemiddelde, en dus
even hard onderworpen aan R-8 als het rendement. Zij wordt hier daarom door
DEZELFDE `clustered_mean` gehaald, op het paneel van gecentreerde kwadraten,
en het interval is de op datum geclusterde standaardfout maal de normale
kwantiel bij het ENE betrouwbaarheidsniveau van deze fase
(`conf/validation/inference.yaml::ci_level`).

Er is bewust GEEN blokbootstrap op deze as. `block_bootstrap_ci` kent alleen
`"sharpe"` en `"mean"` en trekt blokken over EEN reeks; een paneel van
symbool-bars platslaan tot een reeks zou blokken over symboolgrenzen laten
lopen en de cross-sectionele afhankelijkheid als tijdsafhankelijkheid
behandelen. De clustering op datum is de correctie die §5 van het meetcontract
voor precies dat probleem voorschrijft.

WAT DEZE DIAGNOSE NIET VASTSTELT
================================
Geen richtingsclaim en geen alfaclaim. Een conditioneel gemiddelde met een
gedefleerde t onder 1 is geen bewijs van afwezigheid, maar het is al helemaal
geen bewijs van aanwezigheid; en een rendement per toestand is geen alfa, want
er is niets weggecontroleerd. De identificatie loopt bovendien op VARIANTIE:
"HOOG" betekent hier "hoge sigma-dak", nooit "hoog rendement".
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ..schemas.config import inference_config
from ..utils.failfast import DataContractError, require
from ..utils.time import assert_utc_index
from ..validation.inference import ClusteredMean, clustered_mean
from .state import StateAssignment, VolState

__all__ = ["StateDiagnostics", "diagnose"]

#: Het ENE betrouwbaarheidsniveau van deze fase. `validation/` heeft
#: ratchet-budget 0 en `regime/` hoort daar niet stilzwijgend van af te wijken:
#: een tweede niveau in omloop brengen zou betekenen dat twee intervallen in
#: hetzelfde rapport verschillend ruim staan zonder dat de lezer het ziet.
_CI_LEVEL = float(inference_config().ci_level)


def _z() -> float:
    """De normale kwantiel bij `_CI_LEVEL`, tweezijdig."""
    return float(stats.norm.ppf(0.5 + _CI_LEVEL / 2.0))


def _ann_vol(mean_squared_deviation: float, bars_per_year: float) -> float:
    """`sqrt(bars_per_year * E[(r - mu)^2])` -- de ENE vol-formule van deze module.

    Staat apart zodat de gepoolde cel en de per-symbool-cel aantoonbaar
    dezelfde schatter gebruiken. Twee routes naar hetzelfde veld zouden twee
    getallen onder een naam opleveren.
    """
    return math.sqrt(max(mean_squared_deviation, 0.0) * float(bars_per_year))


@dataclass(frozen=True)
class StateDiagnostics:
    """Wat de toestand draagt, per as, met de onzekerheid ernaast."""

    #: Per toestandsnaam: bezetting, episodes en de gerealiseerde volatiliteit
    #: met haar geclusterde interval.
    separation_vol: Mapping[str, dict[str, float]]
    #: Per toestandsnaam: het conditionele rendement met SE, interval en de
    #: drie t-lezingen van §5.
    separation_return: Mapping[str, dict[str, float]]
    #: `StateAssignment.occupancy()` op het UITGELIJNDE paneel.
    occupancy: pd.DataFrame
    #: `StateAssignment.episodes()` op datzelfde paneel.
    episodes: pd.DataFrame
    #: `StateAssignment.mean_duration()`. RULING P31: aangeroepen, nooit
    #: nagebouwd -- R-3 verbiedt een tweede implementatie ook wanneer zij
    #: hetzelfde getal geeft.
    mean_duration: pd.DataFrame
    #: `(3, 3)` rij-genormaliseerde overgangsmatrix, rijen = VolState-volgorde.
    transition_matrix: np.ndarray
    #: symbool -> toestandsnaam -> cel. De tekens spreken elkaar per symbool
    #: tegen en dat mag niet in de pool verdwijnen.
    per_symbol: Mapping[str, Mapping[str, dict[str, float]]]
    bars_per_year: float
    n_bars: int
    n_symbols: int
    source: str
    ordering: str
    params: Mapping[str, float]
    ci_level: float

    @property
    def t_years(self) -> float:
        """De lengte van het meetvenster in jaren. Elke gerapporteerde grootheid
        hoort met haar venster te reizen (meetcontract §10)."""
        return self.n_bars / float(self.bars_per_year)

    def to_dict(self) -> dict[str, Any]:
        return {
            "separation_vol": {k: dict(v) for k, v in self.separation_vol.items()},
            "separation_return": {
                k: dict(v) for k, v in self.separation_return.items()},
            "occupancy": {
                str(i): {str(c): float(v) for c, v in row.items()}
                for i, row in self.occupancy.to_dict("index").items()},
            "episodes": {
                str(i): {str(c): int(v) for c, v in row.items()}
                for i, row in self.episodes.to_dict("index").items()},
            "mean_duration_bars": {
                str(i): {str(c): float(v) for c, v in row.items()}
                for i, row in self.mean_duration.to_dict("index").items()},
            "transition_matrix": {
                "states": [state.name for state in VolState],
                "rows": [[float(x) for x in row] for row in self.transition_matrix],
            },
            "per_symbol": {
                symbol: {state: dict(cell) for state, cell in states.items()}
                for symbol, states in self.per_symbol.items()},
            "window": {
                "n_bars": int(self.n_bars),
                "bars_per_year": float(self.bars_per_year),
                "t_years": float(self.t_years),
                "n_symbols": int(self.n_symbols),
            },
            "assignment": {
                "source": self.source,
                "ordering": self.ordering,
                "params": {k: float(v) for k, v in self.params.items()},
            },
            "ci_level": float(self.ci_level),
        }


def _transition_matrix(states: pd.DataFrame) -> np.ndarray:
    """Rij-genormaliseerde overgangskansen, gepoold over symbolen.

    Overgangen worden PER KOLOM geteld (bar `t` tegen bar `t+1` binnen hetzelfde
    symbool); er loopt dus geen overgang van het einde van het ene symbool naar
    het begin van het volgende.
    """
    values = states.to_numpy(dtype="float64")
    current, following = values[:-1], values[1:]
    valid = np.isfinite(current) & np.isfinite(following)
    counts = np.zeros((len(VolState), len(VolState)), dtype="float64")
    for origin in VolState:
        at_origin = valid & (current == float(int(origin)))
        for target in VolState:
            counts[int(origin), int(target)] = float(
                np.sum(at_origin & (following == float(int(target)))))
    totals = counts.sum(axis=1)
    missing = [state.name for state in VolState if totals[int(state)] <= 0.0]
    require(
        not missing,
        f"Toestand(en) {missing} komen op dit venster nooit voor met een "
        f"opvolgende bar, dus hun rij in de overgangsmatrix bestaat niet. Een "
        f"rij nullen zou lezen als 'deze toestand gaat nergens heen'; dat is "
        f"een bewering en de meting is 'geen waarneming'.",
        DataContractError,
        states_without_transitions=missing, n_bars=int(len(states)),
    )
    return counts / totals[:, None]


def _return_cell(
    estimate: ClusteredMean, *, bars_per_year: float, n_episodes: int
) -> dict[str, float]:
    """De richtingscel: het conditionele gemiddelde met alle drie de lezingen.

    Het rendement wordt REKENKUNDIG geannualiseerd (`mu * bars_per_year`),
    net als de nulmeting van §4.5 waartegen deze tabel wordt gelegd, en de
    standaardfout schaalt mee zodat `ann_return / se` exact de geclusterde t
    blijft.
    """
    scale = float(bars_per_year)
    half_width = _z() * estimate.se * scale
    ann_return = estimate.mean * scale
    return {
        "n_bars": int(estimate.n_obs),
        "n_episodes": int(n_episodes),
        "n_clusters": int(estimate.n_clusters),
        "n_units": int(estimate.n_units),
        "mean_per_bar_bp": estimate.mean * 1e4,
        "ann_return": ann_return,
        "se": estimate.se * scale,
        "se_pooled": estimate.se_pooled * scale,
        "ci_low": ann_return - half_width,
        "ci_high": ann_return + half_width,
        "t_stat_pooled": estimate.t_stat_pooled,
        "t_stat_clustered": estimate.t_stat,
        "t_stat_neff_deflated": estimate.t_stat_neff_deflated,
        "rho_bar": estimate.rho_bar,
        "n_effective": estimate.n_effective,
        "neff_factor": estimate.neff_factor,
    }


def _vol_cell(
    estimate: ClusteredMean,
    *,
    bars_per_year: float,
    occupancy: float,
    n_episodes: int,
    mean_duration_bars: float,
) -> dict[str, float]:
    """De volatiliteitscel: de as waarop de toestand aantoonbaar scheidt.

    `occupancy` is hier het GEPOOLDE aandeel symbool-bars van deze toestand in
    alle toegewezen symbool-bars; de per-symbool-bezetting staat in het veld
    `StateDiagnostics.occupancy`. `mean_duration_bars` is het gemiddelde OVER
    SYMBOLEN van hun eigen gemiddelde episodelengte -- dus een gemiddelde van
    gemiddelden, en niet `n_bars / n_episodes`. Die tweede zou een symbool met
    veel bars zwaarder laten wegen dan een symbool met lange episodes, en het
    is de EPISODE die de effectieve steekproefomvang draagt (Q7).
    """
    half_width = _z() * estimate.se
    return {
        "n_bars": int(estimate.n_obs),
        "occupancy": float(occupancy),
        "n_episodes": int(n_episodes),
        "mean_duration_bars": float(mean_duration_bars),
        "ann_vol": _ann_vol(estimate.mean, bars_per_year),
        "ann_vol_ci_low": _ann_vol(estimate.mean - half_width, bars_per_year),
        "ann_vol_ci_high": _ann_vol(estimate.mean + half_width, bars_per_year),
        "mean_squared_deviation": estimate.mean,
        "se_squared_deviation": estimate.se,
    }


def _per_symbol(
    states: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    bars_per_year: float,
    occupancy: pd.DataFrame,
    episodes: pd.DataFrame,
    mean_duration: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """Dezelfde twee assen per symbool, met een op datum geclusterde t.

    Zonder deze tabel leest een gepoolde richtingscel als een uitspraak over
    het universum, terwijl de nulmeting laat zien dat de TEKENS elkaar per
    symbool tegenspreken (DOTUSDT +131,6 bp/dag in HOOG tegen SOLUSDT
    -45,1 bp). Dat is geen detail maar de bevinding zelf.

    Elke cel loopt door DEZELFDE `_return_cell`/`_vol_cell` als de gepoolde
    tabel, dus ook hier staat er geen enkel gemiddelde zonder standaardfout en
    interval (R-8). Met een kolom is `N = 1`, dus `neff_factor` is per
    constructie 1,0 en de gedefleerde t valt samen met de gepoolde: er is bij
    een enkel symbool geen cross-sectionele breedte om voor te defleren, en dat
    eerlijk zeggen is beter dan een deflatie suggereren die er niet is.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for symbol in states.columns:
        cells: dict[str, dict[str, float]] = {}
        for state in VolState:
            in_state = states[[symbol]] == float(int(state))
            direction = clustered_mean(
                returns[[symbol]].where(in_state), bars_per_year=bars_per_year)
            dispersion = clustered_mean(
                (returns[[symbol]] - direction.mean).where(in_state) ** 2,
                bars_per_year=bars_per_year)
            n_episodes = int(episodes.at[symbol, state.name])
            cells[state.name] = {
                **_vol_cell(
                    dispersion, bars_per_year=bars_per_year,
                    occupancy=float(occupancy.at[symbol, state.name]),
                    n_episodes=n_episodes,
                    mean_duration_bars=float(
                        mean_duration.at[symbol, state.name])),
                **_return_cell(
                    direction, bars_per_year=bars_per_year,
                    n_episodes=n_episodes),
            }
        out[str(symbol)] = cells
    return out


def _aligned(
    assignment: StateAssignment, forward_returns: pd.DataFrame
) -> tuple[StateAssignment, pd.DataFrame, Sequence[str]]:
    """Snijd toestanden en rendementen tot precies de cellen die beide hebben.

    Een symbool-bar met een toestand maar zonder rendement telt in GEEN van
    beide assen mee, en zou de bezetting dus overdrijven als hij bleef staan.
    De uitgelijnde toewijzing wordt teruggegeven als een `StateAssignment`,
    zodat `occupancy`/`episodes`/`mean_duration` van STAP 5 komen en niet van
    hier (R-3, ruling P31).
    """
    columns = [c for c in assignment.states.columns if c in forward_returns.columns]
    require(
        len(columns) > 0,
        "Toestanden en rendementen delen geen enkel symbool; er is dan niets "
        "te conditioneren. Controleer of beide panelen dezelfde kolomnamen "
        "dragen.",
        DataContractError,
        state_columns=list(map(str, assignment.states.columns)),
        return_columns=list(map(str, forward_returns.columns)),
    )
    index = assignment.states.index.intersection(forward_returns.index)
    require(
        len(index) > 0,
        "Toestanden en rendementen overlappen op geen enkele bar; de diagnose "
        "zou over een leeg venster rapporteren.",
        DataContractError,
        n_state_bars=int(len(assignment.states)),
        n_return_bars=int(len(forward_returns)),
    )
    returns = forward_returns.loc[index, columns].astype("float64")
    states = assignment.states.loc[index, columns].where(returns.notna())
    return replace(assignment, states=states), returns, columns


def diagnose(
    assignment: StateAssignment,
    forward_returns: pd.DataFrame,
    *,
    bars_per_year: float,
) -> StateDiagnostics:
    """Meet wat de toestand draagt: volatiliteit, richting, en persistentie.

    Parameters
    ----------
    assignment
        De toewijzing uit `regime/state.py`. Haar `lag` staat in
        `assignment.params` en reist mee in het artefact: een diagnose op een
        ONGELAGDE toewijzing meet iets anders dan een diagnose op een gelagde,
        en het verschil hoort zichtbaar te zijn zonder de code te lezen.
    forward_returns
        Het rendement dat OP de toestand volgt, per symbool. De toestand van
        bar `t` is een functie van sigma-dak op `t-1`, dus het rendement van
        bar `t` ligt per constructie in de toekomst van de toestand.
    bars_per_year
        VERPLICHT en zonder default. Er is precies een annualisatie
        (meetcontract §1) en die staat in `conf/backtest/default.yaml`; een
        default hier zou er een tweede in omloop brengen.
    """
    assert_utc_index(forward_returns, name="diagnose(forward_returns)")
    require(
        float(bars_per_year) > 0.0,
        "bars_per_year moet positief zijn; annualiseren met nul bestaat niet.",
        DataContractError,
        bars_per_year=float(bars_per_year),
    )
    aligned, returns, columns = _aligned(assignment, forward_returns)
    states = aligned.states
    occupancy, episodes = aligned.occupancy(), aligned.episodes()
    mean_duration = aligned.mean_duration()

    assigned = int(np.isfinite(states.to_numpy(dtype="float64")).sum())
    require(
        assigned > 0,
        "Geen enkele bar draagt een toestand op dit venster -- de opstartfase "
        "van het expanding kwantiel beslaat het hele venster. Er valt niets te "
        "diagnosticeren; verklein `min_periods` of vergroot het venster.",
        DataContractError,
        n_bars=int(len(states)), n_symbols=len(columns),
    )

    separation_vol: dict[str, dict[str, float]] = {}
    separation_return: dict[str, dict[str, float]] = {}
    for state in VolState:
        in_state = states == float(int(state))
        conditional = returns.where(in_state)
        direction = clustered_mean(conditional, bars_per_year=bars_per_year)
        deviation = (returns - direction.mean).where(in_state) ** 2
        dispersion = clustered_mean(deviation, bars_per_year=bars_per_year)
        n_episodes = int(episodes[state.name].sum())
        separation_return[state.name] = _return_cell(
            direction, bars_per_year=bars_per_year, n_episodes=n_episodes)
        separation_vol[state.name] = _vol_cell(
            dispersion,
            bars_per_year=bars_per_year,
            occupancy=direction.n_obs / assigned,
            n_episodes=n_episodes,
            mean_duration_bars=float(mean_duration[state.name].mean()),
        )

    return StateDiagnostics(
        separation_vol=separation_vol,
        separation_return=separation_return,
        occupancy=occupancy,
        episodes=episodes,
        mean_duration=mean_duration,
        transition_matrix=_transition_matrix(states),
        per_symbol=_per_symbol(
            states, returns, bars_per_year=bars_per_year, occupancy=occupancy,
            episodes=episodes, mean_duration=mean_duration),
        bars_per_year=float(bars_per_year),
        n_bars=int(len(states)),
        n_symbols=len(columns),
        source=assignment.source,
        ordering=assignment.ordering,
        params=dict(assignment.params),
        ci_level=_CI_LEVEL,
    )
