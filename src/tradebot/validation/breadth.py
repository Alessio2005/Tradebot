# src/tradebot/validation/breadth.py
"""Breedte — de noemer van de fundamentele wet. Fase 11, breedte en tijdschaal.

TWEE GROOTHEDEN, TWEE NAMEN (AD-30)
===================================
In deze repository heetten twee verschillende grootheden allebei "N_eff":

* **het ontwerpeffect** van een gepoold, gelijkgewogen gemiddelde (Kish):
  `N / (1 + (N-1) rho_bar)`. Het zegt hoeveel onafhankelijke waarnemingen het
  gemiddelde van N gecorreleerde reeksen waard is. Het woont in
  `validation/inference.py::effective_breadth` en hoort daar: `clustered_mean`
  defleert er een gepoolde t mee.
* **het aantal onafhankelijke weddenschappen** uit de fundamentele wet
  (`SR ≈ IC · sqrt(BR)`): de participatieratio van de eigenwaarden van de
  correlatiematrix, `(sum lambda)^2 / sum lambda^2`. Dat getal is per
  constructie begrensd door de rang van de matrix.

Het verschil is geen afronding. Op het dollar-neutrale residu van de zes
gecertificeerde namen (`W_DEV`) geeft het ontwerpeffect **123,58** en de
participatieratio **4,353**: het residu sommeert over de namen tot nul, dus de
noemer van de rho-formule gaat naar nul, terwijl de rang vijf is
(`Prompts-fases/fase_11_breedte_en_tijdschaal.md` §2.1).

WAAROM HIER GEEN IMPLEMENTATIE STAAT
====================================
De participatieratio bestaat al: `portfolio/covariance.py::effective_n_assets`.
Dat bestand blijft door DI-10 bewust ongewijzigd, omdat
`portfolio/legacy_sizing.py` ervan afhangt en de Phase 3-baseline herrekenbaar
moet blijven. Een tweede berekening hier zou een tweede implementatie van
dezelfde grootheid zijn (R-3). Deze module geeft haar dus alleen de naam
waaronder de validatielaag haar gebruikt, en bewaakt de invoer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt

from ..portfolio.covariance import effective_n_assets
from ..schemas.config import StrictModel, inference_config, validate_mapping
from ..utils.failfast import DataContractError, require
from .inference import calibrate_block_length, circular_block_indices, effective_breadth

__all__ = [
    "BreadthConfig",
    "BreadthRow",
    "breadth_config",
    "breadth_measurement",
    "construct",
    "independent_bets",
]

Construction = Literal["directional", "dollar_neutral", "beta_hedged_ew"]


# --------------------------------------------------------------------------- #
# Configuratie — conf/research/breadth.yaml, gezet vóór de eerste meting.
# Het schema staat hier en niet in schemas/config.py: dat bestand staat op zijn
# LOC-cap, en deze keuzes horen bij één fase.
# --------------------------------------------------------------------------- #
class SignalClockConfig(StrictModel):
    """De schatter van de signaalklok (stap 3)."""

    iact_window_c: PositiveFloat
    iact_max_lag: PositiveInt


class WallConfig(StrictModel):
    """De IC-muur en de simulatie die haar controleert (stap 5)."""

    horizons_bars: tuple[PositiveInt, ...]
    t_hurdle: PositiveFloat
    simulation_ic_grid: tuple[Annotated[float, Field(gt=0.0, lt=1.0)], ...]
    simulation_n_obs: PositiveInt
    simulation_seed: int
    formula_tolerance: PositiveFloat


class FeasibilityConfig(StrictModel):
    """De haalbaarheidspoort van H-11.2 (stap 7)."""

    primary_track: str
    reference_k: PositiveInt
    null_paths: PositiveInt
    null_seed: int


class BreadthConfig(StrictModel):
    """Alle keuzes van fase 11, breedte en tijdschaal."""

    constructions: tuple[Construction, ...] = Field(min_length=1)
    signal_clock: SignalClockConfig
    wall: WallConfig
    feasibility: FeasibilityConfig


def breadth_config(path: str | Path) -> BreadthConfig:
    """Laad en valideer `conf/research/breadth.yaml`.

    Het bestand draagt één top-level sleutel `breadth:`. `load_config` pelt zo'n
    wrapper alleen af voor domeinen in `schemas/config.py::DOMAIN_SCHEMAS`, en dat
    bestand staat op zijn LOC-cap; daarom gebeurt het afpellen hier, met dezelfde
    validatie (`validate_mapping`) en dezelfde foutklasse.
    """
    import yaml

    source = Path(path)
    require(source.is_file(), f"Configuratiebestand bestaat niet: {source}",
            DataContractError, path=str(source))
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    require(isinstance(raw, dict) and set(raw) == {"breadth"},
            f"{source} hoort precies één top-level sleutel `breadth:` te dragen.",
            DataContractError, path=str(source))
    return validate_mapping(BreadthConfig, raw["breadth"], source=str(source))


def independent_bets(correlation_matrix: np.ndarray) -> float:
    """Het aantal onafhankelijke weddenschappen in een correlatiematrix.

    De participatieratio van de eigenwaarden, begrensd door de rang. De
    berekening staat in `portfolio/covariance.py::effective_n_assets` en blijft
    daar (DI-10, AD-30); deze functie weigert alleen invoer die geen
    correlatiematrix is, want `effective_n_assets` symmetriseert stilzwijgend en
    zou een scheve of onvolledige matrix een getal geven.
    """
    c = np.asarray(correlation_matrix, dtype=np.float64)
    require(
        c.ndim == 2 and c.shape[0] == c.shape[1] and c.shape[0] >= 2,
        "Onafhankelijke weddenschappen vragen een vierkante correlatiematrix met "
        "minstens twee namen; één naam is geen cross-sectie.",
        DataContractError,
        shape=tuple(c.shape),
    )
    require(
        bool(np.isfinite(c).all()),
        "De correlatiematrix bevat een niet-eindige waarde. Een ontbrekende "
        "correlatie is geen nul: zij wordt hier niet ingevuld.",
        DataContractError,
    )
    require(
        bool(np.allclose(c, c.T)),
        "De correlatiematrix is niet symmetrisch. Symmetriseren zou een getal "
        "opleveren voor een matrix die geen correlatiematrix is.",
        DataContractError,
    )
    return float(effective_n_assets(c))


# --------------------------------------------------------------------------- #
# Stap 2 — de constructies en de meting per constructie.
# --------------------------------------------------------------------------- #
def construct(returns: pd.DataFrame, construction: str) -> pd.DataFrame:
    """Het rendementspaneel zoals een constructie het ziet.

    * `directional`: de ruwe rendementen — het boek draagt de marktfactor.
    * `dollar_neutral`: het residu na het gelijkgewogen mandje; per bar
      sommeert het tot nul en het verliest dus precies één rang.
    * `beta_hedged_ew`: het residu na de bèta tegen het gelijkgewogen mandje.
      De bèta is op het VOLLE aangeleverde venster geschat: dit is diagnostiek
      van tweede momenten en geen besluitgrootheid. Een handelbare hedge schat
      causaal en is van de zusterfase (haar arm 2).
    """
    require(
        construction in {"directional", "dollar_neutral", "beta_hedged_ew"},
        f"Onbekende constructie '{construction}'. Een constructie die niet in "
        f"conf/research/breadth.yaml staat, is niet vooraf vastgelegd.",
        DataContractError,
        construction=construction,
    )
    if construction == "directional":
        return returns
    basket = returns.mean(axis=1)
    if construction == "dollar_neutral":
        return returns.sub(basket, axis=0)
    beta = returns.apply(lambda column: np.cov(column, basket)[0, 1] / np.var(basket, ddof=1))
    return returns - np.outer(basket.to_numpy(), beta.to_numpy())


@dataclass(frozen=True)
class BreadthRow:
    """Eén breedtemeting: één paneel, één constructie, beide grootheden."""

    construction: str
    n_obs: int
    n_names: int
    rank: int
    rho_bar: float
    #: Het aantal onafhankelijke weddenschappen (AD-30). Dit is de breedte.
    independent_bets: float
    independent_bets_ci: tuple[float, float]
    #: Kish' ontwerpeffect. GEEN breedte (DI-35); staat erbij omdat de
    #: repository het tot fase 11 zo rapporteerde. `None` wanneer de noemer
    #: 1 + (N-1) rho_bar niet positief is.
    design_effect: float | None
    design_effect_ci: tuple[float, float] | None
    design_effect_undefined_replicates: int
    block_length: int
    n_boot: int
    seed: int
    ci_level: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cross_product_series(values: np.ndarray) -> np.ndarray:
    """De reeks waarvan het gemiddelde rho_bar is: per bar het gemiddelde van
    z_i * z_j over de paren i < j. Haar afhankelijkheid in de tijd is die van
    de correlatieschatting, en dus de reeks waarop de bloklengte hoort."""
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    upper = np.triu_indices(values.shape[1], k=1)
    return (z[:, :, None] * z[:, None, :])[:, upper[0], upper[1]].mean(axis=1)


def _design_effect_or_none(corr: np.ndarray) -> float | None:
    try:
        return float(effective_breadth(corr))
    except DataContractError:
        return None


def breadth_measurement(
    panel: pd.DataFrame,
    *,
    construction: str,
    n_boot: int,
    seed: int,
    ci_level: float,
    block_length: int | None,
) -> BreadthRow:
    """Beide grootheden op één paneel, met een circulaire blokbootstrap over datums.

    De bootstrap-indices komen uit `validation/inference.py` (R-3). Zonder
    opgegeven `block_length` wordt hij gekalibreerd op de kruisproductreeks,
    met dezelfde kalibratie als de rest van de inferentiekern. Een replicatie
    waarin het ontwerpeffect niet gedefinieerd is, wordt GETELD en niet
    ingevuld.
    """
    values = panel.to_numpy(dtype=np.float64)
    require(
        values.ndim == 2 and values.shape[1] >= 2 and bool(np.isfinite(values).all()),
        "Een breedtemeting vraagt een volledig paneel met minstens twee namen. "
        "Een ontbrekende waarde is geen nul en wordt hier niet ingevuld; snij het "
        "venster eerst op de bars waarop elke naam bestaat.",
        DataContractError,
        shape=tuple(values.shape),
    )
    n_obs = int(values.shape[0])
    corr = np.corrcoef(values, rowvar=False)
    length = int(block_length) if block_length is not None else calibrate_block_length(
        _cross_product_series(values))
    batch = int(inference_config().bootstrap_batch_size)
    rng = np.random.default_rng(int(seed))
    bets: list[float] = []
    designs: list[float] = []
    undefined = 0
    remaining = int(n_boot)
    while remaining > 0:
        size = min(batch, remaining)
        for idx in circular_block_indices(n_obs, length, size, rng):
            replicate = np.corrcoef(values[idx], rowvar=False)
            bets.append(independent_bets(replicate))
            value = _design_effect_or_none(replicate)
            if value is None:
                undefined += 1
            else:
                designs.append(value)
        remaining -= size
    tail = (1.0 - float(ci_level)) / 2.0
    point_design = _design_effect_or_none(corr)
    design_ci = (
        (float(np.quantile(designs, tail)), float(np.quantile(designs, 1.0 - tail)))
        if point_design is not None and undefined == 0 else None
    )
    return BreadthRow(
        construction=construction,
        n_obs=n_obs,
        n_names=int(values.shape[1]),
        rank=int(np.linalg.matrix_rank(values)),
        rho_bar=float(corr[~np.eye(corr.shape[0], dtype=bool)].mean()),
        independent_bets=independent_bets(corr),
        independent_bets_ci=(float(np.quantile(bets, tail)),
                             float(np.quantile(bets, 1.0 - tail))),
        design_effect=point_design,
        design_effect_ci=design_ci,
        design_effect_undefined_replicates=int(undefined),
        block_length=int(length),
        n_boot=int(n_boot),
        seed=int(seed),
        ci_level=float(ci_level),
    )
