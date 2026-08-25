# src/tradebot/portfolio/constraints.py
"""Executie-constraints van de portefeuillelaag — L8, GEEN risicoautoriteit.

PHASE 5 HERBEDRADING (fase-opdracht §4.2, exit-criterium 11).
--------------------------------------------------------------
Dit bestand droeg tot Phase 5 een tweede risicoregime:

    max_weight:   0.40   ->  duplicaat van `risk.max_concentration`  (0.40)
    max_leverage: 1.00   ->  duplicaat van `risk.gross_cap`          (1.50)

`reports/phase5_sovereign_wiring_audit.md` §3.2 (D1, D2) classificeert beide als
*legacy duplicate*. De twee `max_leverage`-waarden LIEPEN bovendien uiteen, en
dat is precies het gevaar: welke van de twee bond, hing af van welk pad je
draaide.

De fase-opdracht laat drie uitwegen en geen vierde: verplaats naar sovereign,
maak een pure adapter, of classificeer expliciet als niet-risk
executie-mechanica. Hier is per constraint gekozen:

| Constraint      | Klasse           | Uitkomst                                  |
|-----------------|------------------|-------------------------------------------|
| `max_weight`    | legacy duplicate | ADAPTER - leest `risk.max_concentration`  |
| `max_leverage`  | legacy duplicate | ADAPTER - leest `risk.gross_cap`          |
| `min_weight`    | execution-only   | BLIJFT - dust-drempel, kostenbeslissing   |
| `max_turnover`  | execution-only   | BLIJFT - kostenbeslissing                 |
| toepassingsorde | bug              | EXPLICIET - was impliciet in de codevolgorde |

WAAROM `min_weight` EN `max_turnover` LEGITIEM LOKAAL BLIJVEN
-------------------------------------------------------------
Beide VERKLEINEN exposure en zijn dus geen bypass van de soevereine laag. En
beide beantwoorden een andere vraag dan risico: een positie van 0,4 % kost meer
aan fees dan zij bijdraagt, en een turnovercap ruilt tracking error tegen
transactiekosten. Dat zijn L8/L9-kostenbeslissingen. Ze staan hier expliciet
geclassificeerd in plaats van impliciet meegelift, want dat onderscheid is wat
exit-criterium 13 vraagt.

WAT DEZE MODULE NIET MEER DOET
------------------------------
Zelf een drempel kiezen. `PortfolioConstraints.from_risk_config()` is de enige
manier om de risico-gerelateerde velden te vullen, en `apply_constraints`
weigert een constraint-set die niet uit een soevereine policy komt zodra
`enforce_provenance=True` - wat de productiepaden zetten.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 13, 14, 19 (L8), 24.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..risk.limits import effective_relative_cap, water_filling_limit
from ..utils.failfast import ConfigContractError, require

if TYPE_CHECKING:
    from ..schemas.config import RiskConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CONSTRAINT_ORDER",
    "PortfolioConstraints",
    "apply_constraints",
    "compute_turnover",
]

#: De toepassingsvolgorde, EXPLICIET. Zij stond voorheen alleen in de
#: regelvolgorde van `apply_constraints`, en `risk/engine.py` verbiedt dat
#: patroon voor de soevereine laag om een reden die hier net zo geldt: een
#: volgorde die je alleen uit de code kunt aflezen, verandert bij een refactor
#: zonder dat iemand een beslissing neemt. Zie `conf/risk/constraint_order`.
CONSTRAINT_ORDER: tuple[str, ...] = (
    "concentration",   # sovereign: risk.max_concentration
    "dust",            # execution: min_weight
    "renormalise",
    "leverage",        # sovereign: risk.gross_cap
    "turnover",        # execution: max_turnover
)


@dataclass(frozen=True)
class PortfolioConstraints:
    """Constraint-set voor portefeuilleconstructie.

    `frozen=True`: een constraint-set die tijdens een run wordt bijgesteld,
    maakt het resultaat onreproduceerbaar - dezelfde reden waarom
    `schemas/config.py::StrictModel` frozen is.

    Attributes
    ----------
    max_weight :
        SOVEREIGN. Maximale fractie per asset. Adapter voor
        `risk.max_concentration`; zet hem niet met de hand.
    max_leverage :
        SOVEREIGN. Maximale som van absolute gewichten. Adapter voor
        `risk.gross_cap`.
    min_weight :
        EXECUTION-ONLY. Dust-drempel: posities hieronder kosten meer aan fees
        dan ze bijdragen. Geen risicolimiet.
    max_turnover :
        EXECUTION-ONLY. Eenzijdige turnovercap. Een kostenbeslissing.
    risk_config_hash :
        De policy waaruit de sovereign velden komen. Leeg = handmatig gezet, en
        dan weigert `apply_constraints` met `enforce_provenance=True`.
    """

    max_weight: float
    max_leverage: float
    min_weight: float = 0.0
    max_turnover: float = 1.0
    risk_config_hash: str = ""
    #: Welke velden uit de soevereine policy komen. Documentatie EN testbaar.
    sovereign_fields: tuple[str, ...] = field(
        default=("max_weight", "max_leverage"), repr=False)

    def __post_init__(self) -> None:
        for name in ("max_weight", "max_leverage"):
            value = float(getattr(self, name))
            require(
                np.isfinite(value) and value > 0.0,
                f"{name} moet eindig en strikt positief zijn.",
                ConfigContractError, key=f"portfolio.{name}", value=value,
            )
        require(
            0.0 <= float(self.min_weight) < float(self.max_weight),
            "min_weight moet in [0, max_weight) liggen.",
            ConfigContractError,
            min_weight=self.min_weight, max_weight=self.max_weight,
        )
        require(
            0.0 < float(self.max_turnover) <= 1.0,
            "max_turnover ligt in (0, 1].",
            ConfigContractError, max_turnover=self.max_turnover,
        )

    @classmethod
    def from_risk_config(
        cls, risk: RiskConfig, *, min_weight: float = 0.0,
        max_turnover: float = 1.0,
    ) -> PortfolioConstraints:
        """De ENIGE weg naar een set met soevereine herkomst.

        De twee risico-gerelateerde velden worden hier gelezen en nergens
        gekozen. Wijzigt `conf/risk/default.yaml`, dan wijzigt deze set mee, en
        de `config_hash` maakt zichtbaar dat dat is gebeurd.
        """
        from ..risk.engine import risk_config_hash

        return cls(
            max_weight=float(risk.max_concentration),
            max_leverage=float(risk.gross_cap),
            min_weight=float(min_weight),
            max_turnover=float(max_turnover),
            risk_config_hash=risk_config_hash(risk),
        )

    @property
    def has_sovereign_provenance(self) -> bool:
        return bool(self.risk_config_hash)


def _concentration_limit(w: pd.Series, max_weight: float) -> float:
    """Het exacte vaste punt van de concentratiecap voor dit boek.

    De cap wordt eerst op `max(cap, 1/n_actief)` gezet - dezelfde vloer die
    `risk.limits.effective_relative_cap` toepast, en om dezelfde reden: een cap
    onder `1/n` is wiskundig onhaalbaar (sommeren over de posities geeft
    `1 <= n * cap`), en het aantal actieve posities is een eigenschap van het
    BOEK op deze bar, niet van de configuratie. Een bar waarop de alpha
    toevallig twee namen aanwijst, mag het platform niet stilleggen.

    Zonder die vloer weigert `water_filling_limit` terecht te rekenen: bij twee
    posities die tot 1 sommeren is elke `cap < 0.5` inconsistent.
    """
    active = int((w.abs() > 1e-12).sum())
    cap = effective_relative_cap(max(active, 1), float(max_weight))
    return water_filling_limit(
        [float(v) for v in w.abs().to_numpy()], cap, key="portfolio.max_weight")


def apply_constraints(
    weights: pd.Series,
    constraints: PortfolioConstraints,
    current_weights: pd.Series | None = None,
    *,
    enforce_provenance: bool = False,
) -> pd.Series:
    """Pas de constraint-set toe in de EXPLICIETE volgorde van `CONSTRAINT_ORDER`.

    Parameters
    ----------
    weights : ruwe doelgewichten (long-only, sommeren tot ~1).
    constraints : de set. Met `enforce_provenance=True` moet hij uit
        `from_risk_config()` komen.
    current_weights : huidige gewichten, voor de turnovercap. `None` slaat die
        stap over.
    enforce_provenance : productiepaden zetten dit op `True`. Dan is een
        handmatig samengestelde constraint-set een `ConfigContractError` in
        plaats van een stille tweede risicopolicy.

    Notes
    -----
    De concentratiecap wordt GESLOTEN opgelost met
    `risk.limits.water_filling_limit`, niet iteratief.

    De vorige implementatie deed twintig rondes clip-en-hernormaliseer. Dat
    convergeert LINEAIR met factor `k * cap`, en een test die de cap na afloop
    controleerde liet zien dat er 2,9e-8 boven de limiet overbleef: het
    iteratieplafond werd bereikt vóór convergentie, en de laatste
    hernormalisatie duwde het geclipte gewicht opnieuw omhoog. Klein, maar het
    is een SCHENDING van een soevereine limiet, en `risk/engine.py::_verify`
    crasht op precies dat soort residu.

    Diezelfde vergelijking wordt in de soevereine laag al exact opgelost. Die
    solver hier hergebruiken is bovendien architecturaal het punt van deze
    fase: de concentratielimiet hoort op één plek te worden berekend.
    """
    require(
        isinstance(constraints, PortfolioConstraints),
        "apply_constraints vereist een PortfolioConstraints.",
        ConfigContractError, got=type(constraints).__name__,
    )
    require(
        (not enforce_provenance) or constraints.has_sovereign_provenance,
        "Deze constraint-set is niet uit een soevereine risicopolicy gebouwd. "
        "Gebruik PortfolioConstraints.from_risk_config(); L8 kiest geen "
        "risicodrempels (fase-opdracht §4.2, exit-criterium 11).",
        ConfigContractError,
        risk_config_hash=constraints.risk_config_hash,
    )

    w = weights.copy().fillna(0.0).clip(lower=0.0)
    assets = w.index

    # 1. concentration — gesloten opgelost (zie Notes).
    if float(w.sum()) > 1e-9:
        w = w.clip(upper=_concentration_limit(w, constraints.max_weight))

    # 2. dust — execution-only drempel.
    w[w < constraints.min_weight] = 0.0

    # 3. renormalise; daarna de cap opnieuw sluiten, want hernormaliseren na
    #    het wegvallen van dust-posities kan de verhoudingen verschuiven.
    total = float(w.sum())
    if total < 1e-9:
        w = pd.Series(1.0 / len(assets), index=assets)
    else:
        w = w / total
        limit = _concentration_limit(w, constraints.max_weight)
        if float(w.max()) > limit + 1e-12:
            w = w.clip(upper=limit)
            total = float(w.sum())
            if total > 1e-9:
                w = w / total

    # 4. leverage (sovereign: risk.gross_cap)
    total = w.sum()
    if total > constraints.max_leverage:
        w = w * constraints.max_leverage / total

    # 5. turnover (execution-only: een kostenbeslissing, geen risicolimiet)
    if current_weights is not None and constraints.max_turnover < 1.0:
        cur = current_weights.reindex(assets).fillna(0.0)
        turnover = float((w - cur).abs().sum()) / 2.0
        if turnover > constraints.max_turnover:
            alpha = constraints.max_turnover / (turnover + 1e-9)
            alpha = float(np.clip(alpha, 0.0, 1.0))
            w = alpha * w + (1 - alpha) * cur
            w = w.clip(lower=0.0)
            s = w.sum()
            if s > 1e-9:
                w /= s

    return w.rename("constrained_weight")


def compute_turnover(
    new_weights: pd.Series,
    old_weights: pd.Series,
) -> float:
    """Eenzijdige turnover als fractie van de portefeuillewaarde.

    `turnover = 0.5 * sum(|w_new - w_old|)`. Een METING, geen limiet.
    """
    combined_idx = new_weights.index.union(old_weights.index)
    new = new_weights.reindex(combined_idx).fillna(0.0)
    old = old_weights.reindex(combined_idx).fillna(0.0)
    return float((new - old).abs().sum() / 2.0)
