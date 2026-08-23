# src/tradebot/risk/contract.py
"""L7 risicocontract — de types waarin elke risicobeslissing wordt uitgedrukt.

Phase 4, stap 2. De prozaversie staat in `docs/RISK_CONTRACT.md`; dit bestand
is de afdwingbare helft ervan.

    decide(desired_exposure, market_state, risk_state) -> RiskDecision

Alles hier is `frozen=True`. Dat is geen stijlkeuze: de engine moet PUUR zijn,
want exit-criterium 2 eist dat twee verschillende alpha-units met een identieke
`a_t`-reeks bit-identieke `permitted_exposure` opleveren. Een mutabele
`RiskState` die tijdens een run door een consument wordt bijgesteld, maakt die
eis onbewijsbaar - en dat is precies hoe `risk/portfolio.py` (`self.equity`,
`self._returns_history`, `update()`) aan controle ontsnapte.

WAT DE RISICOLAAG NIET MAG WETEN
--------------------------------
`market_state` draagt uitsluitend GEMETEN marktgrootheden. Er is bewust geen
veld voor modelnaam, verwachte alpha, meta-label-confidence of
strategie-identiteit: audit sectie 14 stelt dat risico altijd alpha overruled,
en een limiet die meeschaalt met modelovertuiging is die regel met een omweg
omgedraaid. De ontkoppelingstest (stap 9) leest deze dataclass en faalt zodra
er een alpha-veld bij komt.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 11.1, 14, 14.1, 19, 24.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = [
    "BOOK_SCOPE",
    "BindingConstraint",
    "ConstraintKind",
    "MarketState",
    "RiskDecision",
    "RiskState",
    "validate_desired_exposure",
]

#: `scope` van een constraint die op het boek als geheel slaat, niet op een symbool.
BOOK_SCOPE = "__book__"


class ConstraintKind(str, Enum):
    """Welke limiet bond. Een enum, geen vrije tekst.

    De reden dat dit geen string is: `SizingDecision.reason` in
    `risk/portfolio.py` was een f-string, en daardoor was er geen enkele manier
    om machinaal vast te stellen welke drempel had gebonden. Een rapport dat
    "welke constraint bond" moet aantonen (deliverable 12) kan niet op
    geformatteerde tekst leunen.
    """

    HALTED = "halted"
    DAILY_LOSS_GOVERNOR = "daily_loss_governor"
    DRAWDOWN_BREAKER = "drawdown_breaker"
    VOL_TARGET = "vol_target"
    PER_ASSET_CAP = "per_asset_cap"
    ADV_CAP = "adv_cap"
    CONCENTRATION_CAP = "concentration_cap"
    CLUSTER_CAP = "cluster_cap"
    GROSS_CAP = "gross_cap"
    NET_CAP = "net_cap"


@dataclass(frozen=True, slots=True)
class BindingConstraint:
    """Eén registratie in het auditspoor: welke limiet bond, waarop, hoeveel.

    Elk veld is verplicht. `config_key` is de sleutel in `conf/risk/` die de
    drempel zette - zonder dat veld is een besluit wel herleidbaar tot een
    getal, maar niet tot een BESLUIT, en dat is wat exit-criterium 6 vraagt.
    """

    kind: ConstraintKind
    scope: str
    measured: float
    threshold: float
    exposure_before: float
    exposure_after: float
    config_key: str

    def __post_init__(self) -> None:
        require(
            abs(self.exposure_after) <= abs(self.exposure_before) + _TOL,
            "Een constraint mag exposure uitsluitend VERKLEINEN. Een limiet die "
            "de gevraagde positie vergroot, is geen risicolimiet maar een tweede "
            "alpha-unit (docs/RISK_CONTRACT.md sectie 5.3).",
            DataContractError,
            kind=self.kind.value,
            scope=self.scope,
            exposure_before=self.exposure_before,
            exposure_after=self.exposure_after,
        )

    def as_record(self) -> dict[str, Any]:
        """Platte, JSON-serialiseerbare rij voor het stressrapport en de ledger."""
        return {
            "kind": self.kind.value,
            "scope": self.scope,
            "measured": float(self.measured),
            "threshold": float(self.threshold),
            "exposure_before": float(self.exposure_before),
            "exposure_after": float(self.exposure_after),
            "config_key": self.config_key,
        }


#: Numerieke tolerantie voor de monotoniteitsinvarianten. Geen beleidsdrempel:
#: dit compenseert uitsluitend float-afronding in opeenvolgende clamps.
_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class MarketState:
    """Gemeten marktstaat op `asof_ts`. Nul strategie-informatie.

    `sigma_hat` is de EX-ANTE, geannualiseerde `sigma_{t+1|t}` uit L2
    (`volatility/ewma.py`, lambda uit `conf/model/volatility.yaml`). Tijdens de
    burn-in levert die estimator NaN, en dat blijft NaN: deze dataclass
    accepteert dat, maar `vol_targeting` crasht erop. De validatie zit daar en
    niet hier, omdat een consument die GEEN vol-targeting draait ook geen
    volledige `sigma_hat` nodig heeft.
    """

    asof_ts: pd.Timestamp
    sigma_hat: Mapping[str, float]
    adv_usd: Mapping[str, float] = field(default_factory=dict)
    cluster: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            isinstance(self.asof_ts, pd.Timestamp),
            "market_state.asof_ts moet een pandas Timestamp zijn.",
            DataContractError,
            got=type(self.asof_ts).__name__,
        )
        require(
            self.asof_ts.tz is not None,
            "market_state.asof_ts moet tijdzone-bewust zijn (UTC). Een naive "
            "timestamp maakt de causaliteit van elke limiet oncontroleerbaar.",
            DataContractError,
            asof_ts=str(self.asof_ts),
        )
        # Read-only maken: een consument die de mapping na constructie aanpast,
        # breekt de puurheid waarop de bit-identiteitstest steunt.
        object.__setattr__(self, "sigma_hat", MappingProxyType(dict(self.sigma_hat)))
        object.__setattr__(self, "adv_usd", MappingProxyType(dict(self.adv_usd)))
        object.__setattr__(self, "cluster", MappingProxyType(dict(self.cluster)))


@dataclass(frozen=True, slots=True)
class RiskState:
    """Toestand die over bars heen leeft. Expliciet, serialiseerbaar, persistent.

    `halted` is een EENRICHTINGSDEUR. Er is in deze module geen enkele methode
    die hem terugzet; `kill_switches.py` levert de handmatige reset als losse,
    expliciete operatorhandeling op de persistente store. Zie exit-criterium 5.
    """

    equity: float
    high_water_mark: float
    day_start_equity: float
    halted: bool = False
    halt_reason: str = ""
    halted_at: str = ""

    def __post_init__(self) -> None:
        for name in ("equity", "high_water_mark", "day_start_equity"):
            value = float(getattr(self, name))
            require(
                np.isfinite(value),
                f"risk_state.{name} is niet-eindig. Een corrupte risicostaat "
                "wordt niet gerepareerd en niet genegeerd: hij crasht.",
                DataContractError,
                field=name,
                value=value,
            )
            require(
                value > 0.0,
                f"risk_state.{name} moet strikt positief zijn.",
                DataContractError,
                field=name,
                value=value,
            )
        require(
            self.high_water_mark >= self.equity - _TOL,
            "De High-Water Mark ligt onder de huidige equity. De HWM is per "
            "constructie causaal en monotoon niet-dalend; dit duidt op een "
            "state die buiten de engine om is bijgewerkt.",
            DataContractError,
            equity=float(self.equity),
            high_water_mark=float(self.high_water_mark),
        )
        require(
            (not self.halted) or bool(self.halt_reason),
            "Een HALTED-staat zonder reden is niet auditbaar.",
            DataContractError,
        )

    @property
    def drawdown(self) -> float:
        """Causale drawdown t.o.v. de High-Water Mark, als positieve fractie."""
        return float(1.0 - self.equity / self.high_water_mark)

    @property
    def daily_loss(self) -> float:
        """Verlies sinds `day_start_equity`, als positieve fractie."""
        return float(1.0 - self.equity / self.day_start_equity)

    def as_record(self) -> dict[str, Any]:
        return {
            "equity": float(self.equity),
            "high_water_mark": float(self.high_water_mark),
            "day_start_equity": float(self.day_start_equity),
            "halted": bool(self.halted),
            "halt_reason": self.halt_reason,
            "halted_at": self.halted_at,
        }


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Het besluit van de soevereine risicolaag, met volledig auditspoor.

    De invariant die deliverable 4 afdwingt - *"de engine geeft nooit `a_t`
    ongewijzigd door zonder expliciete registratie dat geen enkele limiet
    bond"* - wordt hier gecontroleerd en niet elders aangenomen.
    """

    permitted_exposure: Mapping[str, float]
    binding_constraints: tuple[BindingConstraint, ...]
    unconstrained: bool
    config_hash: str
    risk_state_out: RiskState

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "permitted_exposure", MappingProxyType(dict(self.permitted_exposure))
        )
        require(
            self.unconstrained == (len(self.binding_constraints) == 0),
            "`unconstrained` en het auditspoor spreken elkaar tegen. Een besluit "
            "waarin niets bond MOET dat expliciet registreren; een besluit waarin "
            "wel iets bond kan niet ongeremd zijn.",
            DataContractError,
            unconstrained=self.unconstrained,
            n_binding=len(self.binding_constraints),
        )
        require(
            bool(self.config_hash),
            "Een risicobesluit zonder config_hash is niet auditbaar (stap 11).",
            DataContractError,
        )

    @property
    def bound_kinds(self) -> tuple[ConstraintKind, ...]:
        """Welke soorten limieten bonden, in toepassingsvolgorde, ontdubbeld."""
        seen: list[ConstraintKind] = []
        for c in self.binding_constraints:
            if c.kind not in seen:
                seen.append(c.kind)
        return tuple(seen)

    def gross(self) -> float:
        return float(sum(abs(v) for v in self.permitted_exposure.values()))

    def net(self) -> float:
        return float(sum(self.permitted_exposure.values()))

    def as_record(self) -> dict[str, Any]:
        """Machineleesbaar auditspoor - de vervanger van `SizingDecision.reason`."""
        return {
            "permitted_exposure": {k: float(v) for k, v in self.permitted_exposure.items()},
            "gross": self.gross(),
            "net": self.net(),
            "unconstrained": bool(self.unconstrained),
            "config_hash": self.config_hash,
            "binding_constraints": [c.as_record() for c in self.binding_constraints],
            "risk_state_out": self.risk_state_out.as_record(),
        }


def validate_desired_exposure(desired_exposure: Mapping[str, float]) -> dict[str, float]:
    """Controleer het L4-contract `a_t in [-1, +1]` en crash bij schending.

    Bewust GEEN clip. Een alpha-unit die 1.4 levert, is kapot; hem stilzwijgend
    terugsnijden naar 1.0 verbergt dat en levert bovendien een positie op die
    niemand heeft gevraagd. Audit sectie 11.1 is hier expliciet: een unit levert
    `a_t in [-1, +1]` en verder niets.
    """
    require(
        len(desired_exposure) > 0,
        "Lege desired_exposure; er valt geen risicobesluit te nemen.",
        DataContractError,
    )
    out: dict[str, float] = {}
    for symbol, value in desired_exposure.items():
        a_t = float(value)
        require(
            np.isfinite(a_t),
            "Niet-eindige gewenste exposure. De risicolaag repareert geen kapotte "
            "alpha-output; zij crasht erop.",
            DataContractError,
            symbol=str(symbol),
            a_t=a_t,
        )
        require(
            abs(a_t) <= 1.0 + _TOL,
            "Gewenste exposure buiten het L4-contract [-1, +1] (audit sectie "
            "11.1). Een unit levert een RICHTING, geen positiegrootte.",
            DataContractError,
            symbol=str(symbol),
            a_t=a_t,
        )
        out[str(symbol)] = a_t
    return out
