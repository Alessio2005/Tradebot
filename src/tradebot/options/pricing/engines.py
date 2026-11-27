"""Welke QuantLib-engine waarvoor (detailontwerp 5.2).

| Gebruik                       | Engine                                         |
|-------------------------------|------------------------------------------------|
| scan over hele ketens         | Barone-Adesi-Whaley (snel)                     |
| beslissing, sizing, risico    | binomiaal Leisen-Reimer, 401 stappen           |
| discreet dividend in looptijd | FD met dividendschema                          |
| referentie, pariteit          | analytisch Europees                            |

Wie de tabel leest: de keten wordt altijd met `Use.SCAN` gebouwd (BAW, FD bij een discreet
dividend); `Use.DECISION` en `settings.decision` worden door `engine_for` alleen op verzoek
gekozen en door de standaardketen NIET gebruikt. De beslissingsengine (`pricing.decision_engine`)
komt alleen in de risicopoort (`RiskGate(..., Method(decision_engine))`: de herwaardering van de
scenario's), niet in sizing of in de IV van de keten. Een discreet dividend
dwingt daar ook FD af.

Binomiaal en BAW kennen alleen een continue yield; met een discreet dividend in de looptijd
is FD de enige Amerikaanse engine die het goed doet. `engine_for` dwingt dat af.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...utils.failfast import ConfigContractError, require

__all__ = ["EngineSettings", "Method", "Use", "engine_for"]


class Method(StrEnum):
    EUROPEAN = "european"
    BINOMIAL_LR = "binomial_lr"
    BINOMIAL_CRR = "binomial_crr"
    BAW = "baw"
    FD = "fd"

    @property
    def american(self) -> bool:
        return self is not Method.EUROPEAN

    @property
    def supports_discrete_dividends(self) -> bool:
        return self is Method.FD


class Use(StrEnum):
    SCAN = "scan"
    DECISION = "decision"


@dataclass(frozen=True, slots=True)
class EngineSettings:
    binomial_steps: int = 401
    fd_time_steps: int = 200
    fd_grid_points: int = 400
    scan: Method = Method.BAW
    decision: Method = Method.BINOMIAL_LR

    def __post_init__(self) -> None:
        require(self.binomial_steps >= 11 and self.binomial_steps % 2 == 1,
                "binomial_steps moet oneven zijn en ten minste 11 (Leisen-Reimer)",
                ConfigContractError, binomial_steps=self.binomial_steps)
        require(self.fd_time_steps >= 10 and self.fd_grid_points >= 50,
                "te grof FD-rooster", ConfigContractError)
        require(self.scan.american and self.decision.american,
                "scan en beslissing waarderen Amerikaans", ConfigContractError)


def engine_for(use: Use, has_discrete_dividend: bool, settings: EngineSettings) -> Method:
    """De engine voor `use`; met een discreet dividend in de looptijd altijd FD."""
    if has_discrete_dividend:
        return Method.FD
    return settings.scan if use is Use.SCAN else settings.decision
