"""Eenheden van Greeks: per aandeel uit de pricer, per positie na schaling.

Een Greek zonder eenheid is een bron van stille fouten: een vega per volpunt optellen bij een
vega per 100 volpunten, of een delta per aandeel bij een delta per contract. Daarom draagt
elke grootheid op positieniveau haar eenheid in het type, en weigert optellen van twee
verschillende typen (een `TradebotContractError`, en voor mypy een typefout).

De enige overgang van "per aandeel" naar "per positie" is `GreeksPerShare.scale`.

Eenheden, overal gelijk (detailontwerp 2.1):

* delta  aandelen-equivalent per 1 USD koersbeweging van de onderliggende;
* gamma  verandering van de delta per 1 USD koersbeweging;
* vega   USD per 1 volpunt (0,01 absolute volatiliteit);
* theta  USD per kalenderdag;
* rho    USD per 1 procentpunt rente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from ..utils.failfast import TradebotContractError, require

__all__ = ["DeltaShares", "GammaShares", "GreeksPerShare", "PositionGreeks", "RhoUSD",
           "ThetaUSD", "VegaUSD"]


@dataclass(frozen=True, slots=True)
class _Unit:
    """Een getal met een eenheid. Alleen optelbaar met hetzelfde type."""

    value: float

    def __add__(self, other: Self) -> Self:
        if type(other) is not type(self):
            raise TradebotContractError(
                f"{type(self).__name__} en {type(other).__name__} zijn niet op te tellen")
        return type(self)(self.value + other.value)

    def __neg__(self) -> Self:
        return type(self)(-self.value)


@dataclass(frozen=True, slots=True)
class DeltaShares(_Unit):
    """Delta in aandelen-equivalent, al geschaald met multiplier, aantal en teken."""


@dataclass(frozen=True, slots=True)
class GammaShares(_Unit):
    """Verandering van `DeltaShares` per 1 USD koersbeweging."""


@dataclass(frozen=True, slots=True)
class VegaUSD(_Unit):
    """USD per 1 volpunt."""


@dataclass(frozen=True, slots=True)
class ThetaUSD(_Unit):
    """USD per kalenderdag."""


@dataclass(frozen=True, slots=True)
class RhoUSD(_Unit):
    """USD per 1 procentpunt rente."""


@dataclass(frozen=True, slots=True)
class PositionGreeks:
    """De Greeks van een positie (of een som van posities), elk met zijn eenheid."""

    delta: DeltaShares
    gamma: GammaShares
    vega: VegaUSD
    theta: ThetaUSD
    rho: RhoUSD

    @classmethod
    def zero(cls) -> PositionGreeks:
        return cls(DeltaShares(0.0), GammaShares(0.0), VegaUSD(0.0), ThetaUSD(0.0), RhoUSD(0.0))

    def __add__(self, other: PositionGreeks) -> PositionGreeks:
        return PositionGreeks(self.delta + other.delta, self.gamma + other.gamma,
                              self.vega + other.vega, self.theta + other.theta,
                              self.rho + other.rho)


@dataclass(frozen=True, slots=True)
class GreeksPerShare:
    """Ruwe Greeks van één optie, per aandeel, in de eenheden van de module-docstring."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def scale(self, qty: int, multiplier: int) -> PositionGreeks:
        """Greeks van `qty` contracten (negatief is short) met `multiplier` aandelen elk."""
        require(multiplier > 0, "multiplier moet positief zijn", multiplier=multiplier)
        f = float(qty * multiplier)
        return PositionGreeks(DeltaShares(self.delta * f), GammaShares(self.gamma * f),
                              VegaUSD(self.vega * f), ThetaUSD(self.theta * f),
                              RhoUSD(self.rho * f))
