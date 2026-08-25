# src/tradebot/execution/context.py
"""Het gedeelde executiecontract van backtest en live — L9/L13.

Phase 5, fase-opdracht §4.3. De volledige `live/`-migratie is formeel Phase 7,
maar de CONTRACTGRENS wordt hier vastgelegd, want zonder die grens is Phase 7
een herontwerp in plaats van een aansluiting.

DE EIS DIE DIT AFDWINGT
-----------------------
> *"De live controller mag uiteindelijk niet zijn eigen risk policy
> interpreteren. Hij moet dezelfde sovereign decision interface consumeren als
> de backtester."*

`ExecutionContext` is die interface. Hij bevat precies wat de soevereine laag
nodig heeft om te beslissen, en niets meer:

* WAT de markt doet   -> `market_state()`  (`MarketState`: sigma, adv, cluster)
* WAT het boek is     -> `risk_state()`    (`RiskState`: equity, hwm, halted)
* WAAR we staan       -> `positions()`, `marks()`
* WANNEER             -> `asof()`, `next_fill_ts()`

Er is bewust GEEN methode die een strategie-identiteit, een modelnaam of een
verwachte alpha doorgeeft. Dat is dezelfde regel die `risk/contract.py` op
`MarketState` legt, hier herhaald op de laag erboven: een executiecontext die
modelvertrouwen kan doorgeven, is een kanaal waarlangs alpha alsnog de
positiegrootte bepaalt.

WAAROM DIT EEN ABC IS EN GEEN PROTOCOL
--------------------------------------
Een `Protocol` wordt structureel gecontroleerd en pas door mypy afgedwongen. Een
ABC crasht bij instantiatie. De parity-eis (exit-criterium 9: *"bit-identiek op
gedeelde replay"*) leunt erop dat beide implementaties dezelfde vragen
beantwoorden; een implementatie die er stilzwijgend één mist, moet niet kunnen
bestaan.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 15, 19 (L9, L13), 23.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..risk.contract import MarketState, RiskState
from ..utils.failfast import DataContractError, require

__all__ = [
    "BacktestExecutionContext",
    "ExecutionContext",
    "LiveExecutionContext",
    "MarketSlice",
]


@dataclass(frozen=True, slots=True)
class MarketSlice:
    """De marktgegevens van één bar, zoals de executielaag ze ziet.

    Dit is het gedeelde replay-formaat. Een live feed en een backtest-store
    vullen hem uit verschillende bronnen, maar de parity-test voedt beide met
    DEZELFDE reeks - en dan moeten de besluiten en orders identiek zijn.
    """

    ts: pd.Timestamp
    #: Markprijs per symbool (de close van deze bar).
    marks: Mapping[str, float]
    #: Ex-ante geannualiseerde sigma_{t+1|t} per symbool (L2).
    sigma_hat: Mapping[str, float]
    #: DAGELIJKSE sigma per symbool, als decimaal. Het impactmodel rekent op
    #: dagvol; die deannualisering hier doen in plaats van in de router houdt
    #: de conversie op één plek.
    sigma_daily: Mapping[str, float]
    #: Causale ADV in quote-valuta per symbool.
    adv_notional: Mapping[str, float]
    #: Verhandeld volume van DEZE bar in quote-valuta. Bepaalt partial fills.
    bar_volume_notional: Mapping[str, float]
    #: Clusterlabels, als gemeten marktstructuur.
    clusters: Mapping[str, str]
    #: Funding rate die op deze bar settelt, per symbool. Leeg = geen funding.
    funding_rate: Mapping[str, float]

    def __post_init__(self) -> None:
        require(
            isinstance(self.ts, pd.Timestamp) and self.ts.tz is not None,
            "Een MarketSlice zonder tijdzone-bewuste timestamp.",
            DataContractError, ts=str(self.ts),
        )
        require(
            len(self.marks) > 0,
            "Een MarketSlice zonder markprijzen.",
            DataContractError, ts=str(self.ts),
        )
        for symbol, price in self.marks.items():
            require(
                np.isfinite(float(price)) and float(price) > 0.0,
                "Niet-positieve markprijs in een MarketSlice.",
                DataContractError, symbol=str(symbol), price=float(price),
            )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.marks))


class ExecutionContext(ABC):
    """Wat backtest en live beide moeten kunnen beantwoorden.

    Elke methode is abstract. Een implementatie die er één vergeet, kan niet
    worden geinstantieerd, en de parity-test kan dus niet groen worden op een
    context die de helft van de vragen niet stelt.
    """

    #: Vrij te kiezen door de implementatie; verschijnt in het auditspoor zodat
    #: een rapport kan zeggen WELKE context het produceerde.
    kind: str = "abstract"

    @abstractmethod
    def asof(self) -> pd.Timestamp:
        """Het beslismoment. Alles wat later is, bestaat nog niet."""

    @abstractmethod
    def next_fill_ts(self) -> pd.Timestamp:
        """Het vroegste moment waarop een nu geplaatste order kan vullen.

        Strikt later dan `asof()`. Dit is waar latency in het contract zit, en
        het is de reden dat lookahead structureel onmogelijk is in plaats van
        per test gecontroleerd.
        """

    @abstractmethod
    def market_state(self) -> MarketState:
        """De GEMETEN marktstaat voor de soevereine laag. Nul alpha-informatie."""

    @abstractmethod
    def risk_state(self) -> RiskState:
        """De boekstaat voor de soevereine laag."""

    @abstractmethod
    def equity(self) -> float:
        """Huidige equity in quote-valuta."""

    @abstractmethod
    def positions(self) -> Mapping[str, float]:
        """Huidige positie per symbool, in stuks (getekend)."""

    @abstractmethod
    def marks(self) -> Mapping[str, float]:
        """Huidige markprijs per symbool."""

    @abstractmethod
    def slice(self) -> MarketSlice:
        """De volledige marktsnede van deze bar."""

    def audit_header(self) -> dict[str, Any]:
        """De regel die elk rapport over deze context draagt."""
        return {
            "context_kind": self.kind,
            "asof_ts": self.asof().isoformat(),
            "next_fill_ts": self.next_fill_ts().isoformat(),
            "n_symbols": len(self.marks()),
            "equity": float(self.equity()),
        }


@dataclass(frozen=True, slots=True)
class _StateBundle:
    """De boekstaat die beide contexts van hun eigenaar krijgen."""

    equity: float
    high_water_mark: float
    day_start_equity: float
    halted: bool
    halt_reason: str
    halted_at: str
    positions: Mapping[str, float]


class _SharedContext(ExecutionContext):
    """De gedeelde implementatie. Backtest en live verschillen alleen in bron.

    Dit is het hart van exit-criterium 9. Als de twee contexts elk hun eigen
    `market_state()` zouden bouwen, zou "bit-identiek op gedeelde replay" een
    toevalligheid zijn die per wijziging opnieuw bewezen moet worden. Door de
    conversie van `MarketSlice` naar `MarketState`/`RiskState` EEN keer te
    schrijven, is pariteit een eigenschap van de code in plaats van een
    meetresultaat.
    """

    __slots__ = ("_latency_bars", "_next_ts", "_slice", "_state")

    def __init__(
        self, market: MarketSlice, state: _StateBundle, *,
        next_fill_ts: pd.Timestamp,
    ) -> None:
        require(
            next_fill_ts > market.ts,
            "next_fill_ts moet strikt na het beslismoment liggen. Een order "
            "die kan vullen op zijn eigen beslisbar is lookahead.",
            DataContractError,
            asof=str(market.ts), next_fill_ts=str(next_fill_ts),
        )
        self._slice = market
        self._state = state
        self._next_ts = next_fill_ts

    def asof(self) -> pd.Timestamp:
        return self._slice.ts

    def next_fill_ts(self) -> pd.Timestamp:
        return self._next_ts

    def market_state(self) -> MarketState:
        return MarketState(
            asof_ts=self._slice.ts,
            sigma_hat=dict(self._slice.sigma_hat),
            adv_usd=dict(self._slice.adv_notional),
            cluster=dict(self._slice.clusters),
        )

    def risk_state(self) -> RiskState:
        return RiskState(
            equity=float(self._state.equity),
            high_water_mark=float(self._state.high_water_mark),
            day_start_equity=float(self._state.day_start_equity),
            halted=bool(self._state.halted),
            halt_reason=self._state.halt_reason,
            halted_at=self._state.halted_at,
        )

    def equity(self) -> float:
        return float(self._state.equity)

    def positions(self) -> Mapping[str, float]:
        return dict(self._state.positions)

    def marks(self) -> Mapping[str, float]:
        return dict(self._slice.marks)

    def slice(self) -> MarketSlice:
        return self._slice


class BacktestExecutionContext(_SharedContext):
    """De context die `backtest/engine.py` per bar bouwt."""

    kind = "backtest"


class LiveExecutionContext(_SharedContext):
    """De context die `live/` in Phase 7 per bar moet bouwen.

    Hij is hier al volledig bruikbaar en wordt door
    `tests/integration/test_backtest_live_parity.py` gedraaid op dezelfde
    replay als de backtest-variant. Wat Phase 7 nog moet doen, is de BRON
    aansluiten: `live/feed.py` levert de `MarketSlice` en `live/state.py` de
    `_StateBundle`. De besluitvorming erboven verandert dan niet.

    Wat Phase 7 expliciet NIET meer mag doen, staat in
    `reports/phase5_sovereign_wiring_audit.md` §3.3: eigen drawdown-breakers,
    eigen notional-limieten en een eigen concentratiecap.
    """

    kind = "live"


def build_context(
    kind: str,
    market: MarketSlice,
    *,
    equity: float,
    high_water_mark: float,
    day_start_equity: float,
    positions: Mapping[str, float],
    next_fill_ts: pd.Timestamp,
    halted: bool = False,
    halt_reason: str = "",
    halted_at: str = "",
) -> ExecutionContext:
    """Bouw een context van het gevraagde soort uit dezelfde invoer.

    Bestaat zodat de parity-test één aanroep kan doen met `kind="backtest"` en
    `kind="live"` en er geen enkele andere weg is waarlangs de twee kunnen
    afwijken.
    """
    bundle = _StateBundle(
        equity=float(equity),
        high_water_mark=float(high_water_mark),
        day_start_equity=float(day_start_equity),
        halted=bool(halted),
        halt_reason=halt_reason,
        halted_at=halted_at,
        positions=dict(positions),
    )
    factories: dict[str, type[_SharedContext]] = {
        "backtest": BacktestExecutionContext,
        "live": LiveExecutionContext,
    }
    require(
        kind in factories,
        "Onbekend soort executiecontext.",
        DataContractError, kind=kind, known=sorted(factories),
    )
    return factories[kind](market, bundle, next_fill_ts=next_fill_ts)
