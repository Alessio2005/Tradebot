# src/tradebot/execution/order_router.py
"""Van soeverein besluit naar fills — L9.

Phase 5, deliverable 8. Dit is de ENIGE plaats waar een order ontstaat, en de
enige plaats waar een fill ontstaat.

DE STRUCTURELE GARANTIE
-----------------------
`build_orders()` neemt een `RiskDecision` als VERPLICHT argument en verifieert
zijn `config_hash` tegen de hash die de router bij constructie kreeg. Er is geen
overload zonder besluit en geen pad dat een target-gewicht rechtstreeks in een
order omzet.

Dat is opzettelijk structureel in plaats van afgesproken. Fase-opdracht §16
eist dat *"backtest order generation sovereign approval vereist"*; een
conventie waaraan callers zich horen te houden is geen eis maar een verzoek.
Hier is het typesysteem de handhaving: zonder `RiskDecision` compileert de
aanroep niet, en met de VERKEERDE `RiskDecision` crasht hij.

WAT HIER WEL EN NIET WORDT GEMODELLEERD
---------------------------------------
Wel: spread, marktimpact, fees (maker/taker), latency, participatielimieten,
partial fills, geweigerde orders, dust-drempels, venue-funding-cap.

Niet: limit-order queues en cancellations. Dat is geen omissie maar een
databeperking, en hij wordt hier expliciet genoemd in plaats van stilzwijgend
weggelaten. `execution/simulator.py` bevat een LOB-queue-simulator die
orderboekdiepte nodig heeft; `docs/DATA_REGISTER.md` §6 stelt vast dat die data
niet bestaat voor dit universum. Op een daily grid is er bovendien geen
intra-bar tijdas waarop een queue betekenis heeft. De router plaatst daarom
uitsluitend TAKER-orders, en dat is de conservatieve keuze: taker betaalt de
hoogste fee en de volledige spread.

DE PRIJS WAARTEGEN WORDT GEVULD
-------------------------------
Nooit tegen mid. Fase-opdracht §12: *"Een fill tegen mid zonder bewijs is niet
toegestaan."* De fillprijs is::

    fill = mid * (1 + side * (half_spread + impact))

waarbij `side` +1 is bij kopen en -1 bij verkopen, zodat beide richtingen
betalen. `half_spread` en `impact` dragen hun eigen herkomststatus mee.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 15, 15.1, 19 (L9).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.accounting import Fill
from ..risk.contract import RiskDecision
from ..utils.failfast import ConfigContractError, DataContractError, require
from .impact_model import ImpactParams, ImpactStatus, square_root_impact

__all__ = [
    "ExecutionReport",
    "Order",
    "OrderRouter",
    "RejectReason",
    "SpreadModel",
    "SpreadStatus",
    "VenueSpec",
]


class SpreadStatus(str, Enum):
    """Herkomst van de half-spread. Reist mee, net als `ImpactStatus`."""

    #: Gemeten uit echte bid/ask-quotes.
    MEASURED = "MEASURED"
    #: Geen quotes beschikbaar; de waarde is een expliciete, geconfigureerde
    #: aanname. Zie `reports/TCA_CALIBRATION_REPORT.md` §8 voor waarom de
    #: OHLC-gebaseerde alternatieven op dit universum niet bruikbaar zijn.
    SPREAD_ASSUMED = "SPREAD_ASSUMED"


class RejectReason(str, Enum):
    """Waarom een order niet (volledig) is uitgevoerd. Een enum, geen tekst."""

    BELOW_MIN_NOTIONAL = "below_min_notional"
    PARTICIPATION_CAP = "participation_cap"
    NO_LIQUIDITY = "no_liquidity"


@dataclass(frozen=True, slots=True)
class SpreadModel:
    """De half-spread en waar hij vandaan komt.

    Geen default. Een half-spread van nul is een fill tegen mid, en die is
    verboden zonder bewijs.
    """

    half_spread_bps: float
    status: SpreadStatus
    source: str

    def __post_init__(self) -> None:
        require(
            np.isfinite(self.half_spread_bps) and self.half_spread_bps > 0.0,
            "Een half-spread van nul of negatief betekent vullen op of binnen "
            "mid. Dat is alleen toegestaan met bewijs, en dan is de status "
            "MEASURED (fase-opdracht §12).",
            ConfigContractError,
            key="execution.assumed_half_spread_bps",
            half_spread_bps=self.half_spread_bps,
        )
        require(
            isinstance(self.status, SpreadStatus) and bool(self.source),
            "Een spread zonder herkomst is niet auditbaar.",
            ConfigContractError,
            status=self.status, source=self.source,
        )

    @property
    def fraction(self) -> float:
        return float(self.half_spread_bps) * 1e-4


@dataclass(frozen=True, slots=True)
class VenueSpec:
    """De mechanische eigenschappen van de handelsplaats.

    Dit is GEEN risicoconfiguratie. Elk veld hier beschrijft wat de exchange
    doet, niet wat wij onszelf toestaan; dat onderscheid is de reden dat
    `funding_cap_abs` hier staat en niet in de boekhouding, en dat
    `max_participation` hier NIET staat - de participatielimiet is een
    risicobesluit en komt uit `conf/risk/adv_participation_cap`.
    """

    maker_fee_bps: float
    taker_fee_bps: float
    #: Absolute cap op de 8-uurs funding rate die de venue hanteert. Bybit
    #: begrenst funding; een backtest die dat niet doet, modelleert een markt
    #: die niet bestaat (zie `backtest/portfolio.py:408`).
    funding_cap_abs: float
    #: Kleinste order die de venue accepteert, in quote-valuta.
    min_notional: float
    #: Aantal bars tussen orderplaatsing en de vroegst mogelijke fill. Op een
    #: daily grid is 1 de enige causale waarde: een besluit op de close van
    #: bar t kan niet vullen binnen bar t.
    latency_bars: int

    def __post_init__(self) -> None:
        for name in ("maker_fee_bps", "taker_fee_bps", "funding_cap_abs",
                     "min_notional"):
            value = float(getattr(self, name))
            require(
                np.isfinite(value) and value >= 0.0,
                f"venue.{name} moet eindig en niet-negatief zijn.",
                ConfigContractError, key=f"venue.{name}", value=value,
            )
        require(
            int(self.latency_bars) >= 1,
            "Latency van nul bars laat een order vullen op de bar waarop het "
            "besluit is genomen. Dat is lookahead, geen snelle executie.",
            ConfigContractError,
            key="venue.latency_bars", latency_bars=self.latency_bars,
        )

    def fee(self, notional: float, *, liquidity: str) -> float:
        bps = self.taker_fee_bps if liquidity == "taker" else self.maker_fee_bps
        return abs(float(notional)) * float(bps) * 1e-4

    def cap_funding(self, rate: float) -> float:
        """Klem de funding rate op wat de venue toestaat."""
        return float(np.clip(float(rate), -self.funding_cap_abs,
                             self.funding_cap_abs))


@dataclass(frozen=True, slots=True)
class Order:
    """Eén order. Draagt de `config_hash` van het besluit dat hem toestond.

    Zonder dat veld is achteraf niet vast te stellen ONDER WELK risicoregime een
    order is ontstaan, en dan is de bewering "de backtest draaide door de
    soevereine laag" niet controleerbaar.
    """

    symbol: str
    ts_decision: pd.Timestamp
    ts_earliest_fill: pd.Timestamp
    target_qty_delta: float
    risk_config_hash: str

    def __post_init__(self) -> None:
        require(
            bool(self.risk_config_hash),
            "Een order zonder risk_config_hash. Elke order draagt het regime "
            "waaronder hij is toegestaan (fase-opdracht §16 punt 5).",
            DataContractError, symbol=self.symbol,
        )
        require(
            self.ts_earliest_fill > self.ts_decision,
            "Een order die kan vullen op of vóór zijn eigen beslismoment is "
            "lookahead.",
            DataContractError,
            symbol=self.symbol,
            ts_decision=str(self.ts_decision),
            ts_earliest_fill=str(self.ts_earliest_fill),
        )

    @property
    def side(self) -> int:
        return 1 if self.target_qty_delta > 0.0 else -1


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Wat er met één order is gebeurd, met de kostenopsplitsing.

    De opsplitsing is niet decoratief: `tca/post_trade.py` sluit de
    roundtrip hierop, en de invariant `arrival - fill == spread + impact` moet
    per order exact opgaan.
    """

    order: Order
    fill: Fill | None
    requested_qty: float
    filled_qty: float
    unfilled_qty: float
    arrival_price: float
    fill_price: float
    spread_cost: float
    impact_cost: float
    fee_cost: float
    participation: float
    impact_status: ImpactStatus
    spread_status: SpreadStatus
    reject_reason: RejectReason | None

    @property
    def is_partial(self) -> bool:
        return abs(self.unfilled_qty) > 1e-12 and abs(self.filled_qty) > 1e-12

    @property
    def is_rejected(self) -> bool:
        return self.fill is None

    def as_record(self) -> dict[str, Any]:
        return {
            "symbol": self.order.symbol,
            "ts_decision": self.order.ts_decision.isoformat(),
            "ts_fill": self.order.ts_earliest_fill.isoformat(),
            "requested_qty": float(self.requested_qty),
            "filled_qty": float(self.filled_qty),
            "unfilled_qty": float(self.unfilled_qty),
            "arrival_price": float(self.arrival_price),
            "fill_price": float(self.fill_price),
            "spread_cost": float(self.spread_cost),
            "impact_cost": float(self.impact_cost),
            "fee_cost": float(self.fee_cost),
            "participation": float(self.participation),
            "impact_status": self.impact_status.value,
            "spread_status": self.spread_status.value,
            "reject_reason": None if self.reject_reason is None
            else self.reject_reason.value,
            "risk_config_hash": self.order.risk_config_hash,
        }


class OrderRouter:
    """Zet een soeverein besluit om in orders, en orders in fills."""

    __slots__ = ("_impact", "_risk_config_hash", "_spread", "_venue")

    def __init__(
        self,
        *,
        venue: VenueSpec,
        spread: SpreadModel,
        impact: ImpactParams,
        risk_config_hash: str,
    ) -> None:
        require(
            isinstance(venue, VenueSpec) and isinstance(spread, SpreadModel)
            and isinstance(impact, ImpactParams),
            "De router vereist een expliciete venue, spread en impactparameter. "
            "Geen daarvan heeft een default (fase-opdracht §23).",
            ConfigContractError,
        )
        require(
            bool(risk_config_hash),
            "De router weigert te bestaan zonder de config_hash van de "
            "soevereine policy. Zonder die hash kan hij niet controleren dat "
            "een besluit uit hetzelfde regime komt.",
            ConfigContractError,
        )
        self._venue = venue
        self._spread = spread
        self._impact = impact
        self._risk_config_hash = str(risk_config_hash)

    @property
    def venue(self) -> VenueSpec:
        return self._venue

    @property
    def risk_config_hash(self) -> str:
        return self._risk_config_hash

    # ------------------------------------------------------------------ #
    # Besluit -> orders
    # ------------------------------------------------------------------ #
    def build_orders(
        self,
        decision: RiskDecision,
        *,
        equity: float,
        marks: Mapping[str, float],
        current_qty: Mapping[str, float],
        ts_decision: pd.Timestamp,
        ts_earliest_fill: pd.Timestamp,
    ) -> list[Order]:
        """De ENIGE manier om een order te maken. Vereist een soeverein besluit.

        `decision.permitted_exposure` is een gewicht per symbool als fractie van
        de equity. Het verschil met de huidige positie is de order.
        """
        require(
            isinstance(decision, RiskDecision),
            "build_orders vereist een RiskDecision. Er is geen pad van een "
            "target-gewicht naar een order dat de soevereine laag omzeilt "
            "(fase-opdracht §16 punt 2).",
            ConfigContractError,
            got=type(decision).__name__,
        )
        require(
            decision.config_hash == self._risk_config_hash,
            "De config_hash van dit besluit komt niet overeen met de policy "
            "waarmee de router is geconstrueerd. Een besluit uit een ander "
            "risicoregime wordt niet uitgevoerd (fase-opdracht §17: "
            "'policy hash mismatch -> FAIL').",
            ConfigContractError,
            decision_hash=decision.config_hash,
            router_hash=self._risk_config_hash,
        )
        require(
            np.isfinite(equity) and equity > 0.0,
            "Orders bouwen op een niet-positieve equity.",
            DataContractError, equity=equity,
        )

        orders: list[Order] = []
        for symbol, weight in decision.permitted_exposure.items():
            price = marks.get(symbol)
            require(
                price is not None and np.isfinite(float(price))
                and float(price) > 0.0,
                "Ontbrekende markprijs voor een symbool in het soevereine "
                "besluit. Er wordt geen order gebouwd op een aangenomen prijs.",
                DataContractError, symbol=symbol,
            )
            target_qty = float(weight) * float(equity) / float(price)  # type: ignore[arg-type]
            delta = target_qty - float(current_qty.get(symbol, 0.0))
            if abs(delta) * float(price) < self._venue.min_notional:  # type: ignore[arg-type]
                continue
            orders.append(Order(
                symbol=symbol,
                ts_decision=ts_decision,
                ts_earliest_fill=ts_earliest_fill,
                target_qty_delta=delta,
                risk_config_hash=self._risk_config_hash,
            ))
        return orders

    # ------------------------------------------------------------------ #
    # Order -> fill
    # ------------------------------------------------------------------ #
    def execute(
        self,
        order: Order,
        *,
        arrival_price: float,
        bar_volume_notional: float,
        adv_notional: float,
        sigma_daily: float,
        participation_cap: float,
    ) -> ExecutionReport:
        """Voer één order uit tegen de bar op `order.ts_earliest_fill`.

        `participation_cap` komt uit de SOEVEREINE policy
        (`risk.adv_participation_cap`) en niet uit de venue: hoeveel van het
        volume je mag zijn, is een risicobesluit.
        """
        require(
            np.isfinite(arrival_price) and arrival_price > 0.0,
            "Uitvoeren tegen een niet-positieve arrival price.",
            DataContractError, symbol=order.symbol, arrival_price=arrival_price,
        )
        require(
            np.isfinite(bar_volume_notional) and bar_volume_notional >= 0.0,
            "Niet-eindig bar-volume.",
            DataContractError, symbol=order.symbol,
            bar_volume_notional=bar_volume_notional,
        )
        require(
            np.isfinite(participation_cap) and 0.0 < participation_cap <= 1.0,
            "participation_cap buiten (0, 1].",
            ConfigContractError, key="risk.adv_participation_cap",
            participation_cap=participation_cap,
        )

        requested = float(order.target_qty_delta)
        side = order.side
        max_notional = float(participation_cap) * float(bar_volume_notional)

        if max_notional <= 0.0:
            return self._rejected(order, arrival_price, requested,
                                  RejectReason.NO_LIQUIDITY)

        requested_notional = abs(requested) * arrival_price
        fillable_notional = min(requested_notional, max_notional)
        reject = (RejectReason.PARTICIPATION_CAP
                  if fillable_notional < requested_notional - 1e-12 else None)

        if fillable_notional < self._venue.min_notional:
            return self._rejected(order, arrival_price, requested,
                                  RejectReason.BELOW_MIN_NOTIONAL)

        filled_qty = side * fillable_notional / arrival_price
        impact = square_root_impact(
            order_notional=fillable_notional,
            adv_notional=adv_notional,
            sigma_daily=sigma_daily,
            params=self._impact,
        )
        # Beide richtingen betalen: kopen boven mid, verkopen eronder.
        slip = self._spread.fraction + impact.impact_fraction
        fill_price = arrival_price * (1.0 + side * slip)
        require(
            fill_price > 0.0,
            "De gemodelleerde slippage duwt de fillprijs door nul. Dat is geen "
            "dure fill maar een kapot model.",
            DataContractError,
            symbol=order.symbol, arrival_price=arrival_price, slip=slip,
        )

        notional = abs(filled_qty) * fill_price
        fee = self._venue.fee(notional, liquidity="taker")
        fill = Fill(
            symbol=order.symbol,
            ts=order.ts_earliest_fill,
            qty=filled_qty,
            price=fill_price,
            fee=fee,
            liquidity="taker",
            order_id=f"{order.symbol}@{order.ts_decision.isoformat()}",
        )
        return ExecutionReport(
            order=order,
            fill=fill,
            requested_qty=requested,
            filled_qty=filled_qty,
            unfilled_qty=requested - filled_qty,
            arrival_price=arrival_price,
            fill_price=fill_price,
            spread_cost=abs(filled_qty) * arrival_price * self._spread.fraction,
            impact_cost=abs(filled_qty) * arrival_price * impact.impact_fraction,
            fee_cost=fee,
            participation=fillable_notional / max(bar_volume_notional, 1e-30),
            impact_status=impact.status,
            spread_status=self._spread.status,
            reject_reason=reject,
        )

    def _rejected(
        self, order: Order, arrival_price: float, requested: float,
        reason: RejectReason,
    ) -> ExecutionReport:
        return ExecutionReport(
            order=order, fill=None, requested_qty=requested, filled_qty=0.0,
            unfilled_qty=requested, arrival_price=arrival_price,
            fill_price=arrival_price, spread_cost=0.0, impact_cost=0.0,
            fee_cost=0.0, participation=0.0, impact_status=self._impact.status,
            spread_status=self._spread.status, reject_reason=reason,
        )
