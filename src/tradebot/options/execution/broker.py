"""Het BrokerAdapter-protocol en zijn datatypen (detailontwerp 10.6).

`submit` neemt alleen een `ApprovedOrder` (uit de risicopoort), `submit_stock` alleen een
`ApprovedStockOrder` (de delta-hedger, ook langs de poort). `modify` en `cancel` werken op de
client-id van beide. `modify` verandert alleen de limiet, nooit het aantal of de legs.
Een aandelenfill is een `Execution` zonder `contract`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ..contracts import OptionContract
from ..risk.gate import ApprovedOrder, ApprovedStockOrder
from .orders import ComboOrder, OrderStatus, StockOrder

__all__ = ["AccountSnapshot", "BrokerAdapter", "BrokerOrder", "BrokerPosition", "Execution",
           "ExecutionKind", "MarginImpact"]


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    equity: float
    cash: float
    buying_power: float
    is_paper: bool
    account_hash: str


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    contract: OptionContract | None     # None: aandelen
    symbol: str
    qty: int
    avg_price: float


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    order: ComboOrder | StockOrder
    status: OrderStatus
    filled: int = 0
    avg_fill: float | None = None


class ExecutionKind(StrEnum):
    FILL = "fill"
    ASSIGNMENT = "assignment"
    EXERCISE = "exercise"
    EXPIRY = "expiry"


@dataclass(frozen=True, slots=True)
class Execution:
    ts: datetime
    kind: ExecutionKind
    client_id: str                       # leeg bij assignment, exercise en expiratie
    contract: OptionContract | None
    symbol: str
    qty: int                             # + gekocht/ontvangen, − verkocht/geleverd
    price: float
    commission: float = 0.0


@dataclass(frozen=True, slots=True)
class MarginImpact:
    initial_change: float
    maintenance_change: float


class BrokerAdapter(Protocol):
    async def connect(self) -> None: ...

    async def account(self) -> AccountSnapshot: ...

    async def positions(self) -> list[BrokerPosition]: ...

    async def open_orders(self) -> list[BrokerOrder]: ...

    async def submit(self, approved: ApprovedOrder, order: ComboOrder) -> BrokerOrder: ...

    async def submit_stock(self, approved: ApprovedStockOrder,
                           order: StockOrder) -> BrokerOrder: ...

    async def modify(self, client_id: str, new_limit: float) -> BrokerOrder: ...

    async def cancel(self, client_id: str) -> None: ...

    async def what_if(self, order: ComboOrder) -> MarginImpact | None: ...

    async def executions(self, since: datetime) -> list[Execution]: ...
