"""De IBKR-adapter (detailontwerp 10.6) via `ib_async`: alleen een paper-account.

* `connect` maakt verbinding met TWS of IB Gateway, kiest het account en weigert alles wat
  geen paper-account is (`live.require_paper`). Live staat uit.
* Een order is altijd een combo (`BAG`), ook met één leg, met `orderRef` = de client-id. De
  limiet heeft het teken van de broker: + betalen, − ontvangen. Actie is altijd `BUY`.
* `modify` verandert alleen `lmtPrice`; legs en aantal liggen vast.
* Executies komen per leg; de `BAG`-regel zelf wordt overgeslagen.

De `ib`-parameter is een `ib_async.IB` of een nep-object met dezelfde methoden (tests).
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ib_async import ComboLeg, Contract, ExecutionFilter, LimitOrder, Option

from ...utils.failfast import DataContractError, require
from ..contracts import OptionContract, Right
from ..live import account_hash, require_paper
from ..risk.gate import ApprovedOrder
from .broker import (
    AccountSnapshot,
    BrokerOrder,
    BrokerPosition,
    Execution,
    ExecutionKind,
    MarginImpact,
)
from .orders import ComboOrder, OrderStatus

__all__ = ["IBKRBroker", "contract_of", "option_of"]

_STATUS = {"PendingSubmit": OrderStatus.SUBMITTED, "PreSubmitted": OrderStatus.SUBMITTED,
           "Submitted": OrderStatus.SUBMITTED, "ApiPending": OrderStatus.SUBMITTED,
           "Filled": OrderStatus.FILLED, "Cancelled": OrderStatus.CANCELLED,
           "ApiCancelled": OrderStatus.CANCELLED, "PendingCancel": OrderStatus.SUBMITTED,
           "Inactive": OrderStatus.REJECTED}


def option_of(c: OptionContract) -> Option:
    return Option(c.underlying, c.expiry.strftime("%Y%m%d"), c.strike, c.right.value[0].upper(),
                  "SMART", multiplier="100", currency="USD")


def contract_of(ib_contract: Any) -> OptionContract:
    exp = datetime.strptime(ib_contract.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
    right = Right.CALL if ib_contract.right.upper().startswith("C") else Right.PUT
    return OptionContract(ib_contract.symbol, exp, float(ib_contract.strike), right)


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if math.isfinite(x) and abs(x) < 1e300 else math.nan


class IBKRBroker:
    def __init__(self, ib: Any, host: str = "127.0.0.1", port: int = 4002,
                 client_id: int = 17, account: str = "") -> None:
        self.ib, self.host, self.port, self.client_id = ib, host, port, client_id
        self.account_id = account
        self._trades: dict[str, Any] = {}
        self._orders: dict[str, BrokerOrder] = {}

    @property
    def account_hash(self) -> str:
        return account_hash(self.account_id)

    async def connect(self) -> None:
        if not self.ib.isConnected():
            await self.ib.connectAsync(self.host, self.port, self.client_id,
                                       account=self.account_id)
        accounts = list(self.ib.managedAccounts())
        require(bool(accounts), "IBKR gaf geen account", DataContractError)
        if not self.account_id:
            require(len(accounts) == 1, "meerdere accounts: kies er één met IB_ACCOUNT",
                    DataContractError, n=len(accounts))
            self.account_id = accounts[0]
        require(self.account_id in accounts, "het gekozen account hoort niet bij deze login",
                DataContractError)
        require_paper(self.account_id)
        self.ib.reqMarketDataType(3)                # vertraagde data (open vraag A)

    async def account(self) -> AccountSnapshot:
        vals = {v.tag: _num(v.value) for v in await self.ib.accountSummaryAsync(self.account_id)
                if v.currency in ("USD", "BASE", "")}
        eq, cash = vals.get("NetLiquidation", math.nan), vals.get("TotalCashValue", math.nan)
        bp = vals.get("BuyingPower", math.nan)
        require(math.isfinite(eq) and math.isfinite(cash), "geen equity of cash van IBKR",
                DataContractError)
        return AccountSnapshot(eq, cash, bp if math.isfinite(bp) else 0.0, True,
                               self.account_hash)

    async def positions(self) -> list[BrokerPosition]:
        out = []
        for p in self.ib.positions(self.account_id):
            c = p.contract
            if c.secType == "OPT":
                out.append(BrokerPosition(contract_of(c), c.symbol, int(p.position),
                                          float(p.avgCost) / 100.0))
            elif c.secType == "STK":
                out.append(BrokerPosition(None, c.symbol, int(p.position), float(p.avgCost)))
        return sorted(out, key=lambda b: (b.symbol, b.contract.key() if b.contract else ""))

    async def open_orders(self) -> list[BrokerOrder]:
        return [self._refresh(cid) for cid in sorted(self._orders)
                if self._refresh(cid).status is OrderStatus.SUBMITTED]

    def _refresh(self, cid: str) -> BrokerOrder:
        bo, trade = self._orders[cid], self._trades[cid]
        st = _STATUS.get(trade.orderStatus.status, OrderStatus.SUBMITTED)
        filled = int(trade.orderStatus.filled or 0)
        avg = _num(trade.orderStatus.avgFillPrice)
        self._orders[cid] = replace(bo, status=st, filled=filled,
                                    avg_fill=avg if filled and math.isfinite(avg) else None)
        return self._orders[cid]

    async def _bag(self, order: ComboOrder) -> Contract:
        opts = [option_of(leg.contract) for leg in order.legs]
        q = await self.ib.qualifyContractsAsync(*opts)
        require(len(q) == len(opts) and all(x is not None and getattr(x, "conId", 0)
                                            for x in q),
                "IBKR kent een leg niet", DataContractError, client_id=order.client_id)
        legs = [ComboLeg(conId=x.conId, ratio=abs(leg.ratio),
                         action="BUY" if leg.ratio > 0 else "SELL", exchange="SMART")
                for x, leg in zip(q, order.legs, strict=True)]
        return Contract(secType="BAG", symbol=order.legs[0].contract.underlying,
                        currency="USD", exchange="SMART", comboLegs=legs)

    def _order(self, order: ComboOrder) -> LimitOrder:
        o = LimitOrder("BUY", order.quantity, round(order.limit, 2), orderRef=order.client_id,
                       tif=order.tif, account=self.account_id)
        o.smartComboRoutingParams = []
        return o

    async def submit(self, approved: ApprovedOrder, order: ComboOrder) -> BrokerOrder:
        if not isinstance(approved, ApprovedOrder):
            raise TypeError("IBKR neemt alleen een ApprovedOrder van de risicopoort aan")
        p = approved.proposal
        require(order.legs == p.legs and order.quantity == p.quantity,
                "order wijkt af van de goedgekeurde order", DataContractError,
                client_id=order.client_id)
        require(order.client_id not in self._orders, "client-id al gebruikt",
                DataContractError, client_id=order.client_id)
        require_paper(self.account_id)
        trade = self.ib.placeOrder(await self._bag(order), self._order(order))
        self._trades[order.client_id] = trade
        self._orders[order.client_id] = BrokerOrder(order, OrderStatus.SUBMITTED)
        return self._refresh(order.client_id)

    async def modify(self, client_id: str, new_limit: float) -> BrokerOrder:
        bo = self._refresh(client_id)
        require(bo.status is OrderStatus.SUBMITTED, "alleen een open order wijzigen",
                DataContractError, client_id=client_id)
        trade = self._trades[client_id]
        trade.order.lmtPrice = round(new_limit, 2)
        self._trades[client_id] = self.ib.placeOrder(trade.contract, trade.order)
        self._orders[client_id] = replace(bo, order=replace(bo.order, limit=new_limit))
        return self._refresh(client_id)

    async def cancel(self, client_id: str) -> None:
        if self._refresh(client_id).status is OrderStatus.SUBMITTED:
            self.ib.cancelOrder(self._trades[client_id].order)

    async def what_if(self, order: ComboOrder) -> MarginImpact | None:
        st = await self.ib.whatIfOrderAsync(await self._bag(order), self._order(order))
        init, maint = _num(st.initMarginChange), _num(st.maintMarginChange)
        if not (math.isfinite(init) and math.isfinite(maint)):
            return None
        return MarginImpact(init, maint)

    async def executions(self, since: datetime) -> list[Execution]:
        flt = ExecutionFilter(acctCode=self.account_id,
                              time=since.astimezone(UTC).strftime("%Y%m%d-%H:%M:%S"))
        out = []
        for f in await self.ib.reqExecutionsAsync(flt):
            c, ex = f.contract, f.execution
            if c.secType != "OPT":
                continue
            qty = int(ex.shares) * (1 if ex.side == "BOT" else -1)
            fee = _num(getattr(f.commissionReport, "commission", 0.0))
            out.append(Execution(f.time, ExecutionKind.FILL, ex.orderRef or "", contract_of(c),
                                 c.symbol, qty, float(ex.price),
                                 fee if math.isfinite(fee) else 0.0))
        return sorted(out, key=lambda e: (e.ts, e.client_id, e.contract.key()))  # type: ignore[union-attr]
