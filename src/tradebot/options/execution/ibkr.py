"""De IBKR-adapter (detailontwerp 10.6) via `ib_async`: alleen een paper-account.

* `connect` maakt verbinding met TWS of IB Gateway, kiest het account en weigert alles wat
  geen paper-account is (`live.require_paper`). Live staat uit.
* Een order is altijd een combo (`BAG`), ook met één leg, met `orderRef` = de client-id. De
  limiet heeft het teken van de broker: + betalen, − ontvangen. Actie is altijd `BUY`.
* Een aandelenorder (de delta-hedger) is `Stock(symbool, "SMART", "USD")` met een gewone
  `LMT`-order (`BUY` of `SELL`, DAY), ook met `orderRef` = de client-id.
* `modify` verandert alleen `lmtPrice`; legs en aantal liggen vast.
* Executies komen per leg, en per aandelenfill (zonder contract); de `BAG`-regel zelf wordt
  overgeslagen.
* Equity, cash en buying power zijn altijd in USD. Een account met een andere basisvaluta
  (bv. EUR) wordt omgerekend met de IDEALPRO-koers van die valuta tegen USD.

De `ib`-parameter is een `ib_async.IB` of een nep-object met dezelfde methoden (tests).
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from ib_async import ComboLeg, Contract, ExecutionFilter, Forex, LimitOrder, Option, Stock

from ...utils.failfast import DataContractError, require
from ..contracts import OptionContract, Right
from ..live import account_hash, require_paper
from ..risk.gate import ApprovedOrder, ApprovedStockOrder
from .broker import (
    AccountSnapshot,
    BrokerOrder,
    BrokerPosition,
    Execution,
    ExecutionKind,
    MarginImpact,
)
from .orders import ComboOrder, OrderStatus, StockOrder

__all__ = ["IBKRBroker", "contract_of", "option_of"]

_STATUS = {"PendingSubmit": OrderStatus.SUBMITTED, "PreSubmitted": OrderStatus.SUBMITTED,
           "Submitted": OrderStatus.SUBMITTED, "ApiPending": OrderStatus.SUBMITTED,
           "Filled": OrderStatus.FILLED, "Cancelled": OrderStatus.CANCELLED,
           "ApiCancelled": OrderStatus.CANCELLED, "PendingCancel": OrderStatus.SUBMITTED,
           "Inactive": OrderStatus.REJECTED}
_ACCOUNT_TAGS = ("NetLiquidation", "TotalCashValue", "BuyingPower")
_USD = ("USD", "BASE", "")
_FIRST_YEAR = 2000                  # een `since` daarvoor betekent: geen tijdfilter
# Na een herstart van de pc start de gateway trager dan de dienst: zolang inloggen duurt,
# geeft de verbinding een time-out. Opnieuw proberen in plaats van crashen.
CONNECT_ATTEMPTS = 12
CONNECT_WAIT_SECONDS = 10.0

log = logging.getLogger(__name__)


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
                 client_id: int = 17, account: str = "",
                 connect_attempts: int = CONNECT_ATTEMPTS,
                 connect_wait: float = CONNECT_WAIT_SECONDS,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.ib, self.host, self.port, self.client_id = ib, host, port, client_id
        self.account_id = account
        self.connect_attempts, self.connect_wait = connect_attempts, connect_wait
        self._sleep = sleep
        self._trades: dict[str, Any] = {}
        self._orders: dict[str, BrokerOrder] = {}

    @property
    def account_hash(self) -> str:
        return account_hash(self.account_id)

    async def connect(self) -> None:
        for attempt in range(1, self.connect_attempts + 1):
            if self.ib.isConnected():
                break
            try:
                await self.ib.connectAsync(self.host, self.port, self.client_id,
                                           account=self.account_id)
                break
            except (TimeoutError, OSError) as e:        # ConnectionRefusedError is een OSError
                if attempt == self.connect_attempts:
                    raise
                log.warning("IB Gateway nog niet klaar (poging %d van %d): %r",
                            attempt, self.connect_attempts, e)
                self.ib.disconnect()
                await self._sleep(self.connect_wait)
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
        rows = [v for v in await self.ib.accountSummaryAsync(self.account_id)
                if v.tag in _ACCOUNT_TAGS]
        usd = [v for v in rows if v.currency in _USD]
        fx = 1.0
        if not usd:                                 # account met een andere basisvaluta (EUR)
            ccys = sorted({v.currency for v in rows})
            require(len(ccys) == 1, "geen equity of cash van IBKR", DataContractError,
                    valuta=ccys)
            fx, usd = await self._usd_per(ccys[0]), rows
        vals = {v.tag: _num(v.value) * fx for v in usd}
        eq, cash = vals.get("NetLiquidation", math.nan), vals.get("TotalCashValue", math.nan)
        bp = vals.get("BuyingPower", math.nan)
        require(math.isfinite(eq) and math.isfinite(cash), "geen equity of cash van IBKR",
                DataContractError)
        return AccountSnapshot(eq, cash, bp if math.isfinite(bp) else 0.0, True,
                               self.account_hash)

    async def _usd_per(self, base: str) -> float:
        """USD per eenheid `base`, uit de IDEALPRO-koers `base`.USD (EUR, GBP, AUD, NZD)."""
        pair = (await self.ib.qualifyContractsAsync(Forex(base + "USD")))[0]
        require(pair is not None, "geen wisselkoers voor de basisvaluta", DataContractError,
                valuta=base)
        ticker = (await self.ib.reqTickersAsync(pair))[0]
        rates = [r for r in (_num(ticker.marketPrice()), _num(ticker.close))
                 if math.isfinite(r) and r > 0]
        require(bool(rates), "geen wisselkoers voor de basisvaluta", DataContractError,
                valuta=base)
        return rates[0]

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

    async def submit_stock(self, approved: ApprovedStockOrder,
                           order: StockOrder) -> BrokerOrder:
        if not isinstance(approved, ApprovedStockOrder):
            raise TypeError("IBKR neemt alleen een ApprovedStockOrder van de risicopoort aan")
        p = approved.proposal
        require(order.symbol == p.symbol and order.qty == p.qty and order.limit == p.limit,
                "order wijkt af van de goedgekeurde order", DataContractError,
                client_id=order.client_id)
        require(order.client_id not in self._orders, "client-id al gebruikt",
                DataContractError, client_id=order.client_id)
        require_paper(self.account_id)
        (stk,) = await self.ib.qualifyContractsAsync(Stock(order.symbol, "SMART", "USD"))
        require(stk is not None and bool(getattr(stk, "conId", 0)), "IBKR kent het aandeel niet",
                DataContractError, client_id=order.client_id)
        lmt = LimitOrder("BUY" if order.qty > 0 else "SELL", abs(order.qty),
                         round(order.limit, 2), orderRef=order.client_id, tif=order.tif,
                         account=self.account_id)
        self._trades[order.client_id] = self.ib.placeOrder(stk, lmt)
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
        # De eerste ronde vraagt "alles" (`datetime.min`). Jaar 1 is geen geldige IBKR-tijd
        # (fout 10314, en dan géén executies); zonder tijd geeft IBKR alles wat hij heeft.
        stamp = (since.astimezone(UTC).strftime("%Y%m%d-%H:%M:%S")
                 if since.year >= _FIRST_YEAR else "")
        flt = ExecutionFilter(acctCode=self.account_id, time=stamp)
        out = []
        for f in await self.ib.reqExecutionsAsync(flt):
            c, ex = f.contract, f.execution
            if c.secType not in ("OPT", "STK"):
                continue
            qty = int(ex.shares) * (1 if ex.side == "BOT" else -1)
            fee = _num(getattr(f.commissionReport, "commission", 0.0))
            out.append(Execution(f.time, ExecutionKind.FILL, ex.orderRef or "",
                                 contract_of(c) if c.secType == "OPT" else None,
                                 c.symbol, qty, float(ex.price),
                                 fee if math.isfinite(fee) else 0.0))
        return sorted(out, key=lambda e: (e.ts, e.client_id,
                                          e.contract.key() if e.contract else e.symbol))
