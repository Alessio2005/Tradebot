"""Een nep-`ib_async.IB` voor de IBKR-tests: geen netwerk, alles in het geheugen."""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from types import SimpleNamespace as NS
from typing import Any

from tests.unit.options.market_helpers import EXPIRIES, STRIKES, cached_chain
from tradebot.options.contracts import Right


class FakeIB:
    def __init__(self, accounts=("DU1234567",), spot=100.0, now=None):
        self.accounts, self.spot, self.connected = list(accounts), spot, False
        self.chain = cached_chain(spot)
        self.now = now
        self.data_type = None
        self.placed: list[tuple[Any, Any]] = []
        self.cancelled: list[Any] = []
        self.fills: list[Any] = []
        self.status = "Submitted"
        self.summary = [NS(tag="NetLiquidation", value="25000", currency="USD"),
                        NS(tag="TotalCashValue", value="24000", currency="USD"),
                        NS(tag="BuyingPower", value="50000", currency="USD"),
                        NS(tag="NetLiquidation", value="1", currency="EUR")]
        self.values: list[Any] = []
        self.held: list[Any] = []
        self.lines = 0
        self.max_lines = 0
        self._ids = itertools.count(1000)
        self.what_if = NS(initMarginChange="500.0", maintMarginChange="400.0")
        self.unknown: set[float] = set()

    # verbinding
    def isConnected(self):
        return self.connected

    async def connectAsync(self, host, port, clientId, account=""):
        self.connected = True
        self.args = (host, port, clientId, account)

    def managedAccounts(self):
        return self.accounts

    def reqMarketDataType(self, t):
        self.data_type = t

    # account
    async def accountSummaryAsync(self, account=""):
        return self.summary

    def accountValues(self, account=""):
        return self.values

    def positions(self, account=""):
        return self.held

    # contracten
    async def qualifyContractsAsync(self, *cs):
        out = []
        for c in cs:
            if getattr(c, "strike", 0) in self.unknown:
                out.append(None)
                continue
            c.conId = next(self._ids)
            out.append(c)
        return out

    # orders
    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        return NS(contract=contract, order=order,
                  orderStatus=NS(status=self.status, filled=0, avgFillPrice=0.0))

    def cancelOrder(self, order):
        self.cancelled.append(order)

    async def whatIfOrderAsync(self, contract, order):
        return self.what_if

    async def reqExecutionsAsync(self, flt=None):
        self.filter = flt
        return self.fills

    # marktdata
    def reqMktData(self, c, generic=""):
        self.lines += 1
        self.max_lines = max(self.max_lines, self.lines)
        if c.secType == "STK":
            return NS(contract=c, last=self.spot, close=self.spot - 1)
        right = Right.CALL if c.right == "C" else Right.PUT
        exp = datetime.strptime(c.lastTradeDateOrContractMonth, "%Y%m%d").date()
        o = self.chain.find(exp, c.strike, right)
        q = o.quote if o is not None else NS(bid=None, ask=None)
        return NS(contract=c, bid=q.bid, ask=q.ask, last=float("nan"), bidSize=10, askSize=10,
                  volume=500, callOpenInterest=5000, putOpenInterest=6000)

    def cancelMktData(self, c):
        self.lines -= 1

    async def reqSecDefOptParamsAsync(self, name, fut, sec, conid):
        return [NS(exchange="CBOE", expirations=[], strikes=[], tradingClass=name),
                NS(exchange="SMART", tradingClass=name,
                   expirations=[e.strftime("%Y%m%d") for e in EXPIRIES] + ["20300101"],
                   strikes=list(STRIKES))]

    async def reqHistoricalDataAsync(self, c, end, dur, size, what, rth):
        start = datetime(2026, 1, 2)
        return [NS(date=(start + timedelta(days=i)).date(), open=100.0, high=101.0, low=99.0,
                   close=100.0 + (i % 5) * 0.1) for i in range(300)]
