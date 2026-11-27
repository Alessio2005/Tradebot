"""De paper-broker voor replay en tests (detailontwerp 10.5). Fictief geld, geen netwerk.

* Deterministisch: geen toeval.
* Fill: een order vult volledig zodra `|limit − mid| ≥ fill_at_concession × |natural − mid|`,
  tegen de limiet. Geen gedeeltelijke fills. Commissie per contract per leg.
* Aandelen (`submit_stock`, de delta-hedger): een order vult volledig op zijn limiet zodra die
  de spot ± de halve `stock_spread_bps` raakt (kopen: limiet ≥ spot × (1 + h), verkopen:
  limiet ≤ spot × (1 − h)). Commissie `commission_per_share` per aandeel, minimaal
  `min_commission` per order.
* Expiratie (`settle`, na de sluiting van de expiratiedag): een optie die ≥ 0,01 ITM sluit,
  wordt uitgeoefend (long) of assigned (short), en wordt aandelen; de rest vervalt.
* Vroege assignment (`early_assign`, bij de opening van een ex-dividenddag): een short call die
  ITM is en waarvan de extrinsieke waarde kleiner is dan het dividend.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from datetime import date, datetime

from ...utils.failfast import DataContractError, require
from ..calendar import expiry_moment
from ..chain.builder import Chain
from ..contracts import OptionContract, Right
from ..positions import MULTIPLIER
from ..rates import CashDividend
from ..risk.gate import ApprovedOrder, ApprovedStockOrder
from .broker import (
    AccountSnapshot,
    BrokerOrder,
    BrokerPosition,
    Execution,
    ExecutionKind,
    MarginImpact,
)
from .orders import TERMINAL, ComboOrder, OrderStatus, StockOrder, combo_prices

__all__ = ["PaperBroker", "settlement_kind"]

ITM_MIN = 0.01


def settlement_kind(c: OptionContract, spot: float, qty: int) -> ExecutionKind:
    """Hoe een leg van `qty` op `c` na de sluiting van de expiratiedag afrekent bij slotkoers
    `spot`: ≥ 0,01 ITM uitgeoefend (long) of assigned (short), anders vervallen."""
    itm = spot - c.strike if c.right is Right.CALL else c.strike - spot
    if itm < ITM_MIN:
        return ExecutionKind.EXPIRY
    return ExecutionKind.EXERCISE if qty > 0 else ExecutionKind.ASSIGNMENT


class PaperBroker:
    def __init__(self, cash: float, fill_at_concession: float = 0.40,
                 commission_per_contract: float = 0.65, account_hash: str = "paper",
                 stock_spread_bps: float = 2.0, commission_per_share: float = 0.005,
                 min_commission: float = 1.0) -> None:
        self.cash = float(cash)
        self.fill_at = fill_at_concession
        self.commission = commission_per_contract
        self.stock_half_spread = stock_spread_bps / 2e4
        self.commission_per_share, self.min_commission = commission_per_share, min_commission
        self.account_hash = account_hash
        self.now: datetime | None = None
        self.chains: Mapping[str, Chain] = {}
        self._orders: dict[str, BrokerOrder] = {}
        self._options: dict[str, tuple[OptionContract, int, float]] = {}
        self._shares: dict[str, tuple[int, float]] = {}
        self._executions: list[Execution] = []
        self._fills = 0                                 # teller voor `exec_id`

    # --- markt ----------------------------------------------------------------------------

    def update_market(self, now: datetime, chains: Mapping[str, Chain]) -> None:
        self.now, self.chains = now, chains
        for cid, bo in list(self._orders.items()):
            if bo.status is OrderStatus.SUBMITTED:
                self._try_fill(cid)

    def _try_fill(self, cid: str) -> None:
        bo = self._orders[cid]
        o = bo.order
        if isinstance(o, StockOrder):
            self._try_fill_stock(bo, o)
            return
        und = o.legs[0].contract.underlying
        chain = self.chains.get(und)
        prices = combo_prices(chain, o.legs) if chain is not None else None
        if prices is None:
            return
        mid, natural = prices
        if abs(o.limit - mid) + 1e-9 < self.fill_at * abs(natural - mid):
            return
        if (natural >= mid and o.limit < mid - 1e-9) or (natural < mid and o.limit > mid + 1e-9):
            return                                     # limiet aan de verkeerde kant van mid
        self._fill(bo, o)

    def _fill(self, bo: BrokerOrder, o: ComboOrder) -> None:
        now = self.now
        assert now is not None
        n_contracts = sum(abs(leg.ratio) for leg in o.legs) * o.quantity
        fee = self.commission * n_contracts
        self.cash -= o.limit * MULTIPLIER * o.quantity + fee
        for i, leg in enumerate(o.legs):
            c = leg.contract
            qty = leg.ratio * o.quantity
            o_px = self._leg_price(c)
            self._book_option(c, qty, o_px)
            self._executions.append(Execution(now, ExecutionKind.FILL, o.client_id, c,
                                              c.underlying, qty, o_px,
                                              fee if i == 0 else 0.0,
                                              self._exec_id(o.client_id)))
        self._orders[o.client_id] = dataclasses.replace(bo, status=OrderStatus.FILLED,
                                                        filled=o.quantity, avg_fill=o.limit)

    def _try_fill_stock(self, bo: BrokerOrder, o: StockOrder) -> None:
        chain = self.chains.get(o.symbol)
        if chain is None:
            return
        h = self.stock_half_spread
        if (o.qty > 0 and o.limit < chain.spot * (1 + h) - 1e-9) or (
                o.qty < 0 and o.limit > chain.spot * (1 - h) + 1e-9):
            return
        assert self.now is not None
        fee = max(self.commission_per_share * abs(o.qty), self.min_commission)
        self._book_shares(o.symbol, o.qty, o.limit)
        self.cash -= fee
        self._executions.append(Execution(self.now, ExecutionKind.FILL, o.client_id, None,
                                          o.symbol, o.qty, o.limit, fee,
                                          self._exec_id(o.client_id)))
        self._orders[o.client_id] = dataclasses.replace(bo, status=OrderStatus.FILLED,
                                                        filled=abs(o.qty), avg_fill=o.limit)

    def _exec_id(self, cid: str) -> str:
        self._fills += 1
        return f"{cid}#{self._fills}"

    def _leg_price(self, c: OptionContract) -> float:
        chain = self.chains.get(c.underlying)
        o = chain.find(c.expiry, c.strike, c.right) if chain is not None else None
        return o.quote.mid if o is not None and o.quote.mid is not None else 0.0

    def _book_option(self, c: OptionContract, qty: int, px: float) -> None:
        k = c.key()
        _, old, avg = self._options.get(k, (c, 0, 0.0))
        new = old + qty
        if new == 0:
            self._options.pop(k, None)
        else:
            self._options[k] = (c, new, px if old == 0 else avg)

    def _book_shares(self, symbol: str, qty: int, px: float) -> None:
        old, avg = self._shares.get(symbol, (0, 0.0))
        new = old + qty
        self.cash -= qty * px
        if new == 0:
            self._shares.pop(symbol, None)
        else:
            self._shares[symbol] = (new, px if old == 0 else avg)

    # --- expiratie en assignment ----------------------------------------------------------

    def settle(self, day: date, closes: Mapping[str, float], ts: datetime) -> list[Execution]:
        """Verwerk alle opties die op `day` expireren tegen de slotkoersen."""
        out = []
        for k, (c, qty, _) in sorted(self._options.items()):
            if c.expiry != day:
                continue
            kind = settlement_kind(c, closes[c.underlying], qty)
            del self._options[k]
            if kind is ExecutionKind.EXPIRY:
                out.append(Execution(ts, ExecutionKind.EXPIRY, "", c, c.underlying, -qty, 0.0))
                continue
            shares = qty * MULTIPLIER * (1 if c.right is Right.CALL else -1)
            self._book_shares(c.underlying, shares, c.strike)
            out.append(Execution(ts, kind, "", c, c.underlying, -qty, c.strike))
        self._executions.extend(out)
        return out

    def early_assign(self, dividends: Mapping[str, CashDividend], ts: datetime) -> list[Execution]:
        """Bij de opening van een ex-dividenddag: short calls met extrinsiek < dividend."""
        out = []
        for k, (c, qty, _) in sorted(self._options.items()):
            div = dividends.get(c.underlying)
            chain = self.chains.get(c.underlying)
            if qty >= 0 or c.right is not Right.CALL or div is None or chain is None:
                continue
            o = chain.find(c.expiry, c.strike, c.right)
            if o is None or o.quote.mid is None or chain.spot <= c.strike:
                continue
            if o.quote.mid - (chain.spot - c.strike) < div.amount:
                del self._options[k]
                self._book_shares(c.underlying, qty * MULTIPLIER, c.strike)
                out.append(Execution(ts, ExecutionKind.ASSIGNMENT, "", c, c.underlying, -qty,
                                     c.strike))
        self._executions.extend(out)
        return out

    # --- BrokerAdapter --------------------------------------------------------------------

    async def connect(self) -> None:
        return None

    async def account(self) -> AccountSnapshot:
        equity = self.cash
        for c, qty, _ in self._options.values():
            equity += qty * self._leg_price(c) * MULTIPLIER
        for sym, (qty, avg) in self._shares.items():
            chain = self.chains.get(sym)
            equity += qty * (chain.spot if chain is not None else avg)
        return AccountSnapshot(equity, self.cash, self.cash, True, self.account_hash)

    async def positions(self) -> list[BrokerPosition]:
        out = [BrokerPosition(c, c.underlying, qty, avg)
               for _, (c, qty, avg) in sorted(self._options.items())]
        out += [BrokerPosition(None, s, q, a) for s, (q, a) in sorted(self._shares.items())]
        return out

    async def open_orders(self) -> list[BrokerOrder]:
        return [bo for bo in self._orders.values() if bo.status not in TERMINAL]

    async def submit(self, approved: ApprovedOrder, order: ComboOrder) -> BrokerOrder:
        if not isinstance(approved, ApprovedOrder):
            raise TypeError("de paper-broker neemt alleen een ApprovedOrder van de risicopoort aan")
        p = approved.proposal
        require(order.legs == p.legs and order.quantity == p.quantity,
                "order wijkt af van de goedgekeurde order", DataContractError,
                client_id=order.client_id)
        require(order.client_id not in self._orders, "client-id al gebruikt",
                DataContractError, client_id=order.client_id)
        require(self.now is not None and all(self.now < expiry_moment(leg.contract.expiry)
                                              for leg in order.legs),
                "verlopen contract", DataContractError, client_id=order.client_id)
        self._orders[order.client_id] = BrokerOrder(order, OrderStatus.SUBMITTED)
        self._try_fill(order.client_id)
        return self._orders[order.client_id]

    async def submit_stock(self, approved: ApprovedStockOrder,
                           order: StockOrder) -> BrokerOrder:
        if not isinstance(approved, ApprovedStockOrder):
            raise TypeError("de paper-broker neemt alleen een ApprovedStockOrder aan")
        p = approved.proposal
        require(order.symbol == p.symbol and order.qty == p.qty and order.limit == p.limit,
                "order wijkt af van de goedgekeurde order", DataContractError,
                client_id=order.client_id)
        require(order.client_id not in self._orders, "client-id al gebruikt",
                DataContractError, client_id=order.client_id)
        self._orders[order.client_id] = BrokerOrder(order, OrderStatus.SUBMITTED)
        self._try_fill(order.client_id)
        return self._orders[order.client_id]

    async def modify(self, client_id: str, new_limit: float) -> BrokerOrder:
        bo = self._orders[client_id]
        require(bo.status is OrderStatus.SUBMITTED, "alleen een open order wijzigen",
                DataContractError, client_id=client_id)
        self._orders[client_id] = dataclasses.replace(
            bo, order=dataclasses.replace(bo.order, limit=new_limit))
        self._try_fill(client_id)
        return self._orders[client_id]

    async def cancel(self, client_id: str) -> None:
        bo = self._orders[client_id]
        if bo.status not in TERMINAL:
            self._orders[client_id] = dataclasses.replace(bo, status=OrderStatus.CANCELLED)

    async def what_if(self, order: ComboOrder) -> MarginImpact | None:
        return None

    async def executions(self, since: datetime) -> list[Execution]:
        return [e for e in self._executions if e.ts >= since]

    def order(self, client_id: str) -> BrokerOrder:
        return self._orders[client_id]
