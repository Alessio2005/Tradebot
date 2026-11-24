"""Cash-secured put en covered call (detailontwerp 7.4). Standaard uit (open vraag C).

`max_loss` is in v1 het theoretische maximum `strike × 100 × aantal`. Bij 25k komen ze daardoor
niet door de sizing: dat is bedoeld, tot open vraag C anders beslist.

Beheer, in deze volgorde:

1. handelsdag vóór expiratie, na `close_before_expiry_time` → CLOSE, STOP;
2. (covered call) ITM, ex-dividend op de volgende handelsdag en extrinsieke waarde kleiner dan
   het dividend → CLOSE, STOP (vroege assignment voorkomen);
3. kosten ≤ (1 − `take_profit_frac`) × credit → CLOSE, NORMAL;
4. DTE ≤ `roll_dte` → ROLL naar de expiratie die `dte` het best haalt, op dezelfde delta, als
   de nieuwe premie de kosten van sluiten overtreft; anders CLOSE, NORMAL.
"""
from __future__ import annotations

from collections.abc import Collection

from ...schemas.options import CashSecuredPutConfig, CoveredCallConfig
from ..chain.builder import Chain, PricedOption
from ..contracts import OptionContract, Right
from ..iv_metrics import IVCondition
from ..positions import (
    MULTIPLIER,
    Action,
    ActionKind,
    Book,
    Candidate,
    LegIntent,
    MarketState,
    PositionStatus,
    StrategyPosition,
    Urgency,
    mark,
)
from .common import (
    Rules,
    close,
    dte,
    entry_blockers,
    expiry_close_due,
    iv_ok,
    pick_expiry,
    size,
    strike_at_delta,
)

__all__ = ["CashSecuredPut", "CoveredCall"]


def _roll_or_close(pos: StrategyPosition, market: MarketState, chain: Chain, cost: float,
                   dte_range: tuple[int, int], target: float, band: tuple[float, float],
                   min_strike: float = 0.0) -> Action:
    (leg,) = pos.intents()
    old = leg.contract
    later = Chain(chain.underlying, chain.spot, chain.asof, chain.carry,
                  tuple(o for o in chain.options if o.contract.expiry > old.expiry))
    expiry = pick_expiry(later, *dte_range, market.today)
    new: PricedOption | None = None
    if expiry is not None:
        rows = Chain(later.underlying, later.spot, later.asof, later.carry,
                     tuple(o for o in later.by_expiry(expiry, old.right)
                           if o.contract.strike >= min_strike))
        new = strike_at_delta(rows, expiry, old.right, target, band)
    if new is not None and new.quote.mid is not None and new.quote.mid > cost:
        return Action(ActionKind.ROLL, pos.position_id, (leg,), Urgency.NORMAL, "roll",
                      (LegIntent(new.contract, -1),))
    return close(pos, Urgency.NORMAL, "dte")


class CashSecuredPut:
    name = "cash_secured_put"

    def __init__(self, cfg: CashSecuredPutConfig, rules: Rules,
                 share_eligible: Collection[str]) -> None:
        self.cfg, self.rules, self.eligible = cfg, rules, frozenset(share_eligible)

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(set(market.chains) & self.eligible):
            chain = market.chains[und]
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            ok, trace = iv_ok(market, und, IVCondition(cfg.iv_condition), self.rules)
            put = strike_at_delta(chain, expiry, Right.PUT, cfg.delta_target, cfg.delta)
            if not ok or put is None or put.quote.mid is None:
                continue
            loss_unit = put.contract.strike * MULTIPLIER
            units = min(size(loss_unit, market, book, und, self.rules),
                        int(book.cash_usd // loss_unit))
            if units <= 0:
                continue
            credit = put.quote.mid
            out.append(Candidate(self.name, und, (LegIntent(put.contract, -1),), units, credit,
                                 loss_unit * units, credit * MULTIPLIER / loss_unit,
                                 (trace, f"dte:{dte(expiry, market.today)}",
                                  f"cash_reserved:{loss_unit * units:g}")))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        cfg = self.cfg
        (leg,) = pos.intents()
        if expiry_close_due(leg.contract.expiry, market.now, market, self.rules):
            return [close(pos, Urgency.STOP, "expiry")]
        chain = market.chains.get(pos.underlying)
        if chain is None:
            return []
        m = mark(chain, (leg,))
        if m is None:
            return []
        cost = -m
        if cost <= (1 - cfg.take_profit_frac) * pos.entry_credit:
            return [close(pos, Urgency.NORMAL, "take_profit")]
        if dte(leg.contract.expiry, market.today) <= cfg.roll_dte:
            return [_roll_or_close(pos, market, chain, cost, cfg.dte, cfg.delta_target,
                                   cfg.delta)]
        return []


class CoveredCall:
    name = "covered_call"

    def __init__(self, cfg: CoveredCallConfig, rules: Rules,
                 share_eligible: Collection[str]) -> None:
        self.cfg, self.rules, self.eligible = cfg, rules, frozenset(share_eligible)

    def _covered(self, book: Book, und: str) -> int:
        return sum(-leg.qty for p in book.positions
                   if p.underlying == und and p.status is not PositionStatus.CLOSED
                   for leg in p.option_legs
                   if leg.qty < 0 and leg.contract.right is Right.CALL)  # type: ignore[union-attr]

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(set(market.chains) & self.eligible):
            free = book.shares.get(und, 0) // MULTIPLIER - self._covered(book, und)
            if free <= 0:
                continue
            chain = market.chains[und]
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            basis = book.cost_basis.get(und, 0.0)
            above = Chain(und, chain.spot, chain.asof, chain.carry,
                          tuple(o for o in chain.options if o.contract.strike >= basis))
            call = strike_at_delta(above, expiry, Right.CALL, cfg.delta_target, cfg.delta)
            if call is None or call.quote.mid is None:
                continue
            loss_unit = call.contract.strike * MULTIPLIER
            units = min(free, size(loss_unit, market, book, und, self.rules))
            if units <= 0:
                continue
            credit = call.quote.mid
            out.append(Candidate(self.name, und, (LegIntent(call.contract, -1),), units, credit,
                                 loss_unit * units, credit * MULTIPLIER / loss_unit,
                                 (f"basis:{basis:g}", f"dte:{dte(expiry, market.today)}")))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def _early_assignment(self, c: OptionContract, chain: Chain, market: MarketState) -> bool:
        o = chain.find(c.expiry, c.strike, c.right)
        ev = market.events.get(c.underlying)
        if o is None or o.quote.mid is None or ev is None or chain.spot <= c.strike:
            return False
        tomorrow = market.calendar.next_trading_day(market.today)
        div = sum(d.amount for d in ev.dividends if d.ex_date == tomorrow)
        return div > 0 and o.quote.mid - (chain.spot - c.strike) < div

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        cfg = self.cfg
        (leg,) = pos.intents()
        if expiry_close_due(leg.contract.expiry, market.now, market, self.rules):
            return [close(pos, Urgency.STOP, "expiry")]
        chain = market.chains.get(pos.underlying)
        if chain is None:
            return []
        if self._early_assignment(leg.contract, chain, market):
            return [close(pos, Urgency.STOP, "early_assignment")]
        m = mark(chain, (leg,))
        if m is None:
            return []
        cost = -m
        if cost <= (1 - cfg.take_profit_frac) * pos.entry_credit:
            return [close(pos, Urgency.NORMAL, "take_profit")]
        if dte(leg.contract.expiry, market.today) <= cfg.roll_dte:
            return [_roll_or_close(pos, market, chain, cost, cfg.dte, cfg.delta_target,
                                   cfg.delta, min_strike=leg.contract.strike)]
        return []
