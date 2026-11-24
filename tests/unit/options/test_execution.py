"""Combo-orders, de price-walk en de paper-broker (fictief geld, geen netwerk)."""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import date, datetime, timedelta

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CFG,
    SETTINGS,
    cached_chain,
    fake,
    fake_chain,
    market,
)
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.execution.broker import ExecutionKind
from tradebot.options.execution.orders import (
    ComboOrder,
    OrderStatus,
    Purpose,
    client_id,
    combo_prices,
    position_id,
    walk_limit,
)
from tradebot.options.execution.paper import PaperBroker
from tradebot.options.positions import Book, LegIntent, Trend, Urgency
from tradebot.options.pricing.engines import Method
from tradebot.options.rates import CashDividend
from tradebot.options.risk.gate import GateState, OrderProposal, RiskGate, Verdict
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.credit import CreditSpread
from tradebot.utils.failfast import DataContractError

GATE = RiskGate(CFG, SETTINGS.engines, Method.BAW)
E = date(2026, 12, 31)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(scope="module")
def spread():
    (c,) = CreditSpread(CFG.strategies.credit_spread, rules_from_config(CFG)).scan(
        market(trend=Trend.UP), Book())
    return c


def approved(c, pid="p1", **kw):
    d = GATE.review(dataclasses.replace(OrderProposal.open(c, pid), **kw), Book(), market(),
                    GateState(equity=25_000.0))
    assert d.verdict is Verdict.APPROVE
    return d.approved


def order_for(ap, limit, cid="c1"):
    p = ap.proposal
    return ComboOrder(cid, p.position_id, p.legs, p.quantity, limit, Purpose.ENTRY,
                      Urgency.NORMAL)


# --- orders --------------------------------------------------------------------------------

def test_ids_are_stable_and_distinct():
    legs = (LegIntent(OptionContract("TST", E, 95.0, Right.PUT), -1),
            LegIntent(OptionContract("TST", E, 90.0, Right.PUT), 1))
    a = position_id("credit_spread", "TST", legs, NOW.date())
    assert a == position_id("credit_spread", "TST", legs[::-1], NOW.date()) and len(a) == 12
    assert a != position_id("credit_spread", "TST", legs, date(2026, 11, 25))
    c0 = client_id(a, Purpose.ENTRY, NOW.date(), 0)
    assert c0 != client_id(a, Purpose.ENTRY, NOW.date(), 1) and len(c0) == 16


def test_combo_prices_and_the_walk():
    ch = fake_chain(fake(95.0, Right.PUT, mid=2.0), fake(90.0, Right.PUT, mid=1.0))
    short = (LegIntent(OptionContract("TST", E, 95.0, Right.PUT), -1),
             LegIntent(OptionContract("TST", E, 90.0, Right.PUT), 1))
    mid, natural = combo_prices(ch, short)
    assert mid == pytest.approx(-1.0) and natural == pytest.approx(-0.98)
    assert combo_prices(ch, (LegIntent(OptionContract("TST", E, 80.0, Right.PUT), 1),)) is None
    # Credit: begin op mid, loop richting natural (omhoog), stop na de concessie.
    s0 = walk_limit(-1.005, -0.98, 0, 0.01, 0.6)
    assert s0.limit == pytest.approx(-1.01) and not s0.done
    assert walk_limit(-1.005, -0.98, 1, 0.01, 0.6).limit == pytest.approx(-1.0)
    assert walk_limit(-1.005, -0.98, 3, 0.01, 0.6).done
    # Debit: begin onder mid en loop omhoog naar de ask.
    d0 = walk_limit(1.005, 1.03, 0, 0.01, 1.0)
    assert d0.limit == pytest.approx(1.0)
    assert walk_limit(1.005, 1.03, 2, 0.01, 1.0).limit == pytest.approx(1.02)


# --- paper-broker --------------------------------------------------------------------------

def test_submit_needs_an_approved_order_that_matches(spread):
    b = PaperBroker(25_000.0)
    ap = approved(spread)
    with pytest.raises(TypeError, match="ApprovedOrder"):
        run(b.submit(ap.proposal, order_for(ap, 0.0)))     # type: ignore[arg-type]
    b.update_market(NOW, {"TST": cached_chain()})
    with pytest.raises(DataContractError, match="wijkt af"):
        run(b.submit(ap, dataclasses.replace(order_for(ap, 0.0), quantity=99)))
    run(b.submit(ap, order_for(ap, -5.0)))
    with pytest.raises(DataContractError, match="al gebruikt"):
        run(b.submit(ap, order_for(ap, -5.0)))
    late = PaperBroker(1.0)
    late.update_market(datetime(2027, 1, 1, 10, tzinfo=NY), {})
    with pytest.raises(DataContractError, match="verlopen"):
        run(late.submit(ap, order_for(ap, 0.0)))


def test_an_order_fills_once_the_limit_has_walked_far_enough(spread):
    b = PaperBroker(25_000.0, fill_at_concession=0.4, commission_per_contract=0.65)
    ch = cached_chain()
    b.update_market(NOW, {"TST": ch})
    ap = approved(spread)
    mid, natural = combo_prices(ch, ap.proposal.legs)
    bo = run(b.submit(ap, order_for(ap, mid - 0.05)))      # aan de verkeerde kant van mid
    assert bo.status is OrderStatus.SUBMITTED
    bo = run(b.modify("c1", mid))                          # op mid: nog geen concessie
    assert bo.status is OrderStatus.SUBMITTED
    bo = run(b.modify("c1", natural))
    assert bo.status is OrderStatus.FILLED and bo.avg_fill == natural
    with pytest.raises(DataContractError, match="open order"):
        run(b.modify("c1", natural))
    run(b.cancel("c1"))
    assert b.order("c1").status is OrderStatus.FILLED
    q = ap.proposal.quantity
    fills = run(b.executions(NOW))
    assert [e.kind for e in fills] == [ExecutionKind.FILL] * 2
    assert sum(e.commission for e in fills) == pytest.approx(0.65 * 2 * q)
    assert b.cash == pytest.approx(25_000.0 - natural * 100 * q - 0.65 * 2 * q)
    pos = run(b.positions())
    assert sorted(p.qty for p in pos) == [-q, q] and run(b.open_orders()) == []
    acct = run(b.account())
    assert acct.is_paper and acct.equity == pytest.approx(b.cash + mid * 100 * q)
    assert run(b.what_if(order_for(ap, 0.0))) is None and run(b.connect()) is None
    assert run(b.executions(NOW + timedelta(days=1))) == []


def test_a_cancelled_order_never_fills_and_missing_quotes_wait(spread):
    b = PaperBroker(25_000.0)
    b.update_market(NOW, {})
    ap = approved(spread)
    run(b.submit(ap, order_for(ap, 0.0)))
    assert b.order("c1").status is OrderStatus.SUBMITTED   # geen keten: wachten
    run(b.cancel("c1"))
    b.update_market(NOW, {"TST": cached_chain()})
    assert b.order("c1").status is OrderStatus.CANCELLED and run(b.open_orders()) == []
    # Een wachtende order vult bij de volgende marktupdate.
    b.update_market(NOW, {})
    _, natural = combo_prices(cached_chain(), ap.proposal.legs)
    run(b.submit(ap, order_for(ap, natural, cid="c2")))
    assert b.order("c2").status is OrderStatus.SUBMITTED
    b.update_market(NOW, {"TST": cached_chain()})
    assert b.order("c2").status is OrderStatus.FILLED


def _hold(b, *rows):
    for c, qty in rows:
        b._book_option(c, qty, 1.0)


def test_settle_exercises_assigns_and_expires():
    b = PaperBroker(10_000.0)
    put = OptionContract("TST", E, 95.0, Right.PUT)
    call = OptionContract("TST", E, 90.0, Right.CALL)
    far = OptionContract("TST", E, 120.0, Right.CALL)
    later = OptionContract("TST", date(2027, 1, 15), 90.0, Right.CALL)
    _hold(b, (put, -1), (call, 2), (far, -1), (later, 1))
    ts = datetime(2026, 12, 31, 16, 30, tzinfo=NY)
    out = b.settle(E, {"TST": 92.0}, ts)
    kinds = {e.contract.key(): e.kind for e in out}
    assert kinds == {put.key(): ExecutionKind.ASSIGNMENT, call.key(): ExecutionKind.EXERCISE,
                     far.key(): ExecutionKind.EXPIRY}
    # Assigned put: +100 aandelen tegen 95; exercised calls: +200 tegen 90.
    shares = {p.symbol: p.qty for p in run(b.positions()) if p.contract is None}
    assert shares == {"TST": 300}
    assert b.cash == pytest.approx(10_000.0 - 100 * 95.0 - 200 * 90.0)
    assert [p.contract for p in run(b.positions()) if p.contract is not None] == [later]
    b.update_market(ts, {})
    # Zonder keten: aandelen tegen de eerste kostprijs (de call op 90), opties op 0.
    assert run(b.account()).equity == pytest.approx(b.cash + 300 * 90.0)
    b._book_option(later, -1, 1.0)
    assert all(p.contract is None for p in run(b.positions()))
    b._book_shares("TST", -300, 95.0)
    assert not [p for p in run(b.positions()) if p.contract is None]


def test_early_assignment_before_a_dividend():
    b = PaperBroker(0.0)
    ch = cached_chain(spot=110.0)
    b.update_market(NOW, {"TST": ch})
    deep = OptionContract("TST", date(2026, 12, 18), 90.0, Right.CALL)
    atm = OptionContract("TST", date(2026, 12, 18), 110.0, Right.CALL)
    long_ = OptionContract("TST", date(2026, 12, 18), 95.0, Right.CALL)
    put = OptionContract("TST", date(2026, 12, 18), 120.0, Right.PUT)
    missing = OptionContract("TST", date(2026, 12, 18), 90.5, Right.CALL)
    _hold(b, (deep, -1), (atm, -1), (long_, 1), (put, -1), (missing, -1))
    div = {"TST": CashDividend(date(2026, 11, 25), 1.0)}
    out = b.early_assign(div, NOW)
    assert [e.contract for e in out] == [deep] and out[0].kind is ExecutionKind.ASSIGNMENT
    assert {p.symbol: p.qty for p in run(b.positions()) if p.contract is None} == {"TST": -100}
    assert b.early_assign({}, NOW) == []
    b.update_market(NOW, {})
    assert b.early_assign(div, NOW) == []
