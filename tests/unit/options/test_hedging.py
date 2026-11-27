"""De delta-hedger: de regel, aandelenorders bij paper en IBKR, de poort en de engine."""
from __future__ import annotations

import asyncio
import dataclasses
import math
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from tests.unit.options.fake_ib import FakeIB
from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import CFG, SETTINGS, cached_chain, market
from tests.unit.options.test_engine_service import BASE, D0, MKT, at
from tradebot.options.calendar import NY
from tradebot.options.contracts import Right
from tradebot.options.data.feed import ReplayFeed
from tradebot.options.engine import service
from tradebot.options.engine.service import Engine, EngineMode, Paths
from tradebot.options.engine.state import EngineState, intents_record
from tradebot.options.execution.broker import Execution, ExecutionKind
from tradebot.options.execution.ibkr import IBKRBroker
from tradebot.options.execution.orders import OrderStatus, Purpose, StockOrder
from tradebot.options.execution.paper import PaperBroker
from tradebot.options.hedging import (
    HedgeRules,
    hedge_orders,
    hedge_rules,
    position_delta,
    stock_limit,
)
from tradebot.options.positions import (
    MULTIPLIER,
    Book,
    Leg,
    LegIntent,
    PositionStatus,
    StrategyPosition,
)
from tradebot.options.pricing.engines import Method
from tradebot.options.risk.gate import (
    ApprovedStockOrder,
    GateState,
    RiskGate,
    SafetyBreach,
    StockProposal,
    Verdict,
)
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.income import CoveredCall
from tradebot.schemas.options import HedgingConfig
from tradebot.utils.failfast import DataContractError

HEDGED = "long_straddle_strangle"        # staat standaard aan; gamma_scalp staat standaard uit
CHAIN = cached_chain()                   # TST, spot 100
EXP = date(2026, 12, 31)


def run(coro):
    return asyncio.run(coro)


def opt(strike, right="call"):
    return CHAIN.find(EXP, strike, Right.CALL if right == "call" else Right.PUT)


def pos(pid="p1", legs=((100.0, "call", 1),), shares=0, strategy=HEDGED,
        status=PositionStatus.OPEN, und="TST"):
    out = [Leg(opt(k, r).contract, und, q, 1.0) for k, r, q in legs]
    if shares:
        out.append(Leg(None, und, shares, 100.0))
    return StrategyPosition(pid, strategy, und, tuple(out), NOW, 1, -100.0, 500.0,
                            status=status)


RULES = HedgeRules(frozenset({HEDGED}), frozenset({"TST"}), 20.0, 0.5)


def delta_of(*legs):
    return sum(opt(k, r).greeks.delta * MULTIPLIER * q for k, r, q in legs)


# --- de regel -------------------------------------------------------------------------------

def test_the_band_and_the_order_size():
    d = delta_of((100.0, "call", 1))
    (h,) = hedge_orders([pos()], {"TST": CHAIN}, 25_000.0, RULES)
    assert h.qty == -round(d) and h.delta_shares == pytest.approx(d) and not h.capped
    assert (h.underlying, h.position_id, h.strategy) == ("TST", "p1", HEDGED)
    # Al gehedged tot binnen de band: niets.
    assert hedge_orders([pos(shares=-round(d) + 15)], {"TST": CHAIN}, 25_000.0, RULES) == []
    assert position_delta(pos(shares=-40), CHAIN) == pytest.approx(d - 40)


def test_only_hedged_strategies_eligible_names_open_positions_and_known_deltas():
    chains = {"TST": CHAIN}
    assert hedge_orders([pos(strategy="credit_spread")], chains, 25_000.0, RULES) == []
    assert hedge_orders([pos(status=PositionStatus.CLOSED)], chains, 25_000.0, RULES) == []
    other = HedgeRules(RULES.strategies, frozenset({"AAPL"}), 20.0, 0.5)
    assert hedge_orders([pos()], chains, 25_000.0, other) == []
    assert hedge_orders([pos()], {}, 25_000.0, RULES) == []
    no_greeks = dataclasses.replace(CHAIN, options=tuple(
        dataclasses.replace(o, greeks=None) for o in CHAIN.options))
    assert hedge_orders([pos()], {"TST": no_greeks}, 25_000.0, RULES) == []
    assert position_delta(pos(), no_greeks) is None


def test_the_notional_cap_shrinks_but_never_reverses():
    # 4 calls ≈ +200 delta; plafond 0,5 × 10 000 / 100 = 50 aandelen.
    (h,) = hedge_orders([pos(legs=((100.0, "call", 4),))], {"TST": CHAIN}, 10_000.0, RULES)
    assert h.qty == -50 and h.capped
    # Al −50 aandelen (op het plafond) en nog steeds long delta: geen order.
    assert hedge_orders([pos(legs=((100.0, "call", 4),), shares=-50)], {"TST": CHAIN},
                        10_000.0, RULES) == []
    # +80 aandelen (boven het plafond) bij short delta: kopen, maar naar +50 is verkopen → niets.
    short = pos(legs=((100.0, "put", 4),), shares=80)
    assert position_delta(short, CHAIN) < -20
    assert hedge_orders([short], {"TST": CHAIN}, 10_000.0, RULES) == []


def test_the_order_goes_to_the_largest_contributor():
    a = pos("a", legs=((100.0, "call", 1),))
    b = pos("b", legs=((100.0, "call", 2),))
    c = pos("c", legs=((100.0, "put", 1),))
    (h,) = hedge_orders([a, b, c], {"TST": CHAIN}, 25_000.0, RULES)
    assert h.position_id == "b"
    (h,) = hedge_orders([b, pos("a", legs=((100.0, "call", 2),))], {"TST": CHAIN}, 25_000.0,
                        RULES)
    assert h.position_id == "a"                      # gelijk: het kleinste id
    (h,) = hedge_orders([pos("x", legs=((100.0, "put", 2),))], {"TST": CHAIN}, 25_000.0, RULES)
    assert h.qty > 0


def test_the_limit_is_rounded_away_from_us():
    assert stock_limit(100.0, 10, 2.0) == 100.01
    assert stock_limit(100.0, -10, 2.0) == 99.99
    assert stock_limit(100.0, 10, 0.0) == 100.0
    assert stock_limit(50.004, -1, 0.0) == 50.0


def test_the_config():
    h = CFG.hedging
    assert (h.enabled, h.strategies, h.band_shares, h.max_notional_frac,
            h.interval_seconds) == (False, ("gamma_scalp",), 20.0, 0.5, 300.0)
    assert (CFG.paper.stock_spread_bps, CFG.paper.commission_per_share,
            CFG.paper.min_commission_per_order) == (2.0, 0.005, 1.0)
    with pytest.raises(ValidationError):
        HedgingConfig(strategies=("gamma_scalp_typo",))
    r = hedge_rules(CFG)
    assert r.eligible == frozenset(CFG.universe.share_eligible) and r.band_shares == 20.0


# --- de risicopoort -------------------------------------------------------------------------

GCFG = CFG.model_copy(update={"universe": CFG.universe.model_copy(
    update={"share_eligible": ("TST",)})})
GATE = RiskGate(GCFG, SETTINGS.engines, Method.BAW)


def stock(qty, limit=None, closing=False, symbol="TST"):
    return StockProposal("p1", HEDGED, symbol, qty,
                         limit if limit is not None else stock_limit(100.0, qty, 2.0), closing)


def review(p, book=None, state=None, mkt=None):
    return GATE.review_stock(p, book or Book(), mkt or market(),
                             state or GateState(equity=25_000.0))


def test_hard_errors_in_a_stock_order():
    for p, why in ((stock(0, 100.0), "van 0"), (stock(10, symbol="AAPL"), "niet als aandeel"),
                   (stock(10, 100.5), "te ver"), (stock(10, math.nan), "te ver")):
        with pytest.raises(SafetyBreach, match=why):
            review(p)


def test_a_closing_stock_order_is_always_approved():
    d = review(stock(-10, closing=True), Book(hedge_shares={"TST": 10}),
               GateState(halted=True, unwind=True))
    assert d.verdict is Verdict.APPROVE and d.reason == "close"
    assert d.approved.proposal.qty == -10
    assert review(stock(4, closing=True), Book(hedge_shares={"TST": -10})).reason == "close"


def test_a_closing_stock_order_must_flatten_the_share_legs():
    for qty, held in ((-10, 0), (10, 10), (-11, 10), (-10, -10), (11, -10)):
        with pytest.raises(SafetyBreach, match="sluitorder"):
            review(stock(qty, closing=True), Book(hedge_shares={"TST": held} if held else {}))


def test_the_notional_cap_on_the_share_legs():
    # Plafond 0,5 × 25 000 = 12 500 USD: 125 aandelen bij spot 100. Leeg boek, beta-grens ruim.
    loose = RiskGate(GCFG.model_copy(update={"risk": GCFG.risk.model_copy(
        update={"max_abs_beta_delta_frac": 10.0})}), SETTINGS.engines, Method.BAW)

    def rev(p, held=0):
        book = Book(shares={"TST": held} if held else {},
                    hedge_shares={"TST": held} if held else {})
        return loose.review_stock(p, book, market(), GateState())

    d = rev(stock(125))
    assert d.verdict is Verdict.APPROVE
    assert (d.checks[1].name, d.checks[1].measured, d.checks[1].threshold) == \
        ("stock_notional_usd", 12_500.0, 12_500.0)
    d = rev(stock(126))
    assert d.verdict is Verdict.REJECT and d.reason == "limits:stock_notional_usd"
    assert rev(stock(10_000)).reason == "limits:stock_notional_usd"      # een fout in de hedger
    assert rev(stock(10), held=200).reason == "limits:stock_notional_usd"
    # Afbouwen mag altijd, ook als het plafond na een koersstijging overschreden blijft.
    assert rev(stock(-50), held=200).verdict is Verdict.APPROVE


def test_a_stock_order_needs_a_chain_greeks_and_no_reconcile_break():
    assert review(stock(10), mkt=dataclasses.replace(market(), chains={})).reason == \
        "quotes:missing"
    assert review(stock(10), state=GateState(blocked_underlyings=frozenset({"TST"}))).reason \
        == "reconcile:diff"
    no_greeks = dataclasses.replace(CHAIN, options=tuple(
        dataclasses.replace(o, greeks=None) for o in CHAIN.options))
    d = review(stock(-10), Book((pos(),)), mkt=market(no_greeks))
    assert d.verdict is Verdict.REJECT and d.reason == "greeks_incomplete"


def test_in_halt_a_stock_order_must_reduce_the_delta_and_the_shares():
    book = Book((pos(),))                               # ≈ +50 delta, geen aandelen
    halt = GateState(halted=True)
    d = review(stock(-50), book)
    assert d.verdict is Verdict.APPROVE and d.reason == "risk_reducing"
    assert d.checks[0].name == "beta_delta_usd"
    # In HALT geen nieuwe aandelen, ook niet als de delta daalt.
    assert review(stock(-50), book, halt).reason == "state:halt"
    assert review(stock(50), book, halt).reason == "state:halt"
    assert review(stock(50), book, GateState(unwind=True)).reason == "state:unwind"
    # Te veel short aandelen (−120 tegen +50 delta): terug naar −50 verkleint beide.
    over = Book((pos(shares=-120),), {"TST": -120}, hedge_shares={"TST": -120})
    d = review(stock(70), over, halt)
    assert d.verdict is Verdict.APPROVE and d.reason == "risk_reducing"
    assert review(stock(70), over, GateState(unwind=True)).verdict is Verdict.APPROVE
    # Voorbij nul (−120 → +30) verkleint de aandelen; de delta stijgt dan wel: geweigerd.
    assert review(stock(150), over, halt).reason == "state:halt"
    # Minder aandelen maar meer delta: geweigerd.
    under = Book((pos(shares=-30),), {"TST": -30}, hedge_shares={"TST": -30})
    assert review(stock(20), under, halt).reason == "state:halt"


def test_a_stock_order_within_the_beta_delta_limit():
    d = review(stock(10))                              # leeg boek: delta stijgt, past
    assert d.verdict is Verdict.APPROVE and d.reason == "hedge"
    big = stock(100, 100.01)                           # 100 aandelen = 10 000 USD > 7 500
    d = review(big)
    assert d.verdict is Verdict.REJECT and d.reason == "limits:beta_delta_usd"


def test_only_the_gate_makes_an_approved_stock_order():
    with pytest.raises(TypeError, match="risicopoort"):
        ApprovedStockOrder(stock(10), NOW, ())


# --- de paper-broker ------------------------------------------------------------------------

def approved_stock(qty, limit=None, book=None, mkt=None):
    d = review(stock(qty, limit), book, mkt=mkt)
    assert d.approved is not None
    return d.approved


def test_paper_fills_a_stock_order_at_its_limit_with_commission():
    b = PaperBroker(25_000.0)
    b.update_market(NOW, {"TST": CHAIN})
    ap = approved_stock(10)
    with pytest.raises(TypeError, match="ApprovedStockOrder"):
        run(b.submit_stock(ap.proposal, StockOrder("s1", "p1", "TST", 10, 100.01,  # type: ignore[arg-type]
                                                   Purpose.HEDGE)))
    with pytest.raises(DataContractError, match="wijkt af"):
        run(b.submit_stock(ap, StockOrder("s1", "p1", "TST", 11, 100.01, Purpose.HEDGE)))
    bo = run(b.submit_stock(ap, StockOrder("s1", "p1", "TST", 10, 100.01, Purpose.HEDGE)))
    assert bo.status is OrderStatus.FILLED and bo.avg_fill == 100.01
    with pytest.raises(DataContractError, match="al gebruikt"):
        run(b.submit_stock(ap, StockOrder("s1", "p1", "TST", 10, 100.01, Purpose.HEDGE)))
    assert b.cash == pytest.approx(25_000.0 - 1000.1 - 1.0)          # minimum 1,00
    (ex,) = run(b.executions(NOW))
    assert (ex.contract, ex.symbol, ex.qty, ex.price, ex.commission) == (
        None, "TST", 10, 100.01, 1.0)
    assert [(p.contract, p.qty) for p in run(b.positions())] == [(None, 10)]
    big = approved_stock(-400, 99.99, Book((pos(legs=((100.0, "call", 8),)),)),
                         dataclasses.replace(market(), capital_usd=100_000.0))  # plafond 500
    run(b.submit_stock(big, StockOrder("s2", "p1", "TST", -400, 99.99, Purpose.HEDGE)))
    assert run(b.executions(NOW))[-1].commission == pytest.approx(2.0)   # 0,005 × 400


def test_paper_waits_until_the_limit_touches_the_spread():
    b = PaperBroker(25_000.0)
    ap = approved_stock(10, 100.0)                      # op spot: binnen de halve spread
    bo = run(b.submit_stock(ap, StockOrder("s1", "p1", "TST", 10, 100.0, Purpose.HEDGE)))
    assert bo.status is OrderStatus.SUBMITTED           # nog geen markt
    b.update_market(NOW, {"TST": CHAIN})
    assert b.order("s1").status is OrderStatus.SUBMITTED
    sell = approved_stock(-10, 100.0)
    run(b.submit_stock(sell, StockOrder("s2", "p1", "TST", -10, 100.0, Purpose.HEDGE)))
    assert b.order("s2").status is OrderStatus.SUBMITTED
    run(b.modify("s1", 100.01))
    assert b.order("s1").status is OrderStatus.FILLED


# --- IBKR -----------------------------------------------------------------------------------

def ib_broker(ib):
    b = IBKRBroker(ib)
    run(b.connect())
    return b


def test_ibkr_sends_a_stock_limit_order():
    ib = FakeIB()
    b = ib_broker(ib)
    ap = approved_stock(-10)
    with pytest.raises(TypeError, match="ApprovedStockOrder"):
        run(b.submit_stock(ap.proposal, StockOrder("s1", "p1", "TST", -10, 99.99,  # type: ignore[arg-type]
                                                   Purpose.HEDGE)))
    with pytest.raises(DataContractError, match="wijkt af"):
        run(b.submit_stock(ap, StockOrder("s1", "p1", "TSTB", -10, 99.99, Purpose.HEDGE)))
    bo = run(b.submit_stock(ap, StockOrder("s1", "p1", "TST", -10, 99.99, Purpose.HEDGE)))
    assert bo.status is OrderStatus.SUBMITTED
    stk, order = ib.placed[0]
    assert (stk.secType, stk.symbol, stk.exchange, stk.currency) == ("STK", "TST", "SMART",
                                                                      "USD")
    assert (order.action, order.totalQuantity, order.lmtPrice, order.orderType, order.tif,
            order.orderRef, order.account) == ("SELL", 10, 99.99, "LMT", "DAY", "s1",
                                               "DU1234567")
    with pytest.raises(DataContractError, match="al gebruikt"):
        run(b.submit_stock(ap, StockOrder("s1", "p1", "TST", -10, 99.99, Purpose.HEDGE)))
    run(b.submit_stock(approved_stock(10), StockOrder("s2", "p1", "TST", 10, 100.01,
                                                      Purpose.EXIT)))
    assert ib.placed[-1][1].action == "BUY"
    assert run(b.modify("s1", 99.98)).order.limit == 99.98
    run(b.cancel("s1"))
    assert ib.cancelled == [order]


def test_ibkr_refuses_an_unknown_stock():
    ib = FakeIB()
    ib.qualifyContractsAsync = _none
    b = ib_broker(ib)
    with pytest.raises(DataContractError, match="kent het aandeel niet"):
        run(b.submit_stock(approved_stock(10), StockOrder("s1", "p1", "TST", 10, 100.01,
                                                          Purpose.HEDGE)))


async def _none(*cs):
    return [None for _ in cs]


def test_ibkr_stock_executions_have_no_contract():
    ib = FakeIB()
    b = ib_broker(ib)
    t = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
    opt_c = NS(secType="OPT", symbol="TST", lastTradeDateOrContractMonth="20261231",
               strike=95.0, right="P")
    ib.fills = [NS(contract=NS(secType="STK", symbol="TST"),
                   execution=NS(shares=10, side="SLD", price=99.99, orderRef="s1"),
                   commissionReport=NS(commission=1.0), time=t),
                NS(contract=opt_c, execution=NS(shares=1, side="BOT", price=1.0, orderRef="c1"),
                   commissionReport=NS(commission=0.65), time=t)]
    ex = run(b.executions(t - timedelta(hours=1)))
    assert [(e.client_id, e.contract is None, e.symbol, e.qty) for e in ex] == [
        ("c1", False, "TST", 1), ("s1", True, "TST", -10)]


# --- covered call: de hedge is geen dekking -------------------------------------------------

def test_hedge_shares_do_not_count_for_a_covered_call():
    cc = CoveredCall(CFG.strategies.covered_call, rules_from_config(CFG), ("TST",))
    rich = market(capital=100 * 1_000_000.0)
    book = Book(shares={"TST": 300}, cost_basis={"TST": 104.0})
    assert cc.scan(rich, book)[0].quantity == 3
    hedged = dataclasses.replace(book, hedge_shares={"TST": 230})
    assert cc.scan(rich, hedged) == []
    assert cc.scan(rich, dataclasses.replace(book, hedge_shares={"TST": -100}))[0].quantity == 4


# --- de engine ------------------------------------------------------------------------------

HCFG = BASE.model_copy(update={
    "universe": BASE.universe.model_copy(update={"share_eligible": ("TSTB",)}),
    "hedging": BASE.hedging.model_copy(update={"enabled": True, "strategies": (HEDGED,)})})


class HedgeRig:
    def __init__(self, tmp, cfg=HCFG, broker=None):
        self.alerts: list[tuple[str, str, str]] = []
        self.broker = broker or PaperBroker(25_000.0)
        self.state = EngineState("replay", "paper", "t0")
        self.e = Engine(cfg, ReplayFeed(MKT), self.broker, Paths(tmp), self.state,
                        lambda *a: self.alerts.append(a), ())
        self.t0 = at(D0, 10, 0)
        run(self.e.tick(self.t0))                       # reconcile en de eerste keten
        self.e._last.pop("hedge", None)                 # de eerste hedge-ronde komt nog

    def add(self, pid="h1", calls=2, strategy=HEDGED):
        chain = self.e.chains["TSTB"]
        atm = min((o for o in chain.options if o.contract.right is Right.CALL
                   and o.greeks is not None and (o.contract.expiry - D0).days > 20),
                  key=lambda o: (abs(o.contract.strike - chain.spot), o.contract.expiry))
        p = StrategyPosition(pid, strategy, "TSTB", (Leg(atm.contract, "TSTB", calls,
                                                         atm.quote.mid),),
                             self.t0, calls, -atm.quote.mid * 100 * calls, 1e4)
        self.state.positions.append(p)
        self.broker._book_option(atm.contract, calls, atm.quote.mid)
        return p, atm

    def tick(self, t):
        run(self.e.tick(t))

    def events(self, name):
        return [r for r in self.e.journal.read() if r["event"] == name]

    def pos(self, pid="h1"):
        return next(p for p in self.state.positions if p.position_id == pid)


def test_the_engine_hedges_and_books_the_shares_on_the_position(tmp_path):
    r = HedgeRig(tmp_path)
    p, atm = r.add()
    r.tick(at(D0, 10, 0, 15))
    (o,) = r.events("hedge_order")
    want = -round(atm.greeks.delta * 100 * 2)
    assert (o["qty"], o["purpose"], o["verdict"], o["reason"]) == (want, "hedge", "approve",
                                                                   "risk_reducing")
    (f,) = r.events("hedge_fill")
    assert f["qty"] == want and f["shares"] == want and f["commission"] == pytest.approx(1.0)
    assert r.pos().shares == want and r.e.book().shares == {"TSTB": want}
    assert r.e.book().hedge_shares == {"TSTB": want}
    (rec,) = [x for x in r.state.orders.values() if x.get("stock")]
    assert rec["status"] == "filled" and rec["client_id"] == f["client_id"]
    # Binnen de band en binnen het interval: geen nieuwe order; de reconciliatie klopt.
    r.tick(at(D0, 10, 6))
    assert len(r.events("hedge_order")) == 1
    assert r.e.mode is EngineMode.RUNNING and not r.events("recon_break")


def test_without_hedging_nothing_happens(tmp_path):
    r = HedgeRig(tmp_path, HCFG.model_copy(update={"hedging": BASE.hedging}))
    r.add()
    r.tick(at(D0, 10, 0, 15))
    assert not r.events("hedge_order")


def test_a_halted_engine_only_reduces_its_shares(tmp_path):
    r = HedgeRig(tmp_path)
    r.add()
    r.e.paths.halt_file.touch()
    r.tick(at(D0, 10, 0, 15))
    assert r.e.mode is EngineMode.HALTED
    (o,) = r.events("hedge_order")                    # zonder aandelen: niets af te bouwen
    assert o["verdict"] == "reject" and o["reason"] == "state:halt"
    assert not r.events("hedge_fill")


def test_a_halted_engine_trims_too_many_shares(tmp_path):
    r = HedgeRig(tmp_path)
    p, atm = r.add()
    want = -round(atm.greeks.delta * 100 * 2)
    over = 2 * want                                   # dubbel gehedged
    r.state.positions[0] = dataclasses.replace(p, legs=p.legs + (Leg(None, "TSTB", over, 100.0),))
    r.broker._book_shares("TSTB", over, 100.0)
    r.e.paths.halt_file.touch()
    r.tick(at(D0, 10, 0, 15))
    (o,) = r.events("hedge_order")
    assert o["verdict"] == "approve" and o["reason"] == "risk_reducing"
    assert abs(r.pos().shares) < abs(over)


def test_a_rejected_hedge_is_journaled_and_not_sent(tmp_path, monkeypatch):
    r = HedgeRig(tmp_path)
    r.add()
    refused = GATE.review_stock(stock(10), Book(), market(), GateState(halted=True))
    monkeypatch.setattr(r.e.gate, "review_stock", lambda *a: refused)
    r.tick(at(D0, 10, 0, 15))
    (o,) = r.events("hedge_order")
    assert o["verdict"] == "reject" and o["client_id"] is None
    assert not [x for x in r.state.orders.values() if x.get("stock")]


def test_closing_a_position_unwinds_its_shares_in_the_same_round(tmp_path):
    r = HedgeRig(tmp_path)
    p, _ = r.add()
    r.tick(at(D0, 10, 0, 15))
    shares = r.pos().shares
    r.e.paths.unwind_file.touch()                     # manage sluit alles, de walk vult
    t = at(D0, 10, 0, 30)
    while r.pos().status is not PositionStatus.CLOSED:
        t += timedelta(seconds=15)
        r.tick(t)
    assert r.pos().shares == 0
    exit_o = [o for o in r.events("hedge_order") if o["purpose"] == "exit"]
    assert len(exit_o) == 1 and exit_o[0]["qty"] == -shares and exit_o[0]["reason"] == "close"
    assert exit_o[0]["ts"] == t.isoformat()
    (closed,) = r.events("position_closed")
    assert closed["ts"] == t.isoformat()
    assert r.e.book().shares == {} and r.e.book().hedge_shares == {}
    # Vlak: de gerealiseerde hedge-P&L is precies de kasstroom van de aandelenfills.
    fills = r.events("hedge_fill")
    cash = sum(-f["qty"] * f["price"] - f["commission"] for f in fills)
    assert r.pos().hedge_realized_usd == pytest.approx(cash)
    assert fills[-1]["hedge_realized_usd"] == round(r.pos().hedge_realized_usd, 2)
    assert "hedge_realized_usd" in closed
    r.tick(at(D0, 10, 2))                             # niets meer te doen
    assert len(r.events("hedge_order")) == 2


def test_a_close_cancels_an_open_hedge_order_first(tmp_path):
    cfg = HCFG.model_copy(update={"paper": HCFG.paper.model_copy(
        update={"stock_spread_bps": 0.0})})             # limiet op spot: de paper vult niet
    r = HedgeRig(tmp_path, cfg)
    p, _ = r.add()
    r.tick(at(D0, 10, 0, 15))
    (o,) = [x for x in r.state.orders.values() if x.get("stock")]
    assert o["status"] == "submitted" and r.pos().shares == 0
    # Een lopende aandelenorder blokkeert het sluiten niet.
    assert not r.e._pending(p.position_id)
    r.state.positions[0] = dataclasses.replace(p, status=PositionStatus.CLOSED)
    r.tick(at(D0, 10, 0, 30))
    assert o["status"] == "cancelled"
    assert r.events("order_cancelled")[0]["reason"] == "position_closed"
    assert len(r.events("hedge_order")) == 1          # geen aandelen: niets terug te draaien


def test_a_status_set_while_submitting_a_stock_order_is_kept(tmp_path):
    cfg = HCFG.model_copy(update={"paper": HCFG.paper.model_copy(
        update={"stock_spread_bps": 0.0})})             # limiet op spot: de paper vult niet
    r = HedgeRig(tmp_path, cfg)
    r.add()
    real = r.broker.submit_stock

    async def submit(approved, order):
        bo = await real(approved, order)
        r.state.orders[order.client_id]["status"] = "cancelled"   # intussen afgehandeld
        return bo

    r.broker.submit_stock = submit
    r.tick(at(D0, 10, 0, 15))
    (o,) = [x for x in r.state.orders.values() if x.get("stock")]
    assert o["status"] == "cancelled"                  # niet terug naar "submitted"


def test_an_open_hedge_order_is_replaced_one_round_later(tmp_path):
    cfg = HCFG.model_copy(update={"paper": HCFG.paper.model_copy(
        update={"stock_spread_bps": 0.0})})
    r = HedgeRig(tmp_path, cfg)
    r.add()
    r.tick(at(D0, 10, 0, 15))
    r.tick(at(D0, 10, 5, 15))                          # volgende hedge-ronde: annuleren
    assert r.events("order_cancelled")[0]["reason"] == "replaced"
    assert len(r.events("hedge_order")) == 1
    r.tick(at(D0, 10, 10, 15))                         # en dan pas opnieuw
    assert len(r.events("hedge_order")) == 2
    # De walk laat aandelenorders met rust.
    run(r.e._walk(at(D0, 10, 10, 30)))
    assert not r.events("order_modified")


def test_blocked_names_are_not_hedged(tmp_path):
    r = HedgeRig(tmp_path)
    r.add()
    r.e.mode, r.e._blocked = EngineMode.BLOCKED, frozenset({"TSTB"})
    run(r.e._hedge(at(D0, 10, 0, 15)))
    assert not r.events("hedge_order")


def test_stock_fills_are_booked_once_even_after_a_restart(tmp_path):
    r = HedgeRig(tmp_path)
    p, _ = r.add()
    r.state.orders["s"] = {"client_id": "s", "stock": True, "position_id": "h1",
                           "purpose": "hedge", "strategy": HEDGED, "underlying": "TSTB",
                           "qty": -60, "limit": 50.0, "filled": 0, "execs": [],
                           "status": "cancelled"}
    o = r.state.orders["s"]
    e1 = Execution(at(D0, 10, 1), ExecutionKind.FILL, "s", None, "TSTB", -40, 50.0, 1.0)
    r.e._book_stock_fill("s", o, [e1], at(D0, 10, 1))
    assert o["status"] == "cancelled" and r.pos().shares == -40      # deel, na annuleren
    r.e._book_stock_fill("s", o, [e1], at(D0, 10, 1))                # herstart: zelfde fill
    assert r.pos().shares == -40
    e2 = dataclasses.replace(e1, ts=at(D0, 10, 2), qty=-20, price=49.0)
    r.e._book_stock_fill("s", o, [e2], at(D0, 10, 2))
    assert o["status"] == "filled" and r.pos().shares == -60
    (leg,) = [x for x in r.pos().legs if x.contract is None]
    assert leg.entry_price == pytest.approx((50.0 * 40 + 49.0 * 20) / 60)


def test_the_share_leg_keeps_its_price_when_reduced_and_resets_when_flipped():
    p = StrategyPosition("x", HEDGED, "TSTB", (), NOW, 1, 0.0, 0.0)
    p = service._with_shares(p, 10, 50.0)
    p = service._with_shares(p, -4, 52.0)
    assert p.shares == 6 and p.legs[-1].entry_price == 50.0
    p = service._with_shares(p, -10, 53.0)
    assert p.shares == -4 and p.legs[-1].entry_price == 53.0
    assert service._with_shares(p, 4, 51.0).legs == ()


def test_trims_and_flips_realise_the_hedge_pnl_after_commission():
    p = StrategyPosition("x", HEDGED, "TSTB", (), NOW, 1, 0.0, 0.0)
    steps = [                       # (qty, prijs, commissie) → (aandelen, gem. prijs, gerealiseerd)
        ((10, 50.0, 1.0), (10, 50.0, -1.0)),             # openen: alleen commissie
        ((10, 52.0, 1.0), (20, 51.0, -2.0)),             # bijkopen: middelen
        ((-5, 53.0, 1.0), (15, 51.0, 7.0)),              # afbouwen: 5 × (53 − 51) − 1
        ((-20, 54.0, 1.0), (-5, 54.0, 51.0)),            # omdraaien: 15 × 3 − 1, rest op 54
        ((-5, 56.0, 1.0), (-10, 55.0, 50.0)),            # short bijkopen: middelen
        ((4, 52.0, 1.0), (-6, 55.0, 61.0)),              # short afbouwen: 4 × (55 − 52) − 1
        ((6, 57.0, 1.0), (0, None, 48.0)),               # sluiten: 6 × (55 − 57) − 1
    ]
    for (qty, px, fee), (shares, avg, realized) in steps:
        p = service._with_shares(p, qty, px, fee)
        assert p.shares == shares and p.hedge_realized_usd == pytest.approx(realized)
        assert [leg.entry_price for leg in p.legs if leg.contract is None] == (
            [] if avg is None else [pytest.approx(avg)])


def test_booking_options_keeps_the_share_leg(tmp_path):
    r = HedgeRig(tmp_path)
    p, atm = r.add()
    other = next(o for o in r.e.chains["TSTB"].options if o.contract != atm.contract
                 and o.contract.right is atm.contract.right)
    r.state.positions[-1] = dataclasses.replace(
        p, legs=p.legs + (Leg(other.contract, "TSTB", 1, 1.0), Leg(None, "TSTB", -30, 50.0)))
    r.state.orders["a"] = {"client_id": "a", "position_id": "h1", "purpose": "adjust",
                           "kind": "close_leg", "strategy": HEDGED, "underlying": "TSTB",
                           "legs": intents_record((LegIntent(other.contract, -1),)),
                           "quantity": 1, "max_loss": 100.0, "variant": "",
                           "urgency": "normal", "reason": "t", "step": 0, "limit": -0.1,
                           "status": "submitted"}
    ex = [Execution(at(D0, 11, 0), ExecutionKind.FILL, "a", other.contract, "TSTB", -1, 1.0)]
    r.e._book_fill("a", r.state.orders["a"], ex, at(D0, 11, 0))
    assert r.pos().shares == -30 and len(r.pos().option_legs) == 1
    r.e._assignment(Execution(at(D0, 16, 30), ExecutionKind.EXERCISE, "", atm.contract,
                              "TSTB", -2, atm.contract.strike), at(D0, 16, 30))
    assert r.pos().status is PositionStatus.CLOSED and r.pos().shares == -30
    p2 = dataclasses.replace(r.pos(), position_id="h2", status=PositionStatus.OPEN,
                             legs=(Leg(atm.contract, "TSTB", 2, 1.0),
                                   Leg(other.contract, "TSTB", 1, 1.0),
                                   Leg(None, "TSTB", 10, 50.0)))
    r.state.positions.append(p2)
    r.e._assignment(Execution(at(D0, 16, 31), ExecutionKind.EXERCISE, "", other.contract,
                              "TSTB", -1, other.contract.strike), at(D0, 16, 31))
    assert r.pos("h2").shares == 10 and len(r.pos("h2").option_legs) == 1


def test_reconcile_allows_whole_lots_but_not_stray_shares(tmp_path):
    r = HedgeRig(tmp_path)
    r.add()
    r.tick(at(D0, 10, 0, 15))
    assert r.e._share_breaks({"TSTB": r.pos().shares + 100}) == []      # een assignment-lot
    r.broker._book_shares("TSTB", 7, 50.0)
    run(r.e._reconcile(at(D0, 10, 1)))
    assert r.e.mode is EngineMode.BLOCKED
    assert r.events("recon_break")[-1]["contracts"] == ["TSTB shares"]
    assert r.e._blocked == frozenset({"TSTB"})


def test_an_exit_waits_for_a_chain_and_an_open_exit(tmp_path):
    r = HedgeRig(tmp_path, HCFG.model_copy(update={"hedging": BASE.hedging}))
    p, _ = r.add()
    r.state.positions[-1] = dataclasses.replace(
        p, legs=p.legs + (Leg(None, "TSTB", -30, 50.0),), status=PositionStatus.CLOSED)
    chain = r.e.chains.pop("TSTB")
    run(r.e._unwind_shares(at(D0, 10, 1)))
    assert not r.events("hedge_order")                # geen keten: wachten
    r.e.chains["TSTB"] = chain
    r.state.orders["x"] = {"client_id": "x", "stock": True, "position_id": "h1",
                           "purpose": "exit", "underlying": "TSTB", "status": "submitted",
                           "submitted_at": at(D0, 10, 1).isoformat()}
    run(r.e._unwind_shares(at(D0, 10, 1)))
    assert not r.events("hedge_order") and r.state.orders["x"]["status"] == "submitted"
    r.state.orders["x"]["status"] = "cancelled"
    run(r.e._unwind_shares(at(D0, 10, 2)))
    (o,) = r.events("hedge_order")
    assert o["purpose"] == "exit" and o["qty"] == 30 and r.pos().shares == 0


def test_no_hedging_outside_the_session(tmp_path):
    r = HedgeRig(tmp_path)
    r.add()
    run(r.e.tick(datetime.combine(D0, time(9, 0), tzinfo=NY)))
    assert not r.events("hedge_order")


# --- exec-id: deelfills in dezelfde seconde, correcties, en een EXIT die niet vult -----------

def test_paper_fills_carry_a_unique_exec_id():
    b = PaperBroker(25_000.0)
    b.update_market(NOW, {"TST": CHAIN})
    run(b.submit_stock(approved_stock(10), StockOrder("s1", "p1", "TST", 10, 100.01,
                                                      Purpose.HEDGE)))
    run(b.submit_stock(approved_stock(5), StockOrder("s2", "p1", "TST", 5, 100.01,
                                                     Purpose.HEDGE)))
    assert [e.exec_id for e in run(b.executions(NOW))] == ["s1#1", "s2#2"]


def test_ibkr_passes_the_exec_id():
    ib = FakeIB()
    b = ib_broker(ib)
    t = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
    ib.fills = [NS(contract=NS(secType="STK", symbol="TST"),
                   execution=NS(shares=10, side="BOT", price=100.0, orderRef="s1",
                                execId="0001.abc.01.01"),
                   commissionReport=NS(commission=1.0), time=t)]
    (e,) = run(b.executions(t - timedelta(hours=1)))
    assert e.exec_id == "0001.abc.01.01"


def _stock_order(r, cid="s", qty=-60, purpose="hedge", status="submitted"):
    r.state.orders[cid] = {"client_id": cid, "stock": True, "position_id": "h1",
                           "purpose": purpose, "strategy": HEDGED, "underlying": "TSTB",
                           "qty": qty, "limit": 50.0, "filled": 0, "execs": [],
                           "status": status, "submitted_at": at(D0, 10, 1).isoformat()}
    return r.state.orders[cid]


def test_two_same_second_partial_fills_are_both_booked(tmp_path):
    r = HedgeRig(tmp_path, HCFG.model_copy(update={"hedging": BASE.hedging}))
    r.add()
    o = _stock_order(r)
    t = at(D0, 10, 1)
    same = [Execution(t, ExecutionKind.FILL, "s", None, "TSTB", -30, 50.0, 0.5, f"00a.{i}.01")
            for i in (1, 2)]
    r.broker._executions.extend(same)
    run(r.e._executions(t))
    run(r.e._executions(t))                                  # opnieuw gelezen: niets erbij
    assert r.pos().shares == -60 and o["status"] == "filled"
    assert len(r.events("hedge_fill")) == 2
    r.e._book_stock_fill("s", o, same, t)                   # herstart: zelfde exec-ids
    assert r.pos().shares == -60


def test_an_ibkr_correction_is_not_booked_twice(tmp_path):
    r = HedgeRig(tmp_path, HCFG.model_copy(update={"hedging": BASE.hedging}))
    r.add()
    o = _stock_order(r)
    t = at(D0, 10, 1)
    first = Execution(t, ExecutionKind.FILL, "s", None, "TSTB", -60, 50.0, 1.0, "00a.1.01")
    fix = dataclasses.replace(first, ts=t + timedelta(seconds=5), price=50.02,
                              exec_id="00a.1.02")
    r.e._book_stock_fill("s", o, [first], t)
    r.e._book_stock_fill("s", o, [fix], t)
    assert r.pos().shares == -60 and len(r.events("hedge_fill")) == 1
    (c,) = r.events("hedge_fill_correction")
    assert c["exec_id"] == "00a.1.02" and r.alerts[-1][1] == "options_hedge_fill_correction"
    r.e._book_stock_fill("s", o, [fix], t)                  # en ook niet een derde keer
    assert len(r.events("hedge_fill_correction")) == 1


def test_an_unfilled_exit_is_replaced_until_flat(tmp_path):
    r = HedgeRig(tmp_path)
    r.broker.stock_half_spread = 0.01                   # 1 %: de limiet raakt hem niet
    p, _ = r.add()
    r.state.positions[-1] = dataclasses.replace(
        p, legs=p.legs + (Leg(None, "TSTB", -30, 50.0),), status=PositionStatus.CLOSED)
    r.broker._book_shares("TSTB", -30, 50.0)
    t = at(D0, 10, 0, 15)
    r.tick(t)
    (first,) = [x for x in r.state.orders.values() if x.get("purpose") == "exit"]
    assert first["status"] == "submitted"
    r.tick(at(D0, 10, 4, 0))                            # nog geen interval: wachten
    assert first["status"] == "submitted"
    r.tick(at(D0, 10, 5, 15))                           # interval voorbij: annuleren
    assert first["status"] == "cancelled"
    assert r.events("order_cancelled")[-1]["reason"] == "replaced"
    r.broker.stock_half_spread = 0.0                    # nu vult hij wel
    r.tick(at(D0, 10, 5, 30))                           # de ronde erna opnieuw
    exits = [x for x in r.events("hedge_order") if x["purpose"] == "exit"]
    assert len(exits) == 2 and r.pos().shares == 0
