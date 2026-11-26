"""De risicopoort: één ingang, harde fouten, toestand, aantallen en pro-forma-grenzen."""
from __future__ import annotations

import ast
import dataclasses
import math
from datetime import date, datetime
from pathlib import Path

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CFG,
    LOW_IV,
    SETTINGS,
    cached_chain,
    market,
    open_position,
)
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.positions import (
    Action,
    ActionKind,
    Book,
    Leg,
    LegIntent,
    PositionStatus,
    Trend,
    Urgency,
    mark,
)
from tradebot.options.pricing.engines import Method
from tradebot.options.risk import gate as gate_mod
from tradebot.options.risk.gate import (
    ApprovedOrder,
    GateState,
    OrderKind,
    OrderProposal,
    RiskGate,
    SafetyBreach,
    Verdict,
)
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.credit import CreditSpread, IronCondor
from tradebot.options.strategies.debit import DebitVertical
from tradebot.options.strategies.short_vol import ShortVol, position_stress_loss_usd, stress_loss

RULES = rules_from_config(CFG)
GATE = RiskGate(CFG, SETTINGS.engines, Method.BAW)
OK = GateState(equity=25_000.0)
SRC = Path(__file__).resolve().parents[3] / "src" / "tradebot" / "options"


@pytest.fixture(scope="module")
def spread():
    (c,) = CreditSpread(CFG.strategies.credit_spread, RULES).scan(market(trend=Trend.UP), Book())
    return c


def proposal(c, pid="p1", **kw):
    return dataclasses.replace(OrderProposal.open(c, pid), **kw)


# --- de sentinel ---------------------------------------------------------------------------

def test_only_the_gate_can_make_an_approved_order(spread):
    with pytest.raises(TypeError, match="risicopoort"):
        ApprovedOrder(proposal(spread), NOW, ())
    callers = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "ApprovedOrder"):
                callers.append(path.relative_to(SRC).as_posix())
    assert set(callers) == {"risk/gate.py"}


# --- entries -------------------------------------------------------------------------------

def test_an_entry_inside_all_limits_is_approved(spread):
    d = GATE.review(proposal(spread), Book(), market(), OK)
    assert d.verdict is Verdict.APPROVE and d.approved.proposal.quantity == spread.quantity
    assert all(c.ok for c in d.checks) and d.reason == "entry"
    assert d.approved.approved_at == NOW


def test_an_entry_is_resized_one_unit_at_a_time(spread):
    tight = CFG.model_copy(update={"risk": CFG.risk.model_copy(
        update={"max_loss_per_underlying_frac": 0.04})})          # 1000 USD bij 25k
    g = RiskGate(tight, SETTINGS.engines, Method.BAW)
    big = proposal(spread, quantity=20, max_loss=spread.max_loss / spread.quantity * 20)
    d = g.review(big, Book(), market(), OK)
    per_unit = spread.max_loss / spread.quantity
    assert d.verdict is Verdict.RESIZE
    assert d.approved.proposal.quantity == math.floor(1000 / per_unit)
    assert d.approved.proposal.max_loss == pytest.approx(per_unit * d.approved.proposal.quantity)


def test_an_entry_that_never_fits_is_rejected(spread):
    tiny = CFG.model_copy(update={"risk": CFG.risk.model_copy(
        update={"max_vega_frac": 1e-6})})
    d = RiskGate(tiny, SETTINGS.engines, Method.BAW).review(proposal(spread), Book(), market(),
                                                            OK)
    assert d.verdict is Verdict.REJECT and d.approved is None
    assert d.reason == "limits:vega_usd"


@pytest.mark.parametrize(("state", "kw", "book_n", "why"), [
    (GateState(halted=True), {}, 0, "state:halt"),
    (GateState(daily_loss_frac=0.02), {}, 0, "daily_loss:entries"),
    (GateState(blocked_underlyings=frozenset({"TST"})), {}, 0, "reconcile:diff"),
    (OK, {"max_loss": math.inf}, 0, "undefined_risk"),
    (OK, {}, 10, "count:open_positions"),
    (GateState(new_positions_today=3), {}, 0, "count:new_today"),
    (GateState(equity=20_000.0, day_trades=(date(2026, 11, 20), date(2026, 11, 23),
                                            date(2026, 11, 24))), {}, 0, "pdt"),
    (GateState(unwind=True), {}, 0, "state:unwind"),
])
def test_entry_gates(spread, state, kw, book_n, why):
    late = datetime(2026, 11, 24, 15, 30, tzinfo=NY)
    m = market(cached_chain(now=late), now=late)
    book = Book(tuple(open_position(spread, position_id=f"o{i}") for i in range(book_n)))
    d = GATE.review(proposal(spread, **kw), book, m, state)
    assert d.verdict is Verdict.REJECT and d.reason == why


def test_closed_positions_do_not_count(spread):
    closed = tuple(dataclasses.replace(open_position(spread, position_id=f"c{i}"),
                                       status=PositionStatus.CLOSED) for i in range(10))
    assert GATE.review(proposal(spread), Book(closed), market(), OK).verdict is Verdict.APPROVE


# --- undefined risk: het boekbrede stressverlies -------------------------------------------

def undefined_gate(frac=0.06):
    cfg = CFG.model_copy(update={"risk": CFG.risk.model_copy(update={
        "allow_undefined_risk": True, "max_undefined_stress_loss_frac": frac})})
    return RiskGate(cfg, SETTINGS.engines, Method.BAW)


@pytest.fixture(scope="module")
def strangle():
    sv = ShortVol(CFG.strategies.short_straddle_strangle.model_copy(
        update={"underlyings": ("TST",)}), RULES)
    (c,) = sv.scan(market(capital=1e6), Book())
    return c


def stress_check(d):
    (c,) = (c for c in d.checks if c.name == "undefined_stress_loss")
    return c


def test_an_undefined_entry_is_checked_on_the_book_stress_loss(strangle):
    m = market(capital=1e6)
    d = undefined_gate().review(proposal(strangle), Book(), m, OK)
    assert d.verdict is Verdict.APPROVE
    unit = stress_loss(*(leg.contract.strike for leg in strangle.legs), 100.0,
                       strangle.net_credit) * 100
    c = stress_check(d)
    assert c.measured == pytest.approx(unit * strangle.quantity) and c.threshold == 60_000.0
    # Posities met beperkt risico tellen niet mee in het stressverlies.
    (spread,) = CreditSpread(CFG.strategies.credit_spread, RULES).scan(market(), Book())
    d = undefined_gate().review(proposal(strangle), Book((open_position(spread),)), m, OK)
    assert stress_check(d).measured == pytest.approx(unit * strangle.quantity)
    # Een entry met beperkt risico krijgt de check niet (het journal blijft gelijk).
    assert all(c.name != "undefined_stress_loss"
               for c in undefined_gate().review(proposal(spread), Book(), market(), OK).checks)


def test_the_book_stress_cap_resizes_or_blocks_a_second_entry(strangle):
    m = market(capital=1e6)
    held = dataclasses.replace(open_position(strangle, position_id="o1"), max_loss=math.inf)
    base = position_stress_loss_usd(held, 100.0)
    unit = position_stress_loss_usd(dataclasses.replace(held, units=1, legs=tuple(
        dataclasses.replace(leg, qty=leg.qty // held.units) for leg in held.legs),
        entry_net=strangle.net_credit * 100), 100.0)
    d = undefined_gate(0.03).review(proposal(strangle), Book((held,)), m, OK)
    assert d.verdict is Verdict.RESIZE
    assert d.approved.proposal.quantity == math.floor((30_000.0 - base) / unit)
    assert math.isinf(d.approved.proposal.max_loss)
    d = undefined_gate(0.02).review(proposal(strangle), Book((held,)), m, OK)
    assert d.verdict is Verdict.REJECT and d.reason == "limits:undefined_stress_loss"
    assert stress_check(d).measured == pytest.approx(base + unit)
    # Na een vroege assignment (één leg plus aandelen, max_loss nog inf): geen crash, de
    # aandelen tellen mee.
    put, call = held.legs
    assigned = dataclasses.replace(held, legs=(call, Leg(None, "TST", 100 * held.units,
                                                         put.contract.strike)))
    d = undefined_gate().review(proposal(strangle), Book((assigned,)), m, OK)
    assert stress_check(d).measured >= position_stress_loss_usd(assigned, 100.0) + unit
    assert position_stress_loss_usd(assigned, 100.0) >= 100 * held.units * 100.0 * 0.20
    # Zonder keten voor een open undefined-risk positie is het stressverlies onbekend: weigeren.
    elsewhere = dataclasses.replace(held, underlying="TSTB")
    d = undefined_gate().review(proposal(strangle), Book((elsewhere,)), m, OK)
    assert d.verdict is Verdict.REJECT and "undefined_stress_loss" in d.reason
    assert math.isinf(stress_check(d).measured)


# --- harde fouten --------------------------------------------------------------------------

def test_hard_errors_are_safety_breaches(spread):
    m = market()
    with pytest.raises(SafetyBreach, match="aantal"):
        GATE.review(proposal(spread, quantity=0), Book(), m, OK)
    other = (LegIntent(OptionContract("XYZ", date(2026, 12, 31), 95.0, Right.PUT), -1),)
    with pytest.raises(SafetyBreach, match="hoort niet"):
        GATE.review(proposal(spread, legs=other), Book(), m, OK)
    old = (LegIntent(OptionContract("TST", date(2026, 11, 20), 95.0, Right.PUT), -1),)
    with pytest.raises(SafetyBreach, match="verlopen"):
        GATE.review(proposal(spread, legs=old), Book(), m, OK)
    with pytest.raises(SafetyBreach, match="max_order_usd"):
        GATE.review(proposal(spread, quantity=1000), Book(), m, OK)
    with pytest.raises(SafetyBreach, match="buiten"):
        GATE.review(proposal(spread, limit_net=0.90), Book(), m, OK)   # meer credit dan kan
    with pytest.raises(SafetyBreach, match="buiten"):
        GATE.review(proposal(spread, limit_net=-0.50), Book(), m, OK)  # debit boven natuurlijk


def test_missing_quotes_are_a_rejection_not_a_breach(spread):
    bare = market(dataclasses.replace(cached_chain(), options=()))
    assert GATE.review(proposal(spread), Book(), bare, OK).reason == "quotes:missing"
    nochain = dataclasses.replace(market(), chains={})
    assert GATE.review(proposal(spread), Book(), nochain, OK).reason == "quotes:missing"


# --- sluiten en bijsturen ------------------------------------------------------------------

def _close(pos, m, kind=ActionKind.CLOSE, legs=None, new=()):
    a = Action(kind, pos.position_id, legs if legs is not None else pos.intents(),
               Urgency.STOP, "stop_loss", new)
    closing = tuple(LegIntent(leg.contract, -leg.ratio) for leg in a.legs) + new
    return OrderProposal.for_action(a, pos, -(mark(m.chains[pos.underlying], closing) or 0.0),
                                    pos.max_loss)


def test_a_full_close_is_always_approved(spread):
    pos = open_position(spread)
    m = market()
    p = _close(pos, m)
    assert p.kind is OrderKind.CLOSE and p.max_loss == 0.0 and p.quantity == pos.units
    assert [leg.ratio for leg in p.legs] == [1, -1]
    for state in (OK, GateState(halted=True), GateState(unwind=True)):
        assert GATE.review(p, Book((pos,)), m, state).verdict is Verdict.APPROVE
    bare = market(dataclasses.replace(cached_chain(), options=()))
    assert GATE.review(p, Book((pos,)), bare, OK).reason == "quotes:missing"


@pytest.fixture(scope="module")
def condor():
    (c,) = IronCondor(CFG.strategies.iron_condor, RULES).scan(market(), Book())
    return open_position(c, position_id="ic")


def test_adjust_and_roll_kinds(condor):
    m = market()
    calls = tuple(leg for leg in condor.intents() if leg.contract.right is Right.CALL)
    new = (LegIntent(OptionContract("TST", calls[0].contract.expiry, 104.0, Right.CALL), -1),
           LegIntent(OptionContract("TST", calls[0].contract.expiry, 109.0, Right.CALL), 1))
    adj = _close(condor, m, ActionKind.ADJUST, calls, new)
    assert adj.kind is OrderKind.ADJUST and len(adj.legs) == 4
    d = GATE.review(adj, Book((condor,)), m, OK)
    assert d.verdict is Verdict.APPROVE and d.reason == "adjust"
    assert GATE.review(adj, Book((condor,)), m, GateState(halted=True)).reason == "state:halt"
    assert GATE.review(adj, Book((condor,)), m, GateState(unwind=True)).reason == "state:unwind"
    leg = _close(condor, m, ActionKind.CLOSE_LEG, calls)
    assert GATE.review(leg, Book((condor,)), m, GateState(halted=True)).verdict is Verdict.APPROVE
    roll = _close(condor, m, ActionKind.ROLL, calls[:1], new[:1])
    assert roll.kind is OrderKind.ROLL


def test_a_change_that_breaks_a_limit_is_rejected_unless_it_reduces_risk(condor):
    m = market()
    calls = tuple(leg for leg in condor.intents() if leg.contract.right is Right.CALL)
    leg = _close(condor, m, ActionKind.CLOSE_LEG, calls)
    strict = CFG.model_copy(update={"risk": CFG.risk.model_copy(
        update={"max_abs_beta_delta_frac": 1e-6})})
    g = RiskGate(strict, SETTINGS.engines, Method.BAW)
    d = g.review(leg, Book((condor,)), m, OK)
    assert d.verdict is Verdict.REJECT and "beta_delta_usd" in d.reason
    # Eerst te veel vega, daarna minder: risicoverkleinend, ook in HALT.
    (dv,) = DebitVertical(CFG.strategies.debit_vertical, RULES).scan(market(iv=LOW_IV),
                                                                      Book())
    long_pos = open_position(dv, position_id="dv")
    vega_cap = CFG.model_copy(update={"risk": CFG.risk.model_copy(
        update={"max_vega_frac": 1e-6})})
    gv = RiskGate(vega_cap, SETTINGS.engines, Method.BAW)
    lng = long_pos.intents()[:1]
    half = _close(long_pos, m, ActionKind.CLOSE_LEG, lng)
    out = gv.review(half, Book((long_pos,)), m, GateState(halted=True))
    assert out.verdict in (Verdict.APPROVE, Verdict.REJECT)
    assert gate_mod.Verdict.APPROVE.value == "approve"


def test_risk_reducing_is_approved_in_halt(condor):
    m = market()
    calls = tuple(leg for leg in condor.intents() if leg.contract.right is Right.CALL)
    shorts = tuple(leg for leg in calls if leg.ratio < 0)
    strict = CFG.model_copy(update={"risk": CFG.risk.model_copy(
        update={"max_gamma_loss_5pct_frac": 1e-6})})
    g = RiskGate(strict, SETTINGS.engines, Method.BAW)
    cover = _close(condor, m, ActionKind.CLOSE_LEG, shorts)        # koop de short call terug
    d = g.review(cover, Book((condor,)), m, GateState(halted=True))
    assert d.verdict is Verdict.APPROVE and d.reason == "risk_reducing"
    assert d.approved.proposal.kind is OrderKind.CLOSE_LEG
    stays = dataclasses.replace(cover, legs=calls[:0] + (LegIntent(shorts[0].contract, 1),))
    assert g.review(stays, Book((condor, )), m, OK).verdict is Verdict.APPROVE


def test_a_change_leaves_other_positions_alone(condor, spread):
    m = market()
    calls = tuple(leg for leg in condor.intents() if leg.contract.right is Right.CALL)
    other = open_position(spread, position_id="other")
    gone = dataclasses.replace(open_position(spread, position_id="gone"),
                               status=PositionStatus.CLOSED)
    leg = _close(condor, m, ActionKind.CLOSE_LEG, calls)
    after = GATE._after(leg, Book((condor, other, gone)))
    assert [x.position_id for x in after] == ["ic", "other"]
    assert after[1] is other and len(after[0].legs) == 2
