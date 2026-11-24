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
