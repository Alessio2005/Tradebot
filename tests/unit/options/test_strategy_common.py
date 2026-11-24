"""De gedeelde keuzeregels, poorten, sizing en de expiratieregel (detailontwerp 7.2, 7.3, 7.5)."""
from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, time

import pandas as pd
import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import CFG, EXPIRIES, fake, fake_chain, market
from tradebot.options.calendar import NY
from tradebot.options.contracts import Right
from tradebot.options.events import Events
from tradebot.options.iv_metrics import IVCondition
from tradebot.options.positions import (
    ActionKind,
    Book,
    Leg,
    LegIntent,
    PositionStatus,
    StrategyPosition,
    Trend,
    Urgency,
    mark,
    position_greek,
    trend_of,
)
from tradebot.options.strategies.common import (
    close,
    dte,
    entry_blockers,
    expiry_close_due,
    iv_ok,
    otm_step,
    parse_time,
    pick_expiry,
    rules_from_config,
    size,
    strike_at_delta,
    strike_nearest,
)
from tradebot.utils.failfast import DataContractError

RULES = rules_from_config(CFG)
E = EXPIRIES[1]


def test_rules_and_times_come_from_the_config():
    assert parse_time("09:05") == time(9, 5)
    assert RULES.close_before_expiry == time(15, 0) and RULES.max_loss_frac_capital == 0.02
    assert RULES.max_loss_per_underlying_frac == 0.05 and RULES.iv.lookback_days == 252


def test_pick_expiry_takes_the_middle_and_the_earliest_on_a_tie():
    ch = fake_chain(fake(100, Right.CALL, date(2026, 12, 30)),   # DTE 36
                    fake(100, Right.CALL, date(2027, 1, 2)),     # DTE 39
                    fake(100, Right.CALL, date(2027, 2, 1)))
    assert pick_expiry(ch, 30, 45, NOW.date()) == date(2026, 12, 30)
    assert pick_expiry(ch, 37, 45, NOW.date()) == date(2027, 1, 2)
    assert pick_expiry(ch, 200, 300, NOW.date()) is None
    assert dte(date(2026, 12, 30), NOW.date()) == 36


def test_strike_at_delta_breaks_ties_to_the_farthest_otm():
    ch = fake_chain(fake(95, Right.PUT, delta=-0.20), fake(90, Right.PUT, delta=-0.30),
                    fake(97, Right.PUT, delta=-0.40), fake(99, Right.PUT, delta=None))
    assert strike_at_delta(ch, E, Right.PUT, 0.25).contract.strike == 90.0
    assert strike_at_delta(ch, E, Right.PUT, 0.25, (0.15, 0.22)).contract.strike == 95.0
    assert strike_at_delta(ch, E, Right.PUT, 0.25, (0.5, 0.6)) is None
    calls = fake_chain(fake(105, Right.CALL, delta=0.30), fake(110, Right.CALL, delta=0.20))
    assert strike_at_delta(calls, E, Right.CALL, 0.25).contract.strike == 110.0


def test_strike_nearest_and_otm_steps():
    ch = fake_chain(*(fake(k, r) for k in (98, 100, 102) for r in (Right.PUT, Right.CALL)))
    assert strike_nearest(ch, E, Right.CALL, 101.0).contract.strike == 100.0
    assert strike_nearest(ch, EXPIRIES[0], Right.CALL, 101.0) is None
    put100 = ch.find(E, 100, Right.PUT)
    call100 = ch.find(E, 100, Right.CALL)
    assert otm_step(ch, put100, 1).contract.strike == 98.0
    assert otm_step(ch, call100, 1).contract.strike == 102.0
    assert otm_step(ch, put100, 2) is None and otm_step(ch, call100, 2) is None


def test_iv_ok_reports_which_rule_decided():
    m = market()
    assert iv_ok(m, "TST", IVCondition.HIGH, RULES) == (True, "iv:high:ivr:ok")
    assert iv_ok(m, "TST", IVCondition.LOW, RULES) == (False, "iv:low:ivr:no")
    assert iv_ok(m, "XYZ", IVCondition.HIGH, RULES) == (False, "iv:none")


def test_entry_blockers():
    until = date(2026, 12, 31)
    assert entry_blockers(market(), "TST", until, RULES) == []
    stale = market(events=Events("TST", date(2026, 11, 1)))
    assert entry_blockers(stale, "TST", until, RULES) == ["events:stale"]
    assert entry_blockers(market(), "XYZ", until, RULES) == ["events:stale"]
    earn = market(events=Events("TST", NOW.date(), (date(2026, 12, 10),)))
    assert entry_blockers(earn, "TST", until, RULES) == ["events:earnings_in_term"]
    lax = dataclasses.replace(RULES, earnings_blackout=False)
    assert entry_blockers(earn, "TST", until, lax) == []


def _pos(max_loss: float, und: str = "TST", status: PositionStatus = PositionStatus.OPEN):
    leg = Leg(fake(100, Right.PUT).contract, und, -1, 1.0)
    return StrategyPosition("x", "credit_spread", und, (leg,), NOW, 1, 100.0, max_loss,
                            status=status)


def test_size_takes_the_smaller_of_budget_and_room():
    m = market()                                     # 25k: budget 500, ruimte 1250
    assert size(100.0, m, Book(), "TST", RULES) == 5
    book = Book((_pos(1000.0), _pos(5000.0, "OTHER"), _pos(9e9, status=PositionStatus.CLOSED)))
    assert size(100.0, m, book, "TST", RULES) == 2   # ruimte 250
    assert size(100.0, m, Book((_pos(2000.0),)), "TST", RULES) == 0
    assert size(math.inf, m, Book(), "TST", RULES) == 0
    assert size(0.0, m, Book(), "TST", RULES) == 0


@pytest.mark.parametrize(("now", "due"), [
    (datetime(2026, 12, 29, 16, 0, tzinfo=NY), False),
    (datetime(2026, 12, 30, 14, 59, tzinfo=NY), False),
    (datetime(2026, 12, 30, 15, 0, tzinfo=NY), True),
    (datetime(2026, 12, 31, 9, 30, tzinfo=NY), True),
])
def test_expiry_close_is_due_on_the_trading_day_before(now, due):
    assert expiry_close_due(date(2026, 12, 31), now, market(), RULES) is due


def test_close_actions():
    p = _pos(100.0)
    a = close(p, Urgency.STOP, "expiry")
    assert a.kind is ActionKind.CLOSE and a.legs == p.intents() and a.urgency is Urgency.STOP
    leg = (LegIntent(fake(100, Right.PUT).contract, -1),)
    assert close(p, Urgency.NORMAL, "x", leg).kind is ActionKind.CLOSE_LEG


def test_positions_validate_and_derive():
    with pytest.raises(DataContractError, match="units"):
        StrategyPosition("x", "s", "TST", (), NOW, 0, 0.0, 0.0)
    with pytest.raises(DataContractError, match="max_loss"):
        StrategyPosition("x", "s", "TST", (), NOW, 1, 0.0, -1.0)
    share = Leg(None, "TST", 100, 99.0)
    opt = Leg(fake(100, Right.PUT).contract, "TST", -2, 1.0)
    p = StrategyPosition("x", "s", "TST", (share, opt), NOW, 2, 200.0, 10.0)
    assert p.option_legs == (opt,) and p.intents()[0].ratio == -1
    assert p.entry_credit == 1.0 and p.expiries() == (E,)


def test_mark_and_position_greeks():
    ch = fake_chain(fake(100, Right.PUT, mid=2.0), fake(95, Right.PUT, mid=0.5),
                    fake(90, Right.PUT, delta=None))
    legs = (LegIntent(ch.options[0].contract, -1), LegIntent(ch.options[1].contract, 1))
    assert mark(ch, legs) == pytest.approx(-1.5)
    assert position_greek(ch, legs, "vega") == pytest.approx(0.0)
    missing = (LegIntent(fake(50, Right.PUT).contract, 1),)
    assert mark(ch, missing) is None and position_greek(ch, missing, "vega") is None
    no_greeks = (LegIntent(ch.options[2].contract, 1),)
    assert position_greek(ch, no_greeks, "delta") is None
    nobid = fake_chain(dataclasses.replace(
        ch.options[0], quote=dataclasses.replace(ch.options[0].quote, bid=None)))
    assert mark(nobid, legs[:1]) is None


def test_trend_against_the_sma50():
    up = pd.DataFrame({"close": [100.0] * 49 + [110.0]})
    assert trend_of(up) is Trend.UP
    assert trend_of(pd.DataFrame({"close": [100.0] * 49 + [90.0]})) is Trend.DOWN
    assert trend_of(pd.DataFrame({"close": [100.0] * 50})) is Trend.FLAT
    assert trend_of(up.iloc[:10]) is None
