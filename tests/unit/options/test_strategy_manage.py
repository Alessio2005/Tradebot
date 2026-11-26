"""Beheerregels per strategie (detailontwerp 7.4): de eerste regel die geldt, wint."""
from __future__ import annotations

import dataclasses
import itertools
from datetime import date, datetime

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CFG,
    LOW_IV,
    cached_chain,
    make_chain,
    market,
    open_position,
)
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.events import Events
from tradebot.options.positions import ActionKind, Book, Leg, StrategyPosition, Trend, Urgency
from tradebot.options.rates import CashDividend
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.credit import CreditSpread, IronCondor
from tradebot.options.strategies.debit import CalendarSpread, DebitVertical, LongVol
from tradebot.options.strategies.income import CashSecuredPut, CoveredCall
from tradebot.options.strategies.short_vol import ShortVol
from tradebot.options.strategies.wheel import (
    TRANSITIONS,
    Wheel,
    WheelEvent,
    WheelPhase,
    transition,
)
from tradebot.utils.failfast import DataContractError

RULES = rules_from_config(CFG)
S = CFG.strategies
RICH = 1_000_000.0


def at(y: int, m: int, d: int, h: int = 10) -> datetime:
    return datetime(y, m, d, h, 0, tzinfo=NY)


def later(spot: float = 100.0, now: datetime = NOW, **kw) -> object:
    return market(cached_chain(spot, now=now), now=now, **kw)


def reasons(strategy, pos, m):
    return [(a.kind, a.urgency, a.reason) for a in strategy.manage(pos, m)]


CLOSE_STOP = ActionKind.CLOSE, Urgency.STOP
CLOSE_NORMAL = ActionKind.CLOSE, Urgency.NORMAL


# --- credit spread -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bull_put():
    (c,) = CreditSpread(S.credit_spread, RULES).scan(market(trend=Trend.UP), Book())
    return open_position(c)


@pytest.mark.parametrize(("kw", "m", "want"), [
    ({}, lambda: later(now=at(2026, 12, 30, 15)), [(*CLOSE_STOP, "expiry")]),
    ({}, lambda: later(90.0), [(*CLOSE_STOP, "stop_loss")]),
    ({"stop_loss_multiple_credit": 20.0}, lambda: later(93.5), [(*CLOSE_STOP, "short_itm")]),
    ({}, lambda: later(115.0), [(*CLOSE_NORMAL, "take_profit")]),
    ({"take_profit_frac": 0.99}, lambda: later(now=at(2026, 12, 11)),
     [(*CLOSE_NORMAL, "dte")]),
    ({}, later, []),
])
def test_credit_spread_exits_in_order(bull_put, kw, m, want):
    s = CreditSpread(S.credit_spread.model_copy(update=kw), RULES)
    assert reasons(s, bull_put, m()) == want


def test_credit_spread_without_a_chain_only_keeps_the_time_rules(bull_put):
    s = CreditSpread(S.credit_spread, RULES)
    m = later(now=at(2026, 12, 11))
    assert reasons(s, bull_put, dataclasses.replace(m, chains={})) == [(*CLOSE_NORMAL, "dte")]
    assert reasons(s, bull_put, dataclasses.replace(later(), chains={})) == []


def test_bear_call_short_itm(bull_put):
    (c,) = CreditSpread(S.credit_spread, RULES).scan(market(trend=Trend.DOWN), Book())
    s = CreditSpread(S.credit_spread.model_copy(update={"stop_loss_multiple_credit": 20.0}),
                     RULES)
    assert reasons(s, open_position(c), later(107.5)) == [(*CLOSE_STOP, "short_itm")]


# --- iron condor ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def condor():
    (c,) = IronCondor(S.iron_condor, RULES).scan(market(), Book())
    return open_position(c)


def test_iron_condor_adjusts_the_untested_side_once(condor):
    s = IronCondor(S.iron_condor, RULES)
    (a,) = s.manage(condor, later(94.0))
    assert a.kind is ActionKind.ADJUST and a.reason == "adjust:put_tested"
    assert {leg.contract.right for leg in a.legs} == {Right.CALL}
    new_short, new_long = a.new_legs
    assert new_short.ratio == -1 and new_long.ratio == 1
    assert new_long.contract.strike - new_short.contract.strike == 5.0
    assert s.manage(dataclasses.replace(condor, adjustments=1), later(94.0)) == []
    assert s.manage(condor, later()) == []
    assert reasons(s, condor, later(85.0)) == [(*CLOSE_STOP, "stop_loss")]


def test_iron_condor_adjustment_needs_the_strikes(condor):
    s = IronCondor(S.iron_condor, RULES)
    assert s.manage(condor, dataclasses.replace(later(94.0), chains={})) == []
    short_calls = make_chain(94.0, strikes=tuple(float(k) for k in range(70, 105)))
    assert s.manage(condor, market(short_calls)) == []
    no_short_put = dataclasses.replace(cached_chain(), options=tuple(
        o for o in cached_chain().options if o.contract.strike != 92.0))
    assert s.manage(condor, market(no_short_put)) == []
    legs = list(condor.legs)
    wide = dataclasses.replace(condor, legs=(
        *legs[:2],
        Leg(dataclasses.replace(legs[2].contract, strike=101.0), "TST", legs[2].qty, 1.0),
        Leg(dataclasses.replace(legs[3].contract, strike=130.0), "TST", legs[3].qty, 0.0)))
    st = IronCondor(S.iron_condor.model_copy(update={"stop_loss_multiple_credit": 20.0}),
                    RULES)
    assert st.manage(wide, later(94.0)) == []


# --- debit vertical en calendar ------------------------------------------------------------

@pytest.fixture(scope="module")
def bull_call():
    (c,) = DebitVertical(S.debit_vertical, RULES).scan(market(iv=LOW_IV), Book())
    return open_position(c)


@pytest.mark.parametrize(("m", "want"), [
    (lambda: later(now=at(2027, 1, 14, 15)), [(*CLOSE_STOP, "expiry")]),
    (lambda: later(125.0), [(*CLOSE_NORMAL, "take_profit")]),
    (lambda: later(92.0), [(*CLOSE_STOP, "stop_loss")]),
    (lambda: later(now=at(2026, 12, 28)), [(*CLOSE_NORMAL, "dte")]),
    (later, []),
    (lambda: dataclasses.replace(later(), chains={}), []),
])
def test_debit_vertical_exits(bull_call, m, want):
    assert reasons(DebitVertical(S.debit_vertical, RULES), bull_call, m()) == want


@pytest.fixture(scope="module")
def calendar():
    (c,) = CalendarSpread(S.calendar_spread, RULES).scan(market(iv=LOW_IV), Book())
    return open_position(c)


@pytest.mark.parametrize(("m", "want"), [
    (lambda: later(now=at(2026, 12, 17, 15)), [(*CLOSE_STOP, "expiry")]),
    (lambda: later(now=at(2026, 12, 16)), [(*CLOSE_NORMAL, "take_profit")]),
    (lambda: later(80.0), [(*CLOSE_STOP, "stop_loss")]),
    (later, []),
])
def test_calendar_exits(calendar, m, want):
    assert reasons(CalendarSpread(S.calendar_spread, RULES), calendar, m()) == want


# --- long straddle -------------------------------------------------------------------------

EARN = Events("TST", NOW.date(), (date(2026, 12, 4),))


@pytest.fixture(scope="module")
def straddle():
    s = LongVol(S.long_straddle_strangle, RULES)
    (c,) = s.scan(market(events=EARN, capital=RICH), Book())
    return open_position(c)


@pytest.mark.parametrize(("m", "want"), [
    (lambda: later(now=at(2026, 12, 3, 15), events=EARN), [(*CLOSE_STOP, "pre_earnings")]),
    (lambda: later(now=at(2026, 12, 2), events=EARN), []),
    (lambda: later(120.0, events=EARN), [(*CLOSE_NORMAL, "take_profit")]),
    (lambda: dataclasses.replace(later(now=at(2026, 12, 3, 15)), events={}), []),
    (lambda: later(now=at(2026, 12, 3, 15), events=Events("TST", NOW.date())), []),
])
def test_long_straddle_exits(straddle, m, want):
    assert straddle.variant == "earnings"
    assert reasons(LongVol(S.long_straddle_strangle, RULES), straddle, m()) == want


def test_low_iv_straddle_ignores_earnings(straddle):
    low = dataclasses.replace(straddle, variant="low_iv")
    s = LongVol(S.long_straddle_strangle, RULES)
    assert s.manage(low, later(now=at(2026, 12, 3, 15), events=EARN)) == []


# --- cash-secured put en covered call --------------------------------------------------------

def _csp(**kw):
    return CashSecuredPut(S.cash_secured_put.model_copy(update=kw), RULES, ("TST",))


@pytest.fixture(scope="module")
def short_put():
    (c,) = _csp().scan(market(capital=RICH), Book(cash_usd=1e7))
    return open_position(c, phase="csp")


def test_csp_rolls_or_closes(short_put):
    s = _csp(take_profit_frac=0.99)
    (roll,) = s.manage(short_put, later(now=at(2026, 12, 14)))
    assert roll.kind is ActionKind.ROLL and roll.new_legs[0].ratio == -1
    assert roll.new_legs[0].contract.expiry > short_put.legs[0].contract.expiry  # type: ignore[union-attr]
    stuck = _csp(take_profit_frac=0.99, dte=(300, 400))
    assert reasons(stuck, short_put, later(now=at(2026, 12, 14))) == [(*CLOSE_NORMAL, "dte")]
    assert reasons(_csp(), short_put, later(120.0)) == [(*CLOSE_NORMAL, "take_profit")]
    assert reasons(_csp(), short_put, later(now=at(2026, 12, 30, 15))) == [
        (*CLOSE_STOP, "expiry")]
    assert _csp().manage(short_put, later()) == []
    assert _csp().manage(short_put, dataclasses.replace(later(), chains={})) == []
    no_leg = dataclasses.replace(cached_chain(), options=())
    assert _csp().manage(short_put, market(no_leg)) == []


def _cc(**kw):
    return CoveredCall(S.covered_call.model_copy(update=kw), RULES, ("TST",))


def _call(strike: float, expiry: date = date(2026, 12, 18), credit: float = 1.0):
    k = OptionContract("TST", expiry, strike, Right.CALL)
    return StrategyPosition("cc", "covered_call", "TST", (Leg(k, "TST", -1, credit),), NOW,
                            1, credit * 100, strike * 100, phase="cc")


DIV = Events("TST", date(2026, 12, 10), dividends=(CashDividend(date(2026, 12, 11), 0.5),))


def test_covered_call_avoids_early_assignment():
    pos = _call(80.0, credit=20.0)
    m = later(now=at(2026, 12, 10), events=DIV)
    assert reasons(_cc(), pos, m) == [(*CLOSE_STOP, "early_assignment")]
    no_div = later(now=at(2026, 12, 10), events=Events("TST", date(2026, 12, 10)))
    assert reasons(_cc(), pos, no_div) == [(*CLOSE_NORMAL, "dte")]
    assert reasons(_cc(), pos, dataclasses.replace(m, events={})) == [(*CLOSE_NORMAL, "dte")]
    otm = _call(120.0, credit=0.5)
    assert reasons(_cc(), otm, later(now=at(2026, 12, 10), events=DIV)) == [
        (*CLOSE_NORMAL, "take_profit")]


def test_covered_call_rolls_up_and_out_and_other_exits():
    pos = _call(104.0, date(2026, 12, 31), credit=1.0)
    (roll,) = _cc(take_profit_frac=0.99).manage(pos, later(now=at(2026, 12, 14)))
    assert roll.kind is ActionKind.ROLL
    assert roll.new_legs[0].contract.strike >= 104.0
    assert reasons(_cc(), pos, later(now=at(2026, 12, 30, 15))) == [(*CLOSE_STOP, "expiry")]
    assert _cc().manage(pos, dataclasses.replace(later(), chains={})) == []
    assert _cc().manage(pos, market(dataclasses.replace(cached_chain(), options=()))) == []
    assert _cc(take_profit_frac=0.99).manage(pos, later()) == []


# --- wheel en short vol --------------------------------------------------------------------

def test_wheel_delegates_by_phase(short_put):
    w = Wheel(_csp(), _cc())
    assert reasons(w, short_put, later(120.0)) == [(*CLOSE_NORMAL, "take_profit")]
    call = _call(120.0, credit=0.5)
    assert reasons(w, call, later(now=at(2026, 12, 10))) == [(*CLOSE_NORMAL, "take_profit")]


def test_wheel_transitions_are_exactly_the_table():
    assert transition(WheelPhase.IDLE, WheelEvent.CSP_OPENED) is WheelPhase.CSP
    assert transition(WheelPhase.CSP, WheelEvent.ASSIGNED) is WheelPhase.SHARES
    assert transition(WheelPhase.CSP, WheelEvent.CSP_CLOSED) is WheelPhase.IDLE
    assert transition(WheelPhase.SHARES, WheelEvent.CC_OPENED) is WheelPhase.CC
    assert transition(WheelPhase.CC, WheelEvent.CC_CLOSED) is WheelPhase.SHARES
    assert transition(WheelPhase.CC, WheelEvent.CALLED_AWAY) is WheelPhase.IDLE
    for p in WheelPhase:
        assert transition(p, WheelEvent.UNWIND) is WheelPhase.IDLE
    allowed = set(TRANSITIONS)
    assert len(allowed) == 10
    for p, e in itertools.product(WheelPhase, WheelEvent):
        if (p, e) not in allowed:
            with pytest.raises(DataContractError, match="geen overgang"):
                transition(p, e)


def short_vol(**kw):
    return ShortVol(S.short_straddle_strangle.model_copy(update={"underlyings": ("TST",), **kw}),
                    RULES)


@pytest.fixture(scope="module")
def strangle():
    (c,) = short_vol().scan(market(capital=RICH), Book())
    return open_position(c)


def test_short_vol_manages_like_a_credit_position(strangle):
    s = short_vol()
    assert s.manage(strangle, later()) == []
    assert reasons(s, strangle, later(80.0)) == [(*CLOSE_STOP, "stop_loss")]
    assert reasons(s, strangle, later(now=at(2026, 12, 30, 15))) == [(*CLOSE_STOP, "expiry")]
    assert reasons(short_vol(take_profit_frac=0.99), strangle, later(now=at(2026, 12, 11))) == [
        (*CLOSE_NORMAL, "dte")]


@pytest.mark.parametrize("spot", [95.0, 104.0])
def test_short_vol_closes_a_short_leg_past_the_delta_exit(strangle, spot):
    # De legs staan op ~0,16 delta; bij 95 of 104 is één kant > 0,30, het stopverlies nog niet.
    assert reasons(short_vol(), strangle, later(spot)) == [(*CLOSE_STOP, "short_delta")]
    assert reasons(short_vol(short_delta_exit=0.5), strangle, later(spot)) == []


def test_short_vol_delta_exit_goes_before_the_time_rules(strangle):
    s = short_vol(take_profit_frac=0.99)
    m = later(106.0, now=at(2026, 12, 11))     # call-delta ~0,34
    assert reasons(s, strangle, m) == [(*CLOSE_STOP, "short_delta")]
    assert reasons(s, strangle, dataclasses.replace(m, chains={})) == [(*CLOSE_NORMAL, "dte")]


def test_manage_is_pure(bull_put):
    s = CreditSpread(S.credit_spread, RULES)
    m = later(93.5)
    assert repr(s.manage(bull_put, m)) == repr(s.manage(bull_put, m))
