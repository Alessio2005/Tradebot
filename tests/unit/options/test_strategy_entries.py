"""Entryregels per strategie (detailontwerp 7.4): welke combo, hoeveel, en wanneer niets."""
from __future__ import annotations

import dataclasses
import math
from datetime import date, timedelta

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CFG,
    HIGH_IV,
    LOW_IV,
    cached_chain,
    make_chain,
    market,
)
from tradebot.options.contracts import Right
from tradebot.options.events import Events
from tradebot.options.iv_metrics import IVRStatus
from tradebot.options.positions import Book, Leg, MarketState, StrategyPosition, Trend
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.credit import CreditSpread, IronCondor
from tradebot.options.strategies.debit import CalendarSpread, DebitVertical, LongVol
from tradebot.options.strategies.income import CashSecuredPut, CoveredCall
from tradebot.options.strategies.registry import build_strategies
from tradebot.options.strategies.short_vol import ShortVol
from tradebot.options.strategies.wheel import Wheel
from tradebot.utils.failfast import ConfigContractError

RULES = rules_from_config(CFG)
S = CFG.strategies
RICH = 1_000_000.0
NARROW = make_chain(strikes=tuple(float(k) for k in range(98, 103)))


def cs(**kw):
    return CreditSpread(S.credit_spread.model_copy(update=kw), RULES)


def legs_of(c):
    return [(leg.contract.expiry, leg.contract.strike, leg.contract.right, leg.ratio)
            for leg in c.legs]


# --- credit spread -------------------------------------------------------------------------

def test_credit_spread_follows_the_trend():
    (bull,) = cs().scan(market(trend=Trend.UP), Book())
    e = date(2026, 12, 31)
    assert legs_of(bull) == [(e, 95.0, Right.PUT, -1), (e, 94.0, Right.PUT, 1)]
    assert bull.max_loss == pytest.approx((1 - bull.net_credit) * 100 * bull.quantity)
    assert bull.rank_score == pytest.approx(bull.net_credit / (1 - bull.net_credit))
    assert bull.quantity == math.floor(500 / ((1 - bull.net_credit) * 100))
    assert "bull_put" in bull.rule_trace and "iv:high:ivr:ok" in bull.rule_trace
    (bear,) = cs().scan(market(trend=Trend.DOWN), Book())
    assert legs_of(bear)[0][2] is Right.CALL and "bear_call" in bear.rule_trace
    assert cs().scan(market(trend=Trend.FLAT), Book()) == []


@pytest.mark.parametrize(("strategy", "m"), [
    (lambda: cs(dte=(300, 400)), market),                       # geen expiratie
    (cs, lambda: market(events=Events("TST", None))),           # events oud
    (cs, lambda: market(iv=LOW_IV)),                            # IV te laag
    (lambda: cs(short_delta=(0.10, 0.12)), lambda: market(NARROW)),       # geen short
    (lambda: cs(min_credit_frac_width=0.99), market),           # geen breedte
    (lambda: cs(min_credit_frac_width=0.99, wing_width_strikes=(1, 100)),
     market),                                                   # strikes op
    (cs, lambda: market(capital=1_000.0)),                      # sizing 0
])
def test_credit_spread_skips(strategy, m):
    assert strategy().scan(m(), Book()) == []


# --- iron condor ---------------------------------------------------------------------------

def ic(**kw):
    return IronCondor(S.iron_condor.model_copy(update=kw), RULES)


def test_iron_condor_has_equal_wings_and_one_sided_max_loss():
    (c,) = ic().scan(market(trend=Trend.FLAT), Book())
    strikes = [leg[1] for leg in legs_of(c)]
    assert [leg[3] for leg in legs_of(c)] == [1, -1, -1, 1]
    assert strikes[1] - strikes[0] == strikes[3] - strikes[2] == 5.0
    assert c.max_loss == pytest.approx((5.0 - c.net_credit) * 100 * c.quantity)
    assert "wing_strikes:5" in c.rule_trace


@pytest.mark.parametrize("kw", [
    dict(dte=(300, 400)), dict(iv_condition="low"), dict(short_delta=0.9),
    dict(wing_width_strikes=(40, 50)), dict(min_credit_frac_width=0.99),
])
def test_iron_condor_skips(kw):
    assert ic(**kw).scan(market(), Book()) == []


def test_iron_condor_skips_without_budget_or_fresh_events():
    assert ic().scan(market(capital=5_000.0), Book()) == []
    assert ic().scan(market(events=Events("TST", None)), Book()) == []


# --- debit vertical ------------------------------------------------------------------------

def dv(**kw):
    return DebitVertical(S.debit_vertical.model_copy(update=kw), RULES)


def test_debit_vertical_follows_the_trend_and_caps_the_debit():
    (bull,) = dv().scan(market(iv=LOW_IV, trend=Trend.UP), Book())
    (lng, sht) = legs_of(bull)
    assert lng[2] is Right.CALL and lng[3] == 1 and sht[1] > lng[1]
    width = sht[1] - lng[1]
    debit = -bull.net_credit
    assert 0 < debit <= 0.5 * width
    assert bull.max_loss == pytest.approx(debit * 100 * bull.quantity)
    assert bull.rank_score == pytest.approx((width - debit) / debit)
    (bear,) = dv().scan(market(iv=LOW_IV, trend=Trend.DOWN), Book())
    assert legs_of(bear)[0][2] is Right.PUT and legs_of(bear)[1][1] < legs_of(bear)[0][1]
    assert dv().scan(market(iv=LOW_IV, trend=Trend.FLAT), Book()) == []


def test_debit_vertical_skips():
    low = market(iv=LOW_IV)
    assert dv().scan(market(iv=HIGH_IV), Book()) == []
    assert dv(dte=(300, 400)).scan(low, Book()) == []
    assert dv(long_delta=(0.95, 0.99)).scan(low, Book()) == []
    assert dv(max_debit_frac_width=0.01).scan(low, Book()) == []
    assert dv().scan(market(iv=LOW_IV, capital=2_000.0), Book()) == []
    assert dv(max_debit_frac_width=0.01).scan(market(NARROW, iv=LOW_IV), Book()) == []


# --- calendar ------------------------------------------------------------------------------

def cal(**kw):
    return CalendarSpread(S.calendar_spread.model_copy(update=kw), RULES)


def test_calendar_sells_near_and_buys_far_at_the_forward():
    (c,) = cal().scan(market(iv=LOW_IV), Book())
    (near, far) = legs_of(c)
    assert near[3] == -1 and far[3] == 1 and near[1] == far[1] == 100.0
    assert near[0] == date(2026, 12, 18) and far[0] == date(2027, 2, 19)
    assert c.net_credit < 0 and c.rank_score > 0 and "ivp:ivr:ok" in c.rule_trace


def _iv(**kw):
    return dataclasses.replace(LOW_IV, **kw)


@pytest.mark.parametrize(("iv", "ok"), [
    (_iv(ivp=80.0), False),
    (_iv(term_slope=None), False),
    (_iv(term_slope=-0.01), False),
    (_iv(status=IVRStatus.INSUFFICIENT_HISTORY, ivr=None, ivp=None, iv_rv_ratio=0.9), True),
    (_iv(status=IVRStatus.INSUFFICIENT_HISTORY, ivr=None, ivp=None, iv_rv_ratio=1.1), False),
    (_iv(status=IVRStatus.INSUFFICIENT_HISTORY, ivr=None, ivp=None, iv_rv_ratio=None), False),
])
def test_calendar_iv_conditions(iv, ok):
    assert bool(cal().scan(market(iv=iv), Book())) is ok


def test_calendar_fallback_can_be_off_and_iv_can_be_missing():
    fb = _iv(status=IVRStatus.INSUFFICIENT_HISTORY, ivr=None, ivp=None, iv_rv_ratio=0.9)
    off = dataclasses.replace(RULES, iv=dataclasses.replace(RULES.iv, fallback_enabled=False))
    strat = CalendarSpread(S.calendar_spread, off)
    assert strat.scan(market(iv=fb), Book()) == []
    m = market()
    assert cal().scan(dataclasses.replace(m, iv={}), Book()) == []


def test_calendar_skips():
    ch = cached_chain()
    no_far = dataclasses.replace(ch, options=tuple(
        o for o in ch.options if not (o.contract.expiry == date(2027, 2, 19)
                                      and o.contract.strike == 100.0)))
    no_greeks = dataclasses.replace(ch, options=tuple(
        dataclasses.replace(o, greeks=None) if o.contract.expiry == date(2026, 12, 18) else o
        for o in ch.options))
    assert cal().scan(market(no_far, iv=LOW_IV), Book()) == []
    assert cal().scan(market(no_greeks, iv=LOW_IV), Book()) == []
    assert cal(short_dte=(300, 400)).scan(market(iv=LOW_IV), Book()) == []
    assert cal().scan(market(iv=LOW_IV, capital=1_000.0), Book()) == []
    only_far = dataclasses.replace(ch, options=tuple(
        o for o in ch.options if o.contract.expiry != date(2026, 12, 18)
        or o.contract.right is Right.PUT))
    assert cal().scan(market(only_far, iv=LOW_IV), Book()) == []


# --- long straddle / strangle --------------------------------------------------------------

def lv(**kw):
    return LongVol(S.long_straddle_strangle.model_copy(update=kw), RULES)


def test_long_straddle_before_earnings():
    earn = date(2026, 12, 4)                                     # over 10 dagen
    m = market(iv=HIGH_IV, events=Events("TST", NOW.date(), (earn,)), capital=RICH)
    (c,) = lv().scan(m, Book())
    (put, call) = legs_of(c)
    assert c.variant == "earnings" and put[1] == call[1] == 100.0
    assert put[0] == date(2026, 12, 18) and put[3] == call[3] == 1   # eerste na earnings, DTE ≥ 14
    assert c.rank_score > 0 and f"earnings:{earn.isoformat()}" in c.rule_trace


def test_long_strangle_on_low_iv():
    (c,) = lv(style="strangle").scan(market(iv=LOW_IV, capital=RICH), Book())
    (put, call) = legs_of(c)
    assert c.variant == "low_iv" and put[1] < 100.0 < call[1] and put[0] == date(2026, 12, 31)


@pytest.mark.parametrize(("strategy", "m"), [
    (lv, lambda: market(iv=HIGH_IV, capital=RICH)),             # niets te doen
    (lambda: lv(low_iv_variant=False), lambda: market(iv=LOW_IV, capital=RICH)),
    (lambda: lv(low_iv_dte=(300, 400)), lambda: market(iv=LOW_IV, capital=RICH)),
    (lv, lambda: market(iv=LOW_IV, events=Events("TST", None), capital=RICH)),
    (lambda: lv(style="strangle", strangle_delta=(0.97, 0.98)),
     lambda: market(iv=LOW_IV, capital=RICH)),
    (lv, lambda: market(iv=LOW_IV)),                            # sizing 0
    (lambda: lv(earnings_min_dte=300, low_iv_variant=False),
     lambda: market(events=Events("TST", NOW.date(), (date(2026, 12, 4),)), capital=RICH)),
    (lambda: lv(low_iv_variant=False),
     lambda: market(events=Events("TST", NOW.date(), (date(2027, 1, 30),)), capital=RICH)),
])
def test_long_vol_skips(strategy, m):
    assert strategy().scan(m(), Book()) == []


def test_long_vol_skips_without_greeks():
    ch = cached_chain()
    bare = dataclasses.replace(ch, options=tuple(dataclasses.replace(o, greeks=None)
                                                 for o in ch.options))
    assert lv().scan(market(bare, iv=LOW_IV, capital=RICH), Book()) == []


# --- cash-secured put, covered call en wheel ------------------------------------------------

def csp(**kw):
    return CashSecuredPut(S.cash_secured_put.model_copy(update=kw), RULES, ("TST",))


def cc(**kw):
    return CoveredCall(S.covered_call.model_copy(update=kw), RULES, ("TST",))


def test_csp_reserves_cash_and_does_not_size_at_25k():
    rich = market(capital=RICH)
    (c,) = csp().scan(rich, Book(cash_usd=20_000.0))
    (leg,) = legs_of(c)
    assert leg[2] is Right.PUT and leg[3] == -1
    assert c.quantity == int(20_000 // (leg[1] * 100))
    assert c.max_loss == pytest.approx(leg[1] * 100 * c.quantity)
    assert csp().scan(market(), Book(cash_usd=1e9)) == []          # 25k: sizing 0
    assert csp().scan(rich, Book()) == []                          # geen cash
    assert CashSecuredPut(S.cash_secured_put, RULES, ()).scan(rich, Book(cash_usd=1e9)) == []
    assert csp().scan(market(iv=LOW_IV, capital=RICH), Book(cash_usd=1e9)) == []
    assert csp(dte=(300, 400)).scan(rich, Book(cash_usd=1e9)) == []
    assert csp(delta=(0.10, 0.12)).scan(market(NARROW, capital=RICH), Book(cash_usd=1e9)) == []


def _short_call(qty: int) -> StrategyPosition:
    k = next(o.contract for o in cached_chain().options if o.contract.right is Right.CALL)
    return StrategyPosition("c", "covered_call", "TST", (Leg(k, "TST", -qty, 1.0),), NOW,
                            qty, 100.0 * qty, 1.0)


def test_covered_call_needs_free_shares_above_the_cost_basis():
    rich = market(capital=100 * RICH)
    book = Book(shares={"TST": 300}, cost_basis={"TST": 104.0})
    (c,) = cc().scan(rich, book)
    (leg,) = legs_of(c)
    assert leg[2] is Right.CALL and leg[1] >= 104.0 and c.quantity == 3
    covered = dataclasses.replace(book, positions=(_short_call(2),))
    assert cc().scan(rich, covered)[0].quantity == 1
    assert cc().scan(rich, dataclasses.replace(book, positions=(_short_call(3),))) == []
    assert cc().scan(rich, Book()) == []
    assert cc().scan(market(), book) == []                          # 25k: sizing 0
    assert cc(dte=(300, 400)).scan(rich, book) == []
    assert cc().scan(rich, dataclasses.replace(book, cost_basis={"TST": 500.0})) == []


def test_wheel_scans_by_phase():
    w = Wheel(csp(), cc())
    rich = market(capital=RICH)
    book = Book(shares={"TST": 100}, cash_usd=1e7, cost_basis={"TST": 90.0})
    (idle,) = w.scan(rich, book)
    assert idle.strategy == "wheel" and idle.variant == "csp"
    (shares,) = w.scan(rich, dataclasses.replace(book, wheel_phase={"TST": "shares"}))
    assert shares.variant == "cc"
    assert w.scan(rich, dataclasses.replace(book, wheel_phase={"TST": "csp"})) == []


# --- short straddle / strangle en de registry ------------------------------------------------

def sv(**kw):
    return ShortVol(S.short_straddle_strangle.model_copy(update=kw), RULES)


def test_short_vol_has_undefined_risk_and_a_stress_size():
    (c,) = sv().scan(market(capital=RICH), Book())
    assert math.isinf(c.max_loss) and [leg[3] for leg in legs_of(c)] == [-1, -1]
    assert any(t.startswith("stress_loss_unit:") for t in c.rule_trace)
    (s,) = sv(style="straddle").scan(market(capital=RICH), Book())
    assert legs_of(s)[0][1] == legs_of(s)[1][1]
    assert sv().scan(market(capital=1_000.0), Book()) == []
    assert sv(iv_condition="low").scan(market(), Book()) == []
    assert sv(dte=(300, 400)).scan(market(), Book()) == []


def test_short_vol_skips_without_quotes():
    ch = cached_chain()
    bare = dataclasses.replace(ch, options=tuple(
        dataclasses.replace(o, quote=dataclasses.replace(o.quote, bid=None)) for o in ch.options))
    assert sv().scan(market(bare), Book()) == []
    only_calls = dataclasses.replace(ch, options=tuple(o for o in ch.options
                                                       if o.contract.right is Right.CALL))
    assert sv(style="straddle").scan(market(only_calls), Book()) == []


def test_registry_order_and_the_undefined_risk_refusal():
    assert [s.name for s in build_strategies(CFG)] == [
        "credit_spread", "iron_condor", "debit_vertical", "calendar_spread",
        "long_straddle_strangle"]
    on = {k: {"enabled": True} for k in type(S).model_fields}
    everything = CFG.model_copy(update={"strategies": S.model_copy(update={
        k: getattr(S, k).model_copy(update=v) for k, v in on.items()})})
    with pytest.raises(ConfigContractError, match="undefined risk"):
        build_strategies(everything)
    allowed = everything.model_copy(update={"risk": CFG.risk.model_copy(
        update={"allow_undefined_risk": True})})
    assert len(build_strategies(allowed)) == 9


def test_scans_are_pure():
    m = market(iv=LOW_IV, capital=RICH)
    for s in build_strategies(CFG):
        assert repr(s.scan(m, Book())) == repr(s.scan(m, Book()))


def test_market_today():
    m = market()
    assert isinstance(m, MarketState) and m.today == NOW.date()
    assert (m.today - timedelta(days=1)) < m.today
