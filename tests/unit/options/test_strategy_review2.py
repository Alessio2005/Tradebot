"""Reviewfixes strategieën (review 2026-11-27, tweede ronde en #6/#24/#25/#26): per punt het
scenario uit de review."""
from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime, timedelta

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CFG,
    HIGH_IV,
    LOW_IV,
    cached_chain,
    fake,
    fake_chain,
    make_chain,
    market,
    open_position,
    skewed,
)
from tradebot.options.calendar import NY
from tradebot.options.contracts import Right
from tradebot.options.events import Events
from tradebot.options.positions import (
    ActionKind,
    Book,
    Leg,
    PositionStatus,
    StrategyPosition,
    Trend,
    Urgency,
)
from tradebot.options.rates import CashDividend
from tradebot.options.selection import admit, selection_rules
from tradebot.options.strategies.common import (
    atm_iv,
    early_assignment_risk,
    entry_blockers,
    natural_mark,
    rules_from_config,
    short_call_dividend_blockers,
)
from tradebot.options.strategies.credit import CreditSpread, IronCondor, _spread
from tradebot.options.strategies.debit import CalendarSpread, DebitVertical, LongVol
from tradebot.options.strategies.income import CashSecuredPut, CoveredCall
from tradebot.options.strategies.short_vol import ShortVol
from tradebot.options.vol_regime import (
    RVForecast,
    VolRegime,
    VolState,
    classify,
    regime_settings,
)
from tradebot.schemas.options import IronCondorConfig, ShortVolConfig

RULES = rules_from_config(CFG)
S = CFG.strategies
TODAY = NOW.date()
RICH = 1_000_000.0
E37 = date(2026, 12, 31)          # DTE 37 vanaf NOW


def _events(earnings=(), dividends=()) -> Events:
    return Events("TST", TODAY, tuple(earnings), tuple(dividends))


def cs(**kw):
    return CreditSpread(S.credit_spread.model_copy(update=kw), RULES)


def ic(**kw):
    return IronCondor(S.iron_condor.model_copy(update=kw), RULES)


def dv(**kw):
    return DebitVertical(S.debit_vertical.model_copy(update=kw), RULES)


def cal(**kw):
    return CalendarSpread(S.calendar_spread.model_copy(update=kw), RULES)


def sv(**kw):
    return ShortVol(S.short_straddle_strangle.model_copy(update={"underlyings": ("TST",), **kw}),
                    RULES)


# --- N1: de earningsdag zelf telt mee in de blackout ----------------------------------------

@pytest.mark.parametrize(("earnings", "blocked"), [
    ((TODAY,), True),                               # vandaag: na de slotbel, de sprong komt morgen
    ((TODAY + timedelta(days=1),), True),
    ((E37,), True),                                 # de expiratiedag zelf
    ((TODAY - timedelta(days=1),), False),          # gisteren: voorbij
    ((E37 + timedelta(days=1),), False),            # na de expiratie
    ((), False),
])
def test_the_earnings_blackout_covers_today_through_expiry(earnings, blocked):
    m = market(events=_events(earnings))
    assert bool(entry_blockers(m, "TST", E37, RULES)) is blocked
    assert bool(cs().scan(m, Book())) is not blocked


def test_earnings_between_keeps_its_open_start_for_other_callers():
    ev = _events((TODAY,))
    assert not ev.earnings_between(TODAY, E37) and ev.earnings_within(TODAY, E37)
    assert ev.next_earnings(TODAY) is None          # next_earnings blijft strikt na `after`


# --- #6: ex-dividend op short calls buiten de covered call ------------------------------------

BIG_DIV = _events(dividends=(CashDividend(date(2026, 12, 10), 50.0),))
SMALL_DIV = _events(dividends=(CashDividend(date(2026, 12, 10), 0.05),))
CASES = [
    (cs, dict(trend=Trend.DOWN)),                       # bear call
    (ic, {}),                                           # condor, callkant
    (sv, dict(capital=RICH)),                           # short strangle
    (dv, dict(trend=Trend.UP, iv=LOW_IV)),              # bull call spread
    (cal, dict(iv=LOW_IV)),                             # calendar, korte leg
]


@pytest.mark.parametrize(("make", "kw"), CASES)
def test_a_short_call_does_not_open_before_a_dividend_it_cannot_survive(make, kw):
    assert make().scan(market(events=SMALL_DIV, **kw), Book())          # extrinsiek ≫ dividend
    assert make().scan(market(events=BIG_DIV, **kw), Book()) == []      # extrinsiek < dividend


@pytest.mark.parametrize(("make", "kw"), CASES)
def test_a_dividend_that_is_already_ex_or_after_the_expiry_does_not_block(make, kw):
    today = _events(dividends=(CashDividend(TODAY, 50.0),))            # ex vandaag: voorbij
    late = _events(dividends=(CashDividend(date(2027, 3, 1), 50.0),))
    assert make().scan(market(events=today, **kw), Book())
    assert make().scan(market(events=late, **kw), Book())


def test_put_spreads_ignore_the_dividend_and_the_helper_is_exact():
    assert cs().scan(market(events=BIG_DIV, trend=Trend.UP), Book())    # bull put: geen short call
    m = market(events=BIG_DIV)
    chain = m.chains["TST"]
    call = fake(100.0, Right.CALL, mid=0.5)
    assert early_assignment_risk(call, 100.0, 0.6) and not early_assignment_risk(call, 100.0, 0.4)
    assert not early_assignment_risk(call, 100.0, 0.0)
    itm = fake(90.0, Right.CALL, mid=10.5)                 # intrinsiek 10, extrinsiek 0,5
    assert early_assignment_risk(itm, 100.0, 0.6)
    put = fake(110.0, Right.PUT, mid=10.5)                 # put: intrinsiek 10
    assert early_assignment_risk(put, 100.0, 0.6)
    blind = dataclasses.replace(fake(100.0, Right.CALL), quote=dataclasses.replace(
        fake(100.0, Right.CALL).quote, bid=None))
    assert not early_assignment_risk(blind, 100.0, 5.0)
    legs = cs().scan(market(events=SMALL_DIV, trend=Trend.DOWN), Book())[0].legs
    assert short_call_dividend_blockers(m, "TST", legs) == ["events:ex_dividend"]
    assert short_call_dividend_blockers(m, "TST", legs[1:]) == []         # long leg: geen risico
    assert short_call_dividend_blockers(dataclasses.replace(m, events={}), "TST", legs) == []
    assert short_call_dividend_blockers(dataclasses.replace(m, chains={}), "TST", legs) == []
    gone = dataclasses.replace(chain, options=())
    assert short_call_dividend_blockers(market(gone, events=BIG_DIV), "TST", legs) == []


# --- N12: forward-vol-regel tegen de RV over [h1, h2] -------------------------------------------

def test_calendar_forward_vol_compares_with_the_rv_over_the_forward_window():
    near = date(2026, 12, 18)
    chain = make_chain(vol=lambda k, e: 0.30 if e == near else 0.25)       # σ_f ≈ 0,228
    # Een stijgende vol: blend 0,15 trekt naar de lange RV 0,30. σ(h₂−h₁) = 0,2072 (oude vergelijking:
    # te laag, σ_f erboven), de verwachte RV over [24 d, 87 d] is 0,2338 (σ_f 0,228 eronder).
    f = RVForecast(0.15, 0.15, 0.30, 0.15, kappa=4.0)
    state = {"TST": VolState(VolRegime.CHEAP, 0.25, f, 0.01, "test")}
    m = dataclasses.replace(market(chain, iv=HIGH_IV), vol=state)
    h1, h2 = 24, 87
    gap = math.sqrt((f.at_horizon(h2) ** 2 * h2 - f.at_horizon(h1) ** 2 * h1) / (h2 - h1))
    assert gap > f.at_horizon(h2 - h1)                    # de twee horizons verschillen echt
    strat = cal(entry_rule="forward_vol", max_fwd_vol_rv_ratio=1.02)
    assert strat._fwd_trace(m, chain, "TST", near, date(2027, 2, 19)) == "fwd:ok"
    assert gap * 1.02 > 0.228 > f.at_horizon(h2 - h1) * 1.02       # σ_f ≈ 0,228


# --- N13: het regime toetst IV30 tegen σ(30 d), niet tegen de blend ---------------------------

def test_classify_measures_the_vrp_at_the_tenor_of_the_atm_iv():
    s = regime_settings(CFG.vol_regime)
    f = RVForecast(0.14, 0.14, 0.20, 0.14, kappa=4.0)       # blend 0,14, σ(30 d) ≈ 0,15
    assert 0.17 / f.forecast >= s.rich_vrp_ratio            # tegen de blend zou het RICH zijn
    st = classify(0.17, f, 0.0, s)
    assert st.regime is VolRegime.NORMAL and st.reason == "vrp_normal"
    assert st.vrp_ratio == pytest.approx(0.17 / f.at_horizon(30))
    assert st.vrp_points == pytest.approx(0.17 - f.at_horizon(30))
    other = classify(0.17, f, 0.0, regime_settings(CFG.vol_regime, tenor_days=7))
    assert other.vrp_ratio == pytest.approx(0.17 / f.at_horizon(7))       # de tenor volgt de config


# --- N14: de earnings-straddle komt niet door zijn eigen event in STRESS --------------------------

def test_the_earnings_straddle_ignores_the_term_slope_its_own_event_causes():
    rules = selection_rules(CFG)
    chain = fake_chain(fake(95.0, Right.PUT))
    f = RVForecast(0.20, 0.20, 0.20, 0.20)
    back = VolState(VolRegime.STRESS, 0.21, f, -0.05, "backwardation")
    spike = VolState(VolRegime.STRESS, 0.30, f, 0.01, "rv_expansion")

    def cand(variant):
        from tradebot.options.positions import Candidate, LegIntent
        leg = LegIntent(fake(95.0, Right.PUT).contract, 1)
        return Candidate("long_straddle_strangle", "TST", (leg,), 1, -1.0, 100.0, 1.0, ("t",),
                         variant)

    ok = admit(cand("earnings"), chain, back, VolRegime.STRESS, rules, TODAY)
    assert ok.admitted and "regime:earnings_term_ignored" in ok.candidate.rule_trace
    assert not admit(cand("low_iv"), chain, back, VolRegime.STRESS, rules, TODAY).admitted
    assert not admit(cand("earnings"), chain, spike, VolRegime.STRESS, rules, TODAY).admitted
    bare = dataclasses.replace(rules, regime_settings=None)
    assert not admit(cand("earnings"), chain, back, VolRegime.STRESS, bare, TODAY).admitted


# --- N15: een roll gaat langs de entry-blockers -------------------------------------------------

def test_a_roll_into_the_earnings_closes_instead():
    chain = cached_chain()
    csp = CashSecuredPut(S.cash_secured_put.model_copy(update={"take_profit_frac": 0.99}), RULES,
                         ("TST",))
    (c,) = CashSecuredPut(S.cash_secured_put, RULES, ("TST",)).scan(
        market(capital=RICH), Book(cash_usd=1e7))
    pos = open_position(c, phase="csp")
    now = datetime(2026, 12, 14, 10, tzinfo=NY)
    clear = market(cached_chain(100.0, now=now), now=now)
    (roll,) = csp.manage(pos, clear)
    assert roll.kind is ActionKind.ROLL
    new_expiry = roll.new_legs[0].contract.expiry
    earn = market(cached_chain(100.0, now=now), now=now,
                  events=Events("TST", now.date(), (new_expiry - timedelta(days=3),)))
    (act,) = csp.manage(pos, earn)
    assert act.kind is ActionKind.CLOSE and act.reason == "dte" and chain is not None
    stale = market(cached_chain(100.0, now=now), now=now,
                   events=Events("TST", now.date() - timedelta(days=9)))
    assert csp.manage(pos, stale)[0].kind is ActionKind.CLOSE


# --- N19: max_loss van debit-posities op de natuurlijke prijs --------------------------------------

def test_debit_strategies_size_and_book_the_risk_at_the_natural_price():
    chain = cached_chain()
    capital = 100_000.0                                       # budget 2 %: 2.000 USD
    for strat, m in ((dv(), market(iv=LOW_IV, trend=Trend.UP, capital=capital)),
                     (cal(), market(iv=LOW_IV, capital=capital)),
                     (LongVol(S.long_straddle_strangle, RULES),
                      market(iv=LOW_IV, capital=capital))):
        c = strat.scan(m, Book())[0]
        natural = natural_mark(chain, c.legs)
        assert natural is not None and natural > -c.net_credit
        assert c.max_loss == pytest.approx(natural * 100 * c.quantity)
        assert c.quantity == math.floor(2_000 / (natural * 100))


def test_natural_mark_needs_bid_and_ask():
    leg = fake(95.0, Right.PUT)
    from tradebot.options.positions import LegIntent
    legs = (LegIntent(leg.contract, 1),)
    assert natural_mark(fake_chain(leg), legs) == pytest.approx(leg.quote.ask)
    assert natural_mark(fake_chain(), legs) is None
    blind = dataclasses.replace(leg, quote=dataclasses.replace(leg.quote, ask=None))
    assert natural_mark(fake_chain(blind), legs) is None


# --- N20: het cash van de CSP is exclusief wat andere CSP's al reserveren -----------------------------

def _csp_position(strike: float, qty: int, strategy="cash_secured_put") -> StrategyPosition:
    k = next(o.contract for o in cached_chain().options
             if o.contract.right is Right.PUT and o.contract.strike == strike)
    return StrategyPosition("r", strategy, "TST", (Leg(k, "TST", -qty, 1.0),), NOW, qty,
                            100.0 * qty, strike * 100 * qty)


def test_csp_cash_excludes_the_reservation_of_open_and_pending_csps():
    csp = CashSecuredPut(S.cash_secured_put, RULES, ("TST",))
    rich = market(capital=RICH)
    (free,) = csp.scan(rich, Book(cash_usd=20_000.0))
    unit = free.legs[0].contract.strike * 100
    full = int(20_000 // unit)
    assert free.quantity == full >= 2
    held = _csp_position(90.0, 1)                       # reserveert 9.000 USD
    left = int((20_000.0 - 9_000.0) // unit)
    assert left < full
    assert csp.scan(rich, Book(positions=(held,), cash_usd=20_000.0))[0].quantity == left
    pending = Book(pending_entries=(_csp_position(90.0, 1),), cash_usd=20_000.0)
    assert csp.scan(rich, pending)[0].quantity == left
    wheel = Book(positions=(_csp_position(90.0, 1, "wheel"),), cash_usd=9_000.0)
    assert csp.scan(rich, wheel) == []                  # alles al gereserveerd
    over = Book(positions=(_csp_position(90.0, 3),), cash_usd=9_000.0)
    assert csp.scan(rich, over) == []                   # reservering boven het cash: niets
    closed = dataclasses.replace(held, status=PositionStatus.CLOSED)
    assert csp.scan(rich, Book(positions=(closed,), cash_usd=20_000.0))[0].quantity == full
    other = _csp_position(90.0, 1, "credit_spread")
    assert csp.scan(rich, Book(positions=(other,), cash_usd=20_000.0))[0].quantity == full


# --- N21: alleen covered_call en wheel dekken aandelen -------------------------------------------------

def _call_position(strategy: str, qty: int = 1) -> StrategyPosition:
    k = next(o.contract for o in cached_chain().options if o.contract.right is Right.CALL)
    return StrategyPosition("c", strategy, "TST", (Leg(k, "TST", -qty, 1.0),), NOW, qty,
                            100.0 * qty, 1.0)


def test_only_covered_call_and_wheel_short_calls_cover_shares():
    cc = CoveredCall(S.covered_call, RULES, ("TST",))
    rich = market(capital=100 * RICH)
    book = Book(shares={"TST": 100}, cost_basis={"TST": 90.0})
    assert cc.scan(rich, book)
    for strategy in ("covered_call", "wheel"):
        assert cc.scan(rich, dataclasses.replace(book, positions=(_call_position(strategy),))) == []
    for strategy in ("credit_spread", "iron_condor", "short_straddle_strangle"):
        assert cc.scan(rich, dataclasses.replace(book, positions=(_call_position(strategy),)))
    pend = dataclasses.replace(book, pending_entries=(_call_position("covered_call"),))
    assert cc.scan(rich, pend) == []                    # een lopende covered call dekt ook


# --- N22: de VRP-poort gebruikt de ATM-IV van de expiratie -----------------------------------------------

def test_the_vrp_gate_uses_the_atm_iv_not_the_skewed_short_leg_iv():
    chain = make_chain(vol=skewed(0.25, -0.3))            # puts onder spot duurder dan ATM
    f = RVForecast(0.25, 0.25, 0.25, 0.25)
    state = {"TST": VolState(VolRegime.RICH, 0.25, f, 0.01, "test")}
    up = dataclasses.replace(market(chain, trend=Trend.UP), vol=state)
    c = cs().scan(up, Book())[0]
    short = chain.find(c.legs[0].contract.expiry, c.legs[0].contract.strike, Right.PUT)
    assert short.iv / 0.25 > 1.05                          # de skew-IV van de short leg
    assert atm_iv(chain, E37) == pytest.approx(0.25 + 0.3 * 0.0, abs=0.01)
    assert cs(min_vrp_ratio=1.05).scan(up, Book()) == []   # ATM-ratio ≈ 1,0: geweigerd
    assert cs(min_vrp_ratio=0.99).scan(up, Book())


def test_atm_iv_needs_both_ivs():
    chain = cached_chain()
    assert atm_iv(chain, E37) == pytest.approx(0.25, abs=0.01)
    assert atm_iv(dataclasses.replace(chain, options=()), E37) is None


# --- N17: een positie met aandelen-legs krijgt geen waarde-exits op de optie-legs --------------------------

def _with_shares(pos: StrategyPosition, qty: int = 100) -> StrategyPosition:
    return dataclasses.replace(
        pos, legs=(*pos.legs, Leg(None, pos.underlying, qty, 100.0)))


def _later(spot: float = 100.0, now: datetime = NOW):
    return market(cached_chain(spot, now=now), now=now)


def test_value_exits_are_silent_for_positions_with_shares():
    (c,) = cs().scan(market(trend=Trend.UP), Book())
    pos = open_position(c)
    crash = _later(90.0)
    assert [a.reason for a in cs().manage(pos, crash)] == ["stop_loss"]
    assert cs().manage(_with_shares(pos), crash) == []
    expiry_day = _later(now=datetime(2026, 12, 30, 15, tzinfo=NY))
    assert [a.reason for a in cs().manage(_with_shares(pos), expiry_day)] == ["expiry"]
    dte_day = _later(now=datetime(2026, 12, 11, 10, tzinfo=NY))
    assert [a.reason for a in cs().manage(_with_shares(pos), dte_day)] == ["dte"]
    (cc,) = ic().scan(market(), Book())
    condor = _with_shares(open_position(cc))
    assert ic().manage(condor, _later(112.0)) == []        # ook geen adjust
    (s,) = sv().scan(market(capital=RICH), Book())
    strangle = _with_shares(open_position(s))
    assert sv().manage(strangle, _later(130.0)) == []      # geen short_delta-stop


def test_debit_value_exits_are_silent_for_positions_with_shares():
    (c,) = dv().scan(market(iv=LOW_IV, trend=Trend.UP), Book())
    pos = open_position(c)
    crash = _later(60.0)
    assert [a.reason for a in dv().manage(pos, crash)] == ["stop_loss"]
    assert dv().manage(_with_shares(pos, -100), crash) == []


# --- #24: de take-profit van de debit vertical is een fractie van de maximale winst --------------------------

def test_the_debit_vertical_take_profit_is_reachable():
    (c,) = dv().scan(market(iv=LOW_IV, trend=Trend.UP), Book())
    pos = open_position(c)
    width = abs(c.legs[0].contract.strike - c.legs[1].contract.strike)
    assert dv().manage(pos, _later()) == []
    # Een debit van 0,55 × breedte: 2 × debit (de oude regel) ligt boven de maximale waarde
    # (de breedte) en kon nooit vuren; 60 % van de maximale winst ligt eronder.
    rich = dataclasses.replace(pos, entry_net=-(0.55 * width) * 100 * pos.units)
    assert 2 * 0.55 * width > width
    deep = _later(130.0)                               # beide legs diep ITM: waarde ≈ breedte
    acts = dv().manage(rich, deep)
    assert [(a.reason, a.urgency) for a in acts] == [("take_profit", Urgency.NORMAL)]
    cfg = S.debit_vertical.model_copy(update={"take_profit_frac_max_gain": 1.0})
    # 100 % van de maximale winst is pas op de expiratiewaarde; ruim onder die waarde: niets
    assert DebitVertical(cfg, RULES).manage(rich, _later(104.0)) == []
    one_leg = dataclasses.replace(pos, legs=pos.legs[:1])
    assert dv().manage(one_leg, deep) == []            # zonder twee legs geen take-profit


# --- #26: de short strikes van condor en strangle liggen binnen een deltaband -----------------------------------

def test_condor_and_strangle_short_strikes_stay_inside_a_delta_band():
    assert ic().scan(market(), Book()) and sv().scan(market(capital=RICH), Book())
    assert ic(short_delta=0.14, short_delta_tolerance=0.001).scan(market(), Book()) == []
    assert sv(strangle_delta=0.14, strangle_delta_tolerance=0.001).scan(
        market(capital=RICH), Book()) == []
    (c,) = ic().scan(market(), Book())
    chain = market().chains["TST"]
    for leg in c.legs:
        if leg.ratio < 0:
            o = chain.find(leg.contract.expiry, leg.contract.strike, leg.contract.right)
            assert abs(abs(o.delta) - 0.16) <= 0.05
    assert IronCondorConfig().short_delta_tolerance == 0.05
    assert ShortVolConfig().strangle_delta_tolerance == 0.05


def test_condor_adjust_uses_the_same_band():
    (c,) = ic().scan(market(), Book())
    pos = open_position(c)
    base = dict(short_delta=0.14, adjust_short_delta=0.05)     # elke short leg is "getest"
    narrow = ic(short_delta_tolerance=0.001, **base)
    assert [a for a in narrow.manage(pos, _later()) if a.kind is ActionKind.ADJUST] == []
    wide = ic(**base)
    acts = [a for a in wide.manage(pos, _later()) if a.kind is ActionKind.ADJUST]
    assert acts and acts[0].new_legs


# --- #25: de credit spread rangschikt alle breedtes -------------------------------------------------------------

def _put_chain(mids: dict[float, float]):
    return fake_chain(*(fake(k, Right.PUT, mid=m) for k, m in mids.items()))


def test_the_credit_spread_ranks_all_widths_on_credit_after_costs_over_max_loss():
    # breedte 1: credit 0,25 (score 0,333); breedte 2: credit 0,70 (score 0,538): breder wint
    chain = _put_chain({95.0: 1.00, 94.0: 0.75, 93.0: 0.30, 92.0: 0.28})
    short = chain.by_expiry(date(2026, 12, 31), Right.PUT)[-1]            # strike 95
    legs, credit, width = _spread(chain, short, range(1, 4), 0.20)
    assert width == 2 and credit == pytest.approx(0.70)
    # kosten per aandeel keren de rangorde: 0,25 → (0,25 − c)/0,75 tegen (0,70 − c)/1,3
    legs, credit, width = _spread(chain, short, range(1, 4), 0.20, cost=0.3)
    assert width == 2                                                        # breed blijft winnen
    # breedte 1 haalt het criterium niet (credit 0,10 < 0,20): alleen breedte 2
    thin = _put_chain({95.0: 1.00, 94.0: 0.90, 93.0: 0.30})
    short = thin.by_expiry(date(2026, 12, 31), Right.PUT)[-1]
    assert _spread(thin, short, range(1, 3), 0.20)[2] == 2
    # een credit ≥ breedte (arbitrage) telt niet; zonder genoeg strikes: geen spread
    arb = _put_chain({95.0: 2.00, 94.0: 0.50})
    short = arb.by_expiry(date(2026, 12, 31), Right.PUT)[-1]
    assert _spread(arb, short, range(1, 3), 0.20) is None
    assert _spread(arb, short, range(5, 7), 0.20) is None


def test_the_credit_spread_costs_come_from_the_commission_config():
    assert RULES.commission_per_contract == CFG.paper.commission_per_contract
    assert natural_mark(cached_chain(), ()) == 0.0
