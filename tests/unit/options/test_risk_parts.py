"""Bouwstenen van de risicopoort: aggregatie, scenario's, marge, PDT en de grenzen."""
from __future__ import annotations

import dataclasses
import math
from datetime import date, datetime

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CAL,
    CFG,
    SETTINGS,
    cached_chain,
    market,
    open_position,
)
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.positions import (
    Book,
    Leg,
    PositionStatus,
    StrategyPosition,
    Trend,
)
from tradebot.options.pricing.engines import Method
from tradebot.options.rates import Carry, CashDividend
from tradebot.options.risk.aggregate import aggregate
from tradebot.options.risk.limits import BookMetrics, CheckResult, check, measure
from tradebot.options.risk.margin import NAKED_FRAC, buying_power, position_buying_power
from tradebot.options.risk.monitor import assess, degross_plan
from tradebot.options.risk.pdt import day_trades_in_window, pdt_blocks_entry
from tradebot.options.risk.scenarios import Scenario, grid, position_values
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.credit import CreditSpread
from tradebot.utils.failfast import DataContractError

RULES = rules_from_config(CFG)
BETA = {"TST": 1.5}
E = date(2026, 12, 31)
ENG = SETTINGS.engines


@pytest.fixture(scope="module")
def bull_put():
    (c,) = CreditSpread(CFG.strategies.credit_spread, RULES).scan(market(trend=Trend.UP), Book())
    return open_position(c)


def _single(strike: float, right: Right, qty: int, strategy: str = "x", **kw):
    k = OptionContract("TST", E, strike, right)
    return StrategyPosition(f"{strategy}{strike}{right.value}{qty}", strategy, "TST",
                            (Leg(k, "TST", qty, 1.0),), NOW, abs(qty), 0.0,
                            kw.pop("max_loss", 100.0), **kw)


# --- aggregatie ----------------------------------------------------------------------------

def test_aggregate_scales_leg_greeks_and_adds_shares(bull_put):
    ch = cached_chain()
    g = aggregate((bull_put,), {"TST": 100}, {"TST": ch}, BETA)
    short = ch.find(E, 95.0, Right.PUT).greeks
    long = ch.find(E, 94.0, Right.PUT).greeks
    n = bull_put.units
    want_delta = (-short.delta + long.delta) * 100 * n + 100
    (u,) = g.per_underlying
    assert u.greeks.delta.value == pytest.approx(want_delta)
    assert g.beta_delta_usd == pytest.approx(want_delta * 100.0 * 1.5)
    want_gamma = (-short.gamma + long.gamma) * 100 * n
    assert g.gamma_loss_5pct == pytest.approx(-0.5 * want_gamma * (0.05 * 100 * 1.5) ** 2)
    assert g.gamma_loss_5pct > 0 and g.complete      # short gamma verliest
    assert g.vega_usd == pytest.approx((-short.vega + long.vega) * 100 * n)
    assert g.theta_usd > 0


def test_aggregate_flags_missing_data(bull_put):
    assert not aggregate((bull_put,), {}, {}, BETA).complete
    bare = dataclasses.replace(cached_chain(), options=())
    assert not aggregate((bull_put,), {}, {"TST": bare}, BETA).complete
    closed = dataclasses.replace(bull_put, status=PositionStatus.CLOSED)
    assert aggregate((closed,), {}, {}, BETA).per_underlying == ()
    with pytest.raises(DataContractError, match="beta"):
        aggregate((bull_put,), {}, {"TST": cached_chain()}, {})


# --- scenario's ----------------------------------------------------------------------------

def test_grid_always_holds_the_base_case():
    g = grid((0.1, -0.1), (10, 0))
    assert Scenario(0.0, 0.0) in g and len(g) == 6


def test_scenarios_revalue_each_leg(bull_put):
    ch = {"TST": cached_chain()}
    iv = market().iv
    sc = grid((-0.2, 0.2), (0, 10))
    def total(s: Scenario) -> float:
        vals = position_values((bull_put,), {}, ch, iv, BETA, NOW, s, Method.BAW, ENG)
        assert vals is not None
        return sum(vals.values())

    base = total(Scenario(0.0, 0.0))
    losses = {s: base - total(s) for s in sc}
    # Diep ITM: meer vol trekt de spread terug richting het midden, dus vol 0 is het ergst.
    assert max(losses, key=losses.__getitem__) == Scenario(-0.2, 0.0)
    assert max(losses.values()) <= bull_put.max_loss + 1e-6
    assert losses[Scenario(0.0, 0.0)] == pytest.approx(0.0)
    up = losses[Scenario(0.2, 0.0)]
    assert up < 0                                    # bull put wint als spot stijgt



def test_scenarios_use_fd_for_dividend_legs(bull_put):
    div = dataclasses.replace(cached_chain(), carry=Carry(0.04, dividends=(
        CashDividend(date(2026, 12, 4), 0.5),)))
    plain = position_values((bull_put,), {}, {"TST": cached_chain()}, market().iv, BETA, NOW,
                            Scenario(-0.1, 0.0), Method.BAW, ENG)
    fd = position_values((bull_put,), {}, {"TST": div}, market().iv, BETA, NOW,
                         Scenario(-0.1, 0.0), Method.BAW, ENG)
    assert plain is not None and fd is not None
    assert next(iter(fd.values())) != pytest.approx(next(iter(plain.values())))


def test_scenarios_need_a_chain_and_a_vol(bull_put):
    iv = market().iv
    assert position_values((bull_put,), {}, {}, iv, BETA, NOW, Scenario(0.0, 0.0),
                           Method.BAW, ENG) is None
    no_iv = dataclasses.replace(cached_chain(), options=tuple(
        dataclasses.replace(o, iv_mid=dataclasses.replace(o.iv_mid, value=None))
        for o in cached_chain().options))
    with_atm = position_values((bull_put,), {}, {"TST": no_iv}, iv, BETA, NOW,
                               Scenario(0.0, 0.0), Method.BAW, ENG)
    assert with_atm is not None
    assert position_values((bull_put,), {}, {"TST": no_iv}, {}, BETA, NOW, Scenario(0.0, 0.0),
                           Method.BAW, ENG) is None
    assert position_values((), {"XYZ": 10}, {}, {}, {"XYZ": 1.0}, NOW, Scenario(0.0, 0.0),
                           Method.BAW, ENG) is None


def test_shares_and_expired_legs_in_scenarios():
    ch = {"TST": cached_chain()}
    vals = position_values((), {"TST": 100, "ZERO": 0}, ch, {}, BETA, NOW,
                           Scenario(0.1, 0.0), Method.BAW, ENG)
    assert vals == {"shares:TST": pytest.approx(100 * 100 * 1.15)}
    old = StrategyPosition("o", "x", "TST",
                           (Leg(OptionContract("TST", date(2026, 11, 20), 90.0, Right.CALL),
                                "TST", 1, 1.0),), NOW, 1, -100.0, 100.0)
    v = position_values((old,), {}, ch, market().iv, BETA, NOW, Scenario(0.0, 0.0),
                        Method.BAW, ENG)
    assert v == {"o": pytest.approx(1000.0)}
    closed = dataclasses.replace(old, status=PositionStatus.CLOSED)
    assert position_values((closed,), {}, {}, {}, BETA, NOW, Scenario(0.0, 0.0), Method.BAW,
                           ENG) == {}


# --- marge ---------------------------------------------------------------------------------

def test_buying_power_per_strategy(bull_put):
    ch = {"TST": cached_chain()}
    assert position_buying_power(bull_put, ch) == bull_put.max_loss
    csp = _single(95.0, Right.PUT, -1, "cash_secured_put", max_loss=9500.0)
    csp = dataclasses.replace(csp, entry_net=150.0)
    assert position_buying_power(csp, ch) == 9350.0
    wheel_csp = dataclasses.replace(csp, strategy="wheel", phase="csp")
    assert position_buying_power(wheel_csp, ch) == 9350.0
    cc = _single(105.0, Right.CALL, -1, "covered_call", max_loss=10500.0)
    assert position_buying_power(cc, ch) == 0.0
    assert position_buying_power(dataclasses.replace(cc, strategy="wheel", phase="cc"), ch) == 0
    naked = _single(105.0, Right.CALL, -1, "short_straddle_strangle", max_loss=math.inf)
    naked = dataclasses.replace(naked, entry_net=200.0)
    assert position_buying_power(naked, ch) == pytest.approx(NAKED_FRAC * 100 * 100 + 200)
    assert position_buying_power(naked, {}) == pytest.approx(NAKED_FRAC * 105 * 100 + 200)
    closed = dataclasses.replace(bull_put, status=PositionStatus.CLOSED)
    assert buying_power((bull_put, closed), ch) == bull_put.max_loss


# --- PDT -----------------------------------------------------------------------------------

def test_pdt_counts_the_last_five_trading_days():
    days = (date(2026, 11, 18), date(2026, 11, 19), date(2026, 11, 23), date(2026, 11, 24))
    # 11-24 terug: 24, 23, 20, 19, 18 (27 en 26 vallen buiten; 21/22 weekend)
    assert day_trades_in_window(days, date(2026, 11, 24), CAL) == 4
    assert day_trades_in_window(days, date(2026, 11, 30), CAL) == 2
    late = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
    scan = datetime(2026, 11, 24, 10, 0, tzinfo=NY)       # de eerste scan van de dag
    assert pdt_blocks_entry(days, late, 24_999.0, CAL)
    assert pdt_blocks_entry(days, scan, 24_999.0, CAL)
    assert not pdt_blocks_entry(days, late, 25_000.0, CAL)
    assert not pdt_blocks_entry(days[:2], late, 1.0, CAL)


# --- grenzen -------------------------------------------------------------------------------

def test_metrics_are_linear_and_checks_compare_with_the_limit():
    a = BookMetrics(1.0, 2.0, 3.0, {"A": 10.0}, 5.0, (1.0, 2.0), True)
    b = BookMetrics(1.0, 1.0, 1.0, {"B": 1.0}, 1.0, (0.5, 0.5), False)
    s = a.plus(b, 2)
    assert (s.beta_delta_usd, s.vega_usd, s.gamma_loss_5pct, s.buying_power) == (3, 4, 5, 7)
    assert s.max_loss_by_underlying == {"A": 10.0, "B": 2.0}
    assert s.scenario_losses == (2.0, 3.0) and not s.complete
    assert a.plus(dataclasses.replace(b, scenario_losses=None)).scenario_losses is None
    assert CheckResult("x", 1.0, 1.0).ok and not CheckResult("x", 1.1, 1.0).ok
    names = [c.name for c in check(a, CFG.risk, 25_000.0)]
    assert names == ["greeks_complete", "beta_delta_usd", "vega_usd", "gamma_loss_5pct",
                     "max_loss_per_underlying", "buying_power", "scenario_loss"]
    none = check(dataclasses.replace(a, scenario_losses=None), CFG.risk, 25_000.0)
    assert none[-1].measured == math.inf and not none[-1].ok


def test_measure_leaves_undefined_risk_out_of_the_loss_per_underlying(bull_put):
    naked = _single(105.0, Right.CALL, -1, "short_straddle_strangle", max_loss=math.inf)
    m = measure((bull_put, naked), {}, market(), BETA, grid((), (0,)), Method.BAW, ENG)
    assert m.max_loss_by_underlying == {"TST": bull_put.max_loss}
    gone = measure((bull_put,), {}, dataclasses.replace(market(), chains={}), BETA,
                   grid((), (0,)), Method.BAW, ENG)
    assert gone.scenario_losses is None and not gone.complete


# --- dagverlies, drawdown en de-grossing ---------------------------------------------------

def test_assess_daily_loss_and_drawdown():
    r = CFG.risk
    calm = assess(24_900.0, 25_000.0, 25_000.0, r, NOW)
    assert calm.daily_loss_frac == pytest.approx(0.004) and not calm.entries_blocked
    stop = assess(24_400.0, 25_000.0, 25_000.0, r, NOW)
    assert stop.entries_blocked and not stop.degross
    deg = assess(23_900.0, 25_000.0, 25_000.0, r, NOW)
    assert deg.degross and deg.halt is None
    dd = assess(22_000.0, 22_100.0, 25_000.0, r, NOW)
    assert dd.halt is not None and dd.halt.config_key == "risk.drawdown_halt_frac"
    assert dd.halt.measured == pytest.approx(0.12)
    assert assess(100.0, 0.0, 0.0, r, NOW).daily_loss_frac == 0.0
    assert assess(0.0, 0.0, 0.0, r, NOW).drawdown_frac == 0.0
    assert assess(26_000.0, 25_000.0, 25_000.0, r, NOW).drawdown_frac == 0.0


def test_degross_closes_the_biggest_contributor_first(bull_put):
    m = market()
    sc = grid(CFG.risk.scenario_spot, CFG.risk.scenario_vol_points)
    assert degross_plan(Book((bull_put,)), m, BETA, sc, CFG.risk, Method.BAW, ENG) == []
    big = dataclasses.replace(bull_put, position_id="big",
                              legs=tuple(dataclasses.replace(leg, qty=leg.qty * 10)
                                         for leg in bull_put.legs),
                              units=bull_put.units * 10, max_loss=bull_put.max_loss * 10)
    plan = degross_plan(Book((bull_put, big)), m, BETA, sc, CFG.risk, Method.BAW, ENG)
    assert plan == ["big"]
    gone = dataclasses.replace(m, chains={})
    assert degross_plan(Book((big,)), gone, BETA, sc, CFG.risk, Method.BAW, ENG) == ["big"]
