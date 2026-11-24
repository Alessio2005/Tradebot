"""Pricing met QuantLib: acceptatietests uit sectie 5.7 van de leidraad.

* Europees = gesloten Black-Scholes(-Merton) tot 1e-10, put-call-pariteit tot 1e-10;
* Amerikaanse call zonder dividend = Europese call; Amerikaanse put ≥ Europese put;
* de ongelijkheid S − K ≤ C − P ≤ S − K·e^{−rT} voor Amerikaans zonder dividend;
* binomiale convergentie en een literatuurwaarde (Hull);
* `evaluationDate` is na elke waardering weer wat hij was.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest
import QuantLib as ql
from scipy.stats import norm

from tests.unit.options.helpers import EXPIRY, NOW, inputs, inputs_with_t
from tradebot.options.calendar import NY, expiry_moment
from tradebot.options.contracts import Right
from tradebot.options.pricing.engines import EngineSettings, Method, Use, engine_for
from tradebot.options.pricing.ql_env import evaluation_date, make_process, scaled_time, to_ql_date
from tradebot.options.pricing.vanilla import OptionInputs, price, value
from tradebot.options.rates import Carry, CashDividend
from tradebot.utils.failfast import ConfigContractError, DataContractError


def bsm(s, k, t, r, q, sigma, right):
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if right is Right.CALL:
        return s * math.exp(-q * t) * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2)
    return k * math.exp(-r * t) * norm.cdf(-d2) - s * math.exp(-q * t) * norm.cdf(-d1)


@pytest.mark.parametrize("right", list(Right))
@pytest.mark.parametrize("t", [2 / 365, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_european_equals_closed_form(right, t, strike):
    carry = Carry(0.04, div_yield=0.015)
    inp = inputs_with_t(t, strike=strike, right=right, carry=carry)
    t_real = (expiry_moment(EXPIRY) - inp.now).total_seconds() / (365 * 86400)
    expected = bsm(100.0, strike, t_real, 0.04, 0.015, 0.3, right)
    assert value(inp, 0.3, Method.EUROPEAN).npv == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("t", [0.05, 0.5])
def test_european_put_call_parity(t):
    carry = Carry(0.04, div_yield=0.01)
    c = value(inputs_with_t(t, right=Right.CALL, strike=105.0, carry=carry), 0.25,
              Method.EUROPEAN).npv
    p = value(inputs_with_t(t, right=Right.PUT, strike=105.0, carry=carry), 0.25,
              Method.EUROPEAN).npv
    assert c - p == pytest.approx(100 * math.exp(-0.01 * t) - 105 * math.exp(-0.04 * t),
                                  abs=1e-10)


def test_american_call_without_dividend_equals_european():
    inp = inputs_with_t(0.25, right=Right.CALL)
    euro = value(inp, 0.3, Method.EUROPEAN).npv
    for m in (Method.BINOMIAL_LR, Method.FD, Method.BAW):
        assert value(inp, 0.3, m).npv == pytest.approx(euro, abs=2e-3), m


@pytest.mark.parametrize("strike", [90.0, 100.0, 120.0])
def test_american_put_is_worth_at_least_the_european_put(strike):
    inp = inputs_with_t(0.5, strike=strike)
    assert value(inp, 0.3, Method.BINOMIAL_LR).npv >= value(inp, 0.3, Method.EUROPEAN).npv - 1e-9


@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_american_parity_inequality_without_dividend(strike):
    t = 0.5
    c = value(inputs_with_t(t, right=Right.CALL, strike=strike), 0.3, Method.BINOMIAL_LR).npv
    p = value(inputs_with_t(t, right=Right.PUT, strike=strike), 0.3, Method.BINOMIAL_LR).npv
    assert 100 - strike - 1e-9 <= c - p <= 100 - strike * math.exp(-0.04 * t) + 1e-9


def test_binomial_converges_with_more_steps():
    inp = inputs_with_t(0.5)
    ref = value(inp, 0.3, Method.BINOMIAL_LR, EngineSettings(binomial_steps=2001)).npv
    errs = [abs(value(inp, 0.3, Method.BINOMIAL_CRR, EngineSettings(binomial_steps=n)).npv - ref)
            for n in (51, 201, 801)]
    assert errs[0] > errs[2] and errs[2] < 5e-3
    assert value(inp, 0.3, Method.BINOMIAL_LR).npv == pytest.approx(ref, abs=1e-3)


def test_hull_american_put_reference_value():
    # Hull, Options, Futures and Other Derivatives: S=50, K=50, r=10 %, σ=40 %, T=5/12.
    # De boom met 5 stappen geeft 4,49; met veel stappen convergeert hij naar 4,28.
    inp = inputs_with_t(5 / 12, spot=50.0, strike=50.0, carry=Carry(0.10))
    assert value(inp, 0.40, Method.BINOMIAL_LR).npv == pytest.approx(4.28, abs=0.01)


def test_discrete_dividend_raises_the_put_and_needs_fd():
    div = Carry(0.04, dividends=(CashDividend(date(2026, 12, 5), 2.0),))
    with_div = inputs(carry=div)
    assert value(with_div, 0.25, Method.FD).npv > value(inputs(), 0.25, Method.FD).npv + 0.5
    with pytest.raises(DataContractError, match="FD"):
        value(with_div, 0.25, Method.BINOMIAL_LR)
    after = inputs(carry=Carry(0.04, dividends=(CashDividend(date(2026, 11, 20), 2.0),)))
    assert value(after, 0.25, Method.BINOMIAL_LR).npv == value(inputs(), 0.25,
                                                               Method.BINOMIAL_LR).npv


def test_baw_has_no_engine_greeks_and_price_matches_value():
    v = value(inputs(), 0.25, Method.BAW)
    assert v.delta is None and v.gamma is None
    assert price(inputs(), 0.25, Method.BINOMIAL_LR) == value(inputs(), 0.25,
                                                              Method.BINOMIAL_LR).npv


def test_expiry_day_is_valued_from_the_day_before():
    morning = inputs(now=datetime(2026, 12, 18, 10, 0, tzinfo=NY), right=Right.CALL)
    st = scaled_time(morning.now, morning.expiry)
    assert st.eval_date == date(2026, 12, 17) and st.t_model == pytest.approx(1 / 365)
    expected = bsm(100.0, 100.0, 6 / 8760, 0.04, 0.0, 0.3, Right.CALL)
    assert value(morning, 0.3, Method.EUROPEAN).npv == pytest.approx(expected, abs=1e-10)


def test_an_expired_option_is_refused():
    late = inputs(now=expiry_moment(EXPIRY) + timedelta(seconds=1))
    assert late.expired
    with pytest.raises(DataContractError, match="verlopen"):
        value(late, 0.25, Method.EUROPEAN)


def test_evaluation_date_is_restored_even_after_an_error():
    settings = ql.Settings.instance()
    before = settings.evaluationDate
    value(inputs(), 0.25, Method.BINOMIAL_LR)
    assert settings.evaluationDate == before
    with pytest.raises(RuntimeError), evaluation_date(date(2020, 1, 2)) as d:
        assert d == to_ql_date(date(2020, 1, 2))
        raise RuntimeError("boem")
    assert settings.evaluationDate == before


def test_two_valuations_on_different_dates_do_not_interfere():
    early = inputs(now=NOW - timedelta(days=10))
    a1 = value(early, 0.25, Method.BINOMIAL_LR).npv
    value(inputs(), 0.25, Method.BINOMIAL_LR)
    assert value(early, 0.25, Method.BINOMIAL_LR).npv == a1


def test_inputs_validation_and_helpers():
    for kw in ({"spot": 0.0}, {"strike": -1.0}):
        with pytest.raises(DataContractError):
            inputs(**kw)
    with pytest.raises(DataContractError, match="tijdzone"):
        OptionInputs(100, 100, Right.PUT, EXPIRY, datetime(2026, 11, 24, 15, 0), Carry(0.04))
    assert inputs(right=Right.CALL, spot=110).intrinsic == 10.0
    assert inputs(right=Right.PUT, spot=110).intrinsic == 0.0
    assert inputs().with_rate(0.05).carry.rate == 0.05


def test_process_refuses_bad_spot_or_vol():
    st = scaled_time(NOW, EXPIRY)
    with evaluation_date(st.eval_date):
        with pytest.raises(DataContractError, match="spot"):
            make_process(0.0, 0.04, 0.0, 0.2, st)
        with pytest.raises(DataContractError, match="volatiliteit"):
            make_process(100.0, 0.04, 0.0, 0.0, st)


def test_engine_choice_and_settings_validation():
    s = EngineSettings()
    assert engine_for(Use.SCAN, False, s) is Method.BAW
    assert engine_for(Use.DECISION, False, s) is Method.BINOMIAL_LR
    assert engine_for(Use.SCAN, True, s) is Method.FD
    assert not Method.EUROPEAN.american and Method.FD.supports_discrete_dividends
    for kw in ({"binomial_steps": 400}, {"fd_grid_points": 10}, {"scan": Method.EUROPEAN}):
        with pytest.raises(ConfigContractError):
            EngineSettings(**kw)
