"""Implied volatility (foutcodes, fallback) en Greeks (tegen finite differences)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta

import pytest

from tests.unit.options.helpers import EXPIRY, inputs
from tradebot.options.calendar import NY, expiry_moment
from tradebot.options.contracts import Right
from tradebot.options.pricing import implied_vol as iv_mod
from tradebot.options.pricing.engines import Method
from tradebot.options.pricing.greeks import Bumps, greeks, spot_bump_fraction
from tradebot.options.pricing.implied_vol import IVSettings, IVStatus, implied_vol
from tradebot.options.pricing.settings import pricer_settings
from tradebot.options.pricing.vanilla import price, value
from tradebot.options.rates import Carry, CashDividend
from tradebot.schemas.options import load_options_config


@pytest.mark.parametrize("method", [Method.BINOMIAL_LR, Method.BAW, Method.FD, Method.EUROPEAN])
def test_iv_round_trip(method):
    inp = inputs(strike=95.0)
    p = value(inp, 0.27, method).npv
    res = implied_vol(inp, p, method)
    assert res.status is IVStatus.OK and res.method is method
    assert res.value == pytest.approx(0.27, abs=1e-6)


def test_iv_error_codes_never_carry_a_number():
    inp = inputs(right=Right.CALL, spot=110.0)
    cases = {
        IVStatus.NO_QUOTE: implied_vol(inp, None, Method.BAW),
        IVStatus.NO_BID: implied_vol(inp, 0.0, Method.BAW),
        IVStatus.BELOW_INTRINSIC: implied_vol(inp, 9.5, Method.BAW),
        IVStatus.ABOVE_UPPER_BOUND: implied_vol(inp, 111.0, Method.BAW),
        IVStatus.EXPIRED: implied_vol(
            inputs(now=expiry_moment(EXPIRY) + timedelta(minutes=1)), 1.0, Method.BAW),
    }
    for status, res in cases.items():
        assert res.status is status and res.value is None and res.method is None


def test_a_mid_just_below_intrinsic_is_lifted_to_half_a_tick_above():
    inp = inputs(right=Right.PUT, spot=60.0)             # diep ITM, intrinsiek 40
    res = implied_vol(inp, 39.995, Method.BINOMIAL_LR)
    assert res.status is IVStatus.OK
    assert price(inp, res.value, Method.BINOMIAL_LR) == pytest.approx(40.005, abs=1e-5)


@pytest.mark.parametrize("right", list(Right))
def test_european_lower_bound_is_the_discounted_forward(right):
    spot = 130.0 if right is Right.CALL else 70.0
    inp = inputs(right=right, spot=spot)
    intrinsic = inp.intrinsic
    res = implied_vol(inp, intrinsic - 0.5, Method.EUROPEAN)   # onder intrinsiek, boven de PV-grens?
    assert res.status in (IVStatus.OK, IVStatus.BELOW_INTRINSIC)
    assert implied_vol(inp, intrinsic - 5.0, Method.EUROPEAN).status is IVStatus.BELOW_INTRINSIC


def test_no_convergence_when_the_price_is_out_of_reach():
    inp = inputs(right=Right.CALL)
    cfg = IVSettings(max_vol=0.5)
    assert implied_vol(inp, 60.0, Method.BINOMIAL_LR, cfg=cfg).status is IVStatus.NO_CONVERGENCE
    tight = IVSettings(max_evaluations=1)
    assert implied_vol(inp, 3.0, Method.BINOMIAL_LR, cfg=tight).status is IVStatus.NO_CONVERGENCE


def test_falls_back_to_baw_when_the_engine_fails(monkeypatch):
    real = iv_mod._solve
    calls = []

    def fake(inp, target, method, engines, cfg):
        calls.append(method)
        return None if method is Method.BINOMIAL_LR else real(inp, target, method, engines, cfg)

    monkeypatch.setattr(iv_mod, "_solve", fake)
    res = implied_vol(inputs(), 2.4, Method.BINOMIAL_LR)
    assert res.status is IVStatus.OK and res.method is Method.BAW
    assert calls == [Method.BINOMIAL_LR, Method.BAW]


def test_no_baw_fallback_with_a_discrete_dividend(monkeypatch):
    monkeypatch.setattr(iv_mod, "_solve", lambda *a: None)
    inp = inputs(carry=Carry(0.04, dividends=(CashDividend(date(2026, 12, 5), 1.0),)))
    assert implied_vol(inp, 2.4, Method.FD).status is IVStatus.NO_CONVERGENCE
    assert implied_vol(inputs(), 2.4, Method.BAW).status is IVStatus.NO_CONVERGENCE


def _fd(inp, method, vol=0.25):
    h, b = 0.01 * inp.spot, 1e-4
    up = price(inp.with_spot(inp.spot + h), vol, method)
    mid = price(inp, vol, method)
    down = price(inp.with_spot(inp.spot - h), vol, method)
    vega = (price(inp, vol + b, method) - price(inp, vol - b, method)) / (2 * b) * 0.01
    return (up - down) / (2 * h), (up - 2 * mid + down) / h**2, vega


@pytest.mark.parametrize("method", [Method.EUROPEAN, Method.BINOMIAL_LR, Method.BAW, Method.FD])
@pytest.mark.parametrize("right", list(Right))
def test_greeks_match_finite_differences(method, right):
    inp = inputs(right=right, strike=102.0)
    g = greeks(inp, 0.25, method)
    delta, gamma, vega = _fd(inp, method)
    assert g.delta == pytest.approx(delta, rel=0.01, abs=1e-3)
    assert g.gamma == pytest.approx(gamma, rel=0.05, abs=2e-3)
    assert g.vega == pytest.approx(vega, rel=0.01, abs=1e-4)
    assert g.theta < 0
    assert (g.rho > 0) == (right is Right.CALL)


@pytest.mark.parametrize("rate", [0.0, 0.00004])
@pytest.mark.parametrize("right", list(Right))
def test_rho_below_the_bump_is_one_sided_from_zero(rate, right):
    # BAW weigert een call bij een negatieve rente: de bump omlaag stopt op 0.
    inp = inputs(right=right, carry=Carry(rate))
    g = greeks(inp, 0.25, Method.BAW)
    up = price(inp.with_rate(rate + 0.0001), 0.25, Method.BAW)
    zero = price(inp.with_rate(0.0), 0.25, Method.BAW)
    assert g.rho == pytest.approx((up - zero) / (rate + 0.0001) * 0.01, rel=1e-12)
    assert (g.rho > 0) == (right is Right.CALL)


def test_european_greeks_have_the_documented_units():
    # Analytische vega van BS per volpunt: S·φ(d1)·sqrt(T)·0,01; theta per kalenderdag.
    inp = inputs(right=Right.CALL)
    g = greeks(inp, 0.25, Method.EUROPEAN)
    one_day = price(inp.with_now(inp.now + timedelta(days=1)), 0.25, Method.EUROPEAN) \
        - price(inp, 0.25, Method.EUROPEAN)
    assert g.theta == pytest.approx(one_day, rel=1e-12)
    assert 0.05 < g.vega < 0.2


def test_theta_near_expiry_uses_half_the_remaining_time():
    inp = inputs(now=datetime(2026, 12, 18, 15, 0, tzinfo=NY), right=Right.CALL)
    g = greeks(inp, 0.25, Method.EUROPEAN)
    half_hour = price(inp.with_now(inp.now + timedelta(minutes=30)), 0.25, Method.EUROPEAN) \
        - price(inp, 0.25, Method.EUROPEAN)
    assert g.theta == pytest.approx(half_hour * 48, rel=1e-9)


def test_vega_bump_shrinks_for_a_tiny_vol():
    g = greeks(inputs(), 0.015, Method.EUROPEAN)    # bump 0,0075 in plaats van 0,01
    assert g.vega > 0


def test_pricer_settings_follow_the_config():
    s = pricer_settings(load_options_config().pricing)
    assert s.engines.binomial_steps == 401 and s.engines.scan is Method.BAW
    assert s.iv.accuracy == 1e-6 and s.bumps.vol == 0.01


def _fragile_pricer(monkeypatch, breaks):
    """Vervang de pricer door een die een QuantLib-fout geeft waar `breaks(method, σ)` waar is."""
    real = iv_mod.vol_pricer

    @contextmanager
    def fragile(inp, method, engines):
        with real(inp, method, engines) as npv:
            def f(sigma):
                if breaks(method, sigma):
                    raise RuntimeError("forward + displacement must be positive")
                return npv(sigma)
            yield f

    monkeypatch.setattr(iv_mod, "vol_pricer", fragile)


def test_a_quantlib_refusal_inside_the_bracket_falls_back_to_the_tree(monkeypatch):
    _fragile_pricer(monkeypatch, lambda m, s: m is Method.BAW and 0.1 < s < 1.0)
    res = implied_vol(inputs(), 2.4, Method.BAW)
    assert res.status is IVStatus.OK and res.method is Method.BINOMIAL_LR


def test_a_deep_otm_put_solves_with_baw():
    inp = inputs(strike=80.0)
    res = implied_vol(inp, price(inp, 0.30, Method.BAW), Method.BAW)
    assert res.status is IVStatus.OK and res.value == pytest.approx(0.30, abs=1e-6)


def test_the_lower_bracket_moves_up_when_quantlib_refuses_it(monkeypatch):
    _fragile_pricer(monkeypatch, lambda m, s: s < 0.02)
    res = implied_vol(inputs(), price(inputs(), 0.25, Method.BAW), Method.BAW)
    assert res.status is IVStatus.OK and res.value == pytest.approx(0.25, abs=1e-6)


def test_no_iv_when_quantlib_refuses_every_lower_bracket(monkeypatch):
    _fragile_pricer(monkeypatch, lambda m, s: s < 0.1)
    assert implied_vol(inputs(), 2.4, Method.BAW).status is IVStatus.NO_CONVERGENCE


def test_a_price_out_of_the_model_range_has_no_fallback(monkeypatch):
    calls = []
    real = iv_mod._solve
    monkeypatch.setattr(iv_mod, "_solve", lambda *a: calls.append(a[2]) or real(*a))
    deep_itm_call = inputs(right=Right.CALL, spot=130.0)       # S − K·e^{−rT} ≈ 30,26
    assert implied_vol(deep_itm_call, 30.05, Method.BAW).status is IVStatus.NO_CONVERGENCE
    assert calls == [Method.BAW]


def _dividend_carry(ex: date) -> Carry:
    return Carry(0.04, dividends=(CashDividend(ex, 1.80),))


@pytest.mark.parametrize("right", list(Right))
def test_theta_the_day_before_the_ex_date_keeps_the_dividend(right):
    # N11: SPY-achtig, ATM, D = 1,80, ex-datum morgen. Zonder correctie viel het dividend uit
    # het venster van `now + 1 dag` en sprong de theta van de call positief.
    now = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
    inp = inputs(spot=600.0, strike=600.0, right=right, now=now,
                 carry=_dividend_carry(date(2026, 11, 25)))
    assert greeks(inp, 0.15, Method.FD).theta < 0
    # drie dagen eerder en een dag na de ex-datum is de theta gewoon negatief
    assert greeks(inp.with_now(now - timedelta(days=3)), 0.15, Method.FD).theta < 0
    assert greeks(inp.with_now(now + timedelta(days=1)), 0.15, Method.FD).theta < 0


def test_theta_drops_the_spot_by_the_dividend_that_goes_ex_in_the_step():
    now = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
    inp = inputs(spot=600.0, strike=600.0, right=Right.CALL, now=now,
                 carry=_dividend_carry(date(2026, 11, 25)))
    g = greeks(inp, 0.15, Method.FD)
    after = price(inp.with_now(now + timedelta(days=1)).with_spot(600.0 - 1.80), 0.15,
                  Method.FD)
    assert g.theta == pytest.approx(after - price(inp, 0.15, Method.FD), rel=1e-12)


def test_baw_gamma_on_the_expiry_day_follows_the_european_gamma():
    # N5: op de expiratiedag is σ√T ≈ 1 %; de vaste bump van 0,5 % smeerde de gammapiek uit.
    now = datetime(2026, 12, 18, 10, 0, tzinfo=NY)
    inp = inputs(spot=100.0, strike=100.0, right=Right.CALL, now=now)
    baw = greeks(inp, 0.25, Method.BAW)
    european = greeks(inp, 0.25, Method.EUROPEAN)
    assert baw.gamma == pytest.approx(european.gamma, rel=0.03)
    h = 0.005 * inp.spot
    fixed = (price(inp.with_spot(100 + h), 0.25, Method.BAW) - 2 * price(inp, 0.25, Method.BAW)
             + price(inp.with_spot(100 - h), 0.25, Method.BAW)) / h**2
    assert abs(fixed / european.gamma - 1) > 0.03        # de oude vaste bump zat ernaast


def test_the_spot_bump_scales_with_sigma_root_t_between_its_limits():
    b = Bumps()
    assert spot_bump_fraction(0.25, 1.0, b) == b.spot_frac                  # lange looptijd
    assert spot_bump_fraction(0.25, 1e-6, b) == b.spot_frac_min             # uren voor expiratie
    mid = float(spot_bump_fraction(0.25, 0.0004, b))
    assert b.spot_frac_min < mid < b.spot_frac
    assert mid == pytest.approx(b.spot_sigma_frac * 0.25 * 0.02)


def test_a_dividend_with_its_ex_date_on_the_expiry_day_is_counted_once():
    # Oud #34: op de expiratiedag zit het dividend al in de spot; het mag er niet nog eens
    # als dividend afgaan. De dag ervoor telt het nog wel.
    carry = Carry(0.04, dividends=(CashDividend(EXPIRY, 1.80),))
    on_expiry = inputs(now=datetime(2026, 12, 18, 10, 0, tzinfo=NY), carry=carry,
                       right=Right.CALL)
    assert on_expiry.discrete_dividends == ()
    plain = inputs(now=on_expiry.now, right=Right.CALL)
    assert price(on_expiry, 0.25, Method.BAW) == pytest.approx(price(plain, 0.25, Method.BAW))
    day_before = inputs(now=datetime(2026, 12, 17, 10, 0, tzinfo=NY), carry=carry,
                        right=Right.CALL)
    assert day_before.discrete_dividends == carry.dividends
    # theta de dag ervoor: het dividend gaat in de stap ex, de spot zakt mee
    assert greeks(day_before, 0.25, Method.FD).theta < 0
