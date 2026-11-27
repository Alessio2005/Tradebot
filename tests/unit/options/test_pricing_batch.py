"""De numpy-kernel (`pricing.batch`) tegen QuantLib op een vast raster.

Het raster: calls en puts, moneyness 0,7-1,3, DTE 0 (expiratiedag, vóór 16:00) t/m 400,
vol 0,05-1,5, r ∈ {0; 0,04; 0,08}, q ∈ {0; 0,02; 0,06}. QuantLib is de referentie; elke
afwijking boven de grens is een fout in de kernel.
"""
from __future__ import annotations

import itertools
from datetime import datetime, timedelta

import numpy as np
import pytest

from tradebot.options.calendar import NY
from tradebot.options.contracts import Right
from tradebot.options.pricing import batch
from tradebot.options.pricing.batch import (
    BatchInputs,
    KernelIVStatus,
    baw_greeks,
    baw_price,
    european_price,
    implied_vols,
)
from tradebot.options.pricing.engines import Method
from tradebot.options.pricing.greeks import greeks
from tradebot.options.pricing.implied_vol import IVSettings, IVStatus, implied_vol
from tradebot.options.pricing.vanilla import OptionInputs, price
from tradebot.options.rates import Carry, CashDividend
from tradebot.utils.failfast import DataContractError

NOW = datetime(2026, 3, 2, 10, 30, tzinfo=NY)       # maandag, vóór de sluiting
SPOT = 100.0
RATES = (0.0, 0.04, 0.08)
YIELDS = (0.0, 0.02, 0.06)
IV_ACCURACY = IVSettings().accuracy


def _grid(moneyness: tuple[float, ...], dtes: tuple[int, ...], vols: tuple[float, ...]
          ) -> tuple[list[OptionInputs], np.ndarray]:
    opts, sigmas = [], []
    for right, m, dte, vol, r, q in itertools.product(list(Right), moneyness, dtes, vols,
                                                      RATES, YIELDS):
        opts.append(OptionInputs(SPOT, round(SPOT * m, 2), right,
                                 NOW.date() + timedelta(days=dte), NOW, Carry(r, q)))
        sigmas.append(vol)
    return opts, np.array(sigmas)


@pytest.fixture(scope="module")
def price_grid() -> tuple[list[OptionInputs], np.ndarray]:
    return _grid((0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3),
                 (0, 1, 2, 7, 30, 90, 180, 400),
                 (0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.0, 1.5))


@pytest.fixture(scope="module")
def small_grid() -> tuple[list[OptionInputs], np.ndarray]:
    return _grid((0.7, 0.9, 1.0, 1.1, 1.3), (0, 1, 7, 30, 400), (0.05, 0.2, 0.5, 1.5))


@pytest.mark.parametrize(("method", "kernel", "bound"), [
    (Method.BAW, baw_price, 1e-9),
    (Method.EUROPEAN, european_price, 1e-10),
])
def test_prices_match_quantlib(price_grid, method, kernel, bound):
    opts, vols = price_grid
    ours = kernel(BatchInputs.from_options(opts), vols)
    ref = np.array([price(o, float(v), method) for o, v in zip(opts, vols)])
    assert not np.isnan(ours).any()
    assert np.max(np.abs(ours - ref)) <= bound * SPOT


def test_put_without_positive_rate_and_call_with_negative_rate_follow_quantlib():
    """QuantLib 1.44: een put met rD >= 1 is Europees; een call met rD > 1 is een fout."""
    expiry = NOW.date() + timedelta(days=90)
    put = OptionInputs(SPOT, 120.0, Right.PUT, expiry, NOW, Carry(-0.01, 0.02))
    call = OptionInputs(SPOT, 120.0, Right.CALL, expiry, NOW, Carry(-0.01, 0.02))
    ours = baw_price(BatchInputs.from_options([put, call]), np.array([0.3, 0.3]))
    assert ours[0] == pytest.approx(price(put, 0.3, Method.EUROPEAN), abs=1e-12)
    assert price(put, 0.3, Method.BAW) == price(put, 0.3, Method.EUROPEAN)
    assert np.isnan(ours[1])
    with pytest.raises(RuntimeError, match="negative interest rates"):
        price(call, 0.3, Method.BAW)


def test_a_critical_price_that_does_not_converge_is_nan(monkeypatch):
    opts = [OptionInputs(SPOT, 110.0, Right.PUT, NOW.date() + timedelta(days=90), NOW,
                         Carry(0.04))]
    monkeypatch.setattr(batch, "_MAX_CRITICAL_ITERATIONS", 0)
    assert np.isnan(baw_price(BatchInputs.from_options(opts), np.array([0.3]))).all()


def _iv_targets(opts: list[OptionInputs], vols: np.ndarray) -> np.ndarray:
    """Prijzen bij de rastervol: afwisselend op centen afgerond (zoals een quote) en exact.

    Exact alleen vanaf een cent: daaronder bestaat geen quote, en ligt de prijs bij `min_vol`
    binnen afronding van het doel (dan beslist QuantLib, zie `_EDGE_FRACTION`).
    """
    exact = np.array([price(o, float(v), Method.BAW) for o, v in zip(opts, vols)])
    use_exact = (np.arange(exact.size) % 2 == 1) & (exact >= 0.01)
    return np.where(use_exact, exact, np.round(exact, 2))


def _special_cases() -> tuple[list[OptionInputs], list[float]]:
    """Elke status van `implied_vol`, ook op de randen."""
    expiry = NOW.date() + timedelta(days=30)
    itm_put = OptionInputs(SPOT, 130.0, Right.PUT, expiry, NOW, Carry(0.04))
    otm_call = OptionInputs(SPOT, 110.0, Right.CALL, expiry, NOW, Carry(0.04, 0.02))
    expired = OptionInputs(SPOT, 100.0, Right.CALL, NOW.date(), NOW + timedelta(hours=6),
                           Carry(0.04))
    at_min_vol = price(otm_call, 0.01, Method.BAW)
    cases = [
        (expired, 1.0),                                  # EXPIRED
        (otm_call, float("nan")),                        # NO_QUOTE
        (otm_call, 0.0),                                 # NO_BID
        (itm_put, 30.0 - 0.02),                          # BELOW_INTRINSIC
        (itm_put, 30.0 - 0.004),                         # binnen één tick: + ½ tick
        (itm_put, 30.0),                                 # op de ondergrens: + ½ tick
        (otm_call, SPOT + 0.01),                         # ABOVE_UPPER_BOUND
        (otm_call, 99.0),                                # boven de waarde bij max_vol
        (otm_call, 1.0),                                 # gewoon OK
    ]
    assert at_min_vol > 0
    cases.append((otm_call, at_min_vol))                 # precies op de rand van het bracket
    return [c[0] for c in cases], [c[1] for c in cases]


def _assert_iv_parity(opts: list[OptionInputs], targets: np.ndarray) -> tuple[int, int]:
    res = implied_vols(BatchInputs.from_options(opts), targets)
    deferred = 0
    for o, p, status, sigma in zip(opts, targets, res.status, res.value):
        ref = implied_vol(o, None if np.isnan(p) else float(p), Method.BAW)
        if status is KernelIVStatus.TO_QUANTLIB:
            deferred += 1
            assert np.isnan(sigma)
            continue
        assert status.value == ref.status.value, (o, p)
        if status is not KernelIVStatus.OK:
            assert np.isnan(sigma)
            continue
        assert ref.method is Method.BAW and ref.value is not None
        assert abs(sigma - ref.value) <= 2 * IV_ACCURACY
        lower = o.intrinsic
        goal = lower + 0.005 if p <= lower else p
        assert abs(price(o, float(sigma), Method.BAW) - goal) <= 1e-8
    return deferred, len(opts)


def test_implied_vols_match_quantlib(small_grid):
    opts, vols = small_grid
    deferred, total = _assert_iv_parity(opts, _iv_targets(opts, vols))
    # Naar QuantLib gaan contracten met een ontspoorde seed bij `min_vol` (lange looptijd,
    # grote carry) en diep-ITM-prijzen zonder vega, die op de rand van het bracket liggen.
    assert deferred <= 0.1 * total


def test_a_realistic_chain_needs_no_quantlib():
    """r = 4,5 %, q = 1,5 %, quotes op centen: de kernel lost alles zelf op."""
    opts, vols = [], []
    for right, m, dte, vol in itertools.product(list(Right), (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2),
                                                (0, 3, 14, 30, 60, 120, 400), (0.15, 0.3, 0.6)):
        opts.append(OptionInputs(SPOT, round(SPOT * m, 2), right,
                                 NOW.date() + timedelta(days=dte), NOW, Carry(0.045, 0.015)))
        vols.append(vol)
    targets = np.round(baw_price(BatchInputs.from_options(opts), np.array(vols)), 2)
    deferred, _ = _assert_iv_parity(opts, targets)
    assert deferred == 0


def test_implied_vols_special_cases_match_quantlib():
    opts, targets = _special_cases()
    res = implied_vols(BatchInputs.from_options(opts), np.array(targets))
    assert {s.value for s in res.status} >= {s.value for s in IVStatus}
    _assert_iv_parity(opts, np.array(targets))


def test_a_price_on_the_edge_of_the_bracket_goes_to_quantlib():
    opts, targets = _special_cases()
    res = implied_vols(BatchInputs.from_options(opts[-1:]), np.array(targets[-1:]))
    assert res.status == (KernelIVStatus.TO_QUANTLIB,)
    assert implied_vol(opts[-1], targets[-1], Method.BAW).status is IVStatus.OK


def test_where_the_kernel_prices_low_vols_quantlib_agrees(price_grid):
    """De IV-solver steunt hierop: geeft de kernel bij een vol een prijs, dan rekent QuantLib
    daar dezelfde prijs en faalt niet (ook bij 1-5 %, waar de seed kan ontsporen)."""
    opts, _ = price_grid
    low = (0.01, 0.0125, 0.015, 0.02, 0.03, 0.04)
    rows = opts[::8] * len(low)
    vols = np.repeat(np.array(low), len(opts[::8]))
    ours = baw_price(BatchInputs.from_options(rows), vols)
    priced = 0
    for o, v, mine in zip(rows, vols, ours):
        if np.isnan(mine):
            continue
        priced += 1
        assert abs(price(o, float(v), Method.BAW) - mine) <= 1e-9 * SPOT
    assert priced >= 0.8 * len(rows)


def test_a_derailed_seed_is_not_a_price():
    """r = 8 %, q = 2 %, 400 dagen, vol 1 %: exp(h) in de seed is ~1e24; QuantLib geeft daar
    een prijs of een fout naargelang de laatste bit van exp en log."""
    put = OptionInputs(SPOT, 90.0, Right.PUT, NOW.date() + timedelta(days=400), NOW,
                       Carry(0.08, 0.02))
    assert np.isnan(baw_price(BatchInputs.from_options([put]), np.array([0.01]))).all()


def test_no_price_at_min_vol_goes_to_quantlib():
    """Hier faalt QuantLib bij 1 % en 2 % en neemt 5 % als ondergrens, of valt Brent in het
    ontspoorde gebied en neemt QuantLib de binomiale boom: de kernel beslist niet."""
    expiry = NOW.date() + timedelta(days=400)
    call = OptionInputs(SPOT, 85.0, Right.CALL, expiry, NOW, Carry(0.0, 0.06))
    put = OptionInputs(SPOT, 90.0, Right.PUT, expiry, NOW, Carry(0.08, 0.02))
    targets = np.array([price(call, 0.25, Method.BAW), price(put, 0.05, Method.BAW)])
    res = implied_vols(BatchInputs.from_options([call, put]), targets)
    assert res.status == (KernelIVStatus.TO_QUANTLIB,) * 2 and np.isnan(res.value).all()
    for vol in (0.01, 0.02):
        with pytest.raises(RuntimeError):
            price(call, vol, Method.BAW)
    assert implied_vol(call, float(targets[0]), Method.BAW).method is Method.BAW


def test_a_low_iv_with_a_sane_seed_is_solved():
    call = OptionInputs(SPOT, 103.0, Right.CALL, NOW.date() + timedelta(days=30), NOW,
                        Carry(0.04))
    target = price(call, 0.03, Method.BAW)
    _assert_iv_parity([call], np.array([target]))
    res = implied_vols(BatchInputs.from_options([call]), np.array([target]))
    assert res.status == (KernelIVStatus.OK,)
    assert res.value[0] == pytest.approx(0.03, abs=1e-9)


def test_a_critical_price_that_does_not_converge_goes_to_quantlib(monkeypatch):
    monkeypatch.setattr(batch, "_MAX_CRITICAL_ITERATIONS", 0)
    opts, targets = _special_cases()
    res = implied_vols(BatchInputs.from_options(opts[-2:-1]), np.array(targets[-2:-1]))
    assert res.status == (KernelIVStatus.TO_QUANTLIB,)


def test_a_solve_that_does_not_converge_goes_to_quantlib(monkeypatch):
    monkeypatch.setattr(batch, "_MAX_IV_ITERATIONS", 1)
    opts, targets = _special_cases()
    res = implied_vols(BatchInputs.from_options(opts[-2:-1]), np.array(targets[-2:-1]))
    assert res.status == (KernelIVStatus.TO_QUANTLIB,)
    assert np.isnan(res.value).all()


def test_kernel_statuses_carry_the_values_of_iv_status():
    assert {s.value for s in IVStatus} | {"to_quantlib"} == {s.value for s in KernelIVStatus}


def test_baw_greeks_match_quantlib(small_grid):
    opts, vols = small_grid
    ours = baw_greeks(BatchInputs.from_options(opts), vols)
    # Ook bij r = 0: de rho-bump gaat daar niet onder nul (eenzijdig), dus QuantLib rekent ook
    # een call, en de kernel geeft dezelfde rho.
    assert any(o.carry.rate == 0.0 and o.right is Right.CALL for o in opts)
    for i, (o, v) in enumerate(zip(opts, vols)):
        ref = greeks(o, float(v), Method.BAW)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            mine, theirs = float(getattr(ours, name)[i]), getattr(ref, name)
            assert abs(mine - theirs) <= 1e-6 * max(1.0, abs(theirs)), (name, o, v)


def test_expired_contracts_have_no_times():
    expired = OptionInputs(SPOT, 100.0, Right.CALL, NOW.date(), NOW + timedelta(hours=6),
                           Carry(0.04))
    inp = BatchInputs.from_options([expired])
    assert inp.expired.all() and np.isnan(inp.t_model).all() and np.isnan(inp.theta_days).all()


def test_a_discrete_dividend_in_the_term_goes_to_quantlib():
    """QuantLib weigert BAW met een discreet dividend (alleen FD); de kernel rekent niet."""
    expiry = NOW.date() + timedelta(days=60)
    carry = Carry(0.04, dividends=(CashDividend(NOW.date() + timedelta(days=20), 0.8),))
    with_div = OptionInputs(SPOT, 100.0, Right.PUT, expiry, NOW, carry)
    after = OptionInputs(SPOT, 100.0, Right.PUT, NOW.date() + timedelta(days=10), NOW, carry)
    plain = OptionInputs(SPOT, 100.0, Right.PUT, expiry, NOW, Carry(0.04))
    inp = BatchInputs.from_options([with_div, after, plain])
    assert inp.has_dividends.tolist() == [True, False, False]
    vols = np.full(3, 0.3)
    for kernel in (baw_price, european_price):
        assert np.isnan(kernel(inp, vols)).tolist() == [True, False, False]
    assert np.isnan(baw_greeks(inp, vols).delta).tolist() == [True, False, False]
    res = implied_vols(inp, np.array([5.0, 2.0, 5.0]))
    assert res.status[0] is KernelIVStatus.TO_QUANTLIB and np.isnan(res.value[0])
    assert res.status[1:] == (KernelIVStatus.OK,) * 2
    with pytest.raises(DataContractError):
        price(with_div, 0.3, Method.BAW)


def test_greeks_at_zero_vol_are_nan_without_warnings():
    put = OptionInputs(SPOT, 100.0, Right.PUT, NOW.date() + timedelta(days=30), NOW,
                       Carry(0.04))
    g = baw_greeks(BatchInputs.from_options([put]), np.array([0.0]))
    assert np.isnan(g.vega).all()
