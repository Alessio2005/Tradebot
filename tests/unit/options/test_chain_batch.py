"""De keten in batches (`pricing.vectorized`) tegen de keten contract voor contract.

Dezelfde statussen, IV binnen 2·iv_accuracy, Greeks binnen 1e-5 relatief, dezelfde volgorde;
en een spy op QuantLib die laat zien dat BAW-contracten niet meer per stuk door QuantLib gaan
en dat de FD-solve met een startwaarde minder waarderingen nodig heeft.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from tests.unit.options.helpers import EXPIRY, NOW, inputs
from tradebot.options.chain import builder as builder_mod
from tradebot.options.chain.builder import Chain, ChainBuilder, RawOption, _escrowed
from tradebot.options.contracts import OptionContract, Quote, Right
from tradebot.options.pricing import implied_vol as iv_mod
from tradebot.options.pricing import vanilla as vanilla_mod
from tradebot.options.pricing.batch import KernelIVStatus
from tradebot.options.pricing.engines import EngineSettings, Method, Use
from tradebot.options.pricing.greeks import greeks
from tradebot.options.pricing.implied_vol import IVResult, IVStatus, implied_vol
from tradebot.options.pricing.settings import pricer_settings
from tradebot.options.pricing.vanilla import OptionInputs, price
from tradebot.options.rates import Carry, CashDividend
from tradebot.schemas.options import load_options_config

SETTINGS = pricer_settings(load_options_config().pricing)
ON, OFF = replace(SETTINGS, vectorized=True), replace(SETTINGS, vectorized=False)
ACC = SETTINGS.iv.accuracy
CARRY = Carry(0.04)
DIV_CARRY = Carry(0.04, dividends=(CashDividend(date(2026, 12, 4), 0.80),))
SPOT = 100.0


def _raw(strike: float, right: Right, expiry: date, carry: Carry, method: Method,
         vol: float, half: float = 0.03, age: timedelta = timedelta(0)) -> RawOption:
    inp = OptionInputs(SPOT, strike, right, expiry, NOW, carry)
    theo = price(inp, vol, method, SETTINGS.engines)
    q = Quote(round(max(theo - half, 0.0), 2), round(theo + half, 2), NOW - age,
              volume=1000, open_interest=5000)
    return RawOption(OptionContract("TST", expiry, strike, right), q)


def _plain_chain() -> list[RawOption]:
    """Calls en puts over strikes 70-130 en drie expiraties, met een smile en wat randgevallen."""
    rows = []
    for expiry in (date(2026, 11, 27), EXPIRY, date(2027, 3, 19)):
        for strike in range(70, 131, 5):
            for right in (Right.CALL, Right.PUT):
                vol = 0.22 + 0.4 * ((strike - SPOT) / SPOT) ** 2
                rows.append(_raw(float(strike), right, expiry, CARRY, Method.BAW, vol))
    c = OptionContract("TST", EXPIRY, 100.0, Right.PUT)
    rows += [
        RawOption(c, Quote(0.0, 0.05, NOW)),                          # bid 0: NO_BID op de bid
        RawOption(c, Quote(None, None, NOW)),                         # geen quote
        RawOption(OptionContract("TST", EXPIRY, 130.0, Right.PUT),    # onder intrinsiek
                  Quote(20.0, 20.1, NOW)),
        RawOption(OptionContract("TST", EXPIRY, 70.0, Right.CALL),    # boven de spot
                  Quote(120.0, 121.0, NOW)),
        _raw(100.0, Right.CALL, EXPIRY, CARRY, Method.BAW, 0.25, age=timedelta(seconds=60)),
        RawOption(OptionContract("TST", NOW.date() - timedelta(days=1), 100.0, Right.PUT),
                  Quote(1.0, 1.1, NOW)),                              # verlopen: valt weg
    ]
    return rows


def _div_chain() -> list[RawOption]:
    rows = [_raw(k, r, EXPIRY, DIV_CARRY, Method.FD, 0.20 + 0.3 * ((k - SPOT) / SPOT) ** 2)
            for k in (90.0, 100.0, 110.0) for r in (Right.CALL, Right.PUT)]
    # zonder dividend in de looptijd: gewoon BAW in dezelfde keten
    rows.append(_raw(100.0, Right.PUT, date(2026, 11, 27), DIV_CARRY, Method.BAW, 0.25))
    return rows


def _build(settings, rows, carry, skip=None, **kw) -> Chain:
    return ChainBuilder(settings, 10.0, spread_iv=True, **kw).build("TST", SPOT, rows, NOW, carry,
                                                                    skip=skip)


GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def _assert_same(on: Chain, off: Chain, settings=SETTINGS, *, greeks_vs_off: bool = True) -> None:
    """Zelfde volgorde, methode en statussen; IV binnen 2·iv_accuracy; de Greeks van `on`
    gelijk aan die van QuantLib bij dezelfde σ, en (met `greeks_vs_off`) binnen 1e-5
    relatief van die van `off`.

    Zonder `greeks_vs_off` alleen het eerste: bij de standaard `iv_accuracy` (1e-6) wijkt de
    Brent-IV van het oude pad tot ~2e-7 af van de wortel, en bij een diep-ITM-optie op de
    intrinsieke waarde verschuift dat vega, theta en rho al ~2e-5 relatief. Die vergelijking
    loopt daarom met een strakke `iv_accuracy`."""
    acc = settings.iv.accuracy
    assert [o.contract for o in on.options] == [o.contract for o in off.options]
    for a, b in zip(on.options, off.options, strict=True):
        assert (a.method, a.stale) == (b.method, b.stale)
        for x, y in ((a.iv_mid, b.iv_mid), (a.iv_bid, b.iv_bid), (a.iv_ask, b.iv_ask)):
            assert (x.status, x.method) == (y.status, y.method), a.contract
            if x.status is IVStatus.OK:
                assert x.value == pytest.approx(y.value, abs=2 * acc)
            else:
                assert x.value is None and y.value is None
        assert (a.greeks is None) == (b.greeks is None)
        if a.greeks is None or b.greeks is None or a.iv is None:
            continue
        c = a.contract
        inp = OptionInputs(on.spot, c.strike, c.right, c.expiry, on.asof, on.carry)
        ref = greeks(inp, a.iv, a.iv_mid.method or a.method, settings.engines, settings.bumps)
        for name in GREEKS:
            ga = getattr(a.greeks, name)
            assert ga == pytest.approx(getattr(ref, name), rel=1e-6, abs=1e-12), (c, name)
            if greeks_vs_off:
                assert ga == pytest.approx(getattr(b.greeks, name), rel=1e-5, abs=1e-9), \
                    (c, name)


@pytest.fixture
def ql_calls(monkeypatch) -> Counter[str]:
    """Telt QuantLib per engine: IV-solves (`solve-`), hun waarderingen (`iv-`) en de
    waarderingen voor Greeks (`greeks-`)."""
    calls: Counter[str] = Counter()
    real_pricer, real_eval = iv_mod.vol_pricer, vanilla_mod._evaluate

    @contextmanager
    def pricer(inp, method, engines=EngineSettings()) -> Iterator[Callable[[float], float]]:
        calls[f"solve-{method.value}"] += 1
        with real_pricer(inp, method, engines) as npv:
            def counted(sigma: float) -> float:
                calls[f"iv-{method.value}"] += 1
                return npv(sigma)
            yield counted

    def evaluate(inp, vol, method, settings, with_greeks):
        calls[f"greeks-{method.value}"] += 1
        return real_eval(inp, vol, method, settings, with_greeks)

    monkeypatch.setattr(iv_mod, "vol_pricer", pricer)
    monkeypatch.setattr(vanilla_mod, "_evaluate", evaluate)
    return calls


def test_the_batch_chain_matches_the_per_contract_chain(ql_calls):
    rows = _plain_chain()
    ql_calls.clear()                                 # de quotes zelf komen uit QuantLib
    on = _build(ON, rows, CARRY, skip=lambda r: r.contract.strike == 125.0)
    assert ql_calls["iv-baw"] == 0 and ql_calls["greeks-baw"] == 0   # niets per stuk
    off = _build(OFF, rows, CARRY, skip=lambda r: r.contract.strike == 125.0)
    assert ql_calls["iv-baw"] > 0 and ql_calls["greeks-baw"] > 0
    _assert_same(on, off, greeks_vs_off=False)
    tight = replace(SETTINGS, iv=replace(SETTINGS.iv, accuracy=1e-10))
    _assert_same(_build(replace(tight, vectorized=True), rows, CARRY),
                 _build(replace(tight, vectorized=False), rows, CARRY), tight)
    statuses = Counter(o.iv_mid.status for o in on.options) + \
        Counter(o.iv_bid.status for o in on.options)
    assert {IVStatus.OK, IVStatus.NO_BID, IVStatus.NO_QUOTE, IVStatus.BELOW_INTRINSIC,
            IVStatus.ABOVE_UPPER_BOUND} <= set(statuses)
    assert any(o.stale for o in on.options)


def test_with_a_discrete_dividend_fd_starts_near_the_seed_and_needs_fewer_evaluations(ql_calls):
    rows = _div_chain()
    ql_calls.clear()
    on = _build(ON, rows, DIV_CARRY)
    fd_on = ql_calls["iv-fd"]
    ql_calls.clear()
    off = _build(OFF, rows, DIV_CARRY)
    fd_off = ql_calls["iv-fd"]
    _assert_same(on, off)
    assert [o.method for o in on.options] == [Method.FD] * 6 + [Method.BAW]
    assert all(o.iv_mid.status is IVStatus.OK for o in on.options)
    assert 0 < fd_on < fd_off


def test_the_decision_engine_still_goes_per_contract():
    rows = _plain_chain()[:6]
    on = _build(ON, rows, CARRY, use=Use.DECISION)
    off = _build(OFF, rows, CARRY, use=Use.DECISION)
    assert {o.method for o in on.options} == {Method.BINOMIAL_LR}
    _assert_same(on, off)


def test_to_quantlib_and_nan_greeks_go_through_quantlib(monkeypatch, ql_calls):
    rows = _plain_chain()[12:16]
    real_ivs, real_greeks = builder_mod.implied_vols, builder_mod.baw_greeks

    def ivs(batch, prices, cfg):
        res = real_ivs(batch, prices, cfg)
        code = res.code.copy()
        code[0] = list(KernelIVStatus).index(KernelIVStatus.TO_QUANTLIB)
        return replace(res, value=np.where(code == code[0], np.nan, res.value), code=code)

    def nan_greeks(batch, vol, bumps):
        g = real_greeks(batch, vol, bumps)
        return replace(g, rho=np.where(np.arange(vol.size) == 0, np.nan, g.rho))

    monkeypatch.setattr(builder_mod, "implied_vols", ivs)
    monkeypatch.setattr(builder_mod, "baw_greeks", nan_greeks)
    ql_calls.clear()
    on = _build(ON, rows, CARRY)
    assert ql_calls["solve-baw"] == 3                # alleen contract 0, mid/bid/ask
    assert ql_calls["greeks-baw"] > 0                # contract 0 én 1 (rho NaN)
    _assert_same(on, _build(OFF, rows, CARRY))


def test_the_escrowed_seed_inputs():
    carry = Carry(0.04, dividends=(CashDividend(date(2026, 12, 4), 1.0),
                                   CashDividend(date(2027, 3, 4), 1.0)))
    e = _escrowed(inputs(carry=carry))               # alleen het dividend in de looptijd
    assert e is not None and e.carry == Carry(0.04) and not e.discrete_dividends
    assert 99.0 < e.spot < 99.01                    # 100 − 1·e^{−r·t}
    big = Carry(0.04, dividends=(CashDividend(date(2026, 12, 4), 150.0),))
    assert _escrowed(inputs(carry=big)) is None


def _count(monkeypatch) -> Counter[str]:
    calls: Counter[str] = Counter()
    real = iv_mod.vol_pricer

    @contextmanager
    def pricer(inp, method, engines=EngineSettings()):
        with real(inp, method, engines) as npv:
            def counted(sigma: float) -> float:
                calls["n"] += 1
                return npv(sigma)
            yield counted

    monkeypatch.setattr(iv_mod, "vol_pricer", pricer)
    return calls


@pytest.mark.parametrize("bracket", [(0.25, 0.35), (0.40, 0.50), (0.10, 0.20)])
def test_a_bracket_gives_the_same_iv_as_the_full_path(monkeypatch, bracket):
    inp = inputs()
    p = price(inp, 0.30, Method.BAW)
    calls = _count(monkeypatch)
    full = implied_vol(inp, p, Method.BAW)
    n_full = calls["n"]
    calls.clear()
    narrow = implied_vol(inp, p, Method.BAW, bracket=bracket)
    assert (narrow.status, narrow.method) == (full.status, full.method)
    assert narrow.value == pytest.approx(full.value, abs=2 * ACC)
    if bracket[0] < 0.30 < bracket[1]:
        assert calls["n"] < n_full


def test_a_bracket_keeps_out_of_range_and_the_fallback(monkeypatch):
    inp = inputs()
    cfg = replace(SETTINGS.iv, max_vol=0.3)          # de prijs van vol 0,5 valt erbuiten
    assert implied_vol(inp, price(inp, 0.5, Method.BAW), Method.BAW, cfg=cfg,
                       bracket=(0.2, 0.3)) == IVResult(IVStatus.NO_CONVERGENCE)
    # QuantLib weigert een punt van het smalle interval: het volle pad met terugval beslist
    real = iv_mod.vol_pricer

    @contextmanager
    def refusing(i, method, engines=EngineSettings()):
        with real(i, method, engines) as npv:
            def f(sigma: float) -> float:
                if method is Method.BAW and sigma == 0.25:
                    raise RuntimeError("QuantLib weigert")
                return npv(sigma)
            yield f

    monkeypatch.setattr(iv_mod, "vol_pricer", refusing)
    p = price(inp, 0.30, Method.BAW)
    assert implied_vol(inp, p, Method.BAW, bracket=(0.25, 0.35)) == implied_vol(inp, p, Method.BAW)


def test_a_bracket_that_does_not_converge_takes_the_full_path():
    inp = inputs()
    p = price(inp, 0.30, Method.BAW)
    cfg = replace(SETTINGS.iv, max_evaluations=1)
    res = implied_vol(inp, p, Method.BAW, cfg=cfg, bracket=(0.2, 0.4))
    assert res == implied_vol(inp, p, Method.BAW, cfg=cfg)


def test_greeks_of_the_batch_match_quantlib_for_one_contract():
    (o,) = _build(ON, [_raw(95.0, Right.PUT, EXPIRY, CARRY, Method.BAW, 0.3)], CARRY).options
    assert o.iv is not None and o.greeks is not None
    ref = greeks(inputs(strike=95.0), o.iv, Method.BAW, SETTINGS.engines, SETTINGS.bumps)
    assert o.greeks.delta == pytest.approx(ref.delta, rel=1e-6)
    assert o.iv_mid.method is Method.BAW
