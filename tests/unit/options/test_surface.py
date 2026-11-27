"""Het volatiliteitsoppervlak: forward uit pariteit, smile, interpolatie en arbitragecontroles."""
from __future__ import annotations

import math
from datetime import date

import pytest

from tests.unit.options.helpers import EXPIRY, NOW
from tradebot.options.calendar import expiry_moment, year_fraction
from tradebot.options.chain.builder import Chain, PricedOption
from tradebot.options.contracts import OptionContract, Quote, Right
from tradebot.options.pricing.engines import Method
from tradebot.options.pricing.implied_vol import IVResult, IVStatus
from tradebot.options.pricing.surface import (
    MIN_VEGA,
    Smile,
    Surface,
    SurfaceStatus,
    build_surface,
    skew_25d,
)
from tradebot.options.rates import Carry
from tradebot.options.units import GreeksPerShare

LATER = date(2027, 1, 15)
STRIKES = (90.0, 92.5, 95.0, 97.5, 100.0, 102.5, 105.0, 107.5, 110.0)


def opt(strike: float, right: Right, iv: float | None, expiry: date = EXPIRY,
        mid: float = 2.0, delta: float = 0.0, vega: float = 0.1) -> PricedOption:
    q = Quote(mid - 0.01, mid + 0.01, NOW)
    res = IVResult(IVStatus.OK, iv, Method.BAW) if iv is not None else IVResult(
        IVStatus.NO_CONVERGENCE)
    g = GreeksPerShare(delta, 0.01, vega, -0.01, 0.01) if iv is not None else None
    return PricedOption(OptionContract("TST", expiry, strike, right), q, res,
                        IVResult(IVStatus.NO_QUOTE), IVResult(IVStatus.NO_QUOTE), g,
                        Method.BAW, False)


def smile_rows(vol, expiry: date = EXPIRY, strikes=STRIKES) -> list[PricedOption]:
    """Puts en calls met gelijke mid (forward = strike bij spot) en IV `vol(strike)`."""
    return [opt(k, r, vol(k), expiry) for k in strikes for r in (Right.PUT, Right.CALL)]


def chain(rows, spot: float = 100.0) -> Chain:
    return Chain("TST", spot, NOW, Carry(0.04), tuple(rows))


def test_a_flat_smile_gives_its_vol_everywhere():
    s = build_surface(chain(smile_rows(lambda k: 0.25) + smile_rows(lambda k: 0.25, LATER)))
    assert s.status is SurfaceStatus.OK and len(s.smiles) == 2
    sm = s.smiles[0]
    # gelijke mids voor put en call: de forward ligt op de strike plus de kleine correctie
    # voor de vroege-uitoefenpremie van de put (N26)
    assert sm.forward == pytest.approx(100.0, abs=0.05)
    for tenor in (30, 40):
        assert s.atm_vol(tenor) == pytest.approx(0.25, abs=1e-9)
    # N4: ver buiten de smiles (24 en 52 dagen) geen vlakke extrapolatie meer
    assert s.atm_vol(5) is None and s.atm_vol(200) is None
    assert s.term_slope() is None                      # 60 dagen niet omsloten
    nearby = s.atm_vol(58)                             # kleine marge: nog wel vlak
    assert nearby == pytest.approx(0.25, abs=1e-9)
    three = build_surface(chain(smile_rows(lambda k: 0.25) + smile_rows(lambda k: 0.25, LATER)
                                + smile_rows(lambda k: 0.25, date(2027, 1, 29))))
    assert three.term_slope() == pytest.approx(0.0, abs=1e-9)   # echt omsloten: 0 is een meting
    # buiten de punten vlak, niet geëxtrapoleerd
    assert sm.vol(1.0) == pytest.approx(sm.vol(sm.k[-1]))


def test_atm_vol_is_linear_in_total_variance_between_expiries():
    s = build_surface(chain(smile_rows(lambda k: 0.20) + smile_rows(lambda k: 0.30, LATER)))
    a, b = s.smiles
    t = 40 / 365.0
    x = (t - a.t) / (b.t - a.t)
    want = math.sqrt(((1 - x) * 0.20**2 * a.t + x * 0.30**2 * b.t) / t)
    assert s.atm_vol(40) == pytest.approx(want, rel=1e-9)
    assert s.term_slope(30, 50) > 0  # type: ignore[operator]
    assert s.term_slope(30, 60) is None  # 60 dagen ligt voorbij de laatste smile (N4)


def test_calendar_arbitrage_is_refused():
    s = build_surface(chain(smile_rows(lambda k: 0.40) + smile_rows(lambda k: 0.15, LATER)))
    assert s.status is SurfaceStatus.ARBITRAGE and s.reason == "kalenderarbitrage"
    assert s.atm_vol(30) is None and s.term_slope() is None


def test_butterfly_arbitrage_is_refused():
    spike = {k: (0.90 if k == 102.5 else 0.20) for k in STRIKES}
    s = build_surface(chain(smile_rows(spike.get)))
    assert s.status is SurfaceStatus.ARBITRAGE
    assert s.reason == f"butterfly-arbitrage in {EXPIRY.isoformat()}"


def test_only_the_expiry_with_butterfly_arbitrage_is_dropped():
    spike = {k: (0.90 if k == 102.5 else 0.20) for k in STRIKES}
    s = build_surface(chain(smile_rows(spike.get) + smile_rows(lambda k: 0.22, LATER)))
    assert s.status is SurfaceStatus.OK and [x.expiry for x in s.smiles] == [LATER]
    assert s.reason == f"weggelaten, butterfly-arbitrage in {EXPIRY.isoformat()}"
    assert s.atm_vol(52) == pytest.approx(0.22, rel=1e-6)
    assert s.atm_vol(30) is None  # één smile op 52 dagen: 30 is te ver om vlak te extrapoleren


def test_too_few_points_or_no_forward_give_no_surface():
    few = smile_rows(lambda k: 0.25, strikes=STRIKES[:2])
    assert build_surface(chain(few)).status is SurfaceStatus.INSUFFICIENT
    calls_only = [opt(k, Right.CALL, 0.25) for k in STRIKES]
    assert build_surface(chain(calls_only)).reason == "geen bruikbare smile"
    # een put die duurder is dan strike + call: forward ≤ 0
    neg = [opt(2.0, Right.PUT, 0.25, mid=10.0), opt(2.0, Right.CALL, 0.25, mid=0.02)]
    assert build_surface(chain(neg, spot=2.0)).status is SurfaceStatus.INSUFFICIENT


def test_points_without_iv_or_with_a_tiny_vega_do_not_count():
    rows = smile_rows(lambda k: 0.25)
    bad = [opt(k, r, None) for k in (85.0, 115.0) for r in (Right.PUT, Right.CALL)]
    tiny = [opt(k, r, 2.0, vega=MIN_VEGA / 2) for k in (80.0, 120.0)
            for r in (Right.PUT, Right.CALL)]
    s = build_surface(chain(rows + bad + tiny))
    assert s.status is SurfaceStatus.OK and len(s.smiles[0].k) == len(STRIKES)


def test_the_smile_call_price_is_black_on_the_forward():
    t = year_fraction(NOW, expiry_moment(EXPIRY))
    sm = Smile(EXPIRY, t, 100.0, math.exp(-0.04 * t), (-0.1, 0.0, 0.1), (0.04 * t,) * 3)
    d1 = 0.5 * math.sqrt(0.04 * t)
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    want = math.exp(-0.04 * t) * 100.0 * (n(d1) - n(-d1))
    assert sm.call(100.0) == pytest.approx(want, rel=1e-9)
    assert Surface("TST", SurfaceStatus.OK, (sm,)).atm_vol(sm.t * 365 - 1) == pytest.approx(0.2)


def test_skew_25d():
    rows = [opt(90.0, Right.PUT, 0.30, delta=-0.20), opt(95.0, Right.PUT, 0.27, delta=-0.27),
            opt(105.0, Right.CALL, 0.21, delta=0.26), opt(110.0, Right.CALL, 0.20, delta=0.15),
            opt(100.0, Right.CALL, None)]
    assert skew_25d(chain(rows), EXPIRY) == pytest.approx(0.06)
    assert skew_25d(chain(rows[:2]), EXPIRY) is None


def test_the_forward_is_corrected_for_the_early_exercise_premium_of_the_put():
    # N26: Amerikaanse prijzen uit BAW (r = 4 %, geen dividend): de echte forward is S·e^{rT}.
    # Put-call-pariteit op de Amerikaanse mids geeft een te lage forward.
    from tradebot.options.pricing.engines import Method
    from tradebot.options.pricing.surface import forward
    from tradebot.options.pricing.vanilla import OptionInputs, price

    carry = Carry(0.04)
    rows = []
    for k in (95.0, 97.5, 100.0, 102.5, 105.0):
        for r in (Right.PUT, Right.CALL):
            mid = price(OptionInputs(100.0, k, r, LATER, NOW, carry), 0.30, Method.BAW)
            rows.append(opt(k, r, 0.30, LATER, mid=mid))
    ch = chain(rows)
    t = year_fraction(NOW, expiry_moment(LATER))
    true_forward = 100.0 * math.exp(0.04 * t)
    uncorrected = 100.0 + (rows[5].quote.mid - rows[4].quote.mid) / math.exp(-0.04 * t)
    assert true_forward - uncorrected > 0.02                     # de oude schatting zat te laag
    assert forward(ch, LATER) == pytest.approx(true_forward, abs=0.01)


def test_the_forward_correction_handles_dividends_missing_ivs_and_expired_options():
    from tradebot.options.pricing.surface import forward
    from tradebot.options.rates import CashDividend

    rows = [opt(k, r, 0.25, LATER, mid=3.0) for k in (99.0, 100.0, 101.0)
            for r in (Right.PUT, Right.CALL)]
    plain = forward(chain(rows), LATER)
    with_div = Chain("TST", 100.0, NOW, Carry(0.04, dividends=(CashDividend(
        date(2026, 12, 20), 0.5),)), tuple(rows))
    assert forward(with_div, LATER) == pytest.approx(plain, abs=0.05)
    no_iv = [opt(k, r, None, LATER, mid=3.0) for k in (99.0, 100.0, 101.0)
             for r in (Right.PUT, Right.CALL)]
    assert forward(chain(no_iv), LATER) == pytest.approx(100.0)   # zonder IV: de kale mid
    stale = Chain("TST", 100.0, NOW, Carry(0.04), tuple(
        opt(k, r, 0.25, date(2026, 11, 20), mid=3.0) for k in (99.0, 100.0, 101.0)
        for r in (Right.PUT, Right.CALL)))
    assert forward(stale, date(2026, 11, 20)) == pytest.approx(100.0, abs=0.01)  # kale mid
