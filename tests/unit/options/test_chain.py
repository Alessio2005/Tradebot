"""Ketenbouw en filters: IV en Greeks per contract, en een reden voor elke afwijzing."""
from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

import pytest

from tests.unit.options.helpers import EXPIRY, NOW, inputs
from tradebot.options.chain.builder import ChainBuilder, RawOption
from tradebot.options.chain.filters import (
    FilterSettings,
    Reject,
    filter_chain,
    filter_settings,
    filtered,
    reject_reason,
)
from tradebot.options.contracts import OptionContract, Quote, Right
from tradebot.options.events import Events
from tradebot.options.pricing.engines import Method, Use
from tradebot.options.pricing.implied_vol import IVStatus
from tradebot.options.pricing.settings import pricer_settings
from tradebot.options.pricing.vanilla import price
from tradebot.options.rates import Carry, CashDividend
from tradebot.schemas.options import load_options_config
from tradebot.utils.failfast import DataContractError

SETTINGS = pricer_settings(load_options_config().pricing)
CARRY = Carry(0.04)


def raw(strike: float = 100.0, right: Right = Right.PUT, vol: float = 0.25,
        expiry: date = EXPIRY, half: float = 0.02, oi: int = 5000, volume: int = 1000,
        age: timedelta = timedelta(0), underlying: str = "TST") -> RawOption:
    theo = price(inputs(strike=strike, right=right, expiry=expiry), vol, Method.BAW)
    q = Quote(max(theo - half, 0.0), theo + half, NOW - age, volume=volume, open_interest=oi)
    return RawOption(OptionContract(underlying, expiry, strike, right), q)


def build(rows, **kw):
    return ChainBuilder(SETTINGS, 10.0, **kw).build("TST", 100.0, rows, NOW, CARRY)


def test_the_builder_recovers_the_quoted_vol_and_fills_greeks():
    chain = build([raw(95.0, vol=0.30), raw(105.0, Right.CALL, vol=0.20)])
    put, call = chain.options
    assert put.iv == pytest.approx(0.30, abs=1e-4) and call.iv == pytest.approx(0.20, abs=1e-4)
    assert put.delta is not None and put.delta < 0 < call.delta  # type: ignore[operator]
    assert put.method is Method.BAW and not put.stale
    assert put.iv_bid.status is IVStatus.NO_QUOTE  # zonder spread_iv geen bid/ask-solve


def test_spread_iv_and_the_decision_engine():
    (o,) = build([raw()], use=Use.DECISION, spread_iv=True).options
    assert o.method is Method.BINOMIAL_LR
    assert o.iv_bid.value < o.iv < o.iv_ask.value  # type: ignore[operator]


def test_a_discrete_dividend_selects_fd():
    carry = Carry(0.04, dividends=(CashDividend(date(2026, 12, 4), 0.5),))
    rows = [raw()]
    (o,) = ChainBuilder(SETTINGS, 10.0).build("TST", 100.0, rows, NOW, carry).options
    assert o.method is Method.FD


def test_no_greeks_without_an_iv_and_stale_quotes_are_marked():
    bad = RawOption(OptionContract("TST", EXPIRY, 100.0, Right.PUT),
                    Quote(0.0, 0.01, NOW - timedelta(seconds=30)))
    (o,) = build([bad]).options
    assert o.iv is None and o.greeks is None and o.delta is None
    assert o.stale


def test_expired_contracts_are_left_out_and_one_chain_has_one_underlying():
    old = RawOption(OptionContract("TST", NOW.date() - timedelta(days=1), 100.0, Right.PUT),
                    Quote(1.0, 1.1, NOW))
    assert build([old, raw()]).expiries() == (EXPIRY,)
    with pytest.raises(DataContractError, match="één onderliggende"):
        build([raw(underlying="XYZ")])


def test_chain_lookup():
    later = date(2027, 1, 15)
    chain = build([raw(105.0, Right.CALL), raw(95.0), raw(100.0, expiry=later), raw(100.0)])
    assert chain.expiries() == (EXPIRY, later)
    assert [o.contract.strike for o in chain.by_expiry(EXPIRY)] == [95.0, 100.0, 105.0]
    assert [o.contract.strike for o in chain.by_expiry(EXPIRY, Right.CALL)] == [105.0]
    assert chain.find(EXPIRY, 95.0, Right.PUT) is not None
    assert chain.find(EXPIRY, 95.0, Right.CALL) is None


@pytest.mark.parametrize(("row", "reason"), [
    (dict(age=timedelta(seconds=60)), Reject.STALE_QUOTE),
    (dict(oi=10), Reject.OPEN_INTEREST),
    (dict(volume=10), Reject.VOLUME),
    (dict(half=0.30), Reject.SPREAD),
    (dict(), None),
])
def test_each_filter_has_its_own_reason(row, reason):
    (o,) = build([raw(**row)]).options
    assert reject_reason(o, FilterSettings()) is reason


def test_no_iv_and_no_bid_are_rejected():
    no_iv = RawOption(OptionContract("TST", EXPIRY, 100.0, Right.PUT), Quote(0.0, 0.01, NOW))
    no_bid = RawOption(OptionContract("TST", EXPIRY, 120.0, Right.CALL),
                       Quote(0.0, 0.10, NOW, volume=1000, open_interest=5000))
    a, b = build([no_iv, no_bid]).options
    assert reject_reason(a, FilterSettings()) is Reject.NO_IV
    assert b.iv is not None and reject_reason(b, FilterSettings()) is Reject.NO_BID


def test_a_wide_spread_passes_when_it_is_small_relative_to_mid():
    # spread 0,40 > 0,10 absoluut, maar < 10 % van een mid van ~ 9.
    (o,) = build([raw(110.0, half=0.20)]).options
    assert o.quote.mid > 4.0 and reject_reason(o, FilterSettings()) is None  # type: ignore[operator]


def test_filter_chain_counts_and_keeps():
    chain = build([raw(), raw(95.0, oi=1), raw(105.0, Right.CALL, oi=1)])
    res = filter_chain(chain, FilterSettings())
    assert len(res.kept) == 1 and res.rejected == Counter({Reject.OPEN_INTEREST: 2})
    only, res2 = filtered(chain, FilterSettings())
    assert only.options == res.kept and res2.rejected == res.rejected


def test_filter_settings_come_from_the_config():
    s = filter_settings(load_options_config().filters)
    assert s == FilterSettings(500, 100, 0.10, 0.10)


def test_events_are_only_usable_when_fresh():
    ev = Events("TST", date(2026, 11, 23), (date(2026, 12, 3), date(2027, 1, 28)))
    assert ev.usable(date(2026, 11, 24), 1) and not ev.usable(date(2026, 11, 26), 1)
    assert not Events("TST", None).usable(date(2026, 11, 24), 1)
    assert ev.earnings_between(date(2026, 11, 24), date(2026, 12, 3))
    assert not ev.earnings_between(date(2026, 12, 3), date(2027, 1, 27))
    assert ev.next_earnings(date(2026, 12, 3)) == date(2027, 1, 28)
    assert ev.next_earnings(date(2027, 2, 1)) is None
