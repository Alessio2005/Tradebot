"""Eenheden, contracten, quotes en carry: de bouwstenen weigeren onzin en mengen geen eenheden."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from tradebot.options.calendar import NY
from tradebot.options.contracts import Exercise, OptionContract, Quote, Right
from tradebot.options.rates import Carry, CashDividend
from tradebot.options.units import (
    DeltaShares,
    GreeksPerShare,
    PositionGreeks,
    RhoUSD,
    ThetaUSD,
    VegaUSD,
)
from tradebot.utils.failfast import DataContractError, TradebotContractError

TS = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
G = GreeksPerShare(delta=0.5, gamma=0.02, vega=0.1, theta=-0.05, rho=0.03)


def test_scale_applies_quantity_multiplier_and_sign():
    pos = G.scale(-2, 100)
    assert pos.delta == DeltaShares(-100.0)
    assert pos.vega == VegaUSD(-20.0)
    assert pos.theta == ThetaUSD(10.0)
    assert pos.rho == RhoUSD(-6.0)
    assert pos.gamma.value == pytest.approx(-4.0)


def test_scale_is_linear_and_a_mirrored_position_cancels():
    assert G.scale(3, 100) + G.scale(-3, 100) == PositionGreeks.zero()
    assert G.scale(1, 100) + G.scale(2, 100) == G.scale(3, 100)


def test_units_of_different_kind_cannot_be_added():
    with pytest.raises(TradebotContractError, match="niet op te tellen"):
        VegaUSD(1.0) + ThetaUSD(1.0)  # type: ignore[operator]
    assert -VegaUSD(2.0) == VegaUSD(-2.0)


def test_scale_refuses_a_non_positive_multiplier():
    with pytest.raises(TradebotContractError, match="multiplier"):
        G.scale(1, 0)


def test_contract_key_and_defaults():
    c = OptionContract("SPY", date(2026, 12, 18), 600.0, Right.CALL)
    assert c.key() == "SPY 2026-12-18 600 C"
    assert c.multiplier == 100 and c.exercise is Exercise.AMERICAN and c.con_id is None


@pytest.mark.parametrize("kw", [{"underlying": ""}, {"strike": 0.0}, {"multiplier": 0}])
def test_contract_refuses_invalid_fields(kw):
    args = {"underlying": "SPY", "expiry": date(2026, 12, 18), "strike": 600.0,
            "right": Right.PUT} | kw
    with pytest.raises(DataContractError):
        OptionContract(**args)


def test_quote_mid_and_validation():
    assert Quote(1.0, 1.2, TS).mid == pytest.approx(1.1)
    assert Quote(None, 1.2, TS).mid is None
    with pytest.raises(DataContractError, match="tijdzone"):
        Quote(1.0, 1.2, datetime(2026, 11, 24, 15, 0))
    with pytest.raises(DataContractError, match="bid"):
        Quote(1.3, 1.2, TS)


def test_carry_validation_and_dividend_window():
    d1, d2 = CashDividend(date(2026, 12, 5), 1.0), CashDividend(date(2027, 1, 5), 1.0)
    carry = Carry(0.04, dividends=(d1, d2))
    assert carry.dividends_in(date(2026, 11, 24), date(2026, 12, 18)) == (d1,)
    assert carry.dividends_in(date(2026, 12, 5), date(2026, 12, 18)) == ()
    with pytest.raises(DataContractError, match="positief"):
        CashDividend(date(2026, 12, 5), 0.0)
    with pytest.raises(DataContractError, match="eenheidsfout"):
        Carry(4.0)
    with pytest.raises(DataContractError, match="dividendyield"):
        Carry(0.04, div_yield=0.5)
    with pytest.raises(DataContractError, match="niet beide"):
        Carry(0.04, div_yield=0.01, dividends=(d1,))
