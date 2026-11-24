"""Gedeelde invoer voor de options-tests: vaste klok, vaste contracten, geen netwerk."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from tradebot.options.calendar import NY, SECONDS_PER_YEAR, expiry_moment
from tradebot.options.contracts import Right
from tradebot.options.pricing.vanilla import OptionInputs
from tradebot.options.rates import Carry

NOW = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
EXPIRY = date(2026, 12, 18)


def inputs(spot: float = 100.0, strike: float = 100.0, right: Right = Right.PUT,
           expiry: date = EXPIRY, now: datetime = NOW, carry: Carry | None = None) -> OptionInputs:
    return OptionInputs(spot, strike, right, expiry, now, carry or Carry(0.04))


def inputs_with_t(t_years: float, **kw: object) -> OptionInputs:
    """Invoer met een looptijd van precies `t_years` tot het expiratiemoment."""
    now = expiry_moment(EXPIRY) - timedelta(seconds=t_years * SECONDS_PER_YEAR)
    return inputs(now=now, **kw)  # type: ignore[arg-type]
