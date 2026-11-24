"""De NYSE-kalender uit QuantLib, de halve dagen uit de config, en looptijden op seconden."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from tradebot.options.calendar import NY, TradingCalendar, expiry_moment, year_fraction
from tradebot.utils.failfast import DataContractError

# Gecontroleerd tegen de NYSE-feestdagenlijst. Wijkt QuantLib na een update af, dan faalt dit.
HOLIDAYS = {
    2026: ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
           "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"],
    2027: ["2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
           "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"],
}
CAL = TradingCalendar([date(2026, 11, 27), date(2026, 12, 24)])


@pytest.mark.parametrize("year", sorted(HOLIDAYS))
def test_weekday_holidays_match_the_nyse(year):
    d, found = date(year, 1, 1), []
    while d.year == year:
        if d.weekday() < 5 and not CAL.is_trading_day(d):
            found.append(d.isoformat())
        d += timedelta(days=1)
    assert found == HOLIDAYS[year]


def test_sessions_close_early_on_half_days():
    assert CAL.session(date(2026, 11, 27))[1] == datetime(2026, 11, 27, 13, 0, tzinfo=NY)
    start, end = CAL.session(date(2026, 11, 30))
    assert (start.hour, start.minute, end.hour) == (9, 30, 16)
    with pytest.raises(DataContractError, match="geen handelsdag"):
        CAL.session(date(2026, 11, 26))


def test_a_half_day_must_be_a_trading_day():
    with pytest.raises(DataContractError, match="halve dag"):
        TradingCalendar([date(2026, 11, 26)])


def test_is_open():
    assert CAL.is_open(datetime(2026, 11, 24, 10, 0, tzinfo=NY))
    assert not CAL.is_open(datetime(2026, 11, 24, 16, 0, tzinfo=NY))
    assert not CAL.is_open(datetime(2026, 11, 27, 14, 0, tzinfo=NY))   # halve dag
    assert not CAL.is_open(datetime(2026, 11, 28, 10, 0, tzinfo=NY))   # zaterdag
    with pytest.raises(DataContractError, match="tijdzone"):
        CAL.is_open(datetime(2026, 11, 24, 10, 0))


def test_neighbouring_trading_days_and_counting():
    assert CAL.next_trading_day(date(2026, 11, 25)) == date(2026, 11, 27)
    assert CAL.previous_trading_day(date(2026, 11, 30)) == date(2026, 11, 27)
    assert CAL.trading_days_between(date(2026, 11, 24), date(2026, 11, 30)) == 3   # 25, 27 en 30 (26: Thanksgiving)
    assert CAL.trading_days_between(date(2026, 11, 30), date(2026, 11, 24)) == 0


def test_time_to_expiry_counts_seconds_until_four_pm():
    expiry = date(2026, 11, 27)
    assert expiry_moment(expiry) == datetime(2026, 11, 27, 16, 0, tzinfo=NY)
    now = datetime(2026, 11, 25, 13, 0, tzinfo=NY)
    assert year_fraction(now, expiry_moment(expiry)) == pytest.approx(51 / 8760, rel=1e-12)
    with pytest.raises(DataContractError, match="tijdzone"):
        year_fraction(datetime(2026, 11, 25, 13, 0), expiry_moment(expiry))
