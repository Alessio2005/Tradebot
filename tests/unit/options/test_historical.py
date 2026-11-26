"""De historische markt: echte koers en IV in, ketens uit het model, IV-historie erbij."""
from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import date, datetime, time
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from tests.unit.options.market_helpers import CFG
from tradebot.options import cli
from tradebot.options.calendar import NY
from tradebot.options.data import historical as hist
from tradebot.options.data.historical import (
    HistoricalSeries,
    HistoricalSpec,
    build_market,
    fetch_history,
    session_fraction,
    strike_step,
    yfinance_history,
)
from tradebot.options.data.store import read_market, write_market
from tradebot.options.data.synthetic import SyntheticSpec, SyntheticUnderlying, generate
from tradebot.options.engine.replay import run_replay
from tradebot.options.rates import CashDividend
from tradebot.utils.failfast import DataContractError

DAYS = pd.bdate_range("2026-11-02", "2026-11-13")       # twee weken, zonder feestdagen


def series(name="TSTA", spot=100.0, iv=0.25, dividends=(), earnings=()):
    bars = pd.DataFrame({"open": [spot + i for i in range(len(DAYS))],
                         "high": [spot + i + 2 for i in range(len(DAYS))],
                         "low": [spot + i - 2 for i in range(len(DAYS))],
                         "close": [spot + i + 1 for i in range(len(DAYS))],
                         "volume": 1e6}, index=[d.date() for d in DAYS])
    ivs = pd.DataFrame({"open": iv, "close": iv + 0.01}, index=DAYS)
    return HistoricalSeries(name, bars, ivs, tuple(dividends), tuple(earnings))


SPEC = HistoricalSpec(date(2026, 11, 11), date(2026, 11, 12), strike_range=0.06, max_days=20,
                      snapshot_times=(time(10, 0), time(15, 0)), thin_wing_k=0.04)


def test_session_fraction_and_strike_step():
    assert session_fraction(time(9, 0)) == 0.0 and session_fraction(time(16, 30)) == 1.0
    assert session_fraction(time(12, 45)) == pytest.approx(0.5)
    assert [strike_step(s) for s in (50.0, 200.0, 400.0)] == [1.0, 2.5, 5.0]


@pytest.fixture(scope="module")
def mkt():
    div = CashDividend(date(2026, 11, 20), 0.5)
    return build_market([series("TSTB", 250.0, dividends=[div]),
                         series("TSTA", earnings=[date(2026, 11, 19)])], SPEC)


def test_the_market_has_the_period_the_snapshots_and_the_calendar(mkt):
    assert mkt.days == (date(2026, 11, 11), date(2026, 11, 12))
    assert len(mkt.snapshots) == 2 * 2 * 2
    assert [s.underlying for s in mkt.snapshots[:2]] == ["TSTA", "TSTB"]
    assert mkt.bars["TSTA"].index[-1] == pd.Timestamp("2026-11-12")     # niets na het einde
    ev = mkt.events["TSTA"]
    assert ev.fetched == SPEC.end and ev.earnings == (date(2026, 11, 19),)
    assert mkt.events["TSTB"].dividends[0].amount == 0.5


def test_spot_and_iv_follow_the_day_from_open_to_close(mkt):
    first = mkt.snapshots[0]                  # TSTA, 11-11 10:00: open 107, close 108
    w = session_fraction(time(10, 0))
    assert first.asof == datetime(2026, 11, 11, 10, 0, tzinfo=NY)
    assert first.spot == pytest.approx(107 + w)
    strikes = {o.contract.strike for o in first.options}
    assert min(strikes) >= first.spot * 0.94 - 1e-9 and max(strikes) <= first.spot * 1.06
    thin = [o for o in first.options if o.quote.open_interest == 100]
    assert thin and len(thin) < len(first.options)
    assert all(o.quote.ask > o.quote.bid >= 0 for o in first.options)


def test_an_expiry_moved_to_today_with_a_dividend_today_is_priced():
    # Vrijdag 3 juli 2026 is dicht: die expiratie valt op donderdag 2 juli, de dag zelf, en
    # het dividend van die dag ligt in het venster van de pricer (FD, niet BAW).
    day = pd.Timestamp("2026-07-02")
    s = HistoricalSeries("TSTA", pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0],
                                               "close": [100.5], "volume": [1e6]}, index=[day]),
                         pd.DataFrame({"open": [0.2], "close": [0.2]}, index=[day]),
                         (CashDividend(date(2026, 7, 2), 0.5),))
    spec = dataclasses.replace(SPEC, start=date(2026, 7, 2), end=date(2026, 7, 2))
    m = build_market([s], spec)
    assert any(o.contract.expiry == date(2026, 7, 2) for o in m.snapshots[0].options)


def test_the_iv_history_is_only_from_before_the_first_day(mkt):
    rows = mkt.iv_history["TSTA"]
    assert rows[0] == ("2026-11-02", pytest.approx(0.26))
    assert rows[-1][0] == "2026-11-10" and len(rows) == 7


def test_the_market_round_trips_through_the_store(mkt, tmp_path):
    write_market(mkt, tmp_path)
    back = read_market(tmp_path)
    assert back.iv_history == mkt.iv_history and back.days == mkt.days
    meta = json.loads((tmp_path / "market.json").read_text())
    del meta["iv_history"]                                # een markt van vóór deze versie
    (tmp_path / "market.json").write_text(json.dumps(meta))
    assert read_market(tmp_path).iv_history == {}


@pytest.mark.parametrize(("make", "match"), [
    (lambda: [], "geen onderliggende"),
    (lambda: [series()], "geen handelsdagen"),
    (lambda: [dataclasses.replace(series(), iv=series().iv * 0)], "positief"),
])
def test_a_bad_history_is_refused(make, match):
    spec = SPEC if match != "geen handelsdagen" else dataclasses.replace(
        SPEC, start=date(2026, 12, 1), end=date(2026, 12, 4))
    with pytest.raises(DataContractError, match=match):
        build_market(make(), spec)


def test_the_replay_starts_with_the_iv_history_of_the_market(tmp_path):
    spec = SyntheticSpec(underlyings=(SyntheticUnderlying("TSTB", 50.0, strike_step=1.0),),
                         start=date(2026, 12, 14), n_days=1, strike_range=0.08, max_days=30)
    m = dataclasses.replace(generate(spec), iv_history={"TSTB": (("2026-12-11", 0.31),)})
    asyncio.run(run_replay(CFG, m, tmp_path))
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["iv_history"]["TSTB"][0] == ["2026-12-11", 0.31]


# --- ophalen -------------------------------------------------------------------------------

def bar(d, v):
    return NS(date=d, open=v, high=v + 1, low=v - 1, close=v + 0.5, volume=10)


class HistIB:
    def __init__(self, missing=(), empty=()):
        self.missing, self.empty, self.asked = set(missing), set(empty), []

    async def qualifyContractsAsync(self, c):
        return [None if c.symbol in self.missing else c]

    async def reqHistoricalDataAsync(self, c, end, dur, size, what, rth):
        self.asked.append((c.symbol, end, dur, what, rth))
        if (c.symbol, what) in self.empty:
            return []
        return [bar(date(2026, 11, 2), 100.0), bar(date(2026, 11, 3), 101.0)]


def no_events(name, start, end):
    return (CashDividend(date(2026, 11, 20), 0.25),), (date(2026, 11, 19),)


def test_fetch_history_asks_ibkr_for_bars_and_iv():
    ib = HistIB()
    out = asyncio.run(fetch_history(ib, ["TSTB", "TSTA"], date(2026, 11, 2), date(2026, 11, 3),
                                    no_events))
    assert [s.name for s in out] == ["TSTA", "TSTB"]
    assert list(out[0].iv.columns) == ["open", "close"] and out[0].bars["close"].iloc[-1] == 101.5
    assert out[0].dividends[0].amount == 0.25 and out[0].earnings == (date(2026, 11, 19),)
    assert ib.asked[0] == ("TSTA", "20261103 23:59:59 US/Eastern", "2 Y", "TRADES", True)
    assert ib.asked[1][3] == "OPTION_IMPLIED_VOLATILITY"


@pytest.mark.parametrize(("ib", "match"), [
    (HistIB(missing={"TSTA"}), "onbekende"),
    (HistIB(empty={("TSTA", "OPTION_IMPLIED_VOLATILITY")}), "geen historische"),
])
def test_fetch_history_refuses_missing_data(ib, match):
    with pytest.raises(DataContractError, match=match):
        asyncio.run(fetch_history(ib, ["TSTA"], date(2026, 11, 2), date(2026, 11, 3), no_events))


class FakeTicker:
    def __init__(self, symbol, earnings=True):
        idx = pd.DatetimeIndex(["2026-05-10", "2026-08-10", "2026-11-10", "2027-06-01"],
                               tz="America/New_York")
        self.dividends = pd.Series([0.2, 0.0, 0.3, 0.4], index=idx)
        self._ed = (pd.DataFrame({"EPS": [1, 2]},
                                 index=pd.DatetimeIndex(["2026-10-30", "2025-01-30"],
                                                        tz="America/New_York"))
                    if earnings else None)

    def get_earnings_dates(self, limit):
        return self._ed


@pytest.mark.parametrize("earnings", [True, False])
def test_yfinance_history_keeps_the_period(monkeypatch, earnings):
    monkeypatch.setattr(hist.yf, "Ticker", lambda s: FakeTicker(s, earnings))
    divs, ed = yfinance_history("TSTA", date(2026, 6, 1), date(2026, 11, 25))
    assert [(d.ex_date, d.amount) for d in divs] == [(date(2026, 11, 10), 0.3)]
    assert ed == ((date(2026, 10, 30),) if earnings else ())


# --- opdrachtregel -------------------------------------------------------------------------

def test_the_historical_command_writes_a_market(monkeypatch, tmp_path, capsys):
    seen = {}

    async def fake_history(cfg, spec, env):
        seen["spec"] = spec
        return [series()]

    monkeypatch.setattr(cli, "_history", fake_history)
    monkeypatch.setattr(cli, "HistoricalSpec",
                        lambda s, e, half_days: dataclasses.replace(SPEC, start=s, end=e,
                                                                    half_days=half_days))
    assert cli.main(["historical", "--out", str(tmp_path), "--start", "2026-11-11",
                     "--end", "2026-11-11"]) == 0
    assert read_market(tmp_path).days == (date(2026, 11, 11),)
    assert seen["spec"].start == date(2026, 11, 11) and "geschreven" in capsys.readouterr().out


def test_history_connects_read_only_and_disconnects(monkeypatch):
    calls = []

    class FakeIB:
        async def connectAsync(self, host, port, client_id, readonly):
            calls.append(("connect", host, port, client_id, readonly))

        def disconnect(self):
            calls.append(("disconnect",))

    async def fake_fetch(ib, names, start, end):
        calls.append(("fetch", tuple(names)))
        return []

    monkeypatch.setattr(cli, "IB", FakeIB)
    monkeypatch.setattr(cli, "fetch_history", fake_fetch)
    out = asyncio.run(cli._history(CFG, SPEC, {"IB_HOST": "gw", "IB_PORT": "4004"}))
    assert out == [] and calls[0] == ("connect", "gw", 4004, 18, True)
    assert calls[1] == ("fetch", tuple(CFG.universe.underlyings)) and calls[2] == ("disconnect",)
