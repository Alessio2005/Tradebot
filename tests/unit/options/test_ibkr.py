"""De IBKR-adapter, de IBKR-feed en de paper-dienst tegen een nep-IB (geen netwerk)."""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace as NS

import pytest

from tests.unit.options.fake_ib import FakeIB
from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import CFG, EXPIRIES, market
from tests.unit.options.test_execution import approved, order_for
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.data.ibkr_feed import (
    BATCH,
    IBKRFeed,
    _int,
    _px,
    project_dividends,
    yfinance_events,
)
from tradebot.options.engine import paper as paper_mod
from tradebot.options.engine.paper import now_ny, run_paper
from tradebot.options.events import Events
from tradebot.options.execution.ibkr import IBKRBroker, contract_of, option_of
from tradebot.options.execution.orders import OrderStatus
from tradebot.options.live import LiveRefused, account_hash
from tradebot.options.positions import Book, Trend
from tradebot.options.rates import CashDividend
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.credit import CreditSpread
from tradebot.utils.failfast import DataContractError

C = OptionContract("TST", date(2026, 12, 31), 95.0, Right.PUT)


def run(coro):
    return asyncio.run(coro)


def broker(ib=None, account=""):
    b = IBKRBroker(ib or FakeIB(), account=account)
    run(b.connect())
    return b


# --- verbinding en account -----------------------------------------------------------------

def test_connect_picks_the_paper_account_and_asks_for_delayed_data():
    ib = FakeIB()
    b = broker(ib)
    assert b.account_id == "DU1234567" and ib.data_type == 3 and ib.args[1] == 4002
    assert b.account_hash == account_hash("DU1234567") and "DU" not in b.account_hash
    run(b.connect())                                    # al verbonden: niet opnieuw
    acct = run(b.account())
    assert (acct.equity, acct.cash, acct.buying_power, acct.is_paper) == (25000, 24000, 50000,
                                                                           True)


@pytest.mark.parametrize(("accounts", "account", "err", "match"), [
    (["U7654321"], "", LiveRefused, "geen IBKR-paper-account"),
    ([], "", DataContractError, "geen account"),
    (["DU1", "DU2"], "", DataContractError, "meerdere accounts"),
    (["DU1"], "DU9", DataContractError, "hoort niet bij"),
])
def test_connect_refuses_anything_but_one_paper_account(accounts, account, err, match):
    with pytest.raises(err, match=match):
        broker(FakeIB(accounts=accounts), account)


def test_the_account_needs_equity_and_cash():
    ib = FakeIB()
    b = broker(ib)
    ib.summary = [NS(tag="NetLiquidation", value="", currency="USD"),
                  NS(tag="BuyingPower", value="inf", currency="USD")]
    with pytest.raises(DataContractError, match="equity"):
        run(b.account())
    ib.summary = [NS(tag="NetLiquidation", value="1", currency="BASE"),
                  NS(tag="TotalCashValue", value="2", currency="")]
    assert run(b.account()).buying_power == 0.0


def test_an_account_in_another_base_currency_is_converted_to_usd():
    ib = FakeIB()
    b = broker(ib)
    ib.summary = [NS(tag="NetLiquidation", value="1000", currency="EUR"),
                  NS(tag="TotalCashValue", value="900", currency="EUR"),
                  NS(tag="BuyingPower", value="4000", currency="EUR"),
                  NS(tag="Cushion", value="0.9", currency="")]
    acct = run(b.account())
    assert ib.fx_asked == ["EURUSD"]
    assert (acct.equity, acct.cash, acct.buying_power) == pytest.approx((1100, 990, 4400))
    ib.fx = (float("nan"), 1.05)                        # geen actuele koers: de slotkoers
    assert run(b.account()).equity == pytest.approx(1050)


@pytest.mark.parametrize("case", ["two_currencies", "no_rows", "no_rate", "no_pair"])
def test_a_foreign_account_without_a_usable_rate_is_refused(case):
    ib = FakeIB()
    b = broker(ib)
    ib.summary = [NS(tag="NetLiquidation", value="1000", currency="EUR"),
                  NS(tag="TotalCashValue", value="900", currency="EUR")]
    match = "wisselkoers"
    if case == "two_currencies":
        ib.summary.append(NS(tag="TotalCashValue", value="5", currency="GBP"))
        match = "equity"
    elif case == "no_rows":
        ib.summary, match = [], "equity"
    elif case == "no_rate":
        ib.fx = (float("nan"), 0.0)
    else:
        ib.unknown.add(0.0)
    with pytest.raises(DataContractError, match=match):
        run(b.account())


def test_positions_are_mapped():
    ib = FakeIB()
    b = broker(ib)
    opt = NS(secType="OPT", symbol="TST", lastTradeDateOrContractMonth="20261231", strike=95.0,
             right="P")
    ib.held = [NS(contract=NS(secType="STK", symbol="TST"), position=100.0, avgCost=99.5),
               NS(contract=opt, position=-2.0, avgCost=125.0),
               NS(contract=NS(secType="CASH", symbol="EUR"), position=1.0, avgCost=1.0)]
    pos = run(b.positions())
    assert [(p.contract, p.qty) for p in pos] == [(None, 100), (C, -2)]
    assert pos[1].avg_price == 1.25
    assert contract_of(option_of(C)) == C
    call = dataclasses.replace(C, right=Right.CALL)
    assert option_of(call).right == "C" and contract_of(option_of(call)) == call


# --- orders --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ap():
    (c,) = CreditSpread(CFG.strategies.credit_spread, rules_from_config(CFG)).scan(
        market(trend=Trend.UP), Book())
    return approved(c)


def test_an_order_is_a_bag_with_the_client_id(ap):
    ib = FakeIB()
    b = broker(ib)
    with pytest.raises(TypeError, match="ApprovedOrder"):
        run(b.submit(ap.proposal, order_for(ap, -0.5)))    # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="wijkt af"):
        run(b.submit(ap, dataclasses.replace(order_for(ap, -0.5), quantity=99)))
    bo = run(b.submit(ap, order_for(ap, -0.504)))
    assert bo.status is OrderStatus.SUBMITTED
    bag, order = ib.placed[0]
    assert bag.secType == "BAG" and len(bag.comboLegs) == 2
    assert {leg.action for leg in bag.comboLegs} == {"BUY", "SELL"}
    assert (order.action, order.lmtPrice, order.orderRef, order.account) == (
        "BUY", -0.5, "c1", "DU1234567")
    with pytest.raises(DataContractError, match="al gebruikt"):
        run(b.submit(ap, order_for(ap, -0.5)))
    assert [o.order.client_id for o in run(b.open_orders())] == ["c1"]
    bo = run(b.modify("c1", -0.48))
    assert bo.order.limit == -0.48 and ib.placed[-1][1].lmtPrice == -0.48
    run(b.cancel("c1"))
    assert ib.cancelled == [order]
    b._trades["c1"].orderStatus = NS(status="Filled", filled=bo.order.quantity,
                                     avgFillPrice=-0.48)
    assert run(b.open_orders()) == []
    with pytest.raises(DataContractError, match="open order"):
        run(b.modify("c1", -0.47))
    run(b.cancel("c1"))
    assert len(ib.cancelled) == 1 and b._refresh("c1").avg_fill == -0.48


def test_an_unknown_leg_is_refused(ap):
    ib = FakeIB()
    ib.unknown = {ap.proposal.legs[0].contract.strike}
    b = broker(ib)
    with pytest.raises(DataContractError, match="kent een leg niet"):
        run(b.submit(ap, order_for(ap, -0.5)))


def test_what_if_and_executions(ap):
    ib = FakeIB()
    b = broker(ib)
    m = run(b.what_if(order_for(ap, -0.5)))
    assert (m.initial_change, m.maintenance_change) == (500.0, 400.0)
    ib.what_if = NS(initMarginChange="1.7976931348623157E308", maintMarginChange="0")
    assert run(b.what_if(order_for(ap, -0.5))) is None
    opt = NS(secType="OPT", symbol="TST", lastTradeDateOrContractMonth="20261231", strike=95.0,
             right="P")
    t = datetime(2026, 11, 24, 15, 0, tzinfo=NY)
    ib.fills = [NS(contract=opt, execution=NS(shares=2, side="SLD", price=1.2, orderRef="c1"),
                   commissionReport=NS(commission=1.3), time=t),
                NS(contract=opt, execution=NS(shares=1, side="BOT", price=1.0, orderRef=""),
                   commissionReport=NS(commission=float("nan")), time=t),
                NS(contract=NS(secType="BAG", symbol="TST"), execution=NS(), time=t,
                   commissionReport=None)]
    ex = run(b.executions(t - timedelta(hours=1)))
    assert [(e.client_id, e.qty, e.commission) for e in ex] == [("", 1, 0.0), ("c1", -2, 1.3)]
    assert ib.filter.time == "20261124-19:00:00" and ib.filter.acctCode == "DU1234567"
    run(b.executions(datetime.min.replace(tzinfo=NY)))       # eerste ronde: geen tijdfilter
    assert ib.filter.time == ""


# --- feed ----------------------------------------------------------------------------------

def _events(name, today):
    return Events(name, today)


def test_the_feed_builds_snapshots_in_batches():
    ib = FakeIB()
    f = IBKRFeed(ib, ("TST",), 0.04, strike_range=0.05, events_source=_events, wait_seconds=0)
    assert f.snapshot("TST", NOW) is None and f.events("TST") is None
    assert f.bars_before("TST", NOW.date()).empty
    run(f.refresh(NOW))
    s = f.snapshot("TST", NOW)
    assert s is not None and s.spot == 100.0 and f.underlyings() == ("TST",)
    assert {r.contract.expiry for r in s.options} == set(EXPIRIES)
    assert {r.contract.strike for r in s.options} == {float(k) for k in range(95, 106)}
    assert all(r.quote.delayed and r.quote.open_interest in (5000, 6000) for r in s.options)
    assert ib.max_lines <= BATCH and ib.lines == 0
    assert f.snapshot("TST", NOW - timedelta(seconds=1)) is None
    assert f.snapshot("TST", NOW + timedelta(days=1)) is None
    bars = f.bars_before("TST", date(2026, 6, 1))
    assert len(bars) == (date(2026, 6, 1) - date(2026, 1, 2)).days
    assert f.events("TST") == Events("TST", NOW.date())


def test_prices_from_ibkr_are_cleaned():
    assert _px(None) is None and _px("x") is None and _px(float("nan")) is None
    assert _px(0.0) is None and _px("1.5") == 1.5 and _int(-1) == 0 and _int(7.0) == 7


def test_the_feed_skips_names_without_a_price_or_chain():
    ib = FakeIB()
    f = IBKRFeed(ib, ("TST",), 0.04, events_source=_events, wait_seconds=0)
    ib.spot = float("nan")
    run(f.refresh(NOW))
    assert f.snapshot("TST", NOW) is None

    async def none(*a):
        return []
    ib2 = FakeIB()
    ib2.reqSecDefOptParamsAsync = none
    ib2.reqHistoricalDataAsync = none
    f2 = IBKRFeed(ib2, ("TST",), 0.04, events_source=_events, wait_seconds=0)
    run(f2.refresh(NOW))
    assert f2.snapshot("TST", NOW) is None and f2.bars_before("TST", NOW.date()).empty


def test_yfinance_events_are_read(monkeypatch):
    import pandas as pd
    today = date(2026, 11, 24)

    class T:
        def __init__(self, sym):
            self.calendar = {"Earnings Date": [date(2027, 1, 28), "?"],
                             "Ex-Dividend Date": date(2026, 12, 5)}
            self.dividends = pd.Series([0.5, 0.6])

    monkeypatch.setattr("tradebot.options.data.ibkr_feed.yf.Ticker", T)
    ev = yfinance_events("AAPL", today)
    assert ev.earnings == (date(2027, 1, 28),) and ev.dividends[0].amount == 0.6

    class Empty:
        def __init__(self, sym):
            self.calendar, self.dividends = None, pd.Series([], dtype=float)

    monkeypatch.setattr("tradebot.options.data.ibkr_feed.yf.Ticker", Empty)
    assert yfinance_events("SPY", today) == Events("SPY", today)


def test_the_dividend_schedule_is_projected_on_the_historic_cadence():
    # N5: alleen het eerstvolgende dividend meenemen liet de latere uit de prijs.
    today = date(2026, 11, 24)
    hist = [(date(2026, 3, 20), 1.50), (date(2026, 6, 19), 1.55), (date(2026, 9, 18), 1.60)]
    out = project_dividends(date(2026, 12, 18), hist, today)
    assert out[0] == CashDividend(date(2026, 12, 18), 1.60) and not out[0].estimated
    assert [d.ex_date for d in out[1:]] == [date(2026, 12, 18) + timedelta(days=91 * i)
                                            for i in (1, 2, 3)]
    assert all(d.estimated and d.amount == 1.60 for d in out[1:])
    assert out[-1].ex_date <= today + timedelta(days=365)
    # Geen aankondiging: rol vanaf de laatste ex-datum door tot de toekomst.
    rolled = project_dividends(None, hist, today)
    assert rolled[0].ex_date == date(2026, 9, 18) + timedelta(days=91) and rolled[0].estimated
    # Maandelijks (mediaan 30 dagen) en zonder historie.
    monthly = [(date(2026, 8, 31), 0.2), (date(2026, 9, 30), 0.2), (date(2026, 10, 30), 0.2)]
    assert project_dividends(None, monthly, today)[0].ex_date == date(2026, 11, 29)
    assert project_dividends(date(2026, 12, 1), [], today) == ()
    stale = project_dividends(None, [(date(2026, 6, 19), 1.5)], today)   # twee kwartalen over
    assert stale[0].ex_date == date(2026, 6, 19) + timedelta(days=182)
    # Een enkel historisch dividend: standaardcadans (kwartaal).
    one = project_dividends(None, [(date(2026, 11, 1), 0.3)], today)
    assert one[0].ex_date == date(2026, 11, 1) + timedelta(days=91)


def test_yfinance_events_project_the_dividend_schedule(monkeypatch):
    import pandas as pd
    today = date(2026, 11, 24)
    idx = pd.DatetimeIndex(["2026-03-20", "2026-06-19", "2026-09-18"], tz="America/New_York")

    class T:
        def __init__(self, sym):
            self.calendar = {"Earnings Date": [], "Ex-Dividend Date": date(2026, 12, 18)}
            self.dividends = pd.Series([1.5, 1.55, 1.6], index=idx)

    monkeypatch.setattr("tradebot.options.data.ibkr_feed.yf.Ticker", T)
    divs = yfinance_events("SPY", today).dividends
    assert len(divs) == 4 and divs[0].ex_date == date(2026, 12, 18) and divs[1].estimated


# --- de paper-dienst -----------------------------------------------------------------------

def test_the_paper_service_runs_until_stopped(tmp_path, monkeypatch):
    ib = FakeIB()
    uni = CFG.universe.model_copy(update={"underlyings": ("TST",), "share_eligible": ()})
    cfg = CFG.model_copy(update={"universe": uni})
    monkeypatch.setattr(paper_mod, "IBKRFeed",
                        lambda ib, names, rate: IBKRFeed(ib, names, rate, events_source=_events,
                                                         wait_seconds=0))
    t = [datetime.combine(date(2026, 11, 24), time(9, 59), tzinfo=NY)]
    stop = asyncio.Event()

    async def sleep(s):
        t[0] += timedelta(seconds=s)
        if t[0] >= datetime.combine(date(2026, 11, 24), time(10, 1), tzinfo=NY):
            stop.set()

    eng = run(run_paper(cfg, ib, tmp_path, stop, clock=lambda: t[0], sleep=sleep,
                        refresh_seconds=60))
    assert eng.state.mode == "paper" and eng.state.account_hash == account_hash("DU1234567")
    events = [r["event"] for r in eng.journal.read()]
    assert "chain" in events and events[-1] == "stopped"
    assert (tmp_path / "state.json").is_file()
    # Een herstart leest dezelfde staat.
    stop2 = asyncio.Event()
    stop2.set()
    again = run(run_paper(cfg, FakeIB(), tmp_path, stop2, clock=lambda: t[0], sleep=sleep))
    assert again.state.created_at == eng.state.created_at
    assert now_ny().tzinfo is NY


# --- de CLI --------------------------------------------------------------------------------

def test_the_cli_paper_command_and_healthcheck(tmp_path, monkeypatch):
    from tradebot.options import cli
    assert cli.main(["paper", "--run-dir", str(tmp_path), "--healthcheck"]) == 1
    now = datetime.now(NY)
    (tmp_path / "heartbeat").write_text(now.isoformat(), encoding="utf-8")
    assert cli.main(["paper", "--run-dir", str(tmp_path), "--healthcheck"]) == 0
    (tmp_path / "heartbeat").write_text((now - timedelta(hours=1)).isoformat(),
                                        encoding="utf-8")
    assert not cli.healthy(tmp_path, now)
    seen = {}

    async def fake_run(cfg, ib, run_dir, stop, account, host, port, client_id):
        seen.update(account=account, host=host, port=port, client_id=client_id,
                    run_dir=run_dir)

    monkeypatch.setattr(cli, "run_paper", fake_run)
    monkeypatch.setenv("IB_PORT", "4004")
    monkeypatch.setenv("IB_ACCOUNT", "DU1")
    assert cli.main(["paper", "--run-dir", str(tmp_path)]) == 0
    assert seen == {"account": "DU1", "host": "127.0.0.1", "port": 4004, "client_id": 17,
                    "run_dir": tmp_path}


# --- verbinden met een gateway die nog opstart -------------------------------------------

def test_connect_waits_for_a_gateway_that_is_still_starting():
    ib, slept = FakeIB(), []
    ib.fail_connects = 2

    async def sleep(s):
        slept.append(s)

    b = IBKRBroker(ib, connect_attempts=3, connect_wait=7.0, sleep=sleep)
    run(b.connect())
    assert ib.connected and slept == [7.0, 7.0] and ib.disconnects == 2


def test_connect_gives_up_after_the_last_attempt():
    ib = FakeIB()
    ib.fail_connects = 5

    async def sleep(s):
        return None

    with pytest.raises(TimeoutError):
        run(IBKRBroker(ib, connect_attempts=2, sleep=sleep).connect())
    assert ib.disconnects == 1
