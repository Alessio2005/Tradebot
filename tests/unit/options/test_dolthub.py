"""De DoltHub-markt: ophalen (met nep-HTTP), cache, Yahoo-koersen, SEC-earnings en bouwen."""
from __future__ import annotations

import asyncio
import dataclasses
import json
from datetime import date, datetime, time

import pandas as pd
import pytest

from tests.unit.options.market_helpers import CFG
from tradebot.options import cli
from tradebot.options.calendar import NY
from tradebot.options.contracts import Right
from tradebot.options.data import dolthub as dh
from tradebot.options.data.dolthub import (
    DoltSpec,
    UnderlyingData,
    build_market,
    cache_days,
    download,
    fetch_day,
    make_market,
    read_cache,
    sec_earnings,
    yahoo_underlying,
)
from tradebot.options.data.store import read_market, write_market
from tradebot.options.engine.replay import replay_config, run_replay
from tradebot.options.engine.report import buy_and_hold
from tradebot.options.rates import CashDividend
from tradebot.utils.failfast import DataContractError

# Ma 2023-01-09 t/m vr 2023-01-20 (16 januari is MLK-dag, geen handelsdag).
DAYS = [d.date() for d in pd.bdate_range("2023-01-09", "2023-01-20") if d.day != 16]


def row(day, sym, expiry, strike, cp, bid, ask, vol=0.25):
    return {"date": day, "act_symbol": sym, "expiration": expiry, "strike": f"{strike:.2f}",
            "call_put": cp, "bid": bid, "ask": ask, "vol": vol, "delta": "0.5",
            "gamma": "0.01", "theta": "-0.01", "vega": "0.1", "rho": "0.01"}


def chain_rows(day, sym="SPY", spot=100.0, expiry="2023-01-27"):
    out = []
    for k in (spot - 5, spot, spot + 5):
        out += [row(day, sym, expiry, k, "Call", "1.00", "1.10"),
                row(day, sym, expiry, k, "Put", "0.90", "1.00")]
    return out


def bars(spot=100.0, days=DAYS):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    return pd.DataFrame({"open": spot, "high": spot + 1, "low": spot - 1, "close": spot,
                         "volume": 1e6}, index=idx)


# --- ophalen ---------------------------------------------------------------------------------

def test_fetch_day_builds_one_query_and_returns_the_rows():
    urls = []

    def get(url):
        urls.append(url)
        return {"query_execution_status": "Success", "rows": [{"a": 1}]}

    assert fetch_day(date(2023, 1, 9), ["SPY", "AAPL"], get) == [{"a": 1}]
    assert "date%3D%272023-01-09%27" in urls[0] and "%27AAPL%27" in urls[0]


def test_fetch_day_retries_on_network_and_query_errors_and_then_gives_up():
    answers = iter([OSError("weg"), {"query_execution_status": "Error",
                                     "query_execution_message": "deadline"},
                    {"query_execution_status": "Success"}])
    sleeps = []

    def get(url):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert fetch_day(date(2023, 1, 9), ["SPY"], get, sleep=sleeps.append, backoff=1.0) == []
    assert sleeps == [1.0, 2.0]

    def bad(url):
        return {"query_execution_status": "Error", "query_execution_message": "kapot"}

    with pytest.raises(DataContractError, match="kapot"):
        fetch_day(date(2023, 1, 9), ["SPY"], bad, retries=1, sleep=lambda s: None)


def test_a_symbol_that_is_not_alphanumeric_never_reaches_the_query():
    with pytest.raises(DataContractError, match="ongeldig symbool"):
        fetch_day(date(2023, 1, 9), ["SPY'; DROP"], lambda u: {})


def test_only_https_urls_are_opened(monkeypatch):
    with pytest.raises(DataContractError, match="alleen https"):
        dh._https_json("file:///etc/passwd", "x", 1)
    seen = []
    monkeypatch.setattr(dh, "_open_json", lambda req, t: seen.append((req, t)) or {"ok": 1})
    assert dh._https_json("https://example.invalid/x", "agent", 7) == {"ok": 1}
    assert seen[0][0].get_header("User-agent") == "agent" and seen[0][1] == 7


def test_download_writes_rows_and_empty_markers_and_resumes(tmp_path):
    calls = []

    def get(url):
        calls.append(url)
        if "2023-01-09" in url:
            return {"query_execution_status": "Success",
                    "rows": chain_rows("2023-01-09")}
        return {"query_execution_status": "Success", "rows": []}

    logs = []
    got = download(tmp_path, ["SPY"], date(2023, 1, 7), date(2023, 1, 10), get, workers=1,
                   log=logs.append)
    assert got == [date(2023, 1, 9)]
    assert (tmp_path / "2023-01-10.empty").exists() and len(calls) == 2   # geen weekend
    assert logs == ["2023-01-09: 6 rijen"]
    assert download(tmp_path, ["SPY"], date(2023, 1, 7), date(2023, 1, 10), get) == got
    assert len(calls) == 2                                              # niets opnieuw


def test_candidate_days_include_the_saturdays_of_2019_only():
    assert dh._days(date(2019, 2, 1), date(2019, 2, 11)) == [date(2019, 2, 9),
                                                             date(2019, 2, 11)]
    assert date(2020, 2, 8) not in dh._days(date(2020, 2, 7), date(2020, 2, 10))


def test_read_cache_needs_at_least_one_day(tmp_path):
    with pytest.raises(DataContractError, match="geen DoltHub-ketens"):
        read_cache(tmp_path, date(2023, 1, 1), date(2023, 2, 1))
    (tmp_path / "2023-01-09.csv").write_text(
        pd.DataFrame(chain_rows("2023-01-09")).to_csv(index=False))
    assert cache_days(tmp_path, date(2023, 1, 10), date(2023, 2, 1)) == []
    df = read_cache(tmp_path, date(2023, 1, 1), date(2023, 2, 1))
    assert list(df.columns) == list(dh.COLUMNS[:8]) and len(df) == 6


# --- Yahoo en SEC ----------------------------------------------------------------------------

def ts(d):
    return int(datetime.combine(d, time(9, 30), tzinfo=NY).timestamp())


def test_yahoo_bars_and_dividends_are_turned_back_before_a_split():
    split = date(2023, 1, 11)
    res = {"chart": {"result": [{
        "timestamp": [ts(date(2023, 1, 9)), ts(date(2023, 1, 10)), ts(date(2023, 1, 11)),
                      ts(date(2023, 1, 11)), ts(date(2023, 1, 12)), ts(date(2023, 1, 13))],
        "events": {"splits": {"x": {"date": ts(split), "numerator": 4.0, "denominator": 1.0}},
                   "dividends": {"a": {"date": ts(date(2023, 1, 10)), "amount": 0.25},
                                 "b": {"date": ts(date(2023, 1, 12)), "amount": 0.0},
                                 "c": {"date": ts(date(2024, 1, 12)), "amount": 0.3}}},
        "indicators": {"quote": [{"open": [25.0, 25.0, 99.0, 100.0, 100.0, 101.0],
                                  "high": [26.0, 26.0, 99.0, 101.0, 101.0, 102.0],
                                  "low": [24.0, 24.0, 99.0, 99.0, 99.0, 100.0],
                                  "close": [25.0, None, 99.0, 100.0, 100.0, 101.0],
                                  "volume": [400.0, 400.0, 1.0, 100.0, None, 100.0]}]}}]}}
    urls = []

    def get(url):
        urls.append(url)
        return res

    u = yahoo_underlying("AAPL", date(2023, 1, 9), date(2023, 1, 12), get,
                         earnings=lambda s, a, b: (date(2023, 1, 12),), today=date(2023, 6, 1))
    assert list(u.bars.index) == [pd.Timestamp("2023-01-09"), pd.Timestamp("2023-01-11"),
                                  pd.Timestamp("2023-01-12")]   # geen None, dubbel: laatste
    assert u.bars.loc["2023-01-09", "close"] == 100.0 and u.bars.loc["2023-01-09", "volume"] == 100
    assert u.bars.loc["2023-01-12", "volume"] == 0.0
    assert u.dividends == (CashDividend(date(2023, 1, 10), 1.0),)
    assert u.earnings == (date(2023, 1, 12),)
    assert "period2=" in urls[0] and "AAPL" in urls[0]


def test_yahoo_without_prices_is_a_data_contract_error():
    with pytest.raises(DataContractError, match="geen koersen van Yahoo"):
        yahoo_underlying("SPY", date(2023, 1, 9), date(2023, 1, 12),
                         lambda u: {"chart": {"result": None}}, earnings=lambda *a: ())
    with pytest.raises(DataContractError, match="ongeldig symbool"):
        yahoo_underlying("SP Y", date(2023, 1, 9), date(2023, 1, 12), lambda u: {})


def test_sec_earnings_reads_8k_item_202_from_recent_and_older_files():
    recent = {"form": ["8-K", "8-K", "10-Q", "8-K"],
              "filingDate": ["2023-04-27", "2023-03-01", "2023-04-28", "2021-01-01"],
              "items": ["2.02,9.01", "5.02", "", "2.02"]}
    old = {"form": ["8-K"], "filingDate": ["2023-01-26"], "items": ["2.02"]}
    pages = {"CIK0000789019.json": {"filings": {"recent": recent, "files": [
        {"name": "old.json", "filingTo": "2023-02-01"},
        {"name": "ancient.json", "filingTo": "2010-01-01"}]}}, "old.json": old}
    asked = []

    def get(url):
        name = url.rsplit("/", 1)[1]
        asked.append(name)
        return pages[name]

    got = sec_earnings("MSFT", date(2023, 1, 1), date(2023, 3, 31), get)
    assert got == (date(2023, 1, 26), date(2023, 4, 27))
    assert "ancient.json" not in asked
    assert sec_earnings("SPY", date(2023, 1, 1), date(2023, 3, 31), get) == ()
    logs = []
    assert sec_earnings("ZZZZ", date(2023, 1, 1), date(2023, 3, 31), get, log=logs.append) == ()
    assert "geen CIK" in logs[0]


# --- markt bouwen ----------------------------------------------------------------------------

def frame(rows):
    return pd.DataFrame(rows)[list(dh.COLUMNS[:8])]


SPEC = DoltSpec(date(2023, 1, 9), date(2023, 1, 20), rate=0.04)


def test_build_market_one_snapshot_per_trading_day_with_saturday_and_gaps():
    rows = [*chain_rows("2023-01-09"), *chain_rows("2023-01-14"),     # zaterdag → vrijdag
            *chain_rows("2023-01-13", spot=50.0),                    # zelfde dag: zaterdag wint
            row("2023-01-09", "SPY", "2023-01-06", 100, "Call", "1", "2"),   # al verlopen
            row("2023-01-09", "SPY", "2023-02-17", 100, "Call", "2.00", "1.00"),  # gekruist
            row("2023-01-09", "SPY", "2023-02-17", 100, "Put", None, "1.00", vol=None),
            *chain_rows("2023-01-09", "QQQ")]
    warn = []
    m = build_market(frame(rows), [UnderlyingData("SPY", bars()),
                                   UnderlyingData("IWM", bars())], SPEC, warn=warn.append)
    assert warn == ["IWM: geen DoltHub-ketens, valt uit de markt"]
    assert m.days == tuple(DAYS) and sorted(m.bars) == ["SPY"] and m.daily_snapshot == "15:15"
    by_day = {s.asof.date(): s for s in m.snapshots}
    assert len(m.snapshots) == len(DAYS)
    assert all(s.asof.time() == time(15, 15) for s in m.snapshots)
    first = by_day[date(2023, 1, 9)]
    crossed = [o for o in first.options if o.contract.expiry == date(2023, 2, 17)]
    assert {o.contract.right for o in crossed} == {Right.CALL, Right.PUT}
    assert all(o.quote.bid is None for o in crossed)
    assert all(o.contract.expiry >= date(2023, 1, 9) for o in first.options)
    real = [o for o in first.options if o.contract.expiry == date(2023, 1, 27)]
    assert len(real) == 6 and all(o.quote.open_interest == 5000 for o in real)
    # Dinsdag: geen keten, wel de contracten van maandag, aangevuld uit QuantLib.
    tue = by_day[date(2023, 1, 10)]
    assert len(tue.options) == 6 and all(o.quote.open_interest == 0 for o in tue.options)
    assert all(o.quote.ask > o.quote.bid >= 0 for o in tue.options)
    # Vrijdag 13: de zaterdagketen (spot 100) wint van de vrijdagketen (spot 50).
    fri = by_day[date(2023, 1, 13)]
    assert sorted({o.contract.strike for o in fri.options
                   if o.quote.open_interest}) == [95.0, 100.0, 105.0]


def test_fill_stops_after_fill_days_and_outside_the_moneyness_band(tmp_path):
    rows = chain_rows("2023-01-09")
    m = build_market(frame(rows), [UnderlyingData("SPY", bars())],
                     dataclasses.replace(SPEC, fill_days=2))
    counts = [len(s.options) for s in m.snapshots]
    assert counts[:3] == [6, 6, 6] and counts[3:] == [0] * (len(DAYS) - 3)
    narrow = build_market(frame(rows), [UnderlyingData("SPY", bars())],
                          dataclasses.replace(SPEC, fill_moneyness=0.01))
    assert {o.contract.strike for o in narrow.snapshots[1].options} == {100.0}
    write_market(m, tmp_path)
    assert len(read_market(tmp_path).snapshots) == len(DAYS)


def test_a_fill_with_a_dividend_in_its_life_prices_with_fd():
    rows = chain_rows("2023-01-09")
    div = (CashDividend(date(2023, 1, 18), 1.0),)
    m = build_market(frame(rows), [UnderlyingData("SPY", bars(), div)], SPEC)
    plain = build_market(frame(rows), [UnderlyingData("SPY", bars())], SPEC)
    put = next(o for o in m.snapshots[1].options
               if o.contract.right is Right.PUT and o.contract.strike == 100.0)
    put0 = next(o for o in plain.snapshots[1].options
                if o.contract.right is Right.PUT and o.contract.strike == 100.0)
    assert put.quote.bid > put0.quote.bid                    # dividend maakt de put duurder
    assert m.events["SPY"].dividends == div


def test_build_market_needs_a_name_with_chains_and_trading_days():
    with pytest.raises(DataContractError, match="geen onderliggende"):
        build_market(frame(chain_rows("2023-01-09", "QQQ")), [UnderlyingData("SPY", bars())],
                     SPEC, warn=lambda s: None)
    with pytest.raises(DataContractError, match="geen handelsdagen"):
        build_market(frame(chain_rows("2023-01-09")),
                     [UnderlyingData("SPY", bars(days=[date(2022, 12, 1)]))], SPEC)


def test_make_market_fetches_reads_and_drops_names_without_chains(tmp_path):
    fetched = []

    def get(url):
        fetched.append(url)
        hit = "2023-01-09" in url
        return {"query_execution_status": "Success",
                "rows": chain_rows("2023-01-09") if hit else []}

    seen = []

    def und(sym, start, end):
        seen.append((sym, start))
        return UnderlyingData(sym, bars())

    logs = []
    m = make_market(tmp_path, SPEC, ("SPY", "QQQ"), get=get, underlying=und,
                    log=logs.append)
    assert sorted(m.bars) == ["SPY"] and seen == [("SPY", date(2022, 8, 12))]
    assert "QQQ: geen DoltHub-ketens, valt uit de markt" in logs and fetched
    n = len(fetched)
    make_market(tmp_path, SPEC, ("SPY",), fetch=False, underlying=und, log=logs.append)
    assert len(fetched) == n


# --- replay en de opdrachtregel ----------------------------------------------------------------

def test_a_daily_snapshot_market_moves_the_scan_and_expiry_close_to_that_time():
    m = build_market(frame(chain_rows("2023-01-09")), [UnderlyingData("SPY", bars())], SPEC)
    cfg = replay_config(CFG, m)
    assert cfg.schedule.scan_times == ("15:15",)
    assert cfg.schedule.close_before_expiry_time == "15:15"
    plain = replay_config(CFG, dataclasses.replace(m, daily_snapshot=None))
    assert plain.schedule == CFG.schedule


def test_buy_and_hold_counts_dividends_as_cash():
    days = DAYS[:3]
    b = bars(days=days)
    b["close"] = [100.0, 110.0, 99.0]
    res = buy_and_hold(b, (CashDividend(days[1], 1.0),), days, 1000.0, "SPY")
    assert res["symbol"] == "SPY" and res["end_equity"] == pytest.approx(1000.0)
    assert res["total_return"] == pytest.approx(0.0) and res["max_drawdown"] > 0.0


def test_replay_on_a_dolthub_market_reports_buy_and_hold(tmp_path):
    m = build_market(frame(chain_rows("2023-01-09")), [UnderlyingData("SPY", bars())],
                     dataclasses.replace(SPEC, end=date(2023, 1, 11)))
    out = asyncio.run(run_replay(CFG, m, tmp_path))
    rec = json.loads((tmp_path / "summary.json").read_text())
    assert rec["buy_hold"]["symbol"] == "SPY" and out.buy_hold["end_equity"] == 25000.0
    assert rec["buy_hold"]["total_return"] == 0.0


def test_the_dolthub_command_builds_and_writes_a_market(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_make(cache, spec, symbols, fetch):
        seen.update(cache=cache, spec=spec, symbols=symbols, fetch=fetch)
        return build_market(frame(chain_rows("2023-01-09")),
                            [UnderlyingData("SPY", bars())], SPEC)

    monkeypatch.setattr(cli, "make_market", fake_make)
    out = tmp_path / "m"
    assert cli.main(["dolthub", "--cache", str(tmp_path / "c"), "--out", str(out),
                     "--start", "2023-01-09", "--end", "2023-01-20", "--symbols", "SPY",
                     "--offline", "--rate", "0.05"]) == 0
    assert seen["fetch"] is False and seen["symbols"] == ("SPY",)
    assert seen["spec"].rate == 0.05 and seen["spec"].start == date(2023, 1, 9)
    assert read_market(out).daily_snapshot == "15:15" and "SPY" in capsys.readouterr().out
