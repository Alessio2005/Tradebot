"""Staat (atomisch, met mode en account), journal en de replay-feed."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, time

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.synthetic_market import SyntheticSpec, SyntheticUnderlying, generate
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.data.feed import ReplayFeed
from tradebot.options.engine.state import (
    EngineState,
    Journal,
    contract_from,
    contract_record,
    intents_from,
    intents_record,
    load_state,
    save_state,
)
from tradebot.options.positions import Leg, LegIntent, PositionStatus, StrategyPosition
from tradebot.utils.failfast import DataContractError

C = OptionContract("TST", date(2026, 12, 31), 95.0, Right.PUT)


def _state():
    pos = StrategyPosition("p1", "credit_spread", "TST",
                           (Leg(C, "TST", -2, 1.25), Leg(None, "TST", 100, 98.0)), NOW, 2, 50.0,
                           math.inf, phase="csp", variant="bull_put")
    closed = StrategyPosition("p0", "debit_vertical", "TST", (), NOW, 1, -30.0, 30.0,
                              status=PositionStatus.CLOSED, hedge_realized_usd=12.5)
    s = EngineState("replay", "paper", NOW.isoformat(), [pos, closed])
    s.orders["c1"] = {"client_id": "c1", "legs": intents_record((LegIntent(C, -1),))}
    s.attempts["k"] = 2
    s.iv_history["TST"] = [("2026-11-23", 0.22)]
    s.equity.append(("2026-11-23", 25_010.0))
    s.high_water_mark = 25_010.0
    s.day_trades.append("2026-11-23")
    s.wheel["TST"] = "csp"
    s.cost_basis["TST"] = 94.0
    s.new_positions["2026-11-23"] = 1
    return s


def test_the_state_round_trips_atomically(tmp_path):
    s = _state()
    path = tmp_path / "run" / "state.json"
    save_state(s, path)
    assert not path.with_name("state.json.tmp").exists()
    back = load_state(path, "replay", "paper")
    assert back == s
    assert [p.position_id for p in back.open_positions()] == ["p1"]
    assert back.day_trade_dates() == (date(2026, 11, 23),)
    assert intents_from(back.orders["c1"]["legs"]) == (LegIntent(C, -1),)
    assert contract_from(contract_record(C)) == C


def test_an_older_state_without_hedge_pnl_loads_as_zero(tmp_path):
    path = tmp_path / "state.json"
    save_state(_state(), path)
    raw = json.loads(path.read_text())
    for p in raw["positions"]:
        del p["hedge_realized_usd"]
    path.write_text(json.dumps(raw))
    back = load_state(path, "replay", "paper")
    assert [p.hedge_realized_usd for p in back.positions] == [0.0, 0.0]


def test_a_foreign_state_is_refused(tmp_path):
    path = tmp_path / "state.json"
    assert load_state(path, "replay", "paper") is None
    save_state(_state(), path)
    with pytest.raises(DataContractError, match="andere mode"):
        load_state(path, "paper", "paper")
    with pytest.raises(DataContractError, match="ander account"):
        load_state(path, "replay", "U123")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 99
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataContractError, match="schema-versie"):
        load_state(path, "replay", "paper")


def test_the_journal_appends_lines(tmp_path):
    j = Journal(tmp_path / "j" / "journal.jsonl")
    assert j.read() == []
    j.write(NOW, "a", x=1)
    j.write(NOW, "b", when=date(2026, 1, 2))
    rows = j.read()
    assert [r["event"] for r in rows] == ["a", "b"] and rows[1]["when"] == "2026-01-02"


SPEC = SyntheticSpec(underlyings=(SyntheticUnderlying("TSTB", 50.0, strike_step=1.0,
                                                      dividend_day=1, dividend=0.2),),
                     start=date(2026, 12, 1), n_days=2,
                     snapshot_times=(time(10, 0), time(15, 0)), strike_range=0.06,
                     max_days=30, warmup_days=25)


def test_the_replay_feed_is_causal():
    m = generate(SPEC)
    f = ReplayFeed(m)
    assert f.underlyings() == ("TSTB",) and f.rate == m.rate
    assert len({s.asof for s in m.snapshots}) == 4
    d1 = date(2026, 12, 1)
    assert f.snapshot("TSTB", datetime.combine(d1, time(9, 59), tzinfo=NY)) is None
    s = f.snapshot("TSTB", datetime.combine(d1, time(14, 0), tzinfo=NY))
    assert s is not None and s.asof.hour == 10
    # De laatste snapshot van gisteren telt vandaag niet.
    assert f.snapshot("TSTB", datetime.combine(date(2026, 12, 2), time(9, 31),
                                               tzinfo=NY)) is None
    assert f.snapshot("XYZ", datetime.combine(d1, time(14, 0), tzinfo=NY)) is None
    bars = f.bars_before("TSTB", d1)
    assert len(bars) == SPEC.warmup_days and bars.index[-1].date() < d1
    assert f.events("TSTB") is not None and f.events("XYZ") is None
