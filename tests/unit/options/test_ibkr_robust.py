"""Robuustheid van de paper-dienst: crash-herstart, herverbinden, assignment, deelfills."""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time, timedelta

from tests.unit.options.fake_ib import FakeIB
from tests.unit.options.market_helpers import CFG
from tests.unit.options.test_engine_service import BASE, D0, MKT, at
from tests.unit.options.test_ibkr import _events
from tradebot.options.calendar import NY
from tradebot.options.data.ibkr_feed import IBKRFeed
from tradebot.options.engine import paper as paper_mod
from tradebot.options.engine.paper import run_paper
from tradebot.options.data.feed import ReplayFeed
from tradebot.options.engine.service import Engine, EngineMode, Paths
from tradebot.options.engine.state import EngineState, load_state
from tradebot.options.execution.paper import PaperBroker
from tradebot.options.execution.orders import OrderStatus


def ticks(engine, start, until, step=15):
    t = start
    while t <= until:
        asyncio.run(engine.tick(t))
        t += timedelta(seconds=step)


def until_order(engine, start, step=15):
    """Tick tot er een order staat; geeft het tijdstip van die tick."""
    t = start
    while not engine.state.orders:
        asyncio.run(engine.tick(t))
        t += timedelta(seconds=step)
    return t


def engine_on(tmp, broker, state):
    return Engine(BASE, ReplayFeed(MKT), broker, Paths(tmp), state, persist=True)


def test_a_restart_after_a_crash_adopts_the_open_order_and_places_no_second_one(tmp_path):
    broker = PaperBroker(25_000.0, 100.0, 0.65)          # vult nooit: de order blijft open
    a = engine_on(tmp_path, broker, EngineState("paper", "t0", "x"))
    t = until_order(a, at(D0, 9, 30))
    (cid,) = a.state.orders                               # één order, nog open
    assert a.state.orders[cid]["status"] == OrderStatus.SUBMITTED.value
    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert cid in saved["orders"] and saved["attempts"]
    assert any(k.endswith("|scan") or "scan" in k for k in saved["done_today"])
    # Crash: geen stop(), geen EOD. Nieuw proces, dezelfde broker (de order staat daar open).
    state = load_state(tmp_path / "state.json", "paper", "t0")
    b = engine_on(tmp_path, broker, state)
    ticks(b, t, t + timedelta(seconds=30))
    assert list(b.state.orders) == [cid] and list(broker._orders) == [cid]
    assert [r["event"] for r in b.journal.read()].count("order_adopted") == 1


def test_an_order_that_is_gone_at_the_broker_is_not_resent_but_closed_in_the_state(tmp_path):
    broker = PaperBroker(25_000.0, 100.0, 0.65)
    a = engine_on(tmp_path, broker, EngineState("paper", "t0", "x"))
    t = until_order(a, at(D0, 9, 30))
    (cid,) = a.state.orders
    asyncio.run(broker.cancel(cid))                       # tijdens de stilstand geannuleerd
    b = engine_on(tmp_path, broker, load_state(tmp_path / "state.json", "paper", "t0"))
    ticks(b, t, t + timedelta(seconds=15))
    assert b.state.orders[cid]["status"] == OrderStatus.CANCELLED.value
    assert [r["event"] for r in b.journal.read()].count("order_orphaned") == 1
    assert len(b.state.orders) == 1 and b.mode is EngineMode.RUNNING


# --- herverbinden (oud #10) ------------------------------------------------------------------

def _service(tmp_path, monkeypatch, ib, outage_at, stop_at, failures, **kw):
    uni = CFG.universe.model_copy(update={"underlyings": ("TST",), "share_eligible": ()})
    cfg = CFG.model_copy(update={"universe": uni})
    monkeypatch.setattr(paper_mod, "IBKRFeed",
                        lambda ib, names, rate: IBKRFeed(ib, names, rate, events_source=_events,
                                                         wait_seconds=0))
    day = date(2026, 11, 24)
    t = [datetime.combine(day, time(9, 59), tzinfo=NY)]
    stop, alerts, sleeps = asyncio.Event(), [], []

    async def sleep(s):
        sleeps.append(s)
        t[0] += timedelta(seconds=s)
        if t[0] >= datetime.combine(day, outage_at, tzinfo=NY) and not ib.dropped:
            ib.drop()
            ib.fail_connects = failures        # pas na de eerste verbinding
        if t[0] >= datetime.combine(day, stop_at, tzinfo=NY):
            stop.set()

    eng = asyncio.run(run_paper(cfg, ib, tmp_path, stop, clock=lambda: t[0], sleep=sleep,
                                refresh_seconds=60, alert=lambda *a: alerts.append(a), **kw))
    return eng, alerts, sleeps


def test_a_temporary_disconnect_reconnects_with_backoff_and_does_not_halt(tmp_path,
                                                                          monkeypatch):
    ib, pulses = FakeIB(), []
    eng, alerts, sleeps = _service(tmp_path, monkeypatch, ib, time(10, 0), time(10, 6), 3,
                                   pulse=lambda: pulses.append(1))
    events = [r["event"] for r in eng.journal.read()]
    assert "ib_disconnected" in events and "ib_reconnected" in events
    assert not eng.halted and eng.mode is EngineMode.STOPPED
    assert [a[1] for a in alerts] == ["options_ib_down", "options_ib_up"]
    assert [s for s in sleeps if s != 15.0][:3] == [5.0, 10.0, 20.0]   # backoff
    assert "recon_ok" in events[events.index("ib_reconnected"):]       # opnieuw gereconcilieerd


def test_the_heartbeat_stays_stale_while_the_connection_is_down(tmp_path, monkeypatch):
    ib = FakeIB()
    eng, _, _ = _service(tmp_path, monkeypatch, ib, time(10, 0), time(10, 3), 10**6,
                         reconnect_timeout=10**6)
    beat = datetime.fromisoformat((tmp_path / "heartbeat").read_text(encoding="utf-8"))
    assert beat < datetime.combine(date(2026, 11, 24), time(10, 1), tzinfo=NY)


def test_a_long_outage_halts_after_the_timeout(tmp_path, monkeypatch):
    ib = FakeIB()
    eng, alerts, _ = _service(tmp_path, monkeypatch, ib, time(10, 0), time(10, 10), 10**6,
                              reconnect_timeout=60.0)
    assert eng.halted
    (halt,) = [r for r in eng.journal.read() if r["event"] == "halt"]
    assert halt["kind"] == "options_ib_disconnected"
    assert any(a[1] == "options_halt" for a in alerts)


# --- gedeeltelijke fills van combo's (oud #13) -----------------------------------------------

def _partial_rig(tmp_path, kind="open", q=4, max_loss=400.0):
    from tests.unit.options.test_engine_service import C1, C2, Rig
    from tradebot.options.engine.state import intents_record
    from tradebot.options.positions import LegIntent
    rig = Rig(tmp_path)
    legs = (LegIntent(C1, -1), LegIntent(C2, 1))
    rig.state.orders["a"] = {"client_id": "a", "position_id": "w1", "purpose": "entry",
                             "kind": kind, "strategy": "credit_spread", "underlying": "TSTB",
                             "legs": intents_record(legs), "quantity": q,
                             "max_loss": max_loss, "variant": "", "urgency": "normal",
                             "reason": "test", "step": 0, "limit": -0.5,
                             "status": "submitted", "submitted_at": at(D0, 11, 0).isoformat()}
    return rig, legs


def _fill(rig, legs, n, eid, only=None, ts=None):
    from tradebot.options.execution.broker import Execution, ExecutionKind
    ts = ts or at(D0, 11, 0)
    ex = [Execution(ts, ExecutionKind.FILL, "a", leg.contract, "TSTB", leg.ratio * n, 1.0, 0.65,
                    f"{eid}{i}") for i, leg in enumerate(legs) if only is None or i == only]
    rig.e._book_combo_fills("a", rig.state.orders["a"], ex, ts)
    return ex


def test_a_partial_fill_books_only_the_filled_part_and_keeps_the_order_open(tmp_path):
    rig, legs = _partial_rig(tmp_path)
    o = rig.state.orders["a"]
    ex = _fill(rig, legs, 1, "e1")
    (p,) = rig.state.open_positions()
    assert p.units == 1 and p.max_loss == 100.0 and p.entry_net == 50.0
    assert o["status"] == "submitted" and not o.get("booked") and o["filled_units"] == 1
    assert rig.state.new_positions == {D0.isoformat(): 1}
    assert len(rig.e._pending_entries()) == 1 and rig.e._pending_entries()[0].units == 3
    rig.e._book_combo_fills("a", o, ex, at(D0, 11, 0, 5))        # dezelfde fills: niets erbij
    assert rig.state.open_positions()[0].units == 1
    _fill(rig, legs, 2, "e2")
    (p,) = rig.state.open_positions()
    assert p.units == 3 and p.max_loss == 300.0 and [x.qty for x in p.option_legs] == [-3, 3]
    assert p.entry_net == 150.0 and o["status"] == "submitted"
    _fill(rig, legs, 1, "e3")
    (p,) = rig.state.open_positions()
    assert p.units == 4 and p.max_loss == 400.0 and o["status"] == "filled" and o["booked"]
    assert rig.e._pending_entries() == ()
    fills = [r for r in rig.e.journal.read() if r["event"] == "fill"]
    assert [f["quantity"] for f in fills] == [1, 2, 1]
    assert rig.state.new_positions == {D0.isoformat(): 1}


def test_a_leg_that_is_not_yet_filled_books_nothing(tmp_path):
    rig, legs = _partial_rig(tmp_path)
    _fill(rig, legs, 2, "e1", only=0)                    # alleen de short leg gemeld
    assert not rig.state.positions
    _fill(rig, legs, 2, "e2", only=1)
    assert rig.state.open_positions()[0].units == 2


def test_a_partially_filled_close_leaves_the_position_open_for_the_rest(tmp_path):
    rig, legs = _partial_rig(tmp_path)
    _fill(rig, legs, 4, "e1")
    from tradebot.options.engine.state import intents_record
    from tradebot.options.execution.broker import Execution, ExecutionKind
    from tradebot.options.positions import LegIntent
    o = dict(rig.state.orders["a"], client_id="b", kind="close", purpose="exit", quantity=4,
             legs=intents_record(tuple(LegIntent(x.contract, -x.ratio) for x in legs)),
             status="submitted", limit=0.2)
    for k in ("booked", "filled_units", "exec_ids", "leg_fills", "day_trade"):
        o.pop(k, None)
    rig.state.orders["b"] = o
    ex = [Execution(at(D0, 11, 5), ExecutionKind.FILL, "b", leg.contract, "TSTB", -leg.ratio * 1,
                    1.0, 0.65, f"c{i}") for i, leg in enumerate(legs)]
    rig.e._book_combo_fills("b", o, ex, at(D0, 11, 5))
    (p,) = rig.state.open_positions()
    assert sorted(x.qty for x in p.option_legs) == [-3, 3] and p.units == 3 and p.max_loss == 400.0
    assert o["status"] == "submitted" and o["filled_units"] == 1


def test_a_cancelled_partial_entry_keeps_its_filled_units_and_a_late_fill_is_still_booked(
        tmp_path):
    rig, legs = _partial_rig(tmp_path)
    _fill(rig, legs, 1, "e1")
    o = rig.state.orders["a"]
    o["status"] = "expired"                               # EOD-cancel van het restant
    _fill(rig, legs, 1, "e2")                             # een late fill van vlak voor het annuleren
    (p,) = rig.state.open_positions()
    assert p.units == 2 and p.max_loss == 200.0 and not o.get("booked")
