"""De engine taak voor taak: HALT, UNWIND, fouten, de walk, reconcile, risico en boeking."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
from datetime import date, datetime, time, timedelta

import pytest

from tests.unit.options.market_helpers import CFG
from tradebot.options.calendar import NY
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.data.feed import ReplayFeed
from tradebot.options.data.synthetic import SyntheticSpec, SyntheticUnderlying, generate
from tradebot.options.engine import service
from tradebot.options.engine.replay import replay_config
from tradebot.options.engine.service import Engine, EngineMode, Paths
from tradebot.options.engine.state import EngineState, intents_record
from tradebot.options.events import Events
from tradebot.options.execution.broker import Execution, ExecutionKind
from tradebot.options.execution.paper import PaperBroker
from tradebot.options.positions import (
    Action,
    ActionKind,
    Leg,
    LegIntent,
    PositionStatus,
    StrategyPosition,
    Urgency,
)
from tradebot.options.pricing.surface import Smile
from tradebot.options.rates import CashDividend
from tradebot.options.risk.gate import SafetyBreach
from tradebot.options.vol_regime import VolRegime
from tradebot.schemas.options import RegimeAllowedConfig

SPEC = SyntheticSpec(underlyings=(SyntheticUnderlying("TSTB", 50.0, strike_step=1.0),),
                     start=date(2026, 12, 14), n_days=2,
                     snapshot_times=(time(10, 0), time(13, 30), time(15, 0)),
                     strike_range=0.12, max_days=60, warmup_days=60)
MKT = generate(SPEC)
D0, D1 = MKT.days
BASE = replay_config(CFG, MKT)


def at(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss), tzinfo=NY)


def cfg_with(section, **kw):
    return BASE.model_copy(update={section: getattr(BASE, section).model_copy(update=kw)})


class Rig:
    def __init__(self, tmp, cfg=BASE, fill=0.40, strategies=None, market=MKT):
        self.alerts: list[tuple[str, str, str]] = []
        self.broker = PaperBroker(25_000.0, fill, 0.65)
        self.state = EngineState("replay", "paper", "t0")
        self.e = Engine(cfg, ReplayFeed(market), self.broker, Paths(tmp), self.state,
                        lambda *a: self.alerts.append(a), strategies)

    def run(self, until, start=None, step=15):
        t = start or datetime.combine(until.date(), time(9, 30), tzinfo=NY)
        while t <= until:
            asyncio.run(self.e.tick(t))
            t += timedelta(seconds=step)

    def events(self, name):
        return [r for r in self.e.journal.read() if r["event"] == name]


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def opened(rig):
    rig.run(at(D0, 10, 1))
    (pos,) = rig.state.open_positions()
    return pos


def test_the_default_alert_logs(caplog):
    with caplog.at_level(logging.WARNING):
        service._no_alert("critical", "k", "tekst")
    assert "k: tekst" in caplog.text


def test_a_normal_morning_opens_one_position(rig):
    pos = opened(rig)
    assert rig.e.mode is EngineMode.RUNNING and pos.strategy == "credit_spread"
    assert rig.state.new_positions == {D0.isoformat(): 1}
    # Dezelfde kandidaat op het volgende scanmoment wordt niet nog eens geopend.
    rig.run(at(D0, 13, 31), start=at(D0, 13, 30))
    assert len(rig.state.open_positions()) == 1


def test_a_halt_file_blocks_entries(rig):
    rig.e.paths.root.mkdir(parents=True, exist_ok=True)
    rig.e.paths.halt_file.touch()
    rig.run(at(D0, 10, 1))
    assert rig.e.mode is EngineMode.HALTED and not rig.state.positions
    assert rig.events("scan_summary")[0]["skipped"] == "halted"


def test_unwind_closes_everything(rig):
    pos = opened(rig)
    rig.e.paths.unwind_file.touch()
    rig.run(at(D0, 10, 10), start=at(D0, 10, 1, 15))
    assert rig.e.mode is EngineMode.UNWINDING
    assert rig.state.positions[0].status is PositionStatus.CLOSED
    unwind = [o for o in rig.state.orders.values() if o["purpose"] == "unwind"]
    assert unwind and unwind[0]["position_id"] == pos.position_id
    assert rig.state.day_trades == [D0.isoformat()]


def test_no_entries_outside_the_entry_window(tmp_path):
    r = Rig(tmp_path, cfg_with("schedule", scan_times=("09:35",)))
    r.run(at(D0, 9, 36))
    assert r.events("scan_summary")[0]["skipped"] == "entry_window"
    assert r.e._entry_window(at(date(2026, 12, 24), 11, 0))
    assert not r.e._entry_window(at(date(2026, 12, 24), 12, 45))     # halve dag


class Boom:
    name = "boom"

    def __init__(self, exc):
        self.exc = exc

    def scan(self, market, book):
        raise self.exc

    def manage(self, pos, market):
        return []


def test_three_failures_in_a_row_halt(tmp_path):
    r = Rig(tmp_path, cfg_with("schedule", scan_times=("10:00", "10:30", "11:00")),
            strategies=(Boom(RuntimeError("stuk")),))
    r.run(at(D0, 11, 1))
    errs = r.events("task_error")
    assert [x["count"] for x in errs] == [1, 2, 3]
    assert r.e.mode is EngineMode.HALTED and r.e.halts.is_halted()
    assert ("critical", "options_halt", "options_task_failures: taak scan faalde 3 keer achter "
            "elkaar") in r.alerts


def test_a_safety_breach_halts_at_once(tmp_path):
    r = Rig(tmp_path, strategies=(Boom(SafetyBreach("te groot")),))
    r.run(at(D0, 10, 0))
    assert r.e.mode is EngineMode.HALTED and r.events("halt")[0]["reason"] == "te groot"


def test_the_walk_gives_up_after_the_maximum_concession(tmp_path):
    r = Rig(tmp_path, fill=1.0)             # de paper-broker vult pas op natural
    r.run(at(D0, 10, 3))
    (o,) = r.state.orders.values()
    assert o["status"] == "cancelled" and not r.state.positions
    assert r.events("order_cancelled")[0]["reason"] == "max_concession"


def fair_cfg(**kw):
    return cfg_with("execution", limit_anchor="fair", **kw)


def richer_smile(rig, f):
    """Het oppervlak dat de engine voor fair gebruikt, met de vol × `f`: fair ≠ mid. De
    IV-metingen (regime, toelating) blijven die van de markt."""
    metrics = rig.e._metrics

    def scaled(und, chain, today):
        out = metrics(und, chain, today)
        s = rig.e._surfaces[und]
        rig.e._surfaces[und] = dataclasses.replace(s, smiles=tuple(
            Smile(x.expiry, x.t, x.forward, x.discount, x.k, tuple(w * f * f for w in x.w))
            for x in s.smiles))
        return out

    rig.e._metrics = scaled


def test_a_fair_anchored_entry_starts_at_the_fair_value(tmp_path):
    # Credit spread: mid −0.24, natural −0.22, de andere kant −0.26. Mid-anker: −0.24.
    m = Rig(tmp_path / "mid")
    m.run(at(D0, 10, 0, 1), step=1)
    assert [o["limit"] for o in m.events("order_submitted")] == [-0.24]
    # Bij 1,1 × de smile-vol is fair ≈ −0.262: begrensd op de andere kant, −0.26.
    r = Rig(tmp_path / "fair", fair_cfg())
    richer_smile(r, 1.1)
    r.run(at(D0, 10, 0, 1), step=1)
    assert [o["limit"] for o in r.events("order_submitted")] == [-0.26]


def test_a_first_limit_past_the_edge_skips_the_entry(tmp_path):
    # Bij 1,5 × de vol ligt fair zo ver voorbij −0.26 dat het plafond (fair + 0,5 × 0,02)
    # al vóór limit_0 ligt: geen order.
    r = Rig(tmp_path, fair_cfg())
    richer_smile(r, 1.5)
    r.run(at(D0, 10, 1))
    (skip,) = r.events("entry_skipped")
    assert skip["reason"] == "max_edge" and not r.state.orders


def test_an_entry_without_a_fair_value_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "combo_fair", lambda *a: None)
    r = Rig(tmp_path, fair_cfg())
    r.run(at(D0, 10, 1))
    (skip,) = r.events("order_skipped")
    assert skip["reason"] == "no_quote" and skip["purpose"] == "entry" and not r.state.orders


def test_the_fair_walk_gives_up_at_the_edge(tmp_path):
    r = Rig(tmp_path, fair_cfg(), fill=1.0)          # de paper-broker vult pas op natural
    r.run(at(D0, 10, 3))
    (o,) = r.state.orders.values()
    assert o["status"] == "cancelled" and not r.state.positions
    assert r.events("order_cancelled")[0]["reason"] == "max_edge"


def test_a_spread_too_wide_against_fair_skips_the_entry(tmp_path):
    r = Rig(tmp_path, fair_cfg(max_spread_frac_fair=1e-6))
    r.run(at(D0, 10, 1))
    (skip,) = r.events("entry_skipped")
    assert skip["reason"] == "spread_vs_fair" and skip["strategy"] == "credit_spread"
    assert r.events("risk_decision")[0]["verdict"] == "approve"
    assert not r.state.orders and not r.events("order_submitted")


def test_open_orders_expire_at_the_end_of_the_day(tmp_path):
    r = Rig(tmp_path, cfg_with("schedule", scan_times=("15:50",), no_entry_last_minutes=0),
            fill=1.0)
    r.run(at(D0, 16, 15), start=at(D0, 15, 0), step=300)
    assert {o["status"] for o in r.state.orders.values()} == {"expired"}
    assert r.state.equity[-1][0] == D0.isoformat()


def test_stop_cancels_and_a_stopped_engine_does_nothing(tmp_path):
    r = Rig(tmp_path, fill=1.0)
    r.run(at(D0, 10, 0))
    closed = []
    r.e.builder.close = lambda: closed.append(True)        # de procespool van de keten
    asyncio.run(r.e.stop(at(D0, 10, 0, 5)))
    (o,) = r.state.orders.values()
    assert o["status"] == "cancelled" and r.e.mode is EngineMode.STOPPED and closed
    n = len(r.e.journal.read())
    asyncio.run(r.e.tick(at(D0, 10, 1)))
    assert len(r.e.journal.read()) == n


def test_a_rejected_entry_sends_nothing(tmp_path):
    r = Rig(tmp_path, cfg_with("risk", max_vega_frac=1e-6))
    r.run(at(D0, 10, 1))
    assert r.events("risk_decision")[0]["verdict"] == "reject" and not r.state.orders


def test_a_reconcile_break_blocks_until_it_is_fixed(rig):
    pos = opened(rig)
    stray = OptionContract("TSTB", date(2027, 1, 15), 40.0, Right.PUT)
    rig.broker._book_option(stray, 1, 0.1)
    rig.run(at(D0, 10, 7), start=at(D0, 10, 6, 15))
    assert rig.e.mode is EngineMode.BLOCKED
    assert rig.e.gate_state(at(D0, 10, 7)).blocked_underlyings == frozenset({"TSTB"})
    assert [a[1] for a in rig.alerts] == ["options_recon_break"]
    rig.run(at(D0, 10, 13), start=at(D0, 10, 12))           # nog steeds: geen tweede alert
    assert len(rig.alerts) == 1
    rig.broker._book_option(stray, -1, 0.1)
    rig.run(at(D0, 10, 18), start=at(D0, 10, 17))
    assert rig.e.mode is EngineMode.RUNNING and pos in rig.state.open_positions()


def test_a_break_while_halted_stays_halted(rig):
    opened(rig)
    rig.e.mode = EngineMode.HALTED
    rig.broker._book_option(OptionContract("TSTB", date(2027, 1, 15), 40.0, Right.PUT), 1, 0.1)
    asyncio.run(rig.e._reconcile(at(D0, 10, 5)))
    assert rig.e.mode is EngineMode.HALTED


def test_daily_loss_stops_entries_and_degrosses(rig):
    opened(rig)
    rig.state.equity.append(("2026-12-11", 26_500.0))       # ~5,6 % verlies vandaag
    rig.e.cfg = rig.e.cfg.model_copy(update={"risk": rig.e.cfg.risk.model_copy(
        update={"max_vega_frac": 1e-6})})                   # het boek past niet meer
    rig.e._book_version += 1
    rig.run(at(D0, 10, 3), start=at(D0, 10, 2))
    assert rig.events("daily_loss_stop")
    assert rig.e.gate_state(at(D0, 10, 3)).daily_loss_frac == 1.0
    acts = rig.events("action")
    assert acts and acts[0]["reason"] == "degross"
    assert rig.state.positions[0].status is PositionStatus.CLOSED


def test_a_drawdown_halts(rig):
    opened(rig)
    rig.state.high_water_mark = 40_000.0
    rig.e._book_version += 1
    rig.run(at(D0, 10, 3), start=at(D0, 10, 2))
    assert rig.e.mode is EngineMode.HALTED and rig.e.halts.load().kind == "options_drawdown"


def test_actions_are_deferred_without_a_chain_or_quote(rig):
    pos = opened(rig)
    now = at(D0, 10, 2)
    odd = dataclasses.replace(pos, underlying="XYZ")
    asyncio.run(rig.e._act(Action(ActionKind.CLOSE, "x", odd.intents(), Urgency.STOP, "t"),
                           odd, now))
    ghost = (LegIntent(OptionContract("TSTB", date(2027, 1, 15), 99.0, Right.CALL), 1),)
    asyncio.run(rig.e._act(Action(ActionKind.CLOSE, pos.position_id, ghost, Urgency.STOP, "t"),
                           pos, now))
    assert [x["reason"] for x in rig.events("action_deferred")] == ["no_chain", "no_quote"]


def test_a_roll_is_sized_on_the_new_strike(rig):
    pos = opened(rig)
    short = next(leg for leg in pos.intents() if leg.ratio < 0)
    new = LegIntent(dataclasses.replace(short.contract, expiry=date(2027, 1, 15)), -1)
    a = Action(ActionKind.ROLL, pos.position_id, (short,), Urgency.NORMAL, "roll", (new,))
    asyncio.run(rig.e._act(a, pos, at(D0, 10, 2)))
    (d,) = [x for x in rig.events("risk_decision") if x["kind"] == "roll"]
    assert d["position_id"] == pos.position_id


C1 = OptionContract("TSTB", date(2027, 1, 15), 48.0, Right.PUT)
C2 = OptionContract("TSTB", date(2027, 1, 15), 47.0, Right.PUT)


def _order(rig, cid, kind, legs, pid="w1", strategy="credit_spread", variant="",
           max_loss=100.0, q=1, limit=-0.5):
    rig.state.orders[cid] = {"client_id": cid, "position_id": pid, "purpose": "entry",
                             "kind": kind, "strategy": strategy, "underlying": "TSTB",
                             "legs": intents_record(legs), "quantity": q,
                             "max_loss": max_loss, "variant": variant, "urgency": "normal",
                             "reason": "test", "step": 0, "limit": limit,
                             "status": "submitted"}
    ex = [Execution(at(D0, 11, 0), ExecutionKind.FILL, cid, leg.contract, "TSTB", leg.ratio * q,
                    1.0, 0.65) for leg in legs]
    rig.e._book_fill(cid, rig.state.orders[cid], ex, at(D0, 11, 0))


def test_booking_opens_changes_and_closes(rig):
    _order(rig, "a", "open", (LegIntent(C1, -1), LegIntent(C2, 1)), max_loss=None)
    (p,) = rig.state.open_positions()
    assert p.max_loss == math.inf and p.entry_net == 50.0
    _order(rig, "b", "close_leg", (LegIntent(C2, -1),), limit=-0.1)
    (p,) = rig.state.open_positions()
    assert [leg.contract for leg in p.option_legs] == [C1] and p.max_loss == 100.0
    _order(rig, "c", "adjust", (LegIntent(C1, 1), LegIntent(C2, -1)), max_loss=None,
           limit=0.2)
    (p,) = rig.state.open_positions()
    assert p.adjustments == 1 and p.max_loss == math.inf
    _order(rig, "d", "close", (LegIntent(C2, 1),), limit=0.1)
    assert not rig.state.open_positions()
    assert rig.state.day_trades == [D0.isoformat()] * 3


@pytest.mark.parametrize(("phase", "after_open", "after_close"), [
    ("csp", "csp", "idle"), ("cc", "cc", "shares")])
def test_booking_follows_the_wheel(rig, phase, after_open, after_close):
    _order(rig, "a", "open", (LegIntent(C1, -1),), strategy="wheel", variant=phase)
    assert rig.state.wheel["TSTB"] == after_open
    _order(rig, "b", "close", (LegIntent(C1, 1),), strategy="wheel", limit=0.1)
    assert rig.state.wheel["TSTB"] == after_close


def _assign(rig, contract, kind=ExecutionKind.ASSIGNMENT):
    rig.e._assignment(Execution(at(D0, 16, 30), kind, "", contract, "TSTB", 1, contract.strike),
                      at(D0, 16, 30))


def _pos(pid, strategy, legs, phase="", status=PositionStatus.OPEN):
    return StrategyPosition(pid, strategy, "TSTB",
                            tuple(Leg(c, "TSTB", q, 1.0) for c, q in legs), at(D0, 10, 0), 1,
                            100.0, 500.0, phase=phase, status=status)


def test_assignments_close_legs_and_alert(rig):
    rig.state.positions += [_pos("gone", "credit_spread", [(C1, -1)],
                                 status=PositionStatus.CLOSED),
                            _pos("s", "credit_spread", [(C1, -1), (C2, 1)])]
    _assign(rig, C1)
    (p,) = rig.state.open_positions()
    assert [leg.contract for leg in p.option_legs] == [C2]
    assert rig.alerts[-1][1] == "options_assignment"
    _assign(rig, C2, ExecutionKind.EXPIRY)
    assert not rig.state.open_positions() and len(rig.alerts) == 1
    _assign(rig, OptionContract("TSTB", date(2027, 1, 15), 10.0, Right.PUT))
    assert [r["event"] for r in rig.e.journal.read()] == ["assignment", "expiry", "assignment"]


def test_wheel_assignments_move_the_phase(rig):
    rig.state.positions.append(_pos("w", "wheel", [(C1, -1)], phase="csp"))
    _assign(rig, C1)
    assert rig.state.wheel["TSTB"] == "shares"
    assert rig.state.cost_basis["TSTB"] == pytest.approx(48.0 - 1.0)
    call = OptionContract("TSTB", date(2027, 1, 15), 52.0, Right.CALL)
    rig.state.positions.append(_pos("c", "wheel", [(call, -1)], phase="cc"))
    _assign(rig, call)
    assert rig.state.wheel["TSTB"] == "idle" and rig.state.cost_basis["TSTB"] == 47.0
    assert not rig.alerts


def test_broker_assignments_and_known_fills_are_read_once(rig):
    opened(rig)
    (pos,) = rig.state.open_positions()
    short = next(leg.contract for leg in pos.option_legs if leg.qty < 0)
    rig.broker._executions.append(Execution(at(D0, 10, 2), ExecutionKind.ASSIGNMENT, "",
                                            short, "TSTB", 1, short.strike))
    rig.broker._executions.append(Execution(at(D0, 10, 2), ExecutionKind.FILL, "unknown",
                                            short, "TSTB", 1, 0.1))
    asyncio.run(rig.e._executions(at(D0, 10, 2)))
    asyncio.run(rig.e._executions(at(D0, 10, 2)))
    assert len(rig.events("assignment")) == 1


def test_end_of_day_dividends_and_shorts_past_expiry(tmp_path):
    m = dataclasses.replace(MKT, events=dict(MKT.events))
    r = Rig(tmp_path, market=m)
    r.run(at(D0, 15, 1), step=60)
    m.events["TSTB"] = Events("TSTB", D0, dividends=(CashDividend(D1, 0.2),
                                                     CashDividend(date(2027, 3, 1), 0.2)))
    stale = OptionContract("TSTB", D0, 50.0, Right.PUT)
    r.state.positions.append(_pos("old", "credit_spread", [(stale, -1)]))
    asyncio.run(r.e.tick(at(D0, 16, 15)))
    assert ("critical", "options_short_past_expiry", stale.key()) in r.alerts
    assert r.state.equity[-1][0] == D0.isoformat()
    m.events.pop("TSTB")
    asyncio.run(r.e._end_of_day(at(D1, 16, 15)))
    assert r.state.iv_history["TSTB"][-1][0] == D0.isoformat()   # geen nieuwe keten op D1


def test_a_halt_file_during_the_day_halts_a_running_engine(rig):
    opened(rig)
    rig.e.paths.halt_file.touch()
    asyncio.run(rig.e.tick(at(D0, 10, 2)))
    assert rig.e.mode is EngineMode.HALTED


def test_a_pending_entry_is_not_sent_twice(tmp_path):
    r = Rig(tmp_path, cfg_with("schedule", scan_times=("10:00", "10:05")).model_copy(
        update={"execution": BASE.execution.model_copy(update={"walk_interval_seconds": 600.0})}),
        fill=1.0)
    r.run(at(D0, 10, 6))
    assert len(r.state.orders) == 1 and len(r.events("candidate")) == 2


def test_the_walk_skips_orders_without_a_chain(rig):
    _order(rig, "a", "open", (LegIntent(C1, -1),))
    rig.state.orders["x"] = dict(rig.state.orders["a"], client_id="x", underlying="XYZ",
                                 status="submitted")
    asyncio.run(rig.e._walk(at(D0, 11, 0)))
    assert rig.state.orders["x"]["step"] == 0


def test_an_assignment_skips_positions_without_the_contract(rig):
    rig.state.positions += [_pos("other", "credit_spread", [(C2, 1)]),
                            _pos("s", "credit_spread", [(C1, -1)])]
    _assign(rig, C1)
    assert [p.position_id for p in rig.state.open_positions()] == ["other"]


def test_the_regime_and_theta_cover_are_journaled(rig):
    opened(rig)
    (ev,) = rig.events("vol_regime")          # één wisseling: van niets naar het regime
    assert ev["underlying"] == "TSTB" and ev["regime"] in {"rich", "normal"}
    (cand,) = rig.events("candidate")
    assert any(t.startswith("regime:") for t in cand["trace"])
    assert any(t.startswith("theta:cover=") for t in cand["trace"])
    summary = rig.events("scan_summary")[0]
    assert summary["regimes"] == {"TSTB": ev["regime"]} and summary["blocked"] == {}


def test_the_market_state_carries_the_vol_state(rig):
    opened(rig)
    m = rig.e.market_state(at(D0, 10, 2))
    assert m.vol == rig.e._vol and m.vol["TSTB"].rv is not None


def test_a_regime_without_the_strategy_blocks_the_entry(tmp_path):
    from tradebot.schemas.options import RegimeAllowedConfig
    none = RegimeAllowedConfig(rich=(), normal=(), cheap=())
    r = Rig(tmp_path, cfg_with("vol_regime", allowed=none))
    r.run(at(D0, 10, 1))
    assert r.state.open_positions() == []
    assert r.events("scan_summary")[0]["blocked"] == {"credit_spread:regime": 1}


def test_too_little_theta_blocks_the_entry(tmp_path):
    r = Rig(tmp_path, cfg_with("theta_harvest", min_theta_gamma_cover=5.0))
    r.run(at(D0, 10, 1))
    assert r.state.open_positions() == []
    assert r.events("scan_summary")[0]["blocked"] == {"credit_spread:theta": 1}


def test_the_heartbeat_holds_exactly_the_last_tick(tmp_path):
    beat = tmp_path / "heartbeat"
    service._write_heartbeat(beat, "2026-12-14T10:00:00.123456-05:00")
    service._write_heartbeat(beat, "2026-12-14T10:00:15-05:00")    # korter: rest weg
    assert beat.read_text(encoding="utf-8") == "2026-12-14T10:00:15-05:00"
    rig = Rig(tmp_path / "run")
    rig.run(at(D0, 9, 31), start=at(D0, 9, 30))
    assert rig.e.paths.heartbeat.read_text(encoding="utf-8") == at(D0, 9, 31).isoformat()


# --- short strangle met harde limieten (aangezet door de eigenaar) ---------------------------

SV_SPEC = dataclasses.replace(SPEC, underlyings=(SyntheticUnderlying("TSTA", 50.0, strike_step=1.0),
                                                 SyntheticUnderlying("TSTB", 50.0, strike_step=1.0)))
SV_MKT = generate(SV_SPEC)


def undefined_cfg(stress_frac=0.06):
    """Alleen de short strangle, op TSTA en TSTB, met undefined risk toegestaan."""
    base = replay_config(CFG, SV_MKT)
    s = base.strategies
    off = {k: getattr(s, k).model_copy(update={"enabled": False}) for k in type(s).model_fields}
    off["short_straddle_strangle"] = s.short_straddle_strangle.model_copy(update={
        "enabled": True, "underlyings": ("TSTA", "TSTB")})
    every = ("short_straddle_strangle",)
    fb = base.iv_metrics.fallback.model_copy(update={"high_iv_ratio": 1.05})
    return base.model_copy(update={
        "strategies": s.model_copy(update=off),
        "risk": base.risk.model_copy(update={
            "allow_undefined_risk": True, "max_undefined_stress_loss_frac": stress_frac,
            "max_scenario_loss_frac": 0.5}),
        "sizing": base.sizing.model_copy(update={"max_loss_frac_capital": 0.05}),
        "iv_metrics": base.iv_metrics.model_copy(update={"fallback": fb}),
        "vol_regime": base.vol_regime.model_copy(update={
            "allowed": RegimeAllowedConfig(rich=every, normal=every, cheap=every)})})


def test_the_book_stress_cap_blocks_a_second_undefined_entry(tmp_path):
    r = Rig(tmp_path / "a", undefined_cfg(), market=SV_MKT)
    r.run(at(D0, 13, 31))
    (pos,) = r.state.open_positions()
    assert pos.strategy == "short_straddle_strangle" and math.isinf(pos.max_loss)
    first, second = r.events("risk_decision")
    assert first["verdict"] == "approve" and second["underlying"] != pos.underlying
    assert second["verdict"] == "reject" and second["reason"] == "limits:undefined_stress_loss"
    (name, measured, threshold) = second["checks"][-1]
    assert name == "undefined_stress_loss" and measured > threshold == 1500.0
    # Met een ruimere grens gaat dezelfde tweede entry wel door: de grens blokkeerde hem.
    wide = Rig(tmp_path / "b", undefined_cfg(0.5), market=SV_MKT)
    wide.run(at(D0, 13, 31))
    assert len(wide.state.open_positions()) == 2


@pytest.fixture
def strangle_rig(tmp_path):
    r = Rig(tmp_path, undefined_cfg(), market=SV_MKT)
    r.run(at(D0, 10, 1))
    return r


def _stress(rig, und):
    rig.e._vol[und] = dataclasses.replace(rig.e._vol["TSTB"], regime=VolRegime.STRESS)


def _stress_closes(rig, now):
    return [a for a in rig.events("action") if a["reason"] == "regime_stress"
            and a["ts"] == now.isoformat()]


def test_stress_on_the_underlying_closes_undefined_risk(strangle_rig):
    r = strangle_rig
    (pos,) = r.state.open_positions()
    now = at(D0, 10, 2)
    asyncio.run(r.e._manage(now))
    assert _stress_closes(r, now) == []
    _stress(r, pos.underlying)
    now = at(D0, 10, 3)
    asyncio.run(r.e._manage(now))
    (a,) = _stress_closes(r, now)
    assert a["position_id"] == pos.position_id and a["urgency"] == "stop"
    (d,) = [x for x in r.events("risk_decision") if x["kind"] == "close"]
    assert d["verdict"] == "approve"


def test_stress_closes_also_while_halted_and_only_undefined_risk(strangle_rig):
    r = strangle_rig
    (pos,) = r.state.open_positions()
    r.e.paths.halt_file.touch()
    asyncio.run(r.e.tick(at(D0, 10, 2)))
    assert r.e.mode is EngineMode.HALTED
    _stress(r, "SPY")                       # STRESS op de marktonderliggende: het hele boek
    finite = dataclasses.replace(pos, max_loss=500.0)
    r.state.positions = [finite]
    now = at(D0, 10, 3)
    asyncio.run(r.e._manage(now))
    assert _stress_closes(r, now) == []
    r.state.positions = [pos]
    now = at(D0, 10, 4)
    asyncio.run(r.e._manage(now))
    (a,) = _stress_closes(r, now)
    assert a["position_id"] == pos.position_id
    assert [x["verdict"] for x in r.events("risk_decision") if x["kind"] == "close"] == [
        "approve"]


def test_stress_blocks_undefined_entries_even_with_the_regime_layer_off(tmp_path):
    cfg = undefined_cfg()
    cfg = cfg.model_copy(update={
        "vol_regime": cfg.vol_regime.model_copy(update={"enabled": False}),
        "schedule": cfg.schedule.model_copy(update={"scan_times": ("14:00",)})})
    r = Rig(tmp_path, cfg, market=SV_MKT)
    r.run(at(D0, 10, 1))
    # Het regime wordt ook zonder de regimelaag berekend: toelating en sluiting lezen het.
    assert set(r.e._vol) == {"TSTA", "TSTB"}
    cands = r.e.strategies[0].scan(r.e.market_state(at(D0, 10, 1)), r.e.book())
    assert cands
    blocked: dict[str, int] = {}
    stress = dict.fromkeys(r.e._vol, VolRegime.STRESS)
    assert r.e._admitted(cands, stress, blocked, D0) == []
    assert blocked == {"short_straddle_strangle:regime": len(cands)}
    normal = dict.fromkeys(r.e._vol, VolRegime.NORMAL)
    assert len(r.e._admitted(cands, normal, {}, D0)) == len(cands)
