"""De engine end-to-end op een kleine synthetische markt, met de paper-broker."""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time

import pytest

from tests.unit.options.market_helpers import CFG
from tradebot.options import cli
from tradebot.options.calendar import NY, TradingCalendar
from tradebot.options.data.store import write_market
from tradebot.options.data.synthetic import SyntheticSpec, SyntheticUnderlying, generate
from tradebot.options.engine.replay import replay_config, run_replay, session_clock

SPEC = SyntheticSpec(underlyings=(SyntheticUnderlying("TSTB", 50.0, strike_step=1.0),),
                     start=date(2026, 12, 14), n_days=3,
                     snapshot_times=(time(10, 0), time(13, 30), time(15, 0)),
                     strike_range=0.12, max_days=60, warmup_days=60)


@pytest.fixture(scope="module")
def mkt():
    return generate(SPEC)


@pytest.fixture(scope="module")
def result(mkt, tmp_path_factory):
    out = tmp_path_factory.mktemp("replay")
    return asyncio.run(run_replay(CFG, mkt, out)), out


def test_the_replay_trades_and_leaves_no_short_past_expiry(result):
    res, out = result
    assert res.positions_opened >= 1 and res.open_short_legs_past_expiry == 0
    assert not res.halted and len(res.equity) == SPEC.n_days
    rows = [json.loads(x) for x in (out / "journal.jsonl").read_text().splitlines()]
    events = {r["event"] for r in rows}
    assert {"chain", "candidate", "risk_decision", "order_submitted", "fill",
            "position_opened", "state_saved", "stopped"} <= events
    assert "task_error" not in events and "recon_break" not in events
    summary = json.loads((out / "summary.json").read_text())
    assert summary["positions_opened"] == res.positions_opened
    assert (out / "state.json").is_file() and (out / "heartbeat").is_file()


def test_the_replay_universe_comes_from_the_market(mkt):
    cfg = replay_config(CFG, mkt)
    assert cfg.universe.underlyings == ("TSTB",) and cfg.universe.beta_to_spy == {"TSTB": 1.0}
    assert cfg.universe.share_eligible == ()


def test_the_session_clock():
    cal = TradingCalendar((date(2026, 11, 27),))
    eod = datetime(2026, 11, 27, 16, 15, tzinfo=NY)
    ts = session_clock(cal, date(2026, 11, 27), 1800, eod)
    assert ts[0].time() == time(9, 30) and ts[-2].time() == time(13, 0) and ts[-1] == eod
    full = session_clock(cal, date(2026, 11, 30), 3600, datetime(2026, 11, 30, 16, 0,
                                                                  tzinfo=NY))
    assert full[-2].time() == time(15, 30) and full[-1].time() == time(16, 0)


def test_the_cli_runs_a_replay(mkt, tmp_path, capsys):
    write_market(mkt, tmp_path / "m")
    code = cli.main(["replay", "--market", str(tmp_path / "m"), "--out", str(tmp_path / "o")])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["open_short_legs_past_expiry"] == 0
    import tradebot.options.__main__ as entry  # de module-ingang laadt zonder te draaien
    assert entry.main is cli.main


def test_parallel_prebuilt_chains_give_the_same_replay(mkt, result, tmp_path):
    res, _ = result
    par = asyncio.run(run_replay(CFG, mkt, tmp_path, workers=2))
    assert par.as_record() | {"journal": ""} == res.as_record() | {"journal": ""}


def test_the_cached_builder_falls_back_when_the_snapshot_differs(mkt):
    from tradebot.options.chain.builder import ChainBuilder
    from tradebot.options.engine.replay import CachedBuilder
    from tradebot.options.pricing.engines import Use
    from tradebot.options.pricing.settings import pricer_settings
    from tradebot.options.rates import Carry

    inner = ChainBuilder(pricer_settings(CFG.pricing), 10.0, Use.SCAN)
    s = mkt.snapshots[0]
    carry = Carry(mkt.rate, dividends=mkt.events[s.underlying].dividends)
    cached = inner.build(s.underlying, s.spot, s.options[:4], s.asof, carry)
    b = CachedBuilder(inner, {(s.underlying, s.asof): cached})
    assert b.build(s.underlying, s.spot, s.options, s.asof, carry) is cached
    other = b.build(s.underlying, s.spot + 1.0, s.options[:2], s.asof, carry)
    assert other is not cached and len(other.options) <= 2


def test_one_prebuild_job_is_a_plain_chain_build(mkt):
    import dataclasses

    from tradebot.options.chain.builder import ChainBuilder
    from tradebot.options.engine.replay import _build_one
    from tradebot.options.pricing.engines import Use
    from tradebot.options.pricing.settings import pricer_settings
    from tradebot.options.rates import Carry

    s = dataclasses.replace(mkt.snapshots[0], options=mkt.snapshots[0].options[:2])
    chain = _build_one((ChainBuilder(pricer_settings(CFG.pricing), 10.0, Use.SCAN), s,
                        Carry(mkt.rate)))
    assert (chain.underlying, chain.asof, chain.spot) == (s.underlying, s.asof, s.spot)
