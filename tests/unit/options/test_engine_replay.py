"""De engine end-to-end op een kleine synthetische markt, met de paper-broker."""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, time

import pytest

from tests.unit.options.market_helpers import CFG
from tests.unit.options.synthetic_market import SyntheticSpec, SyntheticUnderlying, generate
from tradebot.options import cli
from tradebot.options.calendar import NY, TradingCalendar
from tradebot.options.data.store import write_market
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
    first = json.loads(capsys.readouterr().out)
    assert first["open_short_legs_past_expiry"] == 0
    import tradebot.options.__main__ as entry  # de module-ingang laadt zonder te draaien
    assert entry.main is cli.main
    # Nog een keer in dezelfde map: geweigerd (het journal zou aangevuld worden).
    journal = (tmp_path / "o" / "journal.jsonl").read_bytes()
    again = ["replay", "--market", str(tmp_path / "m"), "--out", str(tmp_path / "o")]
    assert cli.main(again) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert (tmp_path / "o" / "journal.jsonl").read_bytes() == journal
    # Met --overwrite: eerst leeg, dan dezelfde uitkomst (geen verdubbelde tellingen).
    assert cli.main([*again, "--overwrite"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first
    assert (tmp_path / "o" / "journal.jsonl").read_bytes() == journal
