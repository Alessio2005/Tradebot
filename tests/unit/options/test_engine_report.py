"""Backteststatistiek (`engine.report`) en de `sweep`-opdracht."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date

import pytest

from tests.unit.options.market_helpers import CFG
from tests.unit.options.test_engine_replay import SPEC
from tradebot.options import cli
from tradebot.options.data.store import write_market
from tradebot.options.data.synthetic import generate
from tradebot.options.engine.report import book_stats, replay_stats, strategy_stats
from tradebot.options.engine.sweep import resolve_sweep_workers, run_sweep
from tradebot.utils.failfast import TradebotContractError


def _trade(pid: str, strategy: str, open_px: float, close_px: float, fee: float = 1.0,
           hedge: float | None = None) -> list[dict]:
    closed = {"event": "position_closed", "position_id": pid}
    if hedge is not None:
        closed["hedge_realized_usd"] = hedge
    return [{"event": "position_opened", "position_id": pid, "strategy": strategy},
            {"event": "fill", "position_id": pid, "price": open_px, "quantity": 1,
             "commission": fee},
            {"event": "chain"},
            {"event": "fill", "position_id": pid, "price": close_px, "quantity": 1,
             "commission": fee},
            closed]


def test_strategy_stats_per_strategy():
    rows = (_trade("a", "csp", -1.0, 0.4)           # +100 - 40 - 2 = 58
            + _trade("b", "csp", -1.0, 3.0, hedge=5.0)    # +100 - 300 - 2 + 5 = -197
            + _trade("c", "ic", -2.0, 0.5)          # +200 - 50 - 2 = 148
            + _trade("d", "ic", -1.0, 0.0)[:2])     # nog open: telt niet mee
    st = strategy_stats(rows)
    assert set(st) == {"csp", "ic"}
    csp = st["csp"]
    assert (csp.closed, csp.win_rate, csp.avg_pnl, csp.total_pnl, csp.largest_loss) == (
        2, 0.5, -69.5, -139.0, -197.0)
    ic = st["ic"]
    assert (ic.closed, ic.win_rate, ic.total_pnl, ic.largest_loss) == (1, 1.0, 148.0, 0.0)


def _opened(pid: str, strategy: str) -> dict:
    return {"event": "position_opened", "position_id": pid, "strategy": strategy}


def test_settled_positions_count_with_their_settlement_value():
    rows = [
        _opened("exp", "wheel"), {"event": "fill", "position_id": "exp", "price": -1.0,
                                  "quantity": 1, "commission": 1.0},        # +100 - 1
        {"event": "expiry", "position_id": "exp", "settle_usd": 0.0},
        {"event": "position_closed", "position_id": "exp", "reason": "expiry", "exit_net": 0.0},
        _opened("itm", "wheel"), {"event": "fill", "position_id": "itm", "price": -1.0,
                                  "quantity": 1, "commission": 1.0},
        {"event": "assignment", "position_id": "itm", "settle_usd": -300.0},
        {"event": "position_closed", "position_id": "itm", "reason": "assignment",
         "exit_net": -300.0},
        _opened("cal", "calendar"),                 # voorste leg toegewezen, later gesloten
        {"event": "fill", "position_id": "cal", "price": 1.0, "quantity": 1, "commission": 0.0},
        {"event": "assignment", "position_id": "cal", "settle_usd": -50.0},
        {"event": "fill", "position_id": "cal", "price": -0.5, "quantity": 1, "commission": 0.0},
        {"event": "position_closed", "position_id": "cal", "reason": "x", "exit_net": 50.0},
        {"event": "expiry", "position_id": None, "settle_usd": 0.0},      # zonder positie
    ]
    st = strategy_stats(rows)
    assert st["wheel"].closed == 2 and st["wheel"].total_pnl == 99.0 - 201.0
    assert st["wheel"].win_rate == 0.5 and st["wheel"].largest_loss == -201.0
    assert st["calendar"].total_pnl == -100.0 - 50.0 + 50.0


def test_the_final_hedge_unwind_after_the_options_close_is_counted():
    rows = [_opened("g", "gamma_scalp"),
            {"event": "hedge_fill", "position_id": "g", "hedge_realized_usd": 3.0},
            {"event": "fill", "position_id": "g", "price": 1.0, "quantity": 1, "commission": 0.0},
            {"event": "position_closed", "position_id": "g", "reason": "x", "exit_net": 120.0,
             "hedge_realized_usd": 3.0},               # aandelen nog open
            {"event": "hedge_fill", "position_id": "g", "hedge_realized_usd": 40.0}]   # afbouw
    assert strategy_stats(rows)["gamma_scalp"].total_pnl == -100.0 + 40.0


def test_a_close_without_its_open_is_skipped():
    rows = [{"event": "position_closed", "position_id": "x", "reason": "r", "exit_net": 1.0}]
    assert strategy_stats(rows) == {}


def test_no_closed_positions_gives_empty_strategies():
    assert strategy_stats([]) == {}
    assert strategy_stats(_trade("a", "csp", -1.0, 0.0)[:2]) == {}


def test_book_stats_on_a_known_series():
    b = book_stats([100.0, 110.0, 99.0, 108.9])
    assert b.total_return == 0.089 and b.max_drawdown == 0.1
    r = [0.1, -0.1, 0.1]
    mean = sum(r) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / 2)
    assert b.sharpe == pytest.approx(mean / sd * math.sqrt(252), abs=1e-5)
    down = math.sqrt(0.01 / 3)
    assert b.sortino == pytest.approx(mean / down * math.sqrt(252), abs=1e-5)


@pytest.mark.parametrize("series", [[], [100.0]])
def test_book_stats_needs_two_points(series):
    b = book_stats(series)
    assert (b.total_return, b.max_drawdown, b.sharpe, b.sortino) == (0.0, 0.0, None, None)


def test_book_stats_single_return_has_no_ratios():
    b = book_stats([100.0, 105.0])
    assert (b.total_return, b.max_drawdown, b.sharpe, b.sortino) == (0.05, 0.0, None, None)


def test_book_stats_zero_stdev_and_zero_downside():
    flat = book_stats([100.0, 100.0, 100.0])
    assert flat.sharpe is None and flat.sortino is None
    up = book_stats([100.0, 101.0, 103.0, 104.0])      # geen daling: downside 0, stdev > 0
    assert up.sortino is None and up.sharpe is not None and up.max_drawdown == 0.0
    steady = book_stats([100.0, 101.0, 102.01, 103.0301])   # gelijke rendementen: stdev ~ 0
    assert steady.sharpe is None


def test_book_stats_empty_account_is_defined():
    b = book_stats([100.0, 50.0, 0.0])
    assert (b.total_return, b.max_drawdown) == (-1.0, 1.0)
    assert b.sharpe is not None and b.sortino is not None   # rendementen -0,5 en -1,0
    c = book_stats([100.0, 0.0, -20.0, 10.0])            # ook daarna door nul heen: geen NaN
    assert (c.total_return, c.max_drawdown, c.sharpe, c.sortino) == (-0.9, 1.2, None, None)
    json.dumps(c.as_record(), allow_nan=False)


def test_replay_stats_record_is_strict_json():
    rec = replay_stats(_trade("a", "csp", -1.0, 0.4), 25_000.0,
                       [("2026-12-14", 25_010.0), ("2026-12-15", 25_020.0)]).as_record()
    json.dumps(rec, allow_nan=False)
    assert rec["strategies"]["csp"]["closed"] == 1 and rec["book"]["total_return"] > 0
    empty = replay_stats([], 25_000.0, []).as_record()
    assert empty == {"strategies": {}, "book": {"total_return": 0.0, "max_drawdown": 0.0,
                                                  "sharpe": None, "sortino": None}}


@pytest.fixture(scope="module")
def markets(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep_markets")
    write_market(generate(SPEC), root / "m1")
    write_market(generate(replace(SPEC, seed=7, start=date(2026, 12, 15))), root / "m2")
    return [root / "m1", root / "m2"]


def test_sweep_with_one_worker_and_two_workers_agree(markets, tmp_path):
    one = run_sweep(CFG, markets, tmp_path / "one", workers=1)
    two = run_sweep(CFG, markets, tmp_path / "two", workers=2)
    assert one == two and list(one["markets"]) == ["m1", "m2"]
    assert json.loads((tmp_path / "one" / "sweep.json").read_text()) == one
    for name in ("m1", "m2"):
        summary = json.loads((tmp_path / "one" / name / "summary.json").read_text())
        assert summary["stats"] == one["markets"][name]["stats"]
        assert (tmp_path / "two" / name / "journal.jsonl").read_bytes() == (
            tmp_path / "one" / name / "journal.jsonl").read_bytes()
    assert "stats" in one["markets"]["m1"] and "equity" not in one["markets"]["m1"]


def test_sweep_arguments_are_checked(markets, tmp_path):
    with pytest.raises(TradebotContractError):
        run_sweep(CFG, [], tmp_path)
    with pytest.raises(TradebotContractError):
        run_sweep(CFG, [markets[0], markets[0]], tmp_path)
    with pytest.raises(TradebotContractError):
        run_sweep(CFG, markets, tmp_path, workers=0)
    assert resolve_sweep_workers(None, 1) == 1 and resolve_sweep_workers(8, 3) == 3
    assert 1 <= resolve_sweep_workers(None, 100) <= 4


def test_the_cli_runs_a_sweep(markets, tmp_path, capsys):
    code = cli.main(["sweep", "--markets", *map(str, markets), "--out", str(tmp_path / "o"),
                     "--workers", "1"])
    assert code == 0
    assert set(json.loads(capsys.readouterr().out)["markets"]) == {"m1", "m2"}
    assert (tmp_path / "o" / "sweep.json").is_file()


def test_the_cli_sweep_fails_when_a_market_halts(markets, tmp_path, monkeypatch, capsys):
    bad = {"markets": {"m1": {"open_short_legs_past_expiry": 0, "halted": True}}}
    monkeypatch.setattr(cli, "run_sweep", lambda *a: bad)
    assert cli.main(["sweep", "--markets", str(markets[0]), "--out", str(tmp_path)]) == 1
    capsys.readouterr()
