"""Het uitvoeringspad van het carryboek (`tradebot.carry`).

Kern: een shadowboek dat vanaf een start bar voor bar stapt, met de staat tussen elke
bar naar schijf en terug, IS de backtest (`programme_v8.simulate`): bit voor bit. Ook als
het paneel dag voor dag groeit. Daarnaast het live-pad tegen een nep-beurs: reconciliatie,
auto-deleveraging, gedeeltelijke fills, en de kill switch.
"""
from __future__ import annotations

import json
import runpy
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.lookahead.test_levered_carry_causality import _mm
from tests.unit.test_programme_v6_smoke import COSTS
from tests.unit.test_programme_v8_smoke import _v8
from tradebot.carry import __main__ as cli
from tradebot.carry import runner
from tradebot.carry.orders import Leg, LegOrder, orders_for, unwind_orders
from tradebot.carry.runner import build_context, init, new_book, settled_end, step, unwind
from tradebot.carry.state import (
    CarryState,
    load_state,
    restore,
    save_state,
    snapshot,
)
from tradebot.carry.venue import (
    LegFill,
    PaperVenue,
    ReconciliationBreak,
    VenueBook,
    reconcile_book,
)
from tradebot.oms.order import OrderSide
from tradebot.systematic import programme_v8 as v8
from tradebot.systematic.leverage import LeverStep
from tradebot.utils.failfast import DataContractError

NAME = v8.CANDIDATES[0]
START = 260


@pytest.fixture(scope="module")
def world():
    m = _mm()
    cfg = _v8(0.5)
    ctx = build_context(m, cfg, COSTS)
    backtest = v8.simulate(NAME, m, cfg, COSTS)
    return m, cfg, ctx, backtest


def _start(ctx) -> str:
    return str(ctx.arrays.index[START].date())


# --------------------------------------------------------------------------- #
# Pariteit met de backtest
# --------------------------------------------------------------------------- #
def test_the_shadow_book_is_the_backtest_bit_for_bit(world):
    _, _, ctx, backtest = world
    st = init(ctx, start=_start(ctx))
    venue = PaperVenue(spot=dict(st.expected_spot), perp_short=dict(st.expected_perp),
                       equity_usd=st.equity)
    records, st = step(ctx, st, venue, settle=0)
    frame = backtest.frame.iloc[START:]
    assert len(records) == len(frame)
    for col in ("net", "gross_leverage", "loan", "uni_mmr", "turnover", "n_held"):
        got = np.array([np.nan if r[col] is None else r[col] for r in records])
        assert np.array_equal(got, frame[col].to_numpy(), equal_nan=True), col
    assert float(frame["turnover"].sum()) > 0.0, "de toets is leeg zonder handel"
    assert st.last_bar == ctx.arrays.index[-1].isoformat()


def test_a_daily_save_and_load_changes_nothing(world, tmp_path: Path):
    """De staat is volledig en exact: na elke bar naar schijf en terug, toch de backtest."""
    _, _, ctx, backtest = world
    path = tmp_path / "state.json"
    save_state(init(ctx, start=_start(ctx)), path)
    n = len(ctx.arrays.index)
    nets = []
    for t in range(START, START + 60):
        st = load_state(path)
        venue = PaperVenue(spot=dict(st.expected_spot), perp_short=dict(st.expected_perp),
                           equity_usd=st.equity)
        records, st = step(ctx, st, venue, settle=n - 1 - t)
        assert len(records) == 1
        nets.append(records[0]["net"])
        save_state(st, path)
    assert nets == backtest.frame["net"].iloc[START:START + 60].tolist()


def test_a_growing_panel_gives_the_same_decisions(world):
    """Elke dag één bar meer in het paneel: het pad is causaal, dus dezelfde boeken."""
    m, cfg, ctx_full, backtest = world
    st = init(ctx_full, start=_start(ctx_full))
    for t in range(START, START + 5):
        ctx = build_context(m.truncate(m.index[t] + pd.Timedelta(hours=1)), cfg, COSTS)
        venue = PaperVenue(spot=dict(st.expected_spot), perp_short=dict(st.expected_perp),
                           equity_usd=st.equity)
        records, st = step(ctx, st, venue, settle=0)
        assert [r["net"] for r in records] == [float(backtest.frame["net"].iloc[t])]


def test_unsettled_bars_wait(world):
    _, _, ctx, _ = world
    st = init(ctx, start=_start(ctx))
    records, st = step(ctx, st, PaperVenue(), settle=7)
    assert records[-1]["bar"] == ctx.arrays.index[-8].isoformat()
    again, _ = step(ctx, st, PaperVenue(), settle=7)
    assert again == []
    with pytest.raises(DataContractError, match="negatieve settle"):
        settled_end(ctx, -1)


def test_init_needs_a_start_inside_the_settled_panel(world):
    _, _, ctx, _ = world
    for start in ("1990-01-01", "2100-01-01"):
        with pytest.raises(DataContractError, match="gesettelde paneel"):
            init(ctx, start=start)
    with pytest.raises(DataContractError, match="Onbekende modus"):
        init(ctx, start=_start(ctx), mode="yolo")


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
def _step(da, db, equity_pre=1_000.0) -> LeverStep:
    da, db = np.asarray(da, float), np.asarray(db, float)
    return LeverStep(row=(), held=np.zeros(da.size), trades=np.abs(da) + np.abs(db), da=da,
                     db=db, equity_pre=equity_pre, liquidated=False)


def test_orders_carry_the_traded_notional_and_reduce_risk_first():
    res = _step([0.10, -0.05, 0.0], [0.10, -0.05, 0.02])
    orders = orders_for(res, ["AUSDT", "BUSDT", "CUSDT"], np.array([10.0, 4.0, 2.0]),
                        np.array([10.1, 4.1, 2.1]))
    got = [(o.symbol, o.leg, o.side, o.reduce_only) for o in orders]
    assert got == [("BUSDT", Leg.PERP, OrderSide.BUY, True),     # short terug
                   ("BUSDT", Leg.SPOT, OrderSide.SELL, True),    # dan pas het onderpand weg
                   ("AUSDT", Leg.SPOT, OrderSide.BUY, False),    # eerst onderpand erbij
                   ("AUSDT", Leg.PERP, OrderSide.SELL, False),   # dan de short
                   ("CUSDT", Leg.PERP, OrderSide.SELL, False)]
    a_spot = orders[2]
    assert a_spot.notional_usd == pytest.approx(100.0) and a_spot.qty == pytest.approx(10.0)
    assert orders[0].as_record()["leg"] == "perp"


def test_an_order_without_a_price_is_refused():
    with pytest.raises(DataContractError, match="zonder geldige koers"):
        orders_for(_step([0.1], [0.0]), ["AUSDT"], np.array([np.nan]), np.array([1.0]))
    with pytest.raises(DataContractError, match="niet dezelfde lengte"):
        orders_for(_step([0.1], [0.0]), ["AUSDT", "BUSDT"], np.array([1.0]), np.array([1.0]))


def test_the_unwind_closes_every_short_before_any_spot():
    orders = unwind_orders({"AUSDT": 2.0, "BUSDT": 0.0}, {"AUSDT": 2.0, "CUSDT": 1.0},
                           {"AUSDT": 10.0}, {"AUSDT": 10.0, "CUSDT": 3.0})
    assert [(o.symbol, o.leg, o.side, o.qty) for o in orders] == [
        ("AUSDT", Leg.PERP, OrderSide.BUY, 2.0), ("CUSDT", Leg.PERP, OrderSide.BUY, 1.0),
        ("AUSDT", Leg.SPOT, OrderSide.SELL, 2.0)]
    assert all(o.reduce_only for o in orders)


# --------------------------------------------------------------------------- #
# Staat
# --------------------------------------------------------------------------- #
def test_the_state_round_trips_exactly(world, tmp_path: Path):
    _, _, ctx, _ = world
    book = new_book(ctx)
    for t in range(START):
        book.step(ctx.arrays.bar(t))
    book.goal[0] = 0.0125
    st = snapshot(book, candidate=NAME, config_hash=ctx.config_hash, mode="paper",
                  last_bar=ctx.arrays.index[START - 1])
    save_state(st, tmp_path / "s.json")
    back = load_state(tmp_path / "s.json")
    assert back == st
    fresh = new_book(ctx)
    restore(back, fresh, config_hash=ctx.config_hash)
    assert np.array_equal(fresh.a, book.a) and np.array_equal(fresh.b, book.b)
    assert np.array_equal(fresh.goal, book.goal, equal_nan=True)
    assert fresh.equity == book.equity and fresh.min_headroom == book.min_headroom


def test_a_fresh_book_stores_no_headroom_and_restores_infinity(world):
    _, _, ctx, _ = world
    st = snapshot(new_book(ctx), candidate=NAME, config_hash="h", mode="paper",
                  last_bar=ctx.arrays.index[0])
    assert st.min_headroom is None and st.a == {} and st.goal == {}
    book = new_book(ctx)
    restore(st, book, config_hash="h")
    assert book.min_headroom == float("inf")


def test_the_state_refuses_another_config_or_unknown_coins(world, tmp_path: Path):
    _, _, ctx, _ = world
    st = snapshot(new_book(ctx), candidate=NAME, config_hash="h", mode="paper",
                  last_bar=ctx.arrays.index[0])
    with pytest.raises(DataContractError, match="andere config"):
        restore(st, new_book(ctx), config_hash="other")
    ghost = replace(st, b={"GHOSTUSDT": 0.1})
    with pytest.raises(DataContractError, match="niet meer kent"):
        restore(ghost, new_book(ctx), config_hash="h")
    with pytest.raises(DataContractError, match="Geen staat"):
        load_state(tmp_path / "missing.json")
    (tmp_path / "v.json").write_text(json.dumps({"version": 99}), encoding="utf-8")
    with pytest.raises(DataContractError, match="Onbekende versie"):
        load_state(tmp_path / "v.json")


def test_a_halted_state_does_not_step(world):
    _, _, ctx, _ = world
    st = init(ctx, start=_start(ctx)).with_halt("kill switch")
    assert st.halted == "kill switch"
    with pytest.raises(DataContractError, match="staat stil"):
        step(ctx, st, PaperVenue())


def test_step_refuses_a_foreign_state(world):
    _, _, ctx, _ = world
    st = init(ctx, start=_start(ctx))
    with pytest.raises(DataContractError, match="andere kandidaat"):
        step(ctx, replace(st, candidate="X"), PaperVenue())
    with pytest.raises(DataContractError, match="wisselt niet van modus"):
        step(ctx, replace(st, mode="live"), PaperVenue())
    with pytest.raises(DataContractError, match="niet in het paneel"):
        step(ctx, replace(st, last_bar="1999-01-01T00:00:00+00:00"), PaperVenue())


# --------------------------------------------------------------------------- #
# Venue
# --------------------------------------------------------------------------- #
def test_reconciliation_tolerates_noise_and_flags_what_only_one_book_has():
    venue = VenueBook(spot={"AUSDT": 1.004, "BUSDT": 3.0}, perp_short={"AUSDT": 0.5},
                      equity_usd=1e3)
    breaks = reconcile_book({"AUSDT": 1.0}, {"AUSDT": 1.0}, venue, prices={"AUSDT": 100.0})
    assert breaks == ["BUSDT spot: expected 0, venue 3",
                      "AUSDT perp: expected 1, venue 0.5"]
    # Onder de dollardrempel is een afwijking geen breuk, ook boven rel_tol.
    assert reconcile_book({"AUSDT": 1.0}, {}, VenueBook({"AUSDT": 1.5}, {}, 1e3),
                          prices={"AUSDT": 10.0}) == []


def test_the_paper_venue_fills_everything_at_the_reference_price():
    venue = PaperVenue()
    order = LegOrder("AUSDT", Leg.SPOT, OrderSide.BUY, 2.0, 20.0, 10.0, False)
    fills = venue.execute([order])
    assert fills == [LegFill(order=order, qty=2.0, price=10.0, fee_usd=0.0)]
    venue.sync({"AUSDT": 2.0}, {}, 99.0)
    assert venue.book() == VenueBook(spot={"AUSDT": 2.0}, perp_short={}, equity_usd=99.0)


@dataclass
class _Exchange:
    """Een nep-beurs met een echte stand: vult wat hij krijgt (of een deel ervan)."""

    spot: dict = field(default_factory=dict)
    perp_short: dict = field(default_factory=dict)
    equity_usd: float = 1e6
    fill_ratio: float = 1.0
    is_paper: bool = False
    executed: list = field(default_factory=list)

    def book(self) -> VenueBook:
        return VenueBook(dict(self.spot), dict(self.perp_short), self.equity_usd)

    def execute(self, orders):
        out = []
        for o in orders:
            q = o.qty * self.fill_ratio
            side = self.spot if o.leg is Leg.SPOT else self.perp_short
            grows = (o.side is OrderSide.BUY) == (o.leg is Leg.SPOT)
            side[o.symbol] = side.get(o.symbol, 0.0) + (q if grows else -q)
            if abs(side[o.symbol]) < 1e-12:
                del side[o.symbol]
            out.append(LegFill(order=o, qty=q, price=o.ref_price, fee_usd=0.0))
        self.executed.append(orders)
        return out


def _live(ctx, start: int):
    st = init(ctx, start=str(ctx.arrays.index[start].date()), mode="live")
    assert st.expected_spot == {} and st.expected_perp == {}
    return st


def test_a_live_book_builds_from_the_exchange_and_reconciles(world):
    _, _, ctx, _ = world
    n = len(ctx.arrays.index)
    st, ex = _live(ctx, START), _Exchange()
    for t in range(START, START + 8):
        records, st = step(ctx, st, ex, settle=n - 1 - t)
        assert records[0]["mode"] == "live"
    assert ex.spot and ex.perp_short, "het boek bouwt de backtestpositie op"
    assert st.expected_spot == ex.spot and st.expected_perp == ex.perp_short
    # Elke batch in de margeveilige volgorde: short terug, spot weg, spot erbij, short erbij.
    rank = {(Leg.PERP, OrderSide.BUY): 0, (Leg.SPOT, OrderSide.SELL): 1,
            (Leg.SPOT, OrderSide.BUY): 2, (Leg.PERP, OrderSide.SELL): 3}
    for batch in ex.executed:
        ranks = [rank[(o.leg, o.side)] for o in batch]
        assert ranks == sorted(ranks)


def test_auto_deleveraging_breaks_the_reconciliation(world):
    """ADL kort de winnende short in zonder order van het boek: de volgende run stopt."""
    _, _, ctx, _ = world
    n = len(ctx.arrays.index)
    st, ex = _live(ctx, START), _Exchange()
    for t in range(START, START + 6):
        _, st = step(ctx, st, ex, settle=n - 1 - t)
    sym = next(iter(ex.perp_short))
    ex.perp_short[sym] *= 0.7
    with pytest.raises(ReconciliationBreak, match=f"{sym} perp"):
        step(ctx, st, ex, settle=n - 1 - (START + 6))


def test_a_partial_fill_is_the_new_truth_not_a_break(world):
    _, _, ctx, _ = world
    n = len(ctx.arrays.index)
    st, ex = _live(ctx, START), _Exchange(fill_ratio=0.5)
    for t in range(START, START + 4):
        _, st = step(ctx, st, ex, settle=n - 1 - t)
    assert st.expected_spot == ex.spot


def test_live_never_trades_on_an_old_bar(world):
    _, _, ctx, _ = world
    st = _live(ctx, START)
    with pytest.raises(DataContractError, match="alleen de nieuwste bar"):
        step(ctx, st, _Exchange(), settle=0)


def test_decide_live_refuses_a_broken_exchange_report(world):
    _, _, ctx, _ = world
    book = new_book(ctx)
    bar = ctx.arrays.bar(START)
    n = len(ctx.arrays.symbols)
    with pytest.raises(DataContractError, match="positieve equity"):
        book.decide_live(bar, np.zeros(n), np.zeros(n), 0.0)
    with pytest.raises(DataContractError, match="gemarkte benen"):
        book.decide_live(bar, -np.ones(n), np.zeros(n), 1e6)


def test_the_kill_switch_flattens_and_stops(world):
    _, _, ctx, _ = world
    st = init(ctx, start=_start(ctx))
    venue = PaperVenue(spot=dict(st.expected_spot), perp_short=dict(st.expected_perp),
                       equity_usd=st.equity)
    assert venue.spot, "er moet iets plat te maken zijn"
    px = dict(zip(ctx.arrays.symbols, ctx.spot_px[START], strict=True))
    pp = dict(zip(ctx.arrays.symbols, ctx.perp_px[START], strict=True))
    fills, halted = unwind(st, venue, spot_price=px, perp_price=pp, reason="test")
    assert halted.halted == "test" and venue.book().spot == {} and venue.book().perp_short == {}
    assert [f.order.leg for f in fills][: len(st.expected_perp)] == [Leg.PERP] * len(
        st.expected_perp)
    ex = _Exchange(spot={"AUSDT": 1.0}, perp_short={"AUSDT": 1.0})
    _, halted = unwind(st, ex, spot_price={"AUSDT": 2.0}, perp_price={"AUSDT": 2.0},
                       reason="live")
    assert ex.spot == {} and ex.perp_short == {} and halted.halted == "live"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_the_cli_runs_a_shadow_book_end_to_end(world, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch, capsys):
    _, _, ctx, backtest = world
    monkeypatch.setattr(cli, "load_context", lambda root, panels: ctx)
    state = tmp_path / "shadow" / "state.json"
    out = cli.main(["init", "--state", str(state), "--start", _start(ctx)])
    assert out["initialised_through"] == ctx.arrays.index[START - 1].isoformat()
    out = cli.main(["step", "--state", str(state)])
    assert out["bars"] == len(ctx.arrays.index) - START - 7      # settle = 7
    log = state.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["net"] for x in log] == backtest.frame["net"].iloc[
        START:START + out["bars"]].tolist()
    status = cli.main(["status", "--state", str(state)])
    assert status["halted"] is None and status["mode"] == "paper"
    out = cli.main(["unwind", "--state", str(state), "--reason", "manual"])
    assert out["halted"] == "manual"
    assert load_state(state).halted == "manual"
    with pytest.raises(SystemExit):
        cli.main(["unwind", "--state", str(state)])
    assert "vraagt een --reason" in capsys.readouterr().err


def test_init_defaults_to_the_frozen_forward_start(world, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    _, _, ctx, _ = world
    fwd = replace(ctx, cfg=ctx.cfg.model_copy(update={"forward": ctx.cfg.forward.model_copy(
        update={"start": _start(ctx)})}))
    monkeypatch.setattr(cli, "load_context", lambda root, panels: fwd)
    out = cli.main(["init", "--state", str(tmp_path / "s.json")])
    assert out["initialised_through"] == ctx.arrays.index[START - 1].isoformat()


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script(world, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                     capsys):
    _, _, ctx, _ = world
    path = tmp_path / "s.json"
    save_state(init(ctx, start=_start(ctx)), path)
    monkeypatch.setattr(sys, "argv", ["tradebot.carry", "status", "--state", str(path)])
    runpy.run_module("tradebot.carry", run_name="__main__")
    assert json.loads(capsys.readouterr().out)["mode"] == "paper"


def test_load_context_reads_the_frozen_config_and_the_panels(monkeypatch: pytest.MonkeyPatch,
                                                             world):
    m, cfg, _, _ = world
    seen = {}

    def fake_market(root, c, *, panel_root=None):
        seen["panel_root"] = panel_root
        return m
    monkeypatch.setattr(runner.v6, "load_market", fake_market)
    monkeypatch.setattr(runner.v6, "costs_for", lambda root, c: COSTS)
    monkeypatch.setattr(runner, "robust_book_v8_config", lambda path: cfg)
    ctx = runner.load_context(Path("/repo"), Path("/panels"))
    assert seen["panel_root"] == Path("/panels") and ctx.candidate == NAME


def test_a_state_as_record_is_json(world):
    _, _, ctx, _ = world
    st = init(ctx, start=_start(ctx))
    rec = st.as_record()
    assert rec["version"] == 1 and isinstance(json.dumps(rec, allow_nan=False), str)
    assert CarryState(**{k: v for k, v in rec.items() if k != "version"}) == st


def test_a_fill_that_closes_a_leg_removes_it_from_the_expected_book():
    close = LegOrder("AUSDT", Leg.PERP, OrderSide.BUY, 2.0, 20.0, 10.0, True)
    add = LegOrder("BUSDT", Leg.SPOT, OrderSide.BUY, 1.0, 5.0, 5.0, False)
    spot, perp = runner.after_fills({"AUSDT": 2.0}, {"AUSDT": 2.0}, [
        LegFill(close, 2.0, 10.0, 0.0), LegFill(add, 1.0, 5.0, 0.0)])
    assert perp == {} and spot == {"AUSDT": 2.0, "BUSDT": 1.0}
