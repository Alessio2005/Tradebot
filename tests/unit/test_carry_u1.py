"""Het Kraken-boek (`carry/u1.py`): shadow, de live-stap, de staat en de CLI, zonder netwerk.

Een synthetische Kraken-markt met twee carryregimes: hoge funding (~26 %/jaar) van maart tot
en met juni 2026 en weer vanaf 5 oktober, daartussen bijna niets. De H3-regels moeten
daarin instappen en uitstappen; de live-stap moet precies hun beslisregel volgen.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_carry_kraken import FakeKraken
from tradebot.carry import u1
from tradebot.carry.orders import Leg
from tradebot.carry.venue import PaperVenue, ReconciliationBreak, VenueBook
from tradebot.data import external_series as ex
from tradebot.data import kraken_feed as kf
from tradebot.schemas.robust_book_v2 import robust_book_v5_config
from tradebot.systematic.harvest import basis_decide
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]
NOW = pd.Timestamp("2026-10-27 06:00", tz="UTC")


class _Resp:
    def __init__(self, payload: dict | None = None, text: str = "") -> None:
        self._payload, self.text = payload, text

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


def _rate(h: pd.Timestamp) -> float:
    hot = (pd.Timestamp("2026-03-01", tz="UTC") <= h < pd.Timestamp("2026-07-01", tz="UTC")
           or h >= pd.Timestamp("2026-10-05", tz="UTC"))
    return 3e-5 if hot else 1e-6


def kraken_get(now: pd.Timestamp):
    """Een nep-Kraken (en nep-FRED) zoals de wereld eruitzag op `now`."""
    starts = pd.date_range(pd.Timestamp("2025-06-01", tz="UTC"), now.floor("D"), freq="D")
    px = {s: b * np.exp(np.cumsum(np.random.default_rng(i).normal(0.0, 0.02, len(starts))))
          for i, (s, b) in enumerate((("BTCUSDT", 60_000.0), ("ETHUSDT", 3_000.0)))}

    def get(url: str, params: dict[str, str]) -> _Resp:
        if url == ex.FRED_URL:
            days = pd.date_range("2019-09-01", now.tz_convert(None).floor("D")
                                 - pd.Timedelta(days=1), freq="D")
            return _Resp(text="observation_date,DTB3\n"
                         + "".join(f"{d.date()},4.0\n" for d in days))
        if url == kf.FUNDING_URL:
            hours = pd.date_range(pd.Timestamp("2025-10-15", tz="UTC"), now, freq="h")
            return _Resp({"result": "success", "rates": [
                {"timestamp": h.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                 "relativeFundingRate": _rate(h)} for h in hours]})
        if url == kf.SPOT_OHLC_URL:
            sym = "BTCUSDT" if params["pair"] == "XBTUSD" else "ETHUSDT"
            rows = [[int(t.timestamp()), str(p), str(p * 1.01), str(p * 0.99), str(p), str(p),
                     str(5e7 / p), 1] for t, p in zip(starts, px[sym])]
            return _Resp({"error": [], "result": {"X": rows[-720:], "last": 1}})
        sym = "BTCUSDT" if "XBT" in url else "ETHUSDT"
        frm, to = int(params["from"]), int(params["to"])
        return _Resp({"candles": [
            {"time": int(t.timestamp()) * 1000, "open": str(p), "high": str(p * 1.01),
             "low": str(p * 0.99), "close": str(p), "volume": 5e7 / p}
            for t, p in zip(starts, px[sym]) if frm <= t.timestamp() <= to],
            "more_candles": False})
    return get


_get = kraken_get(NOW)


@pytest.fixture(scope="module")
def ctx() -> u1.U1Context:
    panels = kf.fetch_panels(_get, now=NOW)
    cfg = robust_book_v5_config(ROOT / "conf/model/robust_book_v5.yaml")
    return u1.build_context(ROOT, kf.kraken_market(panels, cfg))


@pytest.fixture(scope="module")
def dtb3() -> pd.Series:
    return ex.fetch_dtb3(pd.Timestamp("2026-10-27"), _get)


def test_the_context_is_h3_with_krakens_fees(ctx):
    assert ctx.symbols == ("BTCUSDT", "ETHUSDT")
    assert ctx.costs.spot.taker_fee == pytest.approx(22e-4)
    assert ctx.costs.perp.taker_fee == pytest.approx(5e-4)
    held = ctx.target.loc["2026-05-01 00:00+00:00"]
    assert held.tolist() == [0.35, 0.35]              # twee slots van 35 % in de carry
    assert ctx.target.loc["2026-09-01 00:00+00:00"].tolist() == [0.0, 0.0]   # uitgestapt
    res = u1.simulate(ctx)
    assert res.audit["n_liquidations"] == 0
    assert res.frame["spot_notional"].loc["2026-05-01 00:00+00:00"] > 0.6


def test_the_shadow_report_measures_from_the_forward_start(ctx, dtb3):
    rep = u1.shadow(ctx, dtb3)
    assert rep["as_of"].startswith("2026-10-27") and rep["forward_days"] == 27
    assert rep["daily"].index[0] == pd.Timestamp("2026-10-01", tz="UTC")
    assert rep["now"]["target"] == {"BTCUSDT": 0.35, "ETHUSDT": 0.35}
    assert rep["now"]["carry_apr"]["BTCUSDT"] > 0.15
    f = rep["forward"]
    assert f["return"] == pytest.approx((1 + rep["daily"]["net"]).prod() - 1)
    assert f["book_only"] + f["cash"] == pytest.approx(rep["daily"]["net"].sum())
    md = u1.render(rep)
    assert "in carry" in md and "| BTCUSDT |" in md and "2026 (" in md
    flat = {**rep, "now": {**rep["now"], "spot_notional": 0.0,
                           "carry_apr": {"BTCUSDT": None, "ETHUSDT": 0.01}}}
    assert "staat in kas" in u1.render(flat) and "| — |" in u1.render(flat)
    empty = u1.shadow(ctx, dtb3, start="2027-01-01")
    assert empty["forward_days"] == 0 and empty["forward"]["return"] == 0.0


def _state(ctx, **kw) -> u1.U1State:
    return u1.U1State(config_hash=ctx.config_hash, mode="dry_run", capital_usd=1e6, **kw)


def test_a_flat_book_with_carry_enters_with_the_backtests_rule(ctx):
    venue = PaperVenue(equity_usd=1e6)
    rec, st = u1.step(ctx, _state(ctx), venue)
    assert rec["action"] == "gehandeld"
    t = len(ctx.market.index) - 1
    new_a, new_b, da, db = basis_decide(
        np.zeros(2), np.zeros(2), ctx.target.iloc[t].to_numpy(), np.ones(2, dtype=bool),
        skip=False, band=ctx.v5.basis.band, hedge_tolerance=ctx.v5.basis.hedge_tolerance,
        margin_floor=ctx.v5.basis.margin_floor)
    spot = {o["symbol"]: o["notional_usd"] for o in rec["orders"] if o["leg"] == "spot"}
    assert spot == pytest.approx({"BTCUSDT": da[0] * 1e6, "ETHUSDT": da[1] * 1e6})
    # Eerst spot kopen (onderpand), dan de short openen.
    assert [o["leg"] for o in rec["orders"]] == ["spot", "spot", "perp", "perp"]
    assert st.expected_spot["BTCUSDT"] == pytest.approx(0.35e6 / ctx.spot_px[t, 0])
    assert st.last_bar == ctx.market.index[t].isoformat() and st.committed == (1.0,)
    again, same = u1.step(ctx, st, venue)
    assert again["action"] == "geen nieuwe bar" and same is st


def test_a_held_book_inside_the_band_does_not_trade(ctx):
    t = len(ctx.market.index) - 1
    q = {s: 0.35e6 / ctx.spot_px[t, i] for i, s in enumerate(ctx.symbols)}
    venue = PaperVenue(spot=dict(q), perp_short=dict(q), equity_usd=1e6)
    st = _state(ctx, expected_spot=q, expected_perp=q,
                last_bar=ctx.market.index[t - 1].isoformat())
    rec, _ = u1.step(ctx, st, venue)
    assert rec["action"] == "geen handel nodig" and rec["orders"] == []


def test_too_little_on_the_exchange_asks_for_a_transfer_and_does_not_trade(ctx):
    venue = PaperVenue(equity_usd=10_000.0)
    rec, st = u1.step(ctx, _state(ctx), venue)
    assert rec["action"] == "wacht op overboeking" and venue.fills == []
    assert rec["funding_request_usd"] == pytest.approx(1e6 * 1.0 - 10_000.0)
    assert st.expected_spot == {} and st.last_bar is not None


def test_waiting_for_cash_still_reduces_risk(ctx):
    t = len(ctx.market.index) - 1
    swap = ctx.target.copy()
    swap.iloc[t] = [0.0, 0.35]                      # BTC eruit, ETH erin
    moved = u1.U1Context(**{**ctx.__dict__, "target": swap})
    q = 0.35e6 / ctx.spot_px[t, 0]
    venue = PaperVenue(spot={"BTCUSDT": q}, perp_short={"BTCUSDT": q}, equity_usd=10_000.0)
    st = _state(ctx, expected_spot={"BTCUSDT": q}, expected_perp={"BTCUSDT": q},
                last_bar=ctx.market.index[t - 1].isoformat())
    rec, st2 = u1.step(moved, st, venue)
    assert rec["action"] == "wacht op overboeking"
    assert [(f.order.symbol, f.order.reduce_only) for f in venue.fills] == [
        ("BTCUSDT", True), ("BTCUSDT", True)]       # de BTC-carry is dicht, ETH wacht
    assert st2.expected_spot == {} and st2.expected_perp == {}
    assert len(rec["orders"]) == 4 and rec["funding_request_usd"] > 0.0


def test_a_week_without_need_sends_the_cash_to_the_fund(ctx):
    flat = ctx.target.copy() * 0.0
    quiet = u1.U1Context(**{**ctx.__dict__, "target": flat})
    st = _state(ctx, committed=(0.0,) * 7)
    rec, st2 = u1.step(quiet, st, PaperVenue(equity_usd=400_000.0))
    assert rec["sweep_out_usd"] == pytest.approx(400_000.0 - 0.05e6)
    assert st2.committed[-8:] == (0.0,) * 8
    rec, _ = u1.step(quiet, _state(ctx, committed=(1.0,) + (0.0,) * 6),
                     PaperVenue(equity_usd=400_000.0))
    assert rec["sweep_out_usd"] == 0.0     # nog geen volle week zonder gebruik


def test_a_venue_that_differs_from_the_book_stops_it(ctx):
    st = _state(ctx, expected_spot={"BTCUSDT": 5.0}, expected_perp={"BTCUSDT": 5.0})
    with pytest.raises(ReconciliationBreak, match="BTCUSDT spot"):
        u1.step(ctx, st, PaperVenue(spot={"BTCUSDT": 4.0}, perp_short={"BTCUSDT": 5.0},
                                    equity_usd=1e6))


def test_a_halted_book_or_a_foreign_config_is_refused(ctx):
    with pytest.raises(DataContractError, match="staat stil"):
        u1.step(ctx, _state(ctx, halted="test"), PaperVenue())
    with pytest.raises(DataContractError, match="andere config"):
        u1.step(ctx, u1.U1State(config_hash="x", mode="dry_run", capital_usd=1.0),
                PaperVenue())


def test_unwind_closes_shorts_first_and_halts(ctx):
    venue = PaperVenue(spot={"BTCUSDT": 2.0}, perp_short={"BTCUSDT": 2.0}, equity_usd=1e6)
    fills, st = u1.unwind(ctx, _state(ctx), venue, reason="handmatig")
    assert [(f.order.leg, f.order.reduce_only) for f in fills] == [(Leg.PERP, True),
                                                                   (Leg.SPOT, True)]
    assert st.halted == "handmatig" and st.expected_spot == {}
    with pytest.raises(DataContractError, match="zonder reden"):
        u1.unwind(ctx, _state(ctx), venue, reason=" ")


def test_the_state_round_trips_and_refuses_other_fields(ctx, tmp_path: Path):
    st = _state(ctx, expected_spot={"BTCUSDT": 1.5}, committed=(0.0, 1.0))
    u1.save_u1_state(st, tmp_path / "s.json")
    assert u1.load_u1_state(tmp_path / "s.json") == st
    (tmp_path / "bad.json").write_text(json.dumps({"mode": "x"}), encoding="utf-8")
    with pytest.raises(DataContractError, match="andere velden"):
        u1.load_u1_state(tmp_path / "bad.json")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli(args: list[str], fake: FakeKraken | None = None) -> dict:
    return u1.main(args, root=ROOT, get=_get, http=fake or FakeKraken(), now=NOW)


def test_cli_shadow_writes_the_report(tmp_path: Path, capsys):
    out = _cli(["shadow", "--out", str(tmp_path)])
    assert out["forward_days"] == 27 and "daily" not in out
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") in capsys.readouterr().out
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["as_of"]
    assert len(pd.read_csv(tmp_path / "shadow_daily.csv")) == 27
    assert _cli(["shadow"])["as_of"] == out["as_of"]


def test_cli_dry_run_step_status_and_unwind(tmp_path: Path, capsys):
    state = tmp_path / "u1.json"
    fake = FakeKraken()
    rec = _cli(["step", "--state", str(state), "--capital-usd", "100000"], fake)
    assert rec["action"] == "gehandeld" and len(rec["fills"]) == 4
    # Dry-run: alleen de publieke metadata van Kraken, geen enkele privé-aanroep.
    assert {p for p in fake.paths() if "private" in p or "/api/v3/" in p} == {
        "/derivatives/api/v3/instruments"}
    assert _cli(["step", "--state", str(state)])["action"] == "geen nieuwe bar"
    st = _cli(["status", "--state", str(state)])
    assert st["mode"] == "dry_run" and st["expected_spot"]
    with pytest.raises(DataContractError, match="--reason"):
        _cli(["unwind", "--state", str(state)])
    out = _cli(["unwind", "--state", str(state), "--reason", "test"])
    assert out == {"closed_legs": 4, "halted": "test"}
    with pytest.raises(DataContractError, match="wisselt niet van modus"):
        _cli(["step", "--state", str(state), "--live"])
    capsys.readouterr()


@pytest.mark.parametrize("args", [["status"], ["step"], ["step", "--capital-usd", "0"],
                                  ["unwind", "--reason", "x"]])
def test_cli_refuses_incomplete_commands(tmp_path: Path, args):
    full = args if args == ["status"] else [args[0], "--state", str(tmp_path / "new.json"),
                                            *args[1:]]
    with pytest.raises(DataContractError, match="vraagt"):
        _cli(full)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["u1", "status"])
    with pytest.raises(DataContractError, match="vraagt --state"):
        runpy.run_module("tradebot.carry.u1", run_name="__main__")


def test_live_needs_all_four_keys_and_then_builds_a_live_venue(ctx, monkeypatch):
    st = _state(ctx)
    for k in ("KRAKEN_SPOT_KEY", "KRAKEN_SPOT_SECRET", "KRAKEN_FUTURES_KEY",
              "KRAKEN_FUTURES_SECRET"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(DataContractError, match="sleutels"):
        u1._venue(ctx, st, live=True, http=None)
    for k in ("KRAKEN_SPOT_KEY", "KRAKEN_SPOT_SECRET", "KRAKEN_FUTURES_KEY",
              "KRAKEN_FUTURES_SECRET"):
        monkeypatch.setenv(k, "a2V5")
    venue = u1._venue(ctx, st, live=True, http=None)
    assert not venue.dry_run and venue.spot_key == "a2V5"
    dry = u1._venue(ctx, _state(ctx, expected_spot={"BTCUSDT": 1.0}), live=False, http=None)
    assert dry.dry_run and dry.book() == VenueBook(spot={"BTCUSDT": 1.0}, perp_short={},
                                                    equity_usd=1e6)
