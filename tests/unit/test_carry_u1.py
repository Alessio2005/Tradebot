"""Het Kraken-boek (`carry/u1.py`): shadow, de live-stap, de staat en de CLI, zonder netwerk.

Een synthetische Kraken-markt met twee carryregimes: hoge funding (~26 %/jaar) van maart tot
en met juni 2026 en weer vanaf 5 oktober, daartussen bijna niets. De H3-regels moeten
daarin instappen en uitstappen; de live-stap moet precies hun beslisregel volgen.
"""
from __future__ import annotations

import json
import runpy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_carry_kraken import FakeKraken
from tradebot.carry import u1
from tradebot.carry.kraken import KrakenError
from tradebot.carry.orders import Leg
from tradebot.carry.safety import Deferred, Limits, SafetyBreach, SingleInstance
from tradebot.carry.venue import LegFill, PaperVenue, ReconciliationBreak, VenueBook
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


#: Ruime grenzen: deze tests toetsen het boek, niet de grenzen (die staan hieronder apart).
LIMITS = Limits(max_capital_usd=1e7, max_order_usd=1e7, max_gross=1.5,
                max_price_deviation=0.05, hedge_tolerance_usd=50.0, max_step_loss=0.02,
                margin_warn=1.6, margin_critical=1.2)


@dataclass
class FakeVenue(PaperVenue):
    """De paper-venue met wat de echte adapter ook heeft (koersen, klok) en fouten op afroep."""

    spot_px: dict[str, float] = field(default_factory=dict)
    perp_px: dict[str, float] = field(default_factory=dict)
    time: float = 0.0
    fail_at: int | None = None        # de zoveelste order (0-gebaseerd) faalt
    book_fails_after: int | None = None
    frozen: bool = False              # vult wel, maar de stand verandert niet
    n_orders: int = 0
    n_books: int = 0

    def prices(self) -> tuple[dict[str, float], dict[str, float]]:
        return dict(self.spot_px), dict(self.perp_px)

    def server_time(self) -> float:
        return self.time

    def book(self) -> VenueBook:
        self.n_books += 1
        if self.book_fails_after is not None and self.n_books > self.book_fails_after:
            raise KrakenError("EService:Unavailable")
        return super().book()

    def execute(self, orders):
        out = []
        for o in orders:
            if self.fail_at is not None and self.n_orders == self.fail_at:
                self.n_orders += 1
                raise KrakenError("EOrder:Insufficient margin")
            self.n_orders += 1
            if self.frozen:
                out.append(LegFill(order=o, qty=o.qty, price=o.ref_price, fee_usd=0.0))
            else:
                out += super().execute([o])
        return out


def _state(ctx, **kw) -> u1.U1State:
    return u1.U1State(config_hash=ctx.config_hash, mode="dry_run", capital_usd=1e6, **kw)


def _live(ctx) -> tuple[dict[str, float], dict[str, float]]:
    t = len(ctx.market.index) - 1
    return (dict(zip(ctx.symbols, ctx.spot_px[t], strict=True)),
            dict(zip(ctx.symbols, ctx.perp_px[t], strict=True)))


def test_a_flat_book_with_carry_enters_with_the_backtests_rule(ctx):
    venue = PaperVenue(equity_usd=1e6)
    rec, st = u1.step(ctx, _state(ctx), venue, limits=LIMITS, live_prices=_live(ctx))
    assert rec["action"] == "gehandeld" and st.halted is None
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
    assert rec["post_trade"] == {"breaks": [], "hedge_gaps": {}}
    assert set(st.marks) == {"BTCUSDT", "ETHUSDT"} and rec["gross_after"] == pytest.approx(1.4)
    again, same = u1.step(ctx, st, venue, limits=LIMITS)
    assert again["action"] == "geen nieuwe bar" and same is st


def test_a_held_book_inside_the_band_does_not_trade(ctx):
    t = len(ctx.market.index) - 1
    q = {s: 0.35e6 / ctx.spot_px[t, i] for i, s in enumerate(ctx.symbols)}
    venue = PaperVenue(spot=dict(q), perp_short=dict(q), equity_usd=1e6)
    st = _state(ctx, expected_spot=q, expected_perp=q,
                last_bar=ctx.market.index[t - 1].isoformat())
    rec, _ = u1.step(ctx, st, venue, limits=LIMITS)
    assert rec["action"] == "geen handel nodig" and rec["orders"] == []


def test_too_little_on_the_exchange_asks_for_a_transfer_and_does_not_trade(ctx):
    venue = PaperVenue(equity_usd=10_000.0)
    rec, st = u1.step(ctx, _state(ctx), venue, limits=LIMITS)
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
    rec, st2 = u1.step(moved, st, venue, limits=LIMITS)
    assert rec["action"] == "wacht op overboeking"
    assert [(f.order.symbol, f.order.reduce_only) for f in venue.fills] == [
        ("BTCUSDT", True), ("BTCUSDT", True)]       # de BTC-carry is dicht, ETH wacht
    assert st2.expected_spot == {} and st2.expected_perp == {} and st2.halted is None
    assert len(rec["orders"]) == 4 and rec["funding_request_usd"] > 0.0


def test_a_week_without_need_sends_the_cash_to_the_fund(ctx):
    flat = ctx.target.copy() * 0.0
    quiet = u1.U1Context(**{**ctx.__dict__, "target": flat})
    st = _state(ctx, committed=(0.0,) * 7)
    rec, st2 = u1.step(quiet, st, PaperVenue(equity_usd=400_000.0), limits=LIMITS)
    assert rec["sweep_out_usd"] == pytest.approx(400_000.0 - 0.05e6)
    assert st2.committed[-8:] == (0.0,) * 8
    rec, _ = u1.step(quiet, _state(ctx, committed=(1.0,) + (0.0,) * 6),
                     PaperVenue(equity_usd=400_000.0), limits=LIMITS)
    assert rec["sweep_out_usd"] == 0.0     # nog geen volle week zonder gebruik


def test_a_venue_that_differs_from_the_book_stops_it(ctx):
    st = _state(ctx, expected_spot={"BTCUSDT": 5.0}, expected_perp={"BTCUSDT": 5.0})
    with pytest.raises(ReconciliationBreak, match="BTCUSDT spot"):
        u1.step(ctx, st, PaperVenue(spot={"BTCUSDT": 4.0}, perp_short={"BTCUSDT": 5.0},
                                    equity_usd=1e6), limits=LIMITS)


def test_a_halted_book_or_a_foreign_config_is_refused(ctx):
    with pytest.raises(DataContractError, match="staat stil"):
        u1.step(ctx, _state(ctx, halted="test"), PaperVenue(), limits=LIMITS)
    with pytest.raises(DataContractError, match="andere config"):
        u1.step(ctx, u1.U1State(config_hash="x", mode="dry_run", capital_usd=1.0),
                PaperVenue(), limits=LIMITS)


# --------------------------------------------------------------------------- #
# De veiligheid in de stap
# --------------------------------------------------------------------------- #
def test_an_unhedged_loss_since_the_last_step_stops_the_book(ctx):
    t = len(ctx.market.index) - 1
    q = 0.35e6 / ctx.spot_px[t, 0]
    # Alleen spot (de short is weg, bv. door ADL) en de koers 10 % lager dan bij de vorige stap.
    then = {"BTCUSDT": (ctx.spot_px[t, 0] * 1.1, ctx.perp_px[t, 0] * 1.1)}
    st = _state(ctx, expected_spot={"BTCUSDT": q}, marks=then,
                last_bar=ctx.market.index[t - 1].isoformat())
    with pytest.raises(SafetyBreach, match="verloren"):
        u1.step(ctx, st, PaperVenue(spot={"BTCUSDT": q}, equity_usd=1e6), limits=LIMITS)
    hedged = _state(ctx, expected_spot={"BTCUSDT": q}, expected_perp={"BTCUSDT": q},
                    marks=then, last_bar=ctx.market.index[t - 1].isoformat())
    rec, _ = u1.step(ctx, hedged, PaperVenue(spot={"BTCUSDT": q}, perp_short={"BTCUSDT": q},
                                             equity_usd=1e6), limits=LIMITS)
    assert abs(rec["marked_pnl_usd"]) < 1e-6      # gehedged: de prijs doet niets


@pytest.mark.parametrize(("change", "match"), [
    ({"max_order_usd": 100_000.0}, "boven de maximale order"),
    ({"max_capital_usd": 500_000.0}, "boven het plafond"),
    ({"max_gross": 1.0}, "bruto 1.40"),
])
def test_hard_limits_refuse_the_decision(ctx, change, match):
    lim = Limits(**{**LIMITS.__dict__, **change})
    venue = PaperVenue(equity_usd=1e6)
    with pytest.raises(SafetyBreach, match=match):
        u1.step(ctx, _state(ctx), venue, limits=lim)
    assert venue.fills == []


def test_a_live_price_far_from_the_close_defers_the_step(ctx):
    spot, perp = _live(ctx)
    shock = ({s: p * 1.08 for s, p in spot.items()}, perp)
    with pytest.raises(Deferred, match="wijkt af"):
        u1.step(ctx, _state(ctx), PaperVenue(equity_usd=1e6), limits=LIMITS,
                live_prices=shock)
    with pytest.raises(Deferred, match="geen live koers"):
        u1.step(ctx, _state(ctx), PaperVenue(equity_usd=1e6), limits=LIMITS,
                live_prices=({}, perp))


def test_a_failure_halfway_repairs_the_hedge_reduce_only_and_halts(ctx):
    venue = FakeVenue(equity_usd=1e6, fail_at=2)     # beide spotkopen lukken, de short niet
    rec, st = u1.step(ctx, _state(ctx), venue, limits=LIMITS)
    assert rec["action"] == "fout bij uitvoering" and "Insufficient margin" in rec["error"]
    assert len(rec["fills"]) == 2 and set(rec["hedge_gaps_before"]) == {"BTCUSDT", "ETHUSDT"}
    assert [o["reduce_only"] for o in rec["repair_orders"]] == [True, True]
    assert all(o["side"] == "SELL" and o["leg"] == "spot" for o in rec["repair_orders"])
    assert rec["hedge_gaps_after"] == {} and venue.spot == {} and venue.perp_short == {}
    assert st.halted.startswith("uitvoering mislukt") and st.expected_spot == {}


def test_a_venue_that_cannot_be_read_after_the_failure_still_halts(ctx):
    venue = FakeVenue(equity_usd=1e6, fail_at=2, book_fails_after=1)
    rec, st = u1.step(ctx, _state(ctx), venue, limits=LIMITS)
    assert "stand niet te lezen" in rec["repair_error"]
    assert st.halted.endswith("stand op de beurs onbekend")
    assert set(st.expected_spot) == {"BTCUSDT", "ETHUSDT"} and st.expected_perp == {}


def test_a_venue_that_cannot_be_read_after_trading_halts(ctx):
    venue = FakeVenue(equity_usd=1e6, book_fails_after=1)
    rec, st = u1.step(ctx, _state(ctx), venue, limits=LIMITS)
    assert rec["action"] == "gehandeld, controle mislukt"
    assert st.halted.startswith("na de uitvoering niet te lezen")
    assert set(st.expected_perp) == {"BTCUSDT", "ETHUSDT"}


def test_a_venue_that_does_not_show_the_fills_halts_after_trading(ctx):
    venue = FakeVenue(equity_usd=1e6, frozen=True)
    rec, st = u1.step(ctx, _state(ctx), venue, limits=LIMITS)
    assert rec["action"] == "gehandeld" and rec["post_trade"]["breaks"]
    assert st.halted.startswith("na de uitvoering wijkt de beurs af")


def test_unwind_closes_shorts_first_and_halts(ctx):
    venue = PaperVenue(spot={"BTCUSDT": 2.0}, perp_short={"BTCUSDT": 2.0}, equity_usd=1e6)
    rec, st = u1.unwind(ctx, _state(ctx), venue, reason="handmatig", limits=LIMITS)
    assert [(f.order.leg, f.order.reduce_only) for f in venue.fills] == [(Leg.PERP, True),
                                                                         (Leg.SPOT, True)]
    assert st.halted == "unwind: handmatig" and st.expected_spot == {}
    assert rec["error"] is None and rec["hedge_gaps_after"] == {}
    with pytest.raises(DataContractError, match="zonder reden"):
        u1.unwind(ctx, _state(ctx), venue, reason=" ", limits=LIMITS)


def test_an_unwind_that_fails_halfway_closes_the_open_short(ctx):
    t = len(ctx.market.index) - 1
    venue = FakeVenue(spot={"BTCUSDT": 2.0, "ETHUSDT": 30.0},
                      perp_short={"BTCUSDT": 2.0, "ETHUSDT": 30.0}, equity_usd=1e6, fail_at=1)
    rec, st = u1.unwind(ctx, _state(ctx), venue, reason="test", limits=LIMITS)
    assert "Insufficient margin" in rec["error"]
    # De BTC-short is terug, de ETH-short niet: de reparatie verkoopt de BTC-spot die nu
    # ongehedged is; de ETH-carry blijft gehedged staan.
    assert st.expected_spot == {"ETHUSDT": 30.0} and st.expected_perp == {"ETHUSDT": 30.0}
    assert st.halted == "unwind: test" and t > 0


def test_resume_takes_the_venue_as_the_truth(ctx):
    venue = PaperVenue(spot={"BTCUSDT": 1.0}, perp_short={"BTCUSDT": 1.0}, equity_usd=1e6)
    rec, st = u1.resume(_state(ctx, halted="x"), venue, reason="onderzocht")
    assert st.halted is None and st.expected_spot == {"BTCUSDT": 1.0}
    assert rec["was_halted"] == "x"
    with pytest.raises(DataContractError, match="staat niet stil"):
        u1.resume(_state(ctx), venue, reason="x")
    with pytest.raises(DataContractError, match="zonder reden"):
        u1.resume(_state(ctx, halted="x"), venue, reason="")
    rec, st = u1.resume(_state(ctx, halted="x"), venue, reason="nieuwe fees", config_hash="n")
    assert st.config_hash == "n" and rec["config_hash"] == {"was": ctx.config_hash, "now": "n"}


def test_cli_resume_can_accept_a_changed_config(ctx, tmp_path: Path):
    state = tmp_path / "u1.json"
    u1.save_u1_state(u1.U1State(config_hash="oud", mode="dry_run", capital_usd=1e5,
                                halted="config veranderd"), state)
    out = _cli(["resume", "--state", str(state), "--reason", "fees", "--accept-config"],
               _priced_fake(ctx))
    assert out["config_hash"] == {"was": "oud", "now": ctx.config_hash}
    assert u1.load_u1_state(state).config_hash == ctx.config_hash


def test_the_state_round_trips_keeps_backups_and_refuses_other_fields(ctx, tmp_path: Path):
    st = _state(ctx, expected_spot={"BTCUSDT": 1.5}, committed=(0.0, 1.0),
                marks={"BTCUSDT": (1.0, 2.0)})
    path = tmp_path / "s.json"
    for i in range(u1.BACKUPS + 3):
        u1.save_u1_state(st, path, now=pd.Timestamp("2026-10-31", tz="UTC")
                         + pd.Timedelta(seconds=i))
    assert u1.load_u1_state(path) == st
    assert len(list((tmp_path / "backups").glob("s.json.*"))) == u1.BACKUPS
    (tmp_path / "old.json").write_text(json.dumps(
        {"config_hash": "h", "mode": "dry_run", "capital_usd": 1.0}), encoding="utf-8")
    assert u1.load_u1_state(tmp_path / "old.json").marks == {}   # een oude staat laadt
    (tmp_path / "bad.json").write_text(json.dumps({"mode": "x"}), encoding="utf-8")
    with pytest.raises(DataContractError, match="andere velden"):
        u1.load_u1_state(tmp_path / "bad.json")
    u1.append_journal(tmp_path / "j.jsonl", {"a": 1})
    u1.append_journal(tmp_path / "j.jsonl", {"a": 2})
    assert (tmp_path / "j.jsonl").read_text(encoding="utf-8").splitlines() == [
        '{"a": 1}', '{"a": 2}']


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _priced_fake(ctx) -> FakeKraken:
    """Een nep-Kraken waarvan de live koersen gelijk zijn aan de laatste close."""
    fake = FakeKraken()
    spot, perp = _live(ctx)
    fake.spot_last = {"XXBTZUSD": spot["BTCUSDT"], "XETHZUSD": spot["ETHUSDT"]}
    fake.perp_mark = {"PF_XBTUSD": perp["BTCUSDT"], "PF_ETHUSD": perp["ETHUSDT"]}
    return fake


def _cli(args: list[str], fake: FakeKraken | None = None) -> dict:
    return u1.main(args, root=ROOT, get=_get, http=fake or FakeKraken(), now=NOW)


def test_cli_shadow_writes_the_report(tmp_path: Path, capsys):
    out = _cli(["shadow", "--out", str(tmp_path)])
    assert out["forward_days"] == 27 and "daily" not in out
    assert (tmp_path / "summary.md").read_text(encoding="utf-8") in capsys.readouterr().out
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["as_of"]
    assert len(pd.read_csv(tmp_path / "shadow_daily.csv")) == 27
    assert _cli(["shadow"])["as_of"] == out["as_of"]


def test_cli_dry_run_step_status_unwind_and_resume(ctx, tmp_path: Path, capsys):
    state = tmp_path / "u1.json"
    fake = _priced_fake(ctx)
    rec = _cli(["step", "--state", str(state), "--capital-usd", "100000"], fake)
    assert rec["action"] == "gehandeld" and len(rec["fills"]) == 4
    # Dry-run: alleen publieke Kraken-aanroepen, geen enkele privé-aanroep.
    assert not [p for p in fake.paths() if "private" in p or p.endswith(
        ("/sendorder", "/accounts", "/openpositions", "/withdrawal"))]
    assert _cli(["step", "--state", str(state)], fake)["action"] == "geen nieuwe bar"
    st = _cli(["status", "--state", str(state)])
    assert st["mode"] == "dry_run" and st["expected_spot"]
    journal = (tmp_path / "u1.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["command"] for x in journal] == ["step", "step"]
    with pytest.raises(DataContractError, match="--reason"):
        _cli(["unwind", "--state", str(state)], fake)
    out = _cli(["unwind", "--state", str(state), "--reason", "test"], fake)
    assert out["action"] == "unwind" and len(out["fills"]) == 4
    assert _cli(["status", "--state", str(state)])["halted"] == "unwind: test"
    with pytest.raises(DataContractError, match="--reason"):
        _cli(["resume", "--state", str(state)], fake)
    assert _cli(["resume", "--state", str(state), "--reason", "ok"], fake)["action"] == "hervat"
    with pytest.raises(DataContractError, match="wisselt niet van modus"):
        _cli(["step", "--state", str(state), "--live"], fake)
    capsys.readouterr()


def test_cli_takes_state_journal_and_mode_from_the_server_config(ctx, tmp_path: Path):
    import yaml
    raw = yaml.safe_load((ROOT / "conf/live/u1.example.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {k: str(tmp_path / Path(v).name) for k, v in raw["paths"].items()}
    cfg = tmp_path / "u1.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rec = _cli(["step", "--config", str(cfg), "--capital-usd", "50000"], _priced_fake(ctx))
    assert rec["action"] == "gehandeld"
    assert (tmp_path / "u1-state.json").is_file() and (tmp_path / "u1-journal.jsonl").is_file()
    assert _cli(["status", "--config", str(cfg)])["mode"] == "dry_run"
    raw["mode"] = "live"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(DataContractError, match="wisselt niet van modus"):
        _cli(["unwind", "--config", str(cfg), "--reason", "x"], _priced_fake(ctx))


def test_cli_refuses_a_second_process_on_the_same_state(ctx, tmp_path: Path):
    state = tmp_path / "u1.json"
    with SingleInstance(state.with_suffix(".lock")), pytest.raises(Deferred,
                                                                    match="ander proces"):
        _cli(["step", "--state", str(state), "--capital-usd", "1000"], _priced_fake(ctx))


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


def test_live_needs_all_four_keys_and_then_builds_a_live_venue(ctx):
    st = _state(ctx)
    with pytest.raises(DataContractError, match="sleutels"):
        u1.make_venue(ctx.symbols, st, live=True, env={})
    env = {k: "a2V5" for k in ("KRAKEN_SPOT_KEY", "KRAKEN_SPOT_SECRET", "KRAKEN_FUTURES_KEY",
                               "KRAKEN_FUTURES_SECRET")}
    venue = u1.make_venue(ctx.symbols, st, live=True, env=env)
    assert not venue.dry_run and venue.spot_key == "a2V5"
    dry = u1.make_venue(ctx.symbols, _state(ctx, expected_spot={"BTCUSDT": 1.0}), live=False)
    assert dry.dry_run and dry.book() == VenueBook(spot={"BTCUSDT": 1.0}, perp_short={},
                                                    equity_usd=1e6)
