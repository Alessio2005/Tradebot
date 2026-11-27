"""De synthetische replay-markt en de opslag ervan: deterministisch en verliesvrij."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, time

import pandas as pd
import pytest

from tradebot.options.calendar import NY, TradingCalendar
from tradebot.options.chain.builder import ChainBuilder
from tradebot.options.chain.filters import FilterSettings, filtered
from tradebot.options.contracts import Quote
from tradebot.options.data import synthetic
from tradebot.options.data.store import read_market, write_market
from tradebot.options.data.synthetic import (
    SyntheticSpec,
    SyntheticUnderlying,
    default_spec,
    generate,
)
from tradebot.options.pricing.settings import pricer_settings
from tradebot.options.pricing.surface import SurfaceStatus, build_surface
from tradebot.options.rates import Carry
from tradebot.schemas.options import load_options_config
from tradebot.utils.failfast import DataContractError

SMALL = SyntheticSpec(
    underlyings=(SyntheticUnderlying("TSTA", 100.0, jump_day=1, jump=-0.05, dividend_day=1,
                                     dividend=0.5, earnings_day=1),
                 SyntheticUnderlying("TSTB", 50.0, strike_step=1.0)),
    start=date(2026, 12, 1), n_days=2, snapshot_times=(time(10, 0), time(15, 0)),
    strike_range=0.08, thin_wing_k=0.06, max_days=40, warmup_days=25)


@pytest.fixture(scope="module")
def market():
    return generate(SMALL)


def test_the_market_is_deterministic_and_causal(market):
    again = generate(SMALL)
    assert [s.spot for s in again.snapshots] == [s.spot for s in market.snapshots]
    assert market.days == (date(2026, 12, 1), date(2026, 12, 2))
    assert [(s.asof.hour, s.underlying) for s in market.snapshots[:2]] == [(10, "TSTA"),
                                                                           (10, "TSTB")]
    bars = market.bars["TSTA"]
    bars = bars[bars.index <= pd.Timestamp(date(2026, 12, 1))]
    assert len(bars) == SMALL.warmup_days + 1 and bars.index[-1].date() == date(2026, 12, 1)
    ev = market.events["TSTA"]
    assert ev.earnings == (date(2026, 12, 2),) and ev.dividends[0].amount == 0.5
    assert market.events["TSTB"].earnings == () and market.events["TSTB"].dividends == ()


def test_quotes_are_in_cents_and_thin_wings_are_marked(market):
    qs = [r.quote for s in market.snapshots for r in s.options]
    assert all(q.bid == round(q.bid, 2) and q.ask > q.bid for q in qs)  # type: ignore[operator]
    assert {q.open_interest for q in qs} == {100, SMALL.open_interest}


def test_strikes_follow_the_step_per_underlying(market):
    a, b = market.snapshots[:2]
    assert {r.contract.strike % 2 for r in a.options} == {0.0}
    assert {r.contract.strike % 1 for r in b.options} == {0.0}
    assert len({r.contract.strike for r in b.options}) > 5


def test_the_synthetic_chain_gives_a_clean_surface_at_the_smile_base(market):
    snap = next(s for s in market.snapshots if s.underlying == "TSTB")
    cfg = load_options_config()
    chain = ChainBuilder(pricer_settings(cfg.pricing), 1e9).build(
        snap.underlying, snap.spot, snap.options, snap.asof, Carry(market.rate))
    chain, _ = filtered(chain, FilterSettings())
    surf = build_surface(chain)
    assert surf.status is SurfaceStatus.OK
    assert surf.atm_vol(30) == pytest.approx(SMALL.smile_base, abs=0.005)


def test_expiries_skip_to_thursday_on_a_friday_holiday():
    cal = TradingCalendar()
    spec = dataclasses.replace(SMALL, weekly_days=30, max_days=30)
    out = synthetic._expiries(cal, date(2027, 3, 15), spec)
    assert date(2027, 3, 25) in out and date(2027, 3, 26) not in out  # Goede Vrijdag
    monthly = synthetic._expiries(cal, date(2026, 12, 1), dataclasses.replace(
        SMALL, weekly_days=7, max_days=80))
    assert monthly == [date(2026, 12, 4), date(2026, 12, 18), date(2027, 1, 15),
                       date(2027, 2, 19)]


def test_bid_is_never_negative():
    q = synthetic._quote(0.001, SMALL, datetime(2026, 12, 1, 10, tzinfo=NY), thin=False)
    assert isinstance(q, Quote) and q.bid == 0.0 and q.ask == 0.01


def test_store_round_trip(market, tmp_path):
    write_market(market, tmp_path)
    back = read_market(tmp_path)
    assert back.rate == market.rate and back.days == market.days
    assert back.events == market.events
    assert back.snapshots == market.snapshots
    for k in market.bars:
        assert back.bars[k].equals(market.bars[k])


def test_store_keeps_missing_bids_and_unfetched_events(market, tmp_path):
    snap = market.snapshots[0]
    r0 = snap.options[0]
    nobid = dataclasses.replace(r0, quote=dataclasses.replace(r0.quote, bid=None, ask=None))
    snaps = (dataclasses.replace(snap, options=(nobid, *snap.options[1:])),
             *market.snapshots[1:])
    events = {**market.events, "TSTB": dataclasses.replace(market.events["TSTB"], fetched=None)}
    m = dataclasses.replace(market, snapshots=snaps, events=events)
    write_market(m, tmp_path)
    back = read_market(tmp_path)
    assert back.snapshots[0].options[0].quote.bid is None
    assert back.events["TSTB"].fetched is None


def test_a_missing_file_is_a_data_contract_error(tmp_path):
    with pytest.raises(DataContractError, match="marktbestand ontbreekt"):
        read_market(tmp_path)


def test_the_cli_writes_the_default_market(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(synthetic, "default_spec", lambda: SMALL)
    assert synthetic.main(["--out", str(tmp_path / "m")]) == 0
    assert read_market(tmp_path / "m").days == (date(2026, 12, 1), date(2026, 12, 2))
    assert "geschreven" in capsys.readouterr().out


def test_the_default_spec_has_two_names_and_fifteen_days():
    spec = default_spec()
    assert [u.name for u in spec.underlyings] == ["TSTA", "TSTB"] and spec.n_days == 15
