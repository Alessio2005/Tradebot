"""De historische simulatiemarkt: echte koers en IV, modelketens."""
from __future__ import annotations

import math
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from tradebot.options.contracts import Right
from tradebot.options.data.historical import (
    build_market,
    load_inputs,
    main,
    strike_step,
)
from tradebot.options.data.store import read_market
from tradebot.options.data.synthetic import SyntheticSpec
from tradebot.utils.failfast import DataContractError

DAYS = pd.bdate_range("2026-10-26", "2026-11-04")     # 2 november is een gewone handelsdag


def frame(iv: float = 0.25, close_step: float = 1.0) -> pd.DataFrame:
    close = 100.0 + close_step * np.arange(len(DAYS))
    return pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1_000_000.0, "iv": iv}, index=DAYS)


@pytest.mark.parametrize(("spot", "step"), [(30.0, 1.0), (50.0, 2.5), (199.0, 2.5), (200.0, 5.0)])
def test_the_strike_step_follows_the_price_level(spot, step):
    assert strike_step(spot) == step


def test_the_market_follows_the_real_prices_and_the_real_iv():
    m = build_market({"AAA": frame(0.25), "BBB": frame(0.40)}, date(2026, 11, 2),
                     date(2026, 11, 3))
    assert m.days == (date(2026, 11, 2), date(2026, 11, 3))
    assert len(m.bars["AAA"]) == len(DAYS)              # opwarmgeschiedenis blijft
    assert len(m.snapshots) == 2 * 2 * 2                # dagen x onderliggenden x momenten
    first = m.snapshots[0]
    bar = m.bars[first.underlying].loc[pd.Timestamp(first.asof.date())]
    frac = 30.0 / 390.0                                 # 10:00 is 30 minuten na de opening
    assert first.spot == pytest.approx(bar["open"] * math.exp(
        math.log(bar["close"] / bar["open"]) * frac))
    assert all(e.fetched == date(2026, 11, 2) and not e.earnings for e in m.events.values())

    def atm_call(und: str) -> float:
        s = next(s for s in m.snapshots if s.underlying == und and s.asof.hour == 10
                 and s.asof.date() == date(2026, 11, 2))
        pick = min((o for o in s.options if o.contract.right is Right.CALL
                    and o.contract.expiry == min(x.contract.expiry for x in s.options)),
                   key=lambda o: abs(o.contract.strike - s.spot))
        return (pick.quote.bid + pick.quote.ask) / 2
    assert atm_call("BBB") > atm_call("AAA")            # hogere IV, duurdere optie


def test_the_spec_and_half_days_can_be_passed():
    spec = SyntheticSpec(underlyings=(), start=date(2026, 11, 2), n_days=0,
                         snapshot_times=(time(10, 0),), weekly_days=7, max_days=14)
    m = build_market({"AAA": frame()}, date(2026, 11, 2), date(2026, 11, 2), spec=spec,
                     snapshot_times=(time(10, 0),), half_days=(date(2026, 11, 27),))
    assert len(m.snapshots) == 1


@pytest.mark.parametrize(("iv", "msg"), [(0.0, "fractie"), (45.0, "fractie")])
def test_the_iv_must_be_a_fraction(iv, msg):
    with pytest.raises(DataContractError, match=msg):
        build_market({"AAA": frame(iv)}, date(2026, 11, 2), date(2026, 11, 2))


def test_prices_must_be_positive():
    bad = frame()
    bad.iloc[0, 0] = 0.0
    with pytest.raises(DataContractError, match="positief"):
        build_market({"AAA": bad}, date(2026, 11, 2), date(2026, 11, 2))


def test_there_must_be_data_and_common_trading_days():
    with pytest.raises(DataContractError, match="geen onderliggenden"):
        build_market({}, date(2026, 11, 2), date(2026, 11, 2))
    with pytest.raises(DataContractError, match="gemeenschappelijke"):
        build_market({"AAA": frame()}, date(2026, 11, 7), date(2026, 11, 8))   # weekend


def test_the_inputs_are_read_from_csv_and_written_as_a_store(tmp_path, capsys):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    frame().rename_axis("date").to_csv(src / "AAA.csv")
    assert set(load_inputs(src)) == {"AAA"}
    rc = main(["--in", str(src), "--out", str(out), "--start", "2026-11-02",
               "--end", "2026-11-02"])
    assert rc == 0 and "1 dagen" in capsys.readouterr().out
    assert read_market(out).days == (date(2026, 11, 2),)


def test_bad_input_files_are_refused(tmp_path):
    with pytest.raises(DataContractError, match="geen CSV"):
        load_inputs(tmp_path)
    frame().drop(columns="iv").rename_axis("date").to_csv(tmp_path / "AAA.csv")
    with pytest.raises(DataContractError, match="mist kolommen"):
        load_inputs(tmp_path)
