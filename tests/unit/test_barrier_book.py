"""Het boek: juiste P&L bij een fill op de barrière, kosten, funding, filters, en geen blik vooruit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.backtest.barrier_book import BookInputs, SizingRule, run_barrier_book
from tradebot.risk.engine import RiskEngine
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
RISK = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)


def _market():
    m = synthetic_market(n=200)
    # Lage volatiliteit en diepe ADV: de risicolaag bindt niet, zodat de rekensom zichtbaar blijft.
    m.sigma_annual.loc[:, :] = 0.05
    m.sigma_daily.loc[:, :] = 0.05 / np.sqrt(365.0)
    m.adv_usd.loc[:, :] = 1e12
    m.funding.loc[:, :] = 0.0
    return m


def _inputs(m, cost_rate=0.0):
    corr = pd.DataFrame(0.0, index=m.grid, columns=list(m.symbols))
    return BookInputs(market=m, avg_corr=corr, cost_rate=cost_rate, impact=None)


def _cand(**over) -> pd.DataFrame:
    row = dict(symbol="BTCUSDT", entry_bar=100, exit_bar=104, side=1.0, fill_return=0.05,
               barrier=0.05, p=0.6, p_low=0.58, p_trade=0.52)
    row.update(over)
    return pd.DataFrame([row])


FIXED = SizingRule(mode="fixed", kelly_multiple=0.25, fixed_risk_fraction=0.01,
                   resize_band=0.25, cost_rt=0.0)


def test_a_target_hit_books_exactly_the_barrier_move() -> None:
    m = _market()
    res = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=100_000.0)
    t = res.trades.iloc[0]
    entry = m.ohlcv["BTCUSDT"]["close"].iloc[100]
    assert t["entry_price"] == pytest.approx(entry)
    assert res.equity.iloc[-1] - 100_000.0 == pytest.approx(t["qty"] * entry * 0.05)


def test_costs_are_charged_on_both_legs() -> None:
    m = _market()
    free = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=100_000.0)
    paid = run_barrier_book(_cand(), _inputs(m, cost_rate=0.00065), RiskEngine(RISK), FIXED,
                            equity0=100_000.0)
    t = paid.trades.iloc[0]
    entry = t["entry_price"]
    fill = entry * 1.05
    expected = abs(t["qty"]) * (entry + fill) * 0.00065
    assert paid.total_fees == pytest.approx(expected)
    assert free.equity.iloc[-1] - paid.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_a_long_pays_positive_funding() -> None:
    m = _market()
    m.funding.loc[m.grid[102], "BTCUSDT"] = 0.001
    res = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=100_000.0)
    assert res.total_funding > 0.0


def test_the_books_balance() -> None:
    m = _market()
    m.funding.loc[:, :] = 1e-4
    cands = pd.concat([_cand(), _cand(symbol="ETHUSDT", side=-1.0, fill_return=-0.05,
                                      entry_bar=101, exit_bar=103)], ignore_index=True)
    res = run_barrier_book(cands, _inputs(m, cost_rate=0.00065), RiskEngine(RISK), FIXED,
                           equity0=100_000.0)
    change = res.equity.iloc[-1] - 100_000.0
    assert change == pytest.approx(res.total_pnl - res.total_fees - res.total_funding
                                   - res.total_impact, rel=1e-9)


def test_the_filter_and_the_one_position_per_symbol_rule() -> None:
    m = _market()
    below = run_barrier_book(_cand(p=0.50), _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    assert below.trades.empty
    overlap = pd.concat([_cand(), _cand(entry_bar=102, exit_bar=106)], ignore_index=True)
    res = run_barrier_book(overlap, _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    assert len(res.trades) == 1


def test_simultaneous_same_direction_trades_are_scaled_for_correlation() -> None:
    m = _market()
    inputs = _inputs(m)
    inputs.avg_corr.loc[:, :] = 0.75
    pair = pd.concat([_cand(), _cand(symbol="ETHUSDT")], ignore_index=True)
    res = run_barrier_book(pair, inputs, RiskEngine(RISK), FIXED, equity0=1e5)
    solo = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    assert res.trades["risk_fraction"].iloc[0] == pytest.approx(
        solo.trades["risk_fraction"].iloc[0] / 1.75)


def test_the_book_never_reads_an_exit_before_it_happens() -> None:
    m = _market()
    a = run_barrier_book(_cand(exit_bar=150, fill_return=0.05), _inputs(m), RiskEngine(RISK),
                         FIXED, equity0=1e5)
    b = run_barrier_book(_cand(exit_bar=160, fill_return=-0.05), _inputs(m), RiskEngine(RISK),
                         FIXED, equity0=1e5)
    pd.testing.assert_series_equal(a.returns.iloc[:150], b.returns.iloc[:150])


def test_a_barrier_exit_records_its_realized_return_and_no_risk_exit() -> None:
    m = _market()
    res = run_barrier_book(_cand(), _inputs(m), RiskEngine(RISK), FIXED, equity0=1e5)
    t = res.trades.iloc[0]
    assert t["risk_exit_bar"] == -1
    assert t["realized_return"] == pytest.approx(0.05)


def test_a_risk_halt_closes_the_open_trade_at_its_real_return() -> None:
    m = _market()
    entry = float(m.ohlcv["BTCUSDT"]["close"].iloc[100])
    frame = m.ohlcv["BTCUSDT"].copy()
    frame.iloc[101:, frame.columns.get_loc("close")] = entry * 0.6  # -40% vlak na de entry
    m = type(m)(**{**m.__dict__, "ohlcv": {**m.ohlcv, "BTCUSDT": frame}})
    big = SizingRule(mode="fixed", kelly_multiple=0.25, fixed_risk_fraction=0.9,
                     resize_band=0.25, cost_rt=0.0)
    res = run_barrier_book(_cand(exit_bar=150), _inputs(m), RiskEngine(RISK), big, equity0=1e5)
    t = res.trades.iloc[0]
    assert res.halted
    assert t["risk_exit_bar"] == 101 < t["exit_bar"]
    assert t["realized_return"] == pytest.approx(-0.4)
    assert t["realized_return"] != pytest.approx(t["fill_return"])


def test_a_long_drawdown_does_not_shrink_a_position_to_dust() -> None:
    """De breaker schaalt de BEDOELDE exposure, niet het al verkleinde gewicht, elke dag opnieuw.

    Het huidige gewicht als `desired` terugvoeren liet de breakerfactor dagelijks op zichzelf
    werken: 53 herschalingen, en op dag 157 een valse risico-exit met het stof dat overbleef
    (-15% in plaats van de echte -5%). Het scenario is geleidelijk (-4% per dag, 4 dagen, dan
    vlak): een eendaagse klap haalt de dagverlieslimiet en halt in plaats van te schalen.
    """
    m = _market()
    entry = float(m.ohlcv["BTCUSDT"]["close"].iloc[100])
    frame = m.ohlcv["BTCUSDT"].copy()
    px = frame["close"].to_numpy().copy()
    for i in range(101, len(px)):
        px[i] = entry * 0.96 ** min(i - 100, 4)
    frame["close"] = px
    m = type(m)(**{**m.__dict__, "ohlcv": {**m.ohlcv, "BTCUSDT": frame}})
    big = SizingRule(mode="fixed", kelly_multiple=0.25, fixed_risk_fraction=0.9,
                     resize_band=0.25, cost_rt=0.0)
    res = run_barrier_book(_cand(exit_bar=160, fill_return=-0.05), _inputs(m), RiskEngine(RISK),
                           big, equity0=1e5)
    t = res.trades.iloc[0]
    assert not res.halted
    assert res.n_resizes < 5
    assert t["risk_exit_bar"] == -1  # de positie leefde door tot haar barrière
    assert t["realized_return"] == pytest.approx(-0.05)
