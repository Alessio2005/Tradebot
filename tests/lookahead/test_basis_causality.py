"""Causaliteit en mechaniek van het basisboek van v5 (`systematic/harvest.py`).

Op de synthetische breedte-markt van v2, met een spotbeen erbij. Wat hier wordt bewezen,
moet op elke markt gelden: het besluit tot en met *t* hangt niet af van data na *t*; een
perfecte hedge verdient precies de funding; een verdwenen been sluit de positie; een
short squeeze boven de marge liquideert, en dat kost wat het moet kosten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.lookahead.test_breadth_causality import CUTS, _cfg, _equal_through, _market
from tradebot.systematic.book import BookResult, CostSpec
from tradebot.systematic.harvest import (
    BasisCosts,
    BasisMarket,
    basis_market,
    carry_estimate,
    combine_accounts,
    harvest_targets,
    run_basis,
)
from tradebot.utils.failfast import DataContractError

FREE = BasisCosts(perp=CostSpec(taker_fee=0.0, half_spread=0.0, impact=None, aum_usd=1e6),
                  spot=CostSpec(taker_fee=0.0, half_spread=0.0, impact=None, aum_usd=1e6))
COSTS = BasisCosts(perp=CostSpec(taker_fee=5.5e-4, half_spread=3e-4, impact=None, aum_usd=1e6),
                   spot=CostSpec(taker_fee=10e-4, half_spread=5e-4, impact=None, aum_usd=1e6))
RISK = {"band": 0.5, "hedge_tolerance": 0.02, "maintenance_margin": 0.02,
        "margin_floor": 0.25}
RULE = {"span": 20, "enter_apr": 0.10, "exit_apr": 0.02, "slots": 4, "notional": 0.7,
        "min_spot_adv_usd": 1e5, "max_abs_basis": 0.05}


def _spot(perp_close: pd.DataFrame, qv: pd.DataFrame, *, seed: int = 11,
          basis_sd: float = 5e-4) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    noise = pd.DataFrame(rng.normal(0.0, basis_sd, size=perp_close.shape),
                         index=perp_close.index, columns=perp_close.columns)
    return {"close": perp_close * (1.0 - noise), "quote_volume": qv * 0.5}


def _basis(perturb_after: int | None = None, *, basis_sd: float = 5e-4) -> BasisMarket:
    m = _market(perturb_after)
    return basis_market(m, _spot(m.book.close, m.quote_volume, basis_sd=basis_sd), _cfg())


def _run(m: BasisMarket, target: pd.DataFrame, costs: BasisCosts = COSTS, lag: int = 1,
         **kw) -> BookResult:
    return run_basis(target, m, costs, lag=lag, **{**RISK, **kw})


@pytest.mark.parametrize("t", CUTS)
def test_the_harvest_decision_through_t_ignores_everything_after_t(t):
    base, pert = _basis(), _basis(perturb_after=t)
    assert _equal_through(carry_estimate(base, 20), carry_estimate(pert, 20), t)
    a, b = harvest_targets(base, **RULE), harvest_targets(pert, **RULE)
    assert _equal_through(a, b, t)
    assert float(a.to_numpy().sum()) > 0.0, "de toets is leeg als er nooit iets gehouden wordt"


@pytest.mark.parametrize("t", CUTS)
def test_the_book_through_t_ignores_everything_after_t(t):
    base, pert = _basis(), _basis(perturb_after=t)
    ra = _run(base, harvest_targets(base, **RULE))
    rb = _run(pert, harvest_targets(pert, **RULE))
    assert _equal_through(ra.frame, rb.frame, t)


def test_slots_hold_at_most_the_notional_and_never_go_short():
    m = _basis()
    tgt = harvest_targets(m, **RULE)
    assert (tgt.to_numpy() >= 0.0).all()
    assert tgt.sum(axis=1).max() <= RULE["notional"] + 1e-12
    assert ((tgt > 0).sum(axis=1) <= RULE["slots"]).all()


def test_a_held_coin_is_not_displaced_by_a_better_one():
    """Hysterese: een slot verandert alleen als zijn munt onder `exit_apr` zakt."""
    m = _basis()
    tgt = harvest_targets(m, **{**RULE, "slots": 1})
    carry = carry_estimate(m, RULE["span"])
    held = tgt.idxmax(axis=1).where(tgt.sum(axis=1) > 0)
    for t in range(1, len(held)):
        prev, cur = held.iloc[t - 1], held.iloc[t]
        if isinstance(prev, str) and isinstance(cur, str) and prev != cur:
            assert not carry[prev].iloc[t] > RULE["exit_apr"] or not (
                m.spot_close[prev].notna().iloc[t] and m.perp.book.close[prev].notna().iloc[t])


def test_the_majors_restriction_only_admits_those_symbols():
    m = _basis()
    tgt = harvest_targets(m, **RULE, symbols=("BTCUSDT",))
    assert set(tgt.columns[(tgt > 0).any()]) <= {"BTCUSDT"}


def test_a_perfect_hedge_earns_exactly_the_funding():
    """Spot = perp, geen kosten: bruto P&L nul, netto = funding op de short."""
    m = _basis(basis_sd=0.0)
    res = _run(m, harvest_targets(m, **RULE), costs=FREE)
    f = res.frame
    assert np.abs(f["gross"].to_numpy()).max() < 1e-12
    assert np.allclose(f["net"], f["funding"], atol=1e-15)
    assert f["funding"].sum() > 0.0
    assert np.abs(f["net_leverage"].to_numpy()).max() < 1e-12


def test_costs_are_charged_on_both_legs():
    m = _basis(basis_sd=0.0)
    tgt = harvest_targets(m, **RULE)
    free, paid = _run(m, tgt, costs=FREE), _run(m, tgt, costs=COSTS)
    first = int(np.flatnonzero(free.frame["turnover"].to_numpy() > 0)[0])
    turnover = float(free.frame["turnover"].iloc[first])
    # Instap: elk been verhandelt de helft van de omzet.
    expected = turnover / 2 * (5.5e-4 + 3e-4) + turnover / 2 * (10e-4 + 5e-4)
    got = -(paid.frame[["fees", "spread"]].iloc[first].sum()) / (1.0 + paid.frame["gross"].iloc[first]
                                                                + paid.frame["funding"].iloc[first])
    assert got == pytest.approx(expected, rel=1e-9)


def test_the_band_keeps_quantities_instead_of_chasing_weights():
    """Binnen de band handelt het boek niet: gelijke hoeveelheden blijven gehedged."""
    m = _basis(basis_sd=0.0)
    tgt = harvest_targets(m, **RULE)
    loose = _run(m, tgt, band=0.99)
    tight = _run(m, tgt, band=1e-6)
    assert loose.frame["turnover"].sum() < tight.frame["turnover"].sum()


def test_a_held_coin_whose_perp_vanishes_is_closed():
    m = _basis()
    sym, delist = "C07USDT", 450
    tgt = pd.DataFrame(0.0, index=m.index, columns=list(m.symbols))
    live = m.perp.book.close[sym].notna() & m.spot_close[sym].notna() & m.spot_adv[sym].notna()
    tgt.loc[live, sym] = 0.2
    res = _run(m, tgt)
    assert res.audit["n_forced_exits"] >= 1
    assert (res.held[sym].iloc[delist + 1:] == 0.0).all()


def test_a_held_coin_whose_spot_vanishes_is_closed():
    m0 = _basis()
    sym, gone = "C09USDT", 380
    spot = {"close": m0.spot_close.copy(), "quote_volume": m0.perp.quote_volume * 0.5}
    spot["close"].loc[spot["close"].index[gone:], sym] = np.nan
    m = basis_market(m0.perp, spot, _cfg())
    tgt = pd.DataFrame(0.0, index=m.index, columns=list(m.symbols))
    tgt.loc[m.index[100]:, sym] = 0.2
    res = _run(m, tgt)
    assert res.audit["n_forced_exits"] >= 1
    assert (res.held[sym].iloc[gone + 1:] == 0.0).all()


def test_a_short_squeeze_beyond_the_margin_liquidates_at_the_high():
    m0 = _basis(basis_sd=0.0)
    sym, bar = "C10USDT", 300
    perp = m0.perp
    high = perp.high.copy()
    high.loc[high.index[bar], sym] = perp.book.close[sym].iloc[bar - 1] * 6.0   # +500 %
    squeezed = BasisMarket(perp=type(perp)(book=perp.book, high=high, low=perp.low,
                                           quote_volume=perp.quote_volume,
                                           half_spread=perp.half_spread, universe=perp.universe),
                           spot_close=m0.spot_close, spot_ret=m0.spot_ret, spot_adv=m0.spot_adv,
                           spot_sigma=m0.spot_sigma, basis=m0.basis)
    tgt = pd.DataFrame(0.0, index=m0.index, columns=list(m0.symbols))
    tgt.loc[m0.index[200]:, sym] = 0.7
    res = _run(squeezed, tgt, costs=FREE)
    assert res.audit["n_liquidations"] == 1
    f = res.frame.iloc[bar]
    b = res.held[sym].iloc[bar]
    rp = perp.book.ret[sym].iloc[bar]
    wallet = 1.0 - b
    # Het perpbeen verliest precies de wallet; de spot houdt zijn eigen rendement.
    assert f["liquidation"] == pytest.approx(b * rp - wallet, rel=1e-9)
    assert f["gross"] + f["liquidation"] == pytest.approx(
        b * m0.spot_ret[sym].iloc[bar] - wallet, rel=1e-9)
    # De volgende herbalancering hedget opnieuw op het doel.
    assert res.held[sym].iloc[bar + 1] == pytest.approx(0.7, rel=1e-9)


def test_a_rally_that_starves_the_futures_wallet_rebalances_the_whole_book():
    """Een brede rally laat de spot groeien en de wallet leeglopen; onder de vloer gaat
    alles terug naar het doel, ook munten die binnen hun eigen band blijven."""
    m = _basis(basis_sd=0.0)
    tgt = harvest_targets(m, **RULE)
    tight_floor = _run(m, tgt, margin_floor=0.4)
    no_floor = _run(m, tgt, margin_floor=1e-9)
    assert tight_floor.frame["turnover"].sum() > no_floor.frame["turnover"].sum()
    wallet = 1.0 - tight_floor.frame["spot_notional"]
    perp = tight_floor.frame["gross_leverage"] - tight_floor.frame["spot_notional"]
    # Na elke herbalancering staat de wallet weer boven de vloer.
    traded = tight_floor.frame["turnover"] > 0
    assert (wallet[traded] >= 0.4 * perp[traded] - 1e-9).all()


def test_without_a_squeeze_nothing_is_liquidated():
    m = _basis()
    res = _run(m, harvest_targets(m, **RULE))
    assert res.audit["n_liquidations"] == 0
    assert (res.frame["liquidation"] == 0.0).all()


def test_lag_two_executes_one_bar_later():
    m = _basis()
    tgt = harvest_targets(m, **RULE)
    one, two = _run(m, tgt, lag=1), _run(m, tgt, lag=2)
    first = lambda r: int(np.flatnonzero(r.frame["turnover"].to_numpy() > 0)[0])  # noqa: E731
    assert first(two) == first(one) + 1


def test_a_skipped_bar_trades_nothing_but_forced_exits():
    m = _basis()
    tgt = harvest_targets(m, **RULE)
    skip = pd.Series(True, index=m.index)
    res = _run(m, tgt, skip=skip)
    assert res.frame["turnover"].sum() == 0.0


def test_a_negative_target_is_refused():
    m = _basis()
    tgt = pd.DataFrame(0.0, index=m.index, columns=list(m.symbols))
    tgt.iloc[100, 0] = -0.1
    with pytest.raises(DataContractError):
        _run(m, tgt)


def test_zero_lag_is_refused():
    m = _basis()
    with pytest.raises(DataContractError):
        _run(m, harvest_targets(m, **RULE), lag=0)


def test_combined_accounts_return_the_share_weighted_net():
    m = _basis()
    a = _run(m, harvest_targets(m, **RULE))
    b = _run(m, harvest_targets(m, **{**RULE, "slots": 2}))
    mix = combine_accounts({"a": (0.3, a), "b": (0.7, b)})
    assert np.allclose(mix.frame["net"], 0.3 * a.frame["net"] + 0.7 * b.frame["net"], atol=1e-15)
    with pytest.raises(DataContractError):
        combine_accounts({"a": (0.5, a), "b": (0.6, b)})


def test_truncate_cuts_every_panel():
    m = _basis()
    cut = m.index[300]
    tr = m.truncate(cut)
    assert tr.index.max() < cut
    assert tr.spot_close.index.equals(tr.index) and tr.basis.index.equals(tr.index)
