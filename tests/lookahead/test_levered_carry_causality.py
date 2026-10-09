"""Causaliteit en mechaniek van het hefboom-basisboek van v6 (`systematic/leverage.py`).

Op de synthetische basismarkt van v5, met mark-price- en spot-extremen erbij. Wat hier
wordt bewezen, moet op elke markt gelden:

* het besluit en het boek tot en met *t* hangen niet af van data na *t*;
* een perfecte hedge verdient precies de funding, min de financiering van de lening;
* de lening is alleen wat de spot boven de eigen equity uitkomt;
* de governor houdt de bruto exposure en de uniMMR binnen hun grens;
* een squeeze op de MARK price liquideert, een wick op alleen de last price niet;
* ontbreekt de mark price, dan valt de toets terug op de last price (conservatief).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.lookahead.test_basis_causality import COSTS, FREE, RULE, _basis
from tests.lookahead.test_breadth_causality import CUTS, _equal_through
from tradebot.systematic.harvest import BasisCosts, BasisMarket, carry_estimate, harvest_targets
from tradebot.systematic.leverage import (
    MarginMarket,
    MarginSpec,
    financing_rate,
    governed_scale,
    levered_targets,
    margin_market,
    run_levered,
    tranche_targets,
    uni_mmr,
)
from tradebot.utils.failfast import DataContractError

MARGIN = MarginSpec(haircut_major=0.05, haircut_alt=0.20, majors=("BTCUSDT",),
                    maintenance_margin=0.02, loan_maintenance=0.05, min_uni_mmr=3.0,
                    gross_cap=4.0, liquidation_fee=0.015)
SLOT_RULE = {k: v for k, v in RULE.items() if k != "notional"}
RISK = {"band": 0.5, "hedge_tolerance": 0.02}


def _mm(perturb_after: int | None = None, *, basis_sd: float = 5e-4,
        mark_scale: float = 1.0) -> MarginMarket:
    b = _basis(perturb_after, basis_sd=basis_sd)
    p = b.perp
    mark = {"high": p.book.close + (p.high - p.book.close) * mark_scale,
            "low": p.book.close - (p.book.close - p.low) * mark_scale}
    spot = {"high": b.spot_close * 1.001, "low": b.spot_close * 0.999}
    return margin_market(b, mark, spot)


def _const_rate(m: MarginMarket, apr: float) -> pd.Series:
    return pd.Series(apr, index=m.index)


def _run(m: MarginMarket, target: pd.DataFrame, *, costs=COSTS, lag: int = 1,
         margin: MarginSpec = MARGIN, rate: float = 0.10, **kw):
    return run_levered(target, m, costs, lag=lag, margin=margin,
                       financing=_const_rate(m, rate), **{**RISK, **kw})


def _targets(m: MarginMarket, leverage: float) -> pd.DataFrame:
    return levered_targets(m.basis, leverage=leverage, **SLOT_RULE)


# --------------------------------------------------------------------------- #
# Causaliteit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t", CUTS)
def test_the_levered_decision_through_t_ignores_everything_after_t(t):
    base, pert = _mm(), _mm(perturb_after=t)
    assert _equal_through(_targets(base, 2.0), _targets(pert, 2.0), t)
    fa = financing_rate(base.basis, floor_apr=0.08, multiplier=1.0, span=20)
    fb = financing_rate(pert.basis, floor_apr=0.08, multiplier=1.0, span=20)
    assert _equal_through(fa.to_frame(), fb.to_frame(), t)


@pytest.mark.parametrize("t", CUTS)
def test_the_levered_book_through_t_ignores_everything_after_t(t):
    base, pert = _mm(), _mm(perturb_after=t)
    ra = run_levered(_targets(base, 2.0), base, COSTS, lag=1, margin=MARGIN,
                     financing=financing_rate(base.basis, floor_apr=0.08, multiplier=1.0,
                                              span=20), **RISK)
    rb = run_levered(_targets(pert, 2.0), pert, COSTS, lag=1, margin=MARGIN,
                     financing=financing_rate(pert.basis, floor_apr=0.08, multiplier=1.0,
                                              span=20), **RISK)
    assert _equal_through(ra.frame, rb.frame, t)
    assert float(ra.frame["turnover"].sum()) > 0.0, "de toets is leeg zonder handel"


# --------------------------------------------------------------------------- #
# Doelen en financiering
# --------------------------------------------------------------------------- #
def test_levered_targets_scale_the_unlevered_slots():
    m = _mm()
    one = harvest_targets(m.basis, **{**SLOT_RULE, "notional": 1.0})
    assert np.allclose(_targets(m, 2.0).to_numpy(), 2.0 * one.to_numpy())
    assert _targets(m, 2.0).sum(axis=1).max() <= 2.0 + 1e-12


def _equity_pre(res, aum: float) -> np.ndarray:
    """De equity op het uitvoeringsmoment van elke bar (vóór de kosten van die bar)."""
    f = res.frame
    end = aum * np.cumprod(1.0 + f["net"].to_numpy())
    start = np.concatenate(([aum], end[:-1]))
    port = f[["gross", "funding", "financing", "liquidation"]].sum(axis=1).to_numpy()
    return start * (1.0 + port)


def test_the_adv_cap_limits_each_leg_to_its_thinnest_liquidity_at_the_running_equity():
    m = _mm(basis_sd=0.0)
    big = FREE.scaled(aum_usd=5e7)
    tgt = _targets(m, 2.0)
    free = _run(m, tgt, costs=big)
    capped = _run(m, tgt, costs=big, adv_cap=0.01)
    assert capped.frame["spot_notional"].mean() < free.frame["spot_notional"].mean() - 1e-6
    adv = np.fmin(m.basis.spot_adv.where(m.basis.spot_adv > 0).ffill(),
                  m.basis.perp.book.adv_usd.where(m.basis.perp.book.adv_usd > 0).ffill())
    eq = _equity_pre(capped, 5e7)
    traded = capped.trades.to_numpy() > 0
    spot = capped.held.shift(-1).to_numpy()          # held[t+1] = positie na de trade op t
    limit = 0.01 * adv.to_numpy() / eq[:, None]
    ok = ~traded[:-1] | (spot[:-1] <= limit[:-1] * (1 + 1e-9))
    assert ok.all(), "een verhandeld been boven 1 % van de ADV tegen de lopende equity"
    with pytest.raises(DataContractError):
        _run(m, tgt, adv_cap=0.0)


@pytest.mark.parametrize("t", CUTS)
def test_the_capped_book_through_t_ignores_everything_after_t(t):
    base, pert = _mm(), _mm(perturb_after=t)
    big = COSTS.scaled(aum_usd=5e7)
    ra = _run(base, _targets(base, 2.0), costs=big, adv_cap=0.01)
    rb = _run(pert, _targets(pert, 2.0), costs=big, adv_cap=0.01)
    assert _equal_through(ra.frame, rb.frame, t)


def test_financing_is_the_floor_or_the_btc_carry_whichever_is_higher():
    m = _mm()
    r = financing_rate(m.basis, floor_apr=0.08, multiplier=1.5, span=20)
    carry = (m.basis.perp.book.funding["BTCUSDT"]
             .where(m.basis.perp.book.close["BTCUSDT"].notna())
             .ewm(span=20, min_periods=20).mean() * 365)
    expected = np.maximum(0.08, (1.5 * carry).fillna(-np.inf))
    assert np.allclose(r.to_numpy(), expected.to_numpy())
    flat = financing_rate(m.basis, floor_apr=0.08, multiplier=1.0, span=20, constant_apr=0.06)
    assert (flat == 0.06).all()


def test_a_perfect_hedge_earns_the_funding_minus_the_financing_of_the_loan():
    """Spot = perp, geen kosten: bruto nul; netto = funding − lening × rente / 365."""
    m = _mm(basis_sd=0.0)
    res = _run(m, _targets(m, 2.0), costs=FREE, rate=0.10)
    f = res.frame
    assert np.abs(f["gross"].to_numpy()).max() < 1e-12
    assert np.allclose(f["financing"], -f["loan"] * 0.10 / 365, atol=1e-15)
    assert np.allclose(f["net"], f["funding"] + f["financing"], atol=1e-14)
    assert f["loan"].max() > 0.5, "bij 2x hefboom wordt er geleend"
    assert (f["financing"] <= 0.0).all()


def test_the_loan_is_only_the_spot_above_the_own_equity():
    m = _mm(basis_sd=0.0)
    f = _run(m, _targets(m, 1.0), costs=FREE).frame
    start_spot = f["spot_notional"].shift(1).fillna(0.0)
    assert np.allclose(f["loan"], np.maximum(start_spot - 1.0, 0.0), atol=1e-12)


# --------------------------------------------------------------------------- #
# De governor
# --------------------------------------------------------------------------- #
def test_uni_mmr_is_haircut_equity_over_maintenance():
    a = np.array([1.0, 1.0])
    h = np.array([0.05, 0.20])
    got = uni_mmr(a, a, h, maintenance_margin=0.02, loan_maintenance=0.05)
    ae = 1.0 - (0.05 + 0.20)
    mm = 0.02 * 2.0 + 0.05 * 1.0
    assert got == pytest.approx(ae / mm)
    assert uni_mmr(np.zeros(2), np.zeros(2), h, maintenance_margin=0.02,
                   loan_maintenance=0.05) == np.inf


@pytest.mark.parametrize("lev", [1.0, 2.0, 3.0, 6.0])
def test_governed_scale_is_the_largest_admissible_scale(lev):
    w = np.full(10, lev / 10)
    h = np.full(10, 0.20)
    s = governed_scale(w, w, h, MARGIN)
    assert 0.0 < s <= 1.0
    ok = lambda x: (2 * x * w.sum() <= MARGIN.gross_cap + 1e-9  # noqa: E731
                    and uni_mmr(x * w, x * w, h, maintenance_margin=0.02,
                                loan_maintenance=0.05) >= 3.0 - 1e-9)
    assert ok(s)
    assert s == 1.0 or not ok(min(1.0, s * 1.001))


def test_the_governor_caps_gross_exposure_and_keeps_the_uni_mmr():
    m = _mm(basis_sd=0.0)
    res = _run(m, _targets(m, 3.0), costs=FREE)   # gevraagd: bruto 6
    f = res.frame
    traded = f["turnover"] > 0
    assert (f.loc[traded, "gross_leverage"] <= 4.0 + 1e-9).all()
    assert (f.loc[traded, "uni_mmr"] >= 3.0 - 1e-9).all()
    assert f["gross_leverage"].max() > 3.5, "de cap bindt, het boek staat niet leeg"


# --------------------------------------------------------------------------- #
# Liquidatie op de mark price
# --------------------------------------------------------------------------- #
def _squeezed(m: MarginMarket, sym: str, bar: int, *, mark: float | None,
              last: float | None) -> MarginMarket:
    p = m.basis.perp
    high = p.high.copy()
    mark_high = m.mark_high.copy()
    prev = p.book.close[sym].iloc[bar - 1]
    if last is not None:
        high.loc[high.index[bar], sym] = prev * last
    if mark is not None:
        mark_high.loc[mark_high.index[bar], sym] = prev * mark
    else:
        mark_high.loc[mark_high.index[bar], sym] = np.nan
    perp = type(p)(book=p.book, high=high, low=p.low, quote_volume=p.quote_volume,
                   half_spread=p.half_spread, universe=p.universe)
    b = m.basis
    basis = BasisMarket(perp=perp, spot_close=b.spot_close, spot_ret=b.spot_ret,
                        spot_adv=b.spot_adv, spot_sigma=b.spot_sigma, basis=b.basis)
    return MarginMarket(basis=basis, mark_high=mark_high, mark_low=m.mark_low,
                        spot_high=m.spot_high, spot_low=m.spot_low)


SYM, BAR = "C10USDT", 300


def _single(m: MarginMarket, size: float) -> pd.DataFrame:
    tgt = pd.DataFrame(0.0, index=m.index, columns=list(m.symbols))
    tgt.loc[m.index[200]:, SYM] = size
    return tgt


def test_a_mark_price_squeeze_liquidates_the_account():
    m0 = _mm(basis_sd=0.0)
    m = _squeezed(m0, SYM, BAR, mark=1.5, last=None)        # mark +50 %, spot niet
    res = _run(m, _single(m, 1.5), costs=FREE, rate=0.0,
               margin=replace(MARGIN, min_uni_mmr=1.0))
    assert res.audit["n_liquidations"] == 1
    f = res.frame.iloc[BAR]
    a = res.frame["spot_notional"].iloc[BAR - 1]
    us = min(m.spot_high[SYM].iloc[BAR] / m.basis.spot_close[SYM].iloc[BAR - 1] - 1.0, 0.5)
    e_s = 1.0 + a * us - a * 0.5
    fee = 0.015 * (a * (1.0 + us) + a * 1.5)
    assert e_s - fee > 0.0, "de toets wil een liquidatie, geen wipe-out"
    assert f["net"] == pytest.approx(max(e_s - fee, 0.0) - 1.0, rel=1e-9)
    assert f["funding"] == 0.0
    # Na de liquidatie hedget de volgende herbalancering opnieuw op het doel.
    assert res.held[SYM].iloc[BAR + 1] > 0.0


def test_a_wick_in_the_last_price_alone_does_not_liquidate():
    m0 = _mm(basis_sd=0.0)
    m = _squeezed(m0, SYM, BAR, mark=1.01, last=4.0)        # alleen de last price wickt
    res = _run(m, _single(m, 1.5), costs=FREE,
               margin=replace(MARGIN, min_uni_mmr=1.0))
    assert res.audit["n_liquidations"] == 0


def test_without_a_mark_price_the_last_price_high_is_the_test():
    m0 = _mm(basis_sd=0.0)
    m = _squeezed(m0, SYM, BAR, mark=None, last=1.5)        # mark ontbreekt die dag
    res = _run(m, _single(m, 1.5), costs=FREE,
               margin=replace(MARGIN, min_uni_mmr=1.0))
    assert res.audit["n_liquidations"] == 1


def test_without_a_squeeze_nothing_is_liquidated_and_headroom_is_reported():
    m = _mm()
    res = _run(m, _targets(m, 2.0))
    assert res.audit["n_liquidations"] == 0
    assert (res.frame["liquidation"] == 0.0).all()
    held = res.frame["n_held"].shift(1).fillna(0) > 0
    assert (res.frame.loc[held, "stress_headroom"] > 0.0).all()
    assert res.audit["min_stress_headroom"] == pytest.approx(
        float(res.frame.loc[held, "stress_headroom"].min()))


def test_a_larger_haircut_lowers_the_headroom():
    m = _mm()
    tgt = _targets(m, 2.0)
    lo = _run(m, tgt).audit["min_stress_headroom"]
    hi = _run(m, tgt, margin=replace(MARGIN, haircut_alt=0.35))
    assert hi.audit["min_stress_headroom"] < lo


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_lag_two_executes_one_bar_later():
    m = _mm()
    tgt = _targets(m, 2.0)
    first = lambda r: int(np.flatnonzero(r.frame["turnover"].to_numpy() > 0)[0])  # noqa: E731
    assert first(_run(m, tgt, lag=2)) == first(_run(m, tgt, lag=1)) + 1


def test_a_negative_target_and_zero_lag_are_refused():
    m = _mm()
    tgt = _targets(m, 1.0)
    with pytest.raises(DataContractError):
        _run(m, tgt, lag=0)
    bad = tgt.copy()
    bad.iloc[100, 0] = -0.1
    with pytest.raises(DataContractError):
        _run(m, bad)


def test_truncate_cuts_every_panel():
    m = _mm()
    cut = m.index[300]
    tr = m.truncate(cut)
    assert tr.index.max() < cut
    for frame in (tr.mark_high, tr.mark_low, tr.spot_high, tr.spot_low):
        assert frame.index.equals(tr.index)


# --------------------------------------------------------------------------- #
# v7: gespreide uitvoering en de gefinancierde tranche
# --------------------------------------------------------------------------- #
TRANCHE_RULE = dict(SLOT_RULE)


def _tranche(m: MarginMarket, rate: pd.Series, spread: float = 0.10) -> pd.DataFrame:
    return tranche_targets(m.basis, financing=rate, lever_spread=spread, **TRANCHE_RULE)


@pytest.mark.parametrize("t", CUTS)
def test_the_tranche_decision_through_t_ignores_everything_after_t(t):
    base, pert = _mm(), _mm(perturb_after=t)
    ra = financing_rate(base.basis, floor_apr=0.05, multiplier=1.0, span=20)
    rb = financing_rate(pert.basis, floor_apr=0.05, multiplier=1.0, span=20)
    assert _equal_through(_tranche(base, ra), _tranche(pert, rb), t)


def test_the_tranche_is_on_only_while_carry_beats_the_financing():
    m = _mm()
    rate = _const_rate(m, 0.05)
    out = _tranche(m, rate, spread=0.10)
    one = harvest_targets(m.basis, **{**SLOT_RULE, "notional": 1.0})
    size = 1.0 / SLOT_RULE["slots"]
    extra = (out - one).round(12)
    assert set(np.unique(extra.to_numpy())) <= {0.0, round(size, 12)}
    assert (extra.to_numpy()[one.to_numpy() == 0.0] == 0.0).all(), "geen tranche zonder basisslot"
    carry = carry_estimate(m.basis, SLOT_RULE["span"]).to_numpy()
    on = extra.to_numpy() > 0
    assert (carry[on] >= 0.05 - 1e-12).all(), "een tranche onder de leenrente"
    # Instap alleen boven rente + spread; daarna hysterese tot de rente.
    first = on & ~np.vstack([np.zeros((1, on.shape[1]), dtype=bool), on[:-1]])
    assert (carry[first] >= 0.15 - 1e-12).all()
    assert on.any(), "de toets is leeg zonder tranche"
    # Een hogere rente geeft nooit meer hefboom.
    dear = _tranche(m, _const_rate(m, 0.20), spread=0.10)
    assert dear.sum().sum() <= out.sum().sum() + 1e-9


def test_slicing_moves_each_coin_at_most_one_step_per_day():
    m = _mm(basis_sd=0.0)
    tgt = _targets(m, 1.0)
    step = 0.1 / 5
    res = _run(m, tgt, costs=FREE, slice_step=step)
    traded = res.trades.to_numpy() / 2.0     # beide benen even veel
    assert traded.max() <= step + 1e-12
    whole = _run(m, tgt, costs=FREE)
    assert (whole.trades.to_numpy() / 2.0).max() > step, "zonder spreiding gaat een slot in één keer"
    # Het boek komt er wel: over de tweede helft staat het gespreide boek minstens 85 %
    # zo groot als het ongespreide (het verschil is de aanloop en padafhankelijke drift).
    half = slice(m.index[len(m.index) // 2], None)
    assert res.held.loc[half].sum(axis=1).mean() >= 0.85 * whole.held.loc[half].sum(axis=1).mean()


def test_slicing_cuts_impact_on_a_convex_cost_curve():
    from tests.unit.test_programme_v2_smoke import IMPACT
    from tradebot.systematic.book import CostSpec
    m = _mm(basis_sd=0.0)
    imp = BasisCosts(perp=CostSpec(taker_fee=0.0, half_spread=0.0, impact=IMPACT, aum_usd=5e7,
                                   impact_eta=1.0),
                     spot=CostSpec(taker_fee=0.0, half_spread=0.0, impact=IMPACT, aum_usd=5e7,
                                   impact_eta=1.0))
    tgt = _targets(m, 1.0)
    whole = _run(m, tgt, costs=imp)
    sliced = _run(m, tgt, costs=imp, slice_step=0.1 / 5)
    assert sliced.frame["impact"].sum() > whole.frame["impact"].sum()   # minder negatief


def test_a_vanishing_coin_is_closed_at_once_even_when_sliced():
    m = _mm()
    sym, delist = "C07USDT", 450
    tgt = pd.DataFrame(0.0, index=m.index, columns=list(m.symbols))
    live = (m.basis.perp.book.close[sym].notna() & m.basis.spot_close[sym].notna()
            & m.basis.spot_adv[sym].notna())
    tgt.loc[live, sym] = 0.2
    res = _run(m, tgt, slice_step=0.01)
    assert res.audit["n_forced_exits"] >= 1
    assert (res.held[sym].iloc[delist + 1:] == 0.0).all()


def test_the_sliced_governor_scales_the_goal_not_only_the_book():
    m = _mm(basis_sd=0.0)
    res = _run(m, _targets(m, 3.0), costs=FREE, slice_step=0.02)
    f = res.frame
    traded = f["turnover"] > 0
    assert (f.loc[traded, "gross_leverage"] <= 4.0 + 1e-9).all()
    assert (f.loc[traded, "uni_mmr"] >= 3.0 - 1e-9).all()


@pytest.mark.parametrize("t", CUTS)
def test_the_sliced_book_through_t_ignores_everything_after_t(t):
    base, pert = _mm(), _mm(perturb_after=t)
    ra = _run(base, _targets(base, 2.0), slice_step=0.02, adv_cap=0.01)
    rb = _run(pert, _targets(pert, 2.0), slice_step=0.02, adv_cap=0.01)
    assert _equal_through(ra.frame, rb.frame, t)


# --------------------------------------------------------------------------- #
# v8: de sprongrisicogrens
# --------------------------------------------------------------------------- #
def test_the_sigma_cap_excludes_jump_coins_from_entry_and_keep():
    m = _mm()
    free = harvest_targets(m.basis, **{**SLOT_RULE, "notional": 1.0})
    sig = np.fmax(m.basis.spot_sigma, m.basis.perp.book.sigma_daily)
    cap = float(sig.stack().quantile(0.6))     # een grens die in deze markt bindt
    capped = harvest_targets(m.basis, **{**SLOT_RULE, "notional": 1.0}, max_sigma=cap)
    held = capped.to_numpy() > 0
    assert held.any() and (free.to_numpy() > 0).sum() > held.sum()
    assert (sig.to_numpy()[held] <= cap + 1e-12).all(), "een munt boven de grens gehouden"
    # Zonder grens: exact het oude besluit (v5-v7 reproduceren).
    assert harvest_targets(m.basis, **{**SLOT_RULE, "notional": 1.0}, max_sigma=None).equals(free)


@pytest.mark.parametrize("t", CUTS)
def test_the_sigma_capped_decision_through_t_ignores_everything_after_t(t):
    base, pert = _mm(), _mm(perturb_after=t)
    kw = {**SLOT_RULE, "notional": 1.0, "max_sigma": 0.03}
    assert _equal_through(harvest_targets(base.basis, **kw), harvest_targets(pert.basis, **kw), t)
    ra = financing_rate(base.basis, floor_apr=0.05, multiplier=1.0, span=20)
    rb = financing_rate(pert.basis, floor_apr=0.05, multiplier=1.0, span=20)
    ta = tranche_targets(base.basis, financing=ra, lever_spread=0.1, max_sigma=0.03, **TRANCHE_RULE)
    tb = tranche_targets(pert.basis, financing=rb, lever_spread=0.1, max_sigma=0.03, **TRANCHE_RULE)
    assert _equal_through(ta, tb, t)
