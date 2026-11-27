"""Gamma scalping (`strategies/vol_trading.py`): entry op goedkope IV, beheer incl. hedge-P&L."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.market_helpers import (
    CFG,
    cached_chain,
    market,
    open_position,
)
from tradebot.options.calendar import NY
from tradebot.options.contracts import Right
from tradebot.options.events import Events
from tradebot.options.positions import ActionKind, Book, Leg, Urgency
from tradebot.options.strategies.common import rules_from_config
from tradebot.options.strategies.registry import build_strategies
from tradebot.options.strategies.vol_trading import GammaScalp
from tradebot.options.vol_regime import RVForecast, VolRegime, VolState
from tradebot.utils.failfast import ConfigContractError

RULES = rules_from_config(CFG)
S = CFG.strategies
RICH = 1_000_000.0
EXPIRY = date(2027, 1, 15)                 # DTE 52: het dichtst bij het midden van [30, 60]
ATM = 101.0                                # forward 100 × e^(0,04 × 52/365) ≈ 100,57


def straddle_iv() -> float:
    chain = cached_chain()
    return (chain.find(EXPIRY, ATM, Right.PUT).iv + chain.find(EXPIRY, ATM, Right.CALL).iv) / 2


def vol(rv: float | None = 0.35) -> dict[str, VolState]:
    """Een vlakke RV-verwachting: σ(h) = `rv` voor elke horizon."""
    f = RVForecast(rv, rv, rv, rv) if rv is not None else None
    return {"TST": VolState(VolRegime.CHEAP, 0.25, f, 0.01, "test")}


def mkt(rv: float | None = 0.35, **kw):
    kw.setdefault("capital", RICH)
    return dataclasses.replace(market(**kw), vol=vol(rv))


def gs(rules=RULES, eligible=("TST",), **kw):
    return GammaScalp(S.gamma_scalp.model_copy(update=kw), rules, eligible)


def at(y: int, m: int, d: int, h: int = 10) -> datetime:
    return datetime(y, m, d, h, 0, tzinfo=NY)


def later(spot: float = 100.0, now: datetime = NOW, rv: float | None = 0.35, base: float = 0.25):
    return dataclasses.replace(market(cached_chain(spot, base, now=now), now=now), vol=vol(rv))


def reasons(strategy, pos, m):
    return [(a.kind, a.urgency, a.reason) for a in strategy.manage(pos, m)]


# --- entry ---------------------------------------------------------------------------------

def test_buys_the_atm_straddle_when_iv_is_cheap():
    (c,) = gs().scan(mkt(), Book())
    put, call = c.legs
    assert (put.contract.right, call.contract.right) == (Right.PUT, Right.CALL)
    assert put.contract.strike == call.contract.strike == ATM
    assert put.contract.expiry == call.contract.expiry == EXPIRY
    assert put.ratio == call.ratio == 1
    iv = straddle_iv()
    assert c.rank_score == pytest.approx(0.35 / iv - 1)
    debit = -c.net_credit
    assert debit > 0 and c.max_loss == pytest.approx(debit * 100 * c.quantity)
    assert c.strategy == "gamma_scalp" and c.rule_trace == (
        f"iv_rv:{iv / 0.35:.3f}", "dte:52", "strike:101")


def test_the_iv_rv_threshold_is_inclusive():
    iv = straddle_iv()
    assert len(gs(max_iv_rv_ratio=iv / 0.35 + 1e-9).scan(mkt(), Book())) == 1
    assert gs(max_iv_rv_ratio=iv / 0.35 - 1e-6).scan(mkt(), Book()) == []


def _no_iv():
    ch = cached_chain()
    return dataclasses.replace(ch, options=tuple(
        dataclasses.replace(o, iv_mid=dataclasses.replace(o.iv_mid, value=None))
        for o in ch.options))


def _only_calls():
    ch = cached_chain()
    return dataclasses.replace(ch, options=tuple(o for o in ch.options
                                                 if o.contract.right is Right.CALL))


def _no_quotes():
    ch = cached_chain()
    return dataclasses.replace(ch, options=tuple(
        dataclasses.replace(o, quote=dataclasses.replace(o.quote, bid=None)) for o in ch.options))


@pytest.mark.parametrize(("strategy", "m"), [
    (lambda: gs(eligible=()), mkt),                                      # geen aandelen
    (gs, lambda: market(capital=RICH)),                                  # geen VolState
    (gs, lambda: mkt(rv=None)),                                          # geen RV-verwachting
    (gs, lambda: mkt(rv=0.27)),                                          # IV niet goedkoop
    (lambda: gs(dte=(300, 400)), mkt),                                   # geen expiratie
    (gs, lambda: mkt(events=Events("TST", None))),                       # events oud
    (gs, lambda: mkt(events=Events("TST", NOW.date(), (date(2027, 1, 5),)))),   # earnings
    (lambda: gs(rules=dataclasses.replace(RULES, earnings_blackout=False)),
     lambda: mkt(events=Events("TST", NOW.date(), (date(2027, 1, 5),)))),       # ook zonder blackout
    (gs, lambda: mkt(capital=25_000.0)),                                 # sizing 0
    (gs, lambda: mkt(chain=_no_iv())),                                   # geen IV
    (gs, lambda: mkt(chain=_only_calls())),                              # geen put
    (gs, lambda: mkt(chain=_no_quotes())),                               # geen mark
])
def test_gamma_scalp_skips(strategy, m):
    assert strategy().scan(m(), Book()) == []


def test_earnings_after_the_expiry_do_not_block():
    m = mkt(events=Events("TST", NOW.date(), (date(2027, 1, 20),)))
    assert len(gs().scan(m, Book())) == 1


# --- beheer --------------------------------------------------------------------------------

CLOSE_STOP = ActionKind.CLOSE, Urgency.STOP
CLOSE_NORMAL = ActionKind.CLOSE, Urgency.NORMAL


@pytest.fixture(scope="module")
def straddle():
    (c,) = gs().scan(mkt(), Book())
    return open_position(c)


def hedged(pos, pnl_frac: float):
    """`pos` met een aandelen-leg waarvan de P&L bij spot 100 `pnl_frac` × debit is."""
    debit = -pos.entry_credit
    qty = pos.units * 100
    return dataclasses.replace(pos, legs=pos.legs + (
        Leg(None, "TST", qty, 100.0 - pnl_frac * debit),))


@pytest.mark.parametrize(("kw", "m", "want"), [
    ({}, lambda: later(now=at(2027, 1, 14, 15)), [(*CLOSE_STOP, "expiry")]),
    ({}, lambda: later(130.0), [(*CLOSE_NORMAL, "take_profit")]),
    ({}, lambda: later(base=0.10), [(*CLOSE_STOP, "stop_loss")]),
    ({}, lambda: later(rv=0.20), [(*CLOSE_NORMAL, "iv_normalized")]),
    ({"stop_loss_frac_debit": 0.1}, lambda: later(now=at(2027, 1, 4)), [(*CLOSE_NORMAL, "dte")]),
    ({}, later, []),
    ({}, lambda: dataclasses.replace(later(), chains={}), []),          # zonder keten: alleen DTE
    ({}, lambda: dataclasses.replace(later(now=at(2027, 1, 4)), chains={}),
     [(*CLOSE_NORMAL, "dte")]),
    ({}, lambda: dataclasses.replace(later(), vol={}), []),             # zonder VolState
    ({}, lambda: later(rv=None), []),                                   # zonder RV-verwachting
    ({}, lambda: dataclasses.replace(later(), chains={"TST": _no_quotes()}), []),
])
def test_gamma_scalp_exits_in_order(straddle, kw, m, want):
    assert reasons(gs(**kw), straddle, m()) == want


@pytest.mark.parametrize(("pnl_frac", "want"), [
    (0.35, [(*CLOSE_NORMAL, "take_profit")]),       # mark ≈ debit, hedge +35 %: ≥ 1,30
    (0.25, []),
    (-0.45, [(*CLOSE_STOP, "stop_loss")]),          # mark ≈ debit, hedge −45 %: ≤ 0,60
    (-0.35, []),
])
def test_the_value_includes_the_realised_hedge_pnl(straddle, pnl_frac, want):
    assert reasons(gs(), hedged(straddle, pnl_frac), later()) == want


@pytest.mark.parametrize(("realized_frac", "want"), [
    (0.35, [(*CLOSE_NORMAL, "take_profit")]),       # alleen vastgelegde hedge-winst
    (0.25, []),
    (-0.45, [(*CLOSE_STOP, "stop_loss")]),
])
def test_the_value_includes_the_hedge_pnl_already_locked_in(straddle, realized_frac, want):
    debit = -straddle.entry_credit
    pos = dataclasses.replace(
        straddle, hedge_realized_usd=realized_frac * debit * straddle.units * 100)
    assert pos.shares == 0
    assert reasons(gs(), pos, later()) == want


def test_realised_and_open_hedge_pnl_add_up(straddle):
    debit = -straddle.entry_credit
    half = 0.175 * debit * straddle.units * 100
    pos = dataclasses.replace(hedged(straddle, 0.175), hedge_realized_usd=half)
    assert reasons(gs(), pos, later()) == [(*CLOSE_NORMAL, "take_profit")]


def test_iv_normalized_uses_sigma_at_the_remaining_dte(straddle):
    # Bij σ(h) gelijk aan de IV van de straddle gaat hij dicht (IV ≥ σ(h)).
    iv = straddle_iv()
    assert reasons(gs(), straddle, later(rv=iv)) == [(*CLOSE_NORMAL, "iv_normalized")]
    assert reasons(gs(), straddle, later(rv=iv + 1e-6)) == []


# --- registry ------------------------------------------------------------------------------

def _with_gamma(hedging: bool):
    return CFG.model_copy(update={
        "strategies": S.model_copy(update={
            "gamma_scalp": S.gamma_scalp.model_copy(update={"enabled": True})}),
        "hedging": CFG.hedging.model_copy(update={"enabled": hedging})})


def test_gamma_scalp_needs_hedging():
    assert not S.gamma_scalp.enabled and "gamma_scalp" not in [s.name for s in
                                                               build_strategies(CFG)]
    with pytest.raises(ConfigContractError, match="delta-hedge"):
        build_strategies(_with_gamma(False))
    unlisted = _with_gamma(True).model_copy(update={"hedging": CFG.hedging.model_copy(
        update={"enabled": True, "strategies": ()})})
    with pytest.raises(ConfigContractError, match=r"hedging\.strategies"):
        build_strategies(unlisted)
    names = [s.name for s in build_strategies(_with_gamma(True))]
    assert names[names.index("long_straddle_strangle") + 1] == "gamma_scalp"


def test_gamma_scalp_is_hedged_and_allowed_when_vol_is_cheap_or_normal():
    assert CFG.hedging.strategies == ("gamma_scalp",) and not CFG.hedging.enabled
    allowed = CFG.vol_regime.allowed
    assert "gamma_scalp" in allowed.cheap and "gamma_scalp" in allowed.normal
    assert "gamma_scalp" not in allowed.rich + allowed.stress + allowed.unknown
