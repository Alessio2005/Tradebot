"""Toelating van kandidaten: het regime en de theta-harvest-poort."""
from __future__ import annotations

import dataclasses
import math

import pytest

from tests.unit.options.market_helpers import CFG, EXPIRIES, fake, fake_chain
from tradebot.options.contracts import OptionContract, Right
from tradebot.options.positions import Candidate, LegIntent
from tradebot.options.selection import SelectionRules, admit, selection_rules, theta_cover
from tradebot.options.vol_regime import RVForecast, VolRegime, VolState

SHORT = fake(95.0, Right.PUT)            # theta −0,01, gamma 0,01 per aandeel
LONG = fake(90.0, Right.PUT)
CHAIN = fake_chain(SHORT, LONG)
RULES = selection_rules(CFG)


def vol(rv: float = 0.20) -> VolState:
    return VolState(VolRegime.RICH, 0.25, RVForecast(rv, rv, rv, rv), 0.01, "vrp_rich")


def cand(strategy: str = "credit_spread", credit: float = 0.5, legs=None,
         max_loss: float = 400.0, qty: int = 2) -> Candidate:
    legs = legs if legs is not None else (LegIntent(SHORT.contract, -1),)
    return Candidate(strategy, "TST", tuple(legs), qty, credit, max_loss, 1.0, ("t",))


def test_the_rules_come_from_the_config():
    assert RULES.regime_enabled and RULES.theta_enabled
    assert RULES.allowed[VolRegime.STRESS] == frozenset()
    assert "iron_condor" in RULES.allowed[VolRegime.RICH]
    assert "iron_condor" not in RULES.allowed[VolRegime.NORMAL]
    assert RULES.min_theta_gamma_cover == 1.0


def test_theta_cover():
    # θ 1 USD/dag, Γ −1 aandeel/USD, S 100, σ 0,20: 365 / (0,5 × 10 000 × 0,04) = 1,825.
    assert theta_cover(1.0, -1.0, 100.0, 0.20) == pytest.approx(1.825)
    assert math.isinf(theta_cover(1.0, 0.0, 100.0, 0.20))


def test_an_allowed_credit_with_enough_theta_is_admitted():
    a = admit(cand(), CHAIN, vol(), VolRegime.RICH, RULES)
    assert a.admitted and a.blocker == ""
    assert a.candidate.rule_trace == ("t", "regime:rich:ok", "theta:cover=1.82:eff=0.0050:ok")


def test_the_regime_blocks_a_strategy_it_does_not_allow():
    a = admit(cand("iron_condor"), CHAIN, vol(), VolRegime.NORMAL, RULES)
    assert not a.admitted and a.blocker == "regime"
    assert a.candidate.rule_trace[-1] == "regime:normal:no"
    assert not admit(cand(), CHAIN, vol(), VolRegime.STRESS, RULES).admitted
    assert not admit(cand(), CHAIN, vol(), VolRegime.UNKNOWN, RULES).admitted


def test_too_little_theta_for_the_expected_rv_is_blocked():
    # σ_RV 0,30: dekking 365 / (0,5 × 10 000 × 0,09) = 0,81 < 1.
    a = admit(cand(), CHAIN, vol(0.30), VolRegime.RICH, RULES)
    assert not a.admitted and a.blocker == "theta"
    assert a.candidate.rule_trace[-1] == "theta:cover=0.81:eff=0.0050:no"


def test_the_theta_efficiency_floor():
    strict = dataclasses.replace(RULES, min_theta_per_max_loss=0.01)
    assert admit(cand(), CHAIN, vol(), VolRegime.RICH, strict).blocker == "theta"


def test_a_credit_without_short_gamma_passes_on_cover():
    legs = (LegIntent(SHORT.contract, -1), LegIntent(LONG.contract, 1))   # Γ netto 0
    a = admit(cand(legs=legs), CHAIN, vol(), VolRegime.RICH,
              dataclasses.replace(RULES, regime_enabled=False))
    # θ netto 0: geen premie per dag, dus geblokkeerd op teken.
    assert a.blocker == "theta" and a.candidate.rule_trace == ("t", "theta:negative:0.00")
    rich_theta = fake(95.0, Right.CALL, delta=0.3)
    chain = fake_chain(SHORT, LONG, dataclasses.replace(
        rich_theta, greeks=dataclasses.replace(rich_theta.greeks, theta=-0.02, gamma=0.01)))
    legs = (LegIntent(rich_theta.contract, -1), LegIntent(LONG.contract, 1))
    a = admit(cand(legs=legs), chain, vol(), VolRegime.RICH, RULES)
    assert a.admitted and "theta:cover=inf" in a.candidate.rule_trace[-1]


@pytest.mark.parametrize("legs, state", [
    ((LegIntent(OptionContract("TST", EXPIRIES[0], 50.0, Right.PUT), -1),), vol()),
    (None, None),
    (None, VolState(VolRegime.NORMAL, 0.2, None, None, "no_rv")),
])
def test_missing_greeks_or_rv_blocks_a_credit(legs, state):
    a = admit(cand(legs=legs), CHAIN, state, VolRegime.NORMAL, RULES)
    assert a.blocker == "theta" and a.candidate.rule_trace[-1] == "theta:no_data"


def test_a_debit_skips_the_theta_gate():
    a = admit(cand("debit_vertical", credit=-1.0, legs=(LegIntent(SHORT.contract, 1),)),
              CHAIN, None, VolRegime.CHEAP, RULES)
    assert a.admitted and a.candidate.rule_trace == ("t", "regime:cheap:ok")


def test_both_gates_can_be_switched_off():
    off = SelectionRules(regime_enabled=False, theta_enabled=False)
    a = admit(cand(), CHAIN, None, VolRegime.UNKNOWN, off)
    assert a.admitted and a.candidate.rule_trace == ("t",)
