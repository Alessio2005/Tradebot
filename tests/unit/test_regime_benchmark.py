# tests/unit/test_regime_benchmark.py
"""Het oordeel van H2: de vijf stop-criteria in de volgorde van de pre-registratie.

De campagne zelf draait de engine en is duur; hij wordt getoetst in
`tests/integration`. Wat hier wordt afgedwongen is de POORT -- de logica die
bepaalt of een uitkomst `PROMOTED`, `UNPROVEN`, `DESCOPED` of `FALSIFIED` heet.
Die logica is waar een fase mis kan gaan zonder dat er iets rood wordt: een
`FALSIFIED` op een onderpowerde toets, of een `PROMOTED` op een spread-aanname.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.validation.regime_benchmark import judge_conditioner
from tradebot.validation.sharpe_difference import (
    SharpeDifferenceResult,
    SharpeTestControls,
)

DM = SharpeDifferenceResult(
    sharpe_a=-0.30, sharpe_b=-0.33, difference=0.03, z_statistic=0.4,
    p_value=0.34, standard_error=0.07, correlation=0.97, n_obs=1200,
    alternative="a-better", significant=False)

#: De controle zoals de pre-registratie hem vóór de run verwacht: de toets houdt
#: zijn niveau, maar ziet het verwachte effect van 0,08 niet.
UNDERPOWERED = SharpeTestControls(
    n_replicates=500, alpha=0.05, size_rejection_rate=0.048,
    power_at_expected_effect=0.09, power_at_mde=0.81, expected_effect=0.08,
    minimum_detectable_effect=0.59, target_power=0.8, size_passed=True,
    power_passed=False)


def _judge(**overrides):
    kwargs = dict(
        rarest_state_obs_per_fold=200.0, min_state_obs_required=100.0,
        net_sharpe_delta=0.05, turnover_ratio=1.1, turnover_threshold=2.0,
        spread_at_which_gain_vanishes=None, spread_threshold=3.0,
        dm=DM, controls=UNDERPOWERED)
    kwargs.update(overrides)
    return judge_conditioner(**kwargs)


class TestOccupancyIsAbsorbing:
    def test_below_the_gate_is_descoped_and_never_falsified(self) -> None:
        """No-go 8: een model dat de Data Adequacy Gate niet haalt, mag geen
        `FALSIFIED` krijgen. Er is niets geschat, dus niets weerlegd."""
        verdict = _judge(rarest_state_obs_per_fold=41.0,
                         net_sharpe_delta=-0.9, turnover_ratio=5.0)
        assert verdict.status == "DESCOPED"
        assert verdict.binding == ("state_occupancy_below_adequacy",)
        assert "41.0" in verdict.rationale

    def test_it_outranks_every_other_criterium(self) -> None:
        verdict = _judge(rarest_state_obs_per_fold=99.9)
        assert verdict.status == "DESCOPED"


class TestNoImprovement:
    def test_is_unproven_and_not_falsified(self) -> None:
        """De power-analyse stelt vóór de run vast dat 0,08 niet zichtbaar is.
        Een niet-verbetering is dan onvermogen om te meten, geen afwezigheid."""
        verdict = _judge(net_sharpe_delta=-0.02)
        assert verdict.status == "UNPROVEN"
        assert verdict.binding == ("no_sharpe_improvement_after_costs",)
        assert "0.08" in verdict.rationale or "0,08" in verdict.rationale

    def test_a_delta_of_exactly_zero_still_binds(self) -> None:
        assert _judge(net_sharpe_delta=0.0).status == "UNPROVEN"

    def test_the_rationale_names_the_measured_power(self) -> None:
        verdict = _judge(net_sharpe_delta=-0.02)
        assert "9.0%" in verdict.rationale


class TestTurnover:
    def test_doubling_without_a_net_gain_is_a_falsification(self) -> None:
        """Dit criterium hangt niet van de Sharpe-power af: het zegt dat de
        extra fees niet zijn terugverdiend, en dat is een economische
        vaststelling."""
        verdict = _judge(turnover_ratio=2.4, net_sharpe_delta=-0.01)
        assert verdict.status == "FALSIFIED"
        assert "turnover_doubles" in verdict.binding
        assert "no_sharpe_improvement_after_costs" in verdict.binding

    def test_doubling_with_a_net_gain_does_not_bind(self) -> None:
        """De netto Sharpe heeft de fees al betaald. Verdubbelde turnover die
        zichzelf terugverdient, is geen falsificatie."""
        verdict = _judge(turnover_ratio=2.4, net_sharpe_delta=0.05)
        assert verdict.status == "PROMOTED"

    def test_just_under_the_threshold_does_not_bind(self) -> None:
        verdict = _judge(turnover_ratio=1.99, net_sharpe_delta=-0.01)
        assert verdict.status == "UNPROVEN"


class TestSpreadSensitivity:
    def test_a_gain_that_dies_at_three_bp_is_a_falsification(self) -> None:
        verdict = _judge(net_sharpe_delta=0.04,
                         spread_at_which_gain_vanishes=2.0)
        assert verdict.status == "FALSIFIED"
        assert verdict.binding == (
            "improvement_disappears_at_three_bp_spread",)

    def test_a_gain_that_survives_ten_bp_promotes(self) -> None:
        verdict = _judge(net_sharpe_delta=0.04,
                         spread_at_which_gain_vanishes=10.0)
        assert verdict.status == "PROMOTED"

    def test_it_cannot_bind_without_a_measured_gain(self) -> None:
        """Zonder winst is er niets dat bij 3 bp kan verdwijnen. Het criterium
        is dan niet van toepassing en niet geschonden -- anders zou elke arm
        zonder verbetering óók een spread-falsificatie krijgen."""
        verdict = _judge(net_sharpe_delta=-0.05,
                         spread_at_which_gain_vanishes=None)
        assert verdict.status == "UNPROVEN"
        assert "improvement_disappears_at_three_bp_spread" not in verdict.binding


class TestPromotion:
    def test_requires_every_criterium_clear(self) -> None:
        verdict = _judge(net_sharpe_delta=0.12, turnover_ratio=1.05,
                         spread_at_which_gain_vanishes=25.0)
        assert verdict.status == "PROMOTED"
        assert verdict.binding == ()
        assert "Phase 2" in verdict.rationale

    @pytest.mark.parametrize(
        "overrides",
        [
            {"rarest_state_obs_per_fold": 50.0},
            {"net_sharpe_delta": -0.01},
            {"net_sharpe_delta": 0.04, "spread_at_which_gain_vanishes": 1.0},
            {"net_sharpe_delta": -0.01, "turnover_ratio": 3.0},
        ],
    )
    def test_any_single_binding_criterium_blocks_it(self, overrides) -> None:
        assert _judge(**overrides).status != "PROMOTED"
