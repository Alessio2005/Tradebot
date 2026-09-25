# tests/unit/test_phase11_carry.py
"""H-11.1, de carrykandidaat: wat vóór de eerste fit wordt vastgelegd. Fase 11, stap 6.

De pre-registratie van H-11.1 pint drie dingen die vooraf uitrekenbaar zijn en
daarom vooraf worden uitgerekend, niet na de meting (R-2, R-10):

* de DSR-drempel -- de geannualiseerde Sharpe die bij `M = 25` en het venster
  `DSR >= 0,95` haalt, gevonden met bisectie op `backtest.metrics.deflated_sharpe`,
  de enige DSR-implementatie (R-3);
* de adequaatheidsvloer per houdduur, nominaal en rho-gedefleerd (AD-20);
* het trialbudget: 5 besteed in fase 10, 3 gepland hier, tegen `m_new = 25`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.utils.failfast import DataContractError
from tradebot.validation.phase11_carry import (
    HOLDING_PERIODS,
    PRIMARY_HOLDING_PERIOD,
    adequacy_by_holding_period,
    dsr_hurdle,
    spent_since_reset,
    trial_budget_after,
)


class TestTheDsrHurdle:
    """De drempel die de pre-registratie noemt, komt uit de ene DSR-functie."""

    def test_it_reproduces_the_ledger_reset_table_on_w_full(self) -> None:
        """`registry/trial_budget.py` noemt 1,67 bij M = 25 op n_obs = 1743."""
        assert dsr_hurdle(n_obs=1743, n_trials=25, bars_per_year=365.0) == \
            pytest.approx(1.67, abs=0.005)

    def test_it_reproduces_the_h10_1_hurdle_on_w_dev(self) -> None:
        """H-10.1 rekende op W_DEV (1.390 bars) een eis van 1,8686."""
        assert dsr_hurdle(n_obs=1390, n_trials=25, bars_per_year=365.0) == \
            pytest.approx(1.8686, abs=5e-4)

    def test_more_trials_raise_the_bar(self) -> None:
        low = dsr_hurdle(n_obs=1390, n_trials=25, bars_per_year=365.0)
        high = dsr_hurdle(n_obs=1390, n_trials=50, bars_per_year=365.0)
        assert high > low


class TestTheAdequacyFloor:
    """Stap 6.2: een poort waarvan vooraf uitrekenbaar is dat zij niet gehaald
    kan worden, hoort niet in een pre-registratie. Reken hem uit."""

    def test_the_nominal_counts_are_the_ones_the_prompt_names(self) -> None:
        table = adequacy_by_holding_period(n_obs=1390, n_eff=1.2651)
        assert [table[h]["nonoverlapping_periods"] for h in HOLDING_PERIODS] == \
            [463, 278, 139]

    def test_the_conservative_reading_is_the_smaller_one(self) -> None:
        """AD-20: twee tellingen, het oordeel valt op de conservatieve. Voor
        een boek is dat de kleinste van de nominale telling en de
        rho-gedefleerde paneltelling."""
        table = adequacy_by_holding_period(n_obs=1390, n_eff=0.5)
        cell = table[10]
        assert cell["effective_bets_rho_deflated"] == pytest.approx(139 * 0.5)
        assert cell["conservative"] == pytest.approx(69.5)
        assert cell["passes_floor"] is False

    def test_every_registered_cell_clears_the_floor_at_the_measured_breadth(
        self,
    ) -> None:
        table = adequacy_by_holding_period(n_obs=1390, n_eff=1.2651)
        assert all(table[h]["passes_floor"] for h in HOLDING_PERIODS)

    def test_h_20_would_not_have_cleared_it(self) -> None:
        """De reden dat het rooster bij 10 ophoudt, uitgerekend."""
        assert int(np.floor(1390 / 20)) < 100

    def test_the_primary_cell_is_one_of_the_registered_ones(self) -> None:
        assert PRIMARY_HOLDING_PERIOD in HOLDING_PERIODS


class TestTheTrialBudget:
    """De besteding komt uit de ledger en de rem is `assert_within_budget`
    (R-3): er is geen tweede budgetcontrole bijgekomen."""

    RESET = ROOT / "artefacts/governance/ledger_reset.json"
    LEDGER = ROOT / "artefacts/governance/hypothesis_ledger.json"

    def test_fase_10_spent_five_and_the_ledger_says_so(self) -> None:
        """H-10.1: 4 trials, H-10.2: 1, H-10.3: 0 (vervallen). H-10.3 draagt
        zelf `stage_c_trials_spent: 5`; de optelling moet daarop uitkomen."""
        assert spent_since_reset(self.LEDGER) == 5

    def test_five_spent_plus_three_planned_is_eight_of_twenty_five(self) -> None:
        budget = trial_budget_after(spent_before=5, planned=len(HOLDING_PERIODS),
                                    reset_path=self.RESET)
        assert budget == {"spent_before": 5, "planned": 3, "spent_after": 8,
                          "m_new": 25, "remaining_after": 17}

    def test_a_plan_over_the_budget_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="budget"):
            trial_budget_after(spent_before=24, planned=3, reset_path=self.RESET)


class TestTheInjectedParameters:
    """Wat bij het bevriezen in de pre-registratie-ID terechtkomt."""

    @pytest.fixture(scope="class")
    def params(self) -> dict:
        from tradebot.validation.phase11_carry import h11_parameters
        return h11_parameters(ROOT)

    def test_they_pin_the_primary_cell_and_the_grid(self, params: dict) -> None:
        assert params["primary_cell"] == {
            "construction": "cross_sectional_dollar_neutral",
            "holding_period": 10, "layer": "L3_execution",
            "metric": "net_sharpe", "window": "W_DEV"}
        assert params["holding_periods"] == [3, 5, 10]
        assert params["dropped_construction"]["name"] == "beta_hedged"

    def test_they_carry_the_policy_that_will_decide(self, params: dict) -> None:
        """AD-27: de pre-registratie noemt het beleid waaronder gemeten wordt."""
        from tradebot.registry.policy_carry import current_policy_hash
        assert params["risk_policy_hash"] == current_policy_hash()

    def test_the_window_is_w_dev_and_the_gate_is_unread(self, params: dict) -> None:
        window = params["window"]
        assert (window["start"], window["end"], window["n_bars"]) == \
            ("2021-11-15", "2025-09-04", 1390)
        assert params["gate_reads_at_freeze"] == []

    def test_the_hurdles_are_computed_not_copied(self, params: dict) -> None:
        dsr = params["dsr"]
        assert dsr["m_new"] == 25 and dsr["n_obs"] == 1390
        assert dsr["required_annual_sharpe_gaussian"] == pytest.approx(1.8686, abs=5e-4)
        assert dsr["t2_hurdle"] == pytest.approx(2.0 / np.sqrt(1390 / 365.0))

    def test_the_adequacy_is_measured_on_w_dev_and_clears_the_floor(
        self, params: dict,
    ) -> None:
        assert 1.0 < params["n_eff_w_dev"] < 2.0
        assert set(params["adequacy"]) == {"3", "5", "10"}
        assert all(cell["passes_floor"] for cell in params["adequacy"].values())

    def test_the_budget_is_the_cumulative_one(self, params: dict) -> None:
        assert params["trial_budget"] == {"spent_before": 5, "planned": 3,
                                          "spent_after": 8, "m_new": 25,
                                          "remaining_after": 17}

    def test_the_same_inputs_give_the_same_id(self, params: dict) -> None:
        from tradebot.validation.phase11_carry import h11_parameters, h11_preregistration
        assert h11_preregistration(ROOT, params).preregistration_id == \
            h11_preregistration(ROOT, h11_parameters(ROOT)).preregistration_id


class TestTheFreeze:
    def test_it_writes_once_and_a_second_freeze_is_a_no_op(
        self, tmp_path: Path,
    ) -> None:
        from tradebot.validation.phase11_carry import freeze_h11
        first = freeze_h11(ROOT, directory=tmp_path, git_sha="deadbee")
        second = freeze_h11(ROOT, directory=tmp_path, git_sha="deadbee")
        assert first == second and first.is_file()
        assert first.name.startswith("preregistration_")
