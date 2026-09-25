"""De haalbaarheidspoort van H-11.2, vóór registratie. Fase 11, stap 7.

Drie bouwstenen, geen enkel echt rendement:

* `null_detection_limit`: het kleinste Sharpe-verschil dat de verschiltoets
  van de kern op deze sample kan onderscheiden, gemeten op synthetische
  rendementen MET GEMIDDELDE NUL uit de covariantie van `W_DEV`. Onder de
  nulhypothese hoort het gemiddelde verschil rond nul te liggen; dat is de
  controle dat de simulatie de toets niet scheef voedt.
* `cost_gain_sharpe`: de Sharpe-winst uit kosten alleen, bij nul
  signaalverval: omzetdaling maal kosten per zijde, geannualiseerd, gedeeld
  door de boekvolatiliteit.
* `impact_multiplier`: de verhouding tussen de omzetkosten op L3 en die op de
  L0-kostenas, en hoe zij met de boekgrootte schaalt. Het impactmodel is de
  wortelwet (`execution/impact_model.py`): impact per eenheid schaalt met de
  wortel van de grootte, fees en spread niet.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.phase11_decision_clock_feasibility import (
    cost_gain_sharpe,
    impact_multiplier,
    null_detection_limit,
)

IDX = pd.date_range("2022-01-01", periods=600, freq="D", tz="UTC")


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(50)
    fast = pd.DataFrame(np.sign(rng.standard_normal((len(IDX), 4))) / 4.0, index=IDX,
                        columns=list("ABCD"))
    slow = pd.DataFrame(np.repeat(fast.to_numpy()[::20], 20, axis=0)[: len(IDX)],
                        index=IDX, columns=list("ABCD"))
    return fast, slow


def test_under_the_null_the_mean_difference_is_about_zero() -> None:
    fast, slow = _panels()
    result = null_detection_limit(fast, slow, np.eye(4) * 1e-4, n_paths=40, seed=1,
                                  n_boot=50, bars_per_year=365.0, ci_level=0.95)
    assert abs(result["mean_delta"]) < 4.0 * result["median_se"] / math.sqrt(40)
    assert result["detectable_two_sided"] > result["detectable_one_sided"] > 0.0


def test_the_null_simulation_is_reproducible() -> None:
    fast, slow = _panels()
    kwargs = {"n_paths": 10, "seed": 3, "n_boot": 50, "bars_per_year": 365.0,
              "ci_level": 0.95}
    assert null_detection_limit(fast, slow, np.eye(4) * 1e-4, **kwargs) == \
        null_detection_limit(fast, slow, np.eye(4) * 1e-4, **kwargs)


def test_the_cost_gain_is_turnover_times_cost_over_volatility() -> None:
    gain = cost_gain_sharpe(turnover_reference=0.15, turnover_held=0.03,
                            cost_per_side=6.5e-4, book_volatility=0.233,
                            bars_per_year=365.0)
    assert gain == pytest.approx(0.12 * 6.5e-4 * 365.0 / 0.233)


def test_a_held_book_with_more_turnover_is_refused() -> None:
    with pytest.raises(DataContractError):
        cost_gain_sharpe(turnover_reference=0.03, turnover_held=0.15,
                         cost_per_side=6.5e-4, book_volatility=0.2, bars_per_year=365.0)


def test_the_impact_multiplier_scales_with_the_root_of_book_size() -> None:
    base = impact_multiplier(fees=1987.05, impact=1267.59, spread=361.28, scale=1.0)
    assert base == pytest.approx(3615.92 / 2348.33, rel=1e-5)
    four = impact_multiplier(fees=1987.05, impact=1267.59, spread=361.28, scale=4.0)
    assert four - 1.0 == pytest.approx(2.0 * (base - 1.0))


# --------------------------------------------------------------------------- #
# De bronnen: het ladderartefact en het beleidsregister.
# --------------------------------------------------------------------------- #
def _ladder(tmp_path: Path) -> Path:
    import json

    rows = [
        {"track": "t", "layer": "L1_sovereign", "mean_gross": 0.1, "net_sharpe": 0.2},
        {"track": "t", "layer": "L3_execution", "cost_fees": 55.0, "cost_impact": 30.0,
         "cost_spread": 10.0, "cost_funding": 5.0, "net_sharpe": -0.2,
         "risk_policy_hash": "old"},
        {"track": "u", "layer": "L3_execution", "cost_fees": float("nan")},
    ]
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def _registry(tmp_path: Path) -> Path:
    import json

    entries = [
        {"config_hash": "old", "audit_header": {"sigma_target": 0.08, "gross_cap": 1.5}},
        {"config_hash": "new", "audit_header": {"sigma_target": 0.20, "gross_cap": 4.0}},
    ]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def test_the_ladder_cost_mix_is_read_for_one_track(tmp_path: Path) -> None:
    from tradebot.validation.phase11_decision_clock_feasibility import ladder_cost_mix

    mix = ladder_cost_mix(_ladder(tmp_path), track="t")
    assert (mix["fees"], mix["impact"], mix["spread"]) == (55.0, 30.0, 10.0)
    assert mix["risk_policy_hash"] == "old"
    assert mix["mean_gross_l1"] == 0.1


def test_a_track_without_a_finite_cost_mix_is_refused(tmp_path: Path) -> None:
    from tradebot.validation.phase11_decision_clock_feasibility import ladder_cost_mix

    with pytest.raises(DataContractError):
        ladder_cost_mix(_ladder(tmp_path), track="u")


def test_book_scales_come_from_the_two_registered_policies(tmp_path: Path) -> None:
    from tradebot.validation.phase11_decision_clock_feasibility import policy_book_scales

    scales = policy_book_scales(_registry(tmp_path), ladder_hash="old", current_hash="new",
                                ladder_mean_gross=0.1)
    assert scales["vol_target"] == pytest.approx(2.5)
    assert scales["gross_cap_bound"] == pytest.approx(40.0)
    same = policy_book_scales(_registry(tmp_path), ladder_hash="new", current_hash="new",
                              ladder_mean_gross=0.1)
    assert same["vol_target"] == pytest.approx(1.0)


def test_an_unregistered_policy_is_refused(tmp_path: Path) -> None:
    from tradebot.validation.phase11_decision_clock_feasibility import policy_book_scales

    with pytest.raises(DataContractError):
        policy_book_scales(_registry(tmp_path), ladder_hash="old", current_hash="elders",
                           ladder_mean_gross=0.1)


# --------------------------------------------------------------------------- #
# De poort zelf, zoals zij vooraf staat (stap 7.3).
# --------------------------------------------------------------------------- #
def _verdict(gain: float, *, met: bool = True) -> dict[str, object]:
    from tradebot.validation.phase11_decision_clock_feasibility import gate_verdict

    return gate_verdict(gain_l0=gain, multipliers={"ladder": 1.5, "gross_cap_bound": 4.0},
                        detectable_two_sided=0.8, detectable_one_sided=0.7,
                        precondition_met=met)


def test_the_gate_is_green_only_when_the_ladder_gain_beats_the_two_sided_limit() -> None:
    assert _verdict(0.6)["gate"] == "GREEN"      # 0,9 > 0,8
    assert _verdict(0.5)["gate"] == "RED"        # 0,75 < 0,8, ook al is 0,75 > 0,7


def test_without_the_new_ladder_only_a_red_that_no_book_can_flip_is_final() -> None:
    assert _verdict(0.1, met=False)["gate"] == "RED"          # 4 * 0,1 < 0,7
    assert _verdict(0.1, met=False)["red_under_every_book"] is True
    assert _verdict(0.3, met=False)["gate"] == "PENDING_7_1"  # 4 * 0,3 > 0,7
    assert _verdict(0.6, met=False)["gate"] == "PENDING_7_1"  # groen hoort op de nieuwe ladder


def test_the_required_multiplier_is_the_limit_over_the_l0_gain() -> None:
    assert _verdict(0.1)["multiplier_required_two_sided"] == pytest.approx(8.0)
