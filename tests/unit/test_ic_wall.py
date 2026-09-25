"""De IC-muur: welke IC de poort vraagt bij een gegeven breedte. Stap 5.

Drie functies en een simulatie:

* `dsr_hurdle` keert `backtest/metrics.py::deflated_sharpe` numeriek om. Het is
  geen tweede DSR-formule (R-3). De eerste test reproduceert het getal dat tot
  nu toe alleen in een ledger-notitie stond: 1,868609 bij M = 25, N = 1.390,
  onder normaliteit. De negatieve controle: de drempel stijgt met M en daalt
  met N.
* `t_hurdle_sharpe` is de t = 2-drempel uit het meetcontract.
* `required_ic` keert de fundamentele wet exact om.
* `simulate_wall` zet een voorspelling met BEKENDE IC op synthetische
  rendementen en meet de gerealiseerde Sharpe met de echte kern. Op
  onafhankelijke namen moet de wet daar kloppen; dat is de controle dat de
  simulatie zelf niet scheef is.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.breadth import (
    dsr_hurdle,
    required_ic,
    simulate_wall,
    t_hurdle_sharpe,
)

N_DEV = 1390


def _dsr(n_obs: int = N_DEV, n_trials: int = 25) -> float:
    return dsr_hurdle(n_obs=n_obs, n_trials=n_trials, skew=0.0, kurtosis=3.0,
                      sr_variance=1.0 / n_obs, bars_per_year=365.0, dsr_target=0.95)


def test_the_dsr_hurdle_reproduces_the_number_that_only_lived_in_a_ledger_note() -> None:
    assert _dsr() == pytest.approx(1.868609, abs=1e-5)


def test_the_hurdle_rises_with_trials_and_falls_with_sample() -> None:
    assert _dsr(n_trials=50) > _dsr(n_trials=25) > _dsr(n_trials=12)
    assert _dsr(n_obs=1743) < _dsr(n_obs=N_DEV)


def test_the_t_hurdle_matches_the_measurement_contract() -> None:
    assert t_hurdle_sharpe(n_obs=N_DEV, bars_per_year=365.0, t=2.0) == pytest.approx(
        1.0249, abs=1e-4)


def test_required_ic_inverts_the_law_exactly() -> None:
    ic = required_ic(1.8686, independent_bets=4.353, horizon_bars=30, bars_per_year=365.0)
    assert ic * math.sqrt(4.353 * 365.0 / 30) == pytest.approx(1.8686)


@pytest.mark.parametrize(("bets", "horizon"), [(0.0, 1), (-1.0, 1), (4.0, 0)])
def test_required_ic_refuses_what_has_no_breadth(bets: float, horizon: int) -> None:
    with pytest.raises(DataContractError):
        required_ic(1.0, independent_bets=bets, horizon_bars=horizon, bars_per_year=365.0)


def test_on_independent_names_the_simulation_obeys_the_law() -> None:
    result = simulate_wall(np.eye(6) * 1e-4, construction="directional", ic=0.05,
                           n_obs=200_000, seed=5, bars_per_year=365.0)
    assert result.ic_measured == pytest.approx(0.05, rel=0.05)
    assert result.independent_bets == pytest.approx(6.0, rel=0.01)
    assert result.ratio == pytest.approx(1.0, abs=0.05)


def test_the_simulation_is_reproducible() -> None:
    cov = np.eye(4) * 1e-4
    a = simulate_wall(cov, construction="directional", ic=0.1, n_obs=20_000, seed=9,
                      bars_per_year=365.0)
    b = simulate_wall(cov, construction="directional", ic=0.1, n_obs=20_000, seed=9,
                      bars_per_year=365.0)
    assert a.to_dict() == b.to_dict()


def test_an_impossible_ic_is_refused() -> None:
    with pytest.raises(DataContractError):
        simulate_wall(np.eye(4), construction="directional", ic=1.5, n_obs=100, seed=1,
                      bars_per_year=365.0)


# --------------------------------------------------------------------------- #
# Stap 6 — AD-29: een pre-registratie noemt haar breedte, horizon en muur.
# --------------------------------------------------------------------------- #
def _declaration(**overrides: object) -> dict[str, object]:
    ic = required_ic(1.8686, independent_bets=4.353, horizon_bars=30, bars_per_year=365.0)
    base: dict[str, object] = {
        "construction": "dollar_neutral", "n_names": 6, "independent_bets": 4.353,
        "horizon_bars": 30, "bars_per_year": 365.0, "sr_required": 1.8686,
        "required_ic": ic, "ic_evidence": "geen: dit is precies wat de hypothese toetst",
    }
    base.update(overrides)
    return base


def test_a_complete_and_consistent_declaration_passes() -> None:
    from tradebot.validation.breadth import assert_ic_wall_declared

    assert_ic_wall_declared({"ic_wall": _declaration()})


@pytest.mark.parametrize("missing", ["construction", "independent_bets", "horizon_bars",
                                     "sr_required", "required_ic", "ic_evidence"])
def test_a_declaration_without_a_field_is_refused(missing: str) -> None:
    from tradebot.validation.breadth import assert_ic_wall_declared

    declaration = _declaration()
    del declaration[missing]
    with pytest.raises(DataContractError):
        assert_ic_wall_declared({"ic_wall": declaration})


def test_a_preregistration_without_the_block_is_refused() -> None:
    from tradebot.validation.breadth import assert_ic_wall_declared

    with pytest.raises(DataContractError):
        assert_ic_wall_declared({"hypothesis": "iets"})


def test_breadth_above_the_number_of_names_is_refused() -> None:
    """De les van DI-35: 123,58 weddenschappen op zes namen bestaat niet."""
    from tradebot.validation.breadth import assert_ic_wall_declared

    with pytest.raises(DataContractError):
        assert_ic_wall_declared({"ic_wall": _declaration(independent_bets=123.58)})


def test_a_required_ic_that_does_not_follow_from_its_own_numbers_is_refused() -> None:
    from tradebot.validation.breadth import assert_ic_wall_declared

    with pytest.raises(DataContractError):
        assert_ic_wall_declared({"ic_wall": _declaration(required_ic=0.05)})
