"""Het budget is de enige rem op de manoeuvre uit stap 3."""
from __future__ import annotations

from pathlib import Path

import pytest

from tradebot.registry.ledger_reset import freeze_reset
from tradebot.registry.trial_budget import assert_within_budget, remaining
from tradebot.utils.failfast import DataContractError

LEDGER = Path("artefacts/governance/hypothesis_ledger.json")


def _reset(tmp_path: Path, m_new: int = 25) -> Path:
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=m_new, rationale="test", git_sha="deadbee", out=out)
    return out


def test_a_plan_inside_the_budget_passes(tmp_path: Path) -> None:
    assert_within_budget(6, reset_path=_reset(tmp_path))


def test_a_plan_over_the_budget_crashes(tmp_path: Path) -> None:
    with pytest.raises(DataContractError):
        assert_within_budget(26, reset_path=_reset(tmp_path))


def test_inheriting_optuna_trials_blows_the_budget(tmp_path: Path) -> None:
    """R4, becijferd. Het CPCV-ensemble draagt 200 x 6 x 2 = 2400 trials.
    Wie het hergebruikt, tilt M van 25 naar 2425 en de DSR-eis van 1,67 naar
    2,36 -- en dat hoort te crashen in plaats van stilzwijgend door te gaan."""
    with pytest.raises(DataContractError):
        assert_within_budget(2400, reset_path=_reset(tmp_path))


def test_a_conditional_branch_counts_as_a_trial(tmp_path: Path) -> None:
    """Nieuw in revisie 2. Stap 9 kent een terugvalpad: haalt de driedelige
    toestand de bezettingspoort niet, dan wordt de tweedelige gemeten. Dat is
    een tweede specificatie op dezelfde data en dus een tweede trial, ook al
    voelt het als hetzelfde experiment.

    Een beslisboom met B takken die op de data wordt doorlopen, kost B trials
    en niet 1. Dit is de rekenregel die revisie 1 ontbrak."""
    path = _reset(tmp_path, m_new=2)
    assert_within_budget(2, reset_path=path)          # k=3 en k=2, beide geboekt
    with pytest.raises(DataContractError):
        assert_within_budget(3, reset_path=path)


def test_remaining_counts_down(tmp_path: Path) -> None:
    path = _reset(tmp_path)
    assert remaining(reset_path=path, ledger_path=LEDGER, booked=0) == 25
    assert remaining(reset_path=path, ledger_path=LEDGER, booked=6) == 19
