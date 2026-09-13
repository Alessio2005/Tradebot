# tests/unit/test_phase10_decision_frequency.py
"""De campagnelogica van H-10.1, op SYNTHETISCHE reeksen.

Waarom synthetisch en niet de echte ladder: deze tests moeten de vier
beslissingen vastpinnen die de campagne neemt, en die zijn onafhankelijk van de
markt. Een test die de ladder draait, meet vooral dat de ladder draait -- hij is
traag, hij faalt zodra de data verschuift, en hij zegt niets over de vraag of de
referentiecel zichzelf toetst.

De vier vastgepinde beslissingen:

1. `k = 1` is de REFERENTIE en wordt niet tegen zichzelf getoetst. Sinds commit
   812a232 heeft `sharpe_difference_test` een schaal-invariante ontaardingstak
   die op twee identieke reeksen nul-met-zekerheid teruggeeft; dat is het juiste
   antwoord, maar het is een ANDER antwoord dan "nul per constructie". De
   campagne mag die tak niet nodig hebben.
2. Een omzetverschil van exact nul heeft geen breakevenniveau. `breakeven_cost_
   bps` gooit dan `DataContractError` -- bij ontwerp -- en de campagne hoort dat
   VOORAF te zien en `None` met een `note` te noteren, niet de uitzondering weg
   te vangen.
3. Elk geserialiseerd record met een Sharpe erin draagt `(n_obs, bars_per_year,
   t_years)`. MEASUREMENT_CONTRACT.md §10; de poort staat in
   `validation/inference.py::require_sharpe_triple` en wordt hier over de HELE
   payload-boom gehaald, niet alleen op het topniveau.
4. Een k-grid zonder 1 wordt geweigerd: dan is er geen referentie om tegen te
   verschillen en zou het "verschil" een vergelijking tussen twee vastgehouden
   sporen zijn.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tradebot.registry.preregistration import StopCriterion
from tradebot.registry.trial_counter import TrialCount
from tradebot.schemas.config import ValidationConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.validation import phase10_decision_frequency_measurement as campaign_module
from tradebot.validation.inference import require_sharpe_triple
from tradebot.validation.phase10_decision_frequency import (
    LadderRead,
    measure_hold_inertness,
)
from tradebot.validation.phase10_decision_frequency_measurement import measure_campaign

ROOT = Path(__file__).resolve().parents[2]

#: 400 bars, waarvan 300 ontwikkeling. Ruim boven `inference.min_obs` (30) en
#: boven `dsr.MIN_OBS_FOR_DSR` (30), zodat geen enkele poort op steekproefgrootte
#: afketst en de tests over de LOGICA gaan.
IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
SPLIT = IDX[300]
BARS_PER_YEAR = 365.0
COST_PER_SIDE_BPS = 6.5
PRIMARY_TRACK = "synthetic_primary"
PRIMARY_LAYER = "L3_execution"
COST_AXIS_LAYER = "L0_vectorized"


@pytest.fixture(scope="module")
def validation() -> ValidationConfig:
    return load_config(ROOT / "conf/validation/default.yaml", ValidationConfig)


@pytest.fixture(scope="module")
def trial_count() -> TrialCount:
    """M = 25, bevroren (AD-24). `dsr_gate` weigert een live M."""
    return TrialCount(value=25, source="frozen", origin="test-reset",
                      seed_total=0, registered_total=25)


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    path = tmp_path / "holdout_lock.json"
    path.write_text(
        f'{{"split_utc": "{SPLIT.isoformat()}", "git_sha": "test", '
        '"frozen_utc": "2026-01-01T00:00:00+00:00", "reads": []}',
        encoding="utf-8")
    return path


CRITERIA = (
    StopCriterion(
        name="no_improvement_on_development",
        metric="delta_net_sharpe_development_best_k", operator="<=",
        threshold=0.0, action="archive", rationale="deel 1"),
    StopCriterion(
        name="sign_flips_on_gate_sample",
        metric="delta_net_sharpe_gate_best_k", operator="<=",
        threshold=0.0, action="archive", rationale="deel 3"),
    StopCriterion(
        name="promotion_requires_all_gates_clear",
        metric="n_binding_stop_criteria", operator="<=",
        threshold=0.0, action="promote", rationale="de conjunctie"),
)


def _read(k: int, *, turnover: float, mu: float, seed: int) -> LadderRead:
    """Eén synthetische ladderlezing: L3-netto, L0-bruto/netto, L0-omzet."""
    rng = np.random.default_rng(seed)
    primary = pd.Series(rng.normal(mu, 0.01, len(IDX)), index=IDX)
    gross = pd.Series(rng.normal(mu, 0.012, len(IDX)), index=IDX)
    turnover_series = pd.Series(np.full(len(IDX), turnover), index=IDX)
    net = gross - turnover_series * (COST_PER_SIDE_BPS / 1e4)
    frame = pd.DataFrame({"A": np.full(len(IDX), 0.5)}, index=IDX)
    return LadderRead(
        k=k,
        primary_net_returns=primary,
        cost_axis_gross_returns=gross,
        cost_axis_net_returns=net,
        cost_axis_turnover=turnover_series,
        execution_audit={"n_sovereign_halted": 300 - k,
                         "n_sovereign_clipped": 399},
        cost_axis_audit={"mean_turnover": float(turnover_series.mean())},
        inertness=measure_hold_inertness(frame, frame, k=k),
    )


def _reads(levels: Mapping[int, float]) -> dict[str, list[LadderRead]]:
    return {PRIMARY_TRACK: [_read(k, turnover=level, mu=0.0004 + 0.0001 * i,
                                 seed=100 + k)
                            for i, (k, level) in enumerate(sorted(levels.items()))]}


def _measure(reads: Mapping[str, Any], lock: Path, validation: ValidationConfig,
             trial_count: TrialCount) -> Any:
    return measure_campaign(
        reads, primary_track=PRIMARY_TRACK, primary_layer=PRIMARY_LAYER,
        cost_axis_layer=COST_AXIS_LAYER, anchor="first_bar", lock_path=lock,
        bars_per_year=BARS_PER_YEAR, cost_per_side_bps=COST_PER_SIDE_BPS,
        trial_count=trial_count, validation=validation, stop_criteria=CRITERIA)


def _mappings(node: Any) -> Iterator[Mapping[str, Any]]:
    """Elke mapping in de payload-boom, inclusief de wortel."""
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _mappings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _mappings(value)


# --------------------------------------------------------------------------- #
# 1 — de referentie toetst zichzelf niet
# --------------------------------------------------------------------------- #
def test_the_reference_k_is_zero_by_construction_and_calls_no_difference_test(
    monkeypatch: pytest.MonkeyPatch, lock: Path, validation: ValidationConfig,
    trial_count: TrialCount,
) -> None:
    calls: list[int] = []
    real = campaign_module.sharpe_difference_test

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(campaign_module, "sharpe_difference_test", spy)
    result = _measure(_reads({1: 0.20, 2: 0.10}), lock, validation, trial_count)

    reference = next(r for r in result.runs if r.k == 1)
    other = next(r for r in result.runs if r.k == 2)
    assert reference.is_reference is True
    assert reference.difference is None
    assert reference.to_dict()["delta_vs_reference"]["delta_sharpe"] == 0.0
    assert other.difference is not None
    # Drie toetsen voor k=2: de primaire cel plus bruto en netto op de kostenas.
    # Zes zou betekenen dat k=1 ook is getoetst -- tegen zichzelf.
    assert len(calls) == 3


# --------------------------------------------------------------------------- #
# 2 — nul omzetverschil: geen breakeven, met een note en zonder uitzondering
# --------------------------------------------------------------------------- #
def test_zero_turnover_delta_gives_a_null_breakeven_with_a_note(
    lock: Path, validation: ValidationConfig, trial_count: TrialCount,
) -> None:
    result = _measure(_reads({1: 0.20, 2: 0.20, 5: 0.20}), lock, validation,
                      trial_count)
    for run in result.runs:
        block = run.breakeven
        assert block.turnover_delta == 0.0
        assert block.breakeven_cost_bps is None
        assert block.note, "een ontbrekend breakevenniveau zonder note is een leeg veld"
        if run.is_reference:
            # De referentie draagt een ANDERE reden: geen meting, een constructie.
            assert "referentie" in block.note.lower()
        else:
            assert "turnover" in block.note.lower()
            assert "exact nul" in block.note.lower()
    metrics = result.stop_criteria_metrics()
    assert metrics["breakeven_cost_bps_development_best_k"] is None


def test_a_tiny_turnover_delta_still_yields_a_level_and_names_its_fraction(
    lock: Path, validation: ValidationConfig, trial_count: TrialCount,
) -> None:
    """De bijna-ontaarde kant: de noemer is klein maar niet nul.

    Dan bestaat het niveau wel, maar het is enorm en het mag niet zonder zijn
    noemer worden gelezen. De campagne noemt de fractie als GEMETEN getal; er
    wordt geen materialiteitsdrempel verzonnen.
    """
    result = _measure(_reads({1: 0.001665, 2: 0.001594}), lock, validation,
                      trial_count)
    block = next(r for r in result.runs if r.k == 2).breakeven
    assert block.turnover_delta > 0.0
    assert block.breakeven_cost_bps is not None
    assert block.turnover_delta_fraction_of_reference == pytest.approx(
        block.turnover_delta / block.turnover_reference)
    assert f"{block.turnover_delta_fraction_of_reference:.6f}" in block.note


# --------------------------------------------------------------------------- #
# 3 — §10 over de hele boom
# --------------------------------------------------------------------------- #
def test_every_serialised_record_carries_the_sharpe_triple(
    lock: Path, validation: ValidationConfig, trial_count: TrialCount,
) -> None:
    payload = _measure(_reads({1: 0.20, 2: 0.10, 5: 0.05}), lock, validation,
                       trial_count).to_dict()
    seen = 0
    for record in _mappings(payload):
        require_sharpe_triple(record, where="test_phase10_decision_frequency")
        seen += 1 if any("sharpe" in k.lower() for k in record) else 0
    assert seen >= 4, "geen enkel record met een Sharpe gevonden; de test bewijst niets"


# --------------------------------------------------------------------------- #
# 4 — een grid zonder de identiteit
# --------------------------------------------------------------------------- #
def test_a_grid_without_k_equals_one_is_refused(
    lock: Path, validation: ValidationConfig, trial_count: TrialCount,
) -> None:
    with pytest.raises(DataContractError):
        _measure(_reads({2: 0.10, 5: 0.05}), lock, validation, trial_count)


def test_a_duplicate_k_is_refused(
    lock: Path, validation: ValidationConfig, trial_count: TrialCount,
) -> None:
    reads = _reads({1: 0.20, 2: 0.10})
    reads[PRIMARY_TRACK].append(_read(2, turnover=0.05, mu=0.001, seed=9))
    with pytest.raises(DataContractError):
        _measure(reads, lock, validation, trial_count)


def test_a_missing_primary_track_is_refused(
    lock: Path, validation: ValidationConfig, trial_count: TrialCount,
) -> None:
    reads = {"some_other_track": _reads({1: 0.2, 2: 0.1})[PRIMARY_TRACK]}
    with pytest.raises(DataContractError):
        _measure(reads, lock, validation, trial_count)
