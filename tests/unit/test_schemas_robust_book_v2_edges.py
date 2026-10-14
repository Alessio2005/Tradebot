"""Randgevallen van het contract van robuust boek v2 .. v8 (`schemas/robust_book_v2.py`).

Elke validator weigert precies wat zijn foutmelding belooft; de grens zelf valt aan de kant
die de vergelijking kiest (strikt of niet-strikt); en de uitgeleverde configs blijven
toegestaan. Er wordt alleen `conf/model/robust_book_v*.yaml` gelezen, en alleen naar
`tmp_path` geschreven.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tradebot.schemas.robust_book_v2 import (
    ROBUST_BOOK_V2_CONFIG_PATH,
    ROBUST_BOOK_V3_CONFIG_PATH,
    ROBUST_BOOK_V5_CONFIG_PATH,
    ROBUST_BOOK_V6_CONFIG_PATH,
    ROBUST_BOOK_V7_CONFIG_PATH,
    ROBUST_BOOK_V8_CONFIG_PATH,
    Backcast,
    Basis,
    Forward,
    Leverage,
    RobustBookV2Config,
    RobustBookV3Config,
    RobustBookV5Config,
    RobustBookV6Config,
    RobustBookV7Config,
    RobustBookV8Config,
    Windows,
    robust_book_v2_config,
    robust_book_v3_config,
    robust_book_v5_config,
    robust_book_v6_config,
    robust_book_v7_config,
    robust_book_v8_config,
)
from tradebot.utils.failfast import ConfigContractError

PATHS = {2: ROBUST_BOOK_V2_CONFIG_PATH, 3: ROBUST_BOOK_V3_CONFIG_PATH,
         5: ROBUST_BOOK_V5_CONFIG_PATH, 6: ROBUST_BOOK_V6_CONFIG_PATH,
         7: ROBUST_BOOK_V7_CONFIG_PATH, 8: ROBUST_BOOK_V8_CONFIG_PATH}


def _raw(version: int) -> dict[str, Any]:
    """De uitgeleverde config als vers, vrij te muteren document."""
    return copy.deepcopy(yaml.safe_load(PATHS[version].read_text(encoding="utf-8")))


def _windows(**over: str) -> Windows:
    base = {"train_start": "2021-01-01", "train_end": "2021-12-31",
            "validate_start": "2022-01-01", "w_dev_end": "2022-06-30",
            "holdout_start": "2022-07-01", "holdout_end": "2022-12-31"}
    return Windows.model_validate({**base, **over})


# --------------------------------------------------------------------------- #
# De uitgeleverde configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("loader", "model"), [
    (robust_book_v2_config, RobustBookV2Config), (robust_book_v3_config, RobustBookV3Config),
    (robust_book_v5_config, RobustBookV5Config), (robust_book_v6_config, RobustBookV6Config),
    (robust_book_v7_config, RobustBookV7Config), (robust_book_v8_config, RobustBookV8Config)])
def test_every_shipped_config_loads_and_satisfies_its_own_validators(loader, model):
    cfg = loader()
    assert type(cfg) is model
    assert cfg.planned_trials >= 1 and cfg.windows.holdout_start == "2025-07-01"


def test_each_version_extends_the_previous_one():
    assert issubclass(RobustBookV3Config, RobustBookV2Config)
    assert issubclass(RobustBookV5Config, RobustBookV3Config)
    assert issubclass(RobustBookV6Config, RobustBookV5Config)
    assert issubclass(RobustBookV7Config, RobustBookV6Config)
    assert issubclass(RobustBookV8Config, RobustBookV7Config)


# --------------------------------------------------------------------------- #
# Windows: strikt oplopende vensters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("key", "value"), [
    ("train_end", "2021-01-01"),          # gelijk aan train_start
    ("train_end", "2020-12-31"),          # vóór train_start
    ("validate_start", "2021-12-31"),     # gelijk aan train_end
    ("w_dev_end", "2022-01-01"),          # gelijk aan validate_start
    ("holdout_start", "2022-06-30"),      # gelijk aan w_dev_end
    ("holdout_end", "2022-07-01"),        # gelijk aan holdout_start
    ("holdout_end", "2022-06-01"),        # vóór holdout_start
])
def test_windows_that_do_not_ascend_strictly_are_refused(key, value):
    with pytest.raises(ValidationError, match="de vensters moeten strikt oplopen"):
        _windows(**{key: value})


def test_windows_one_day_apart_are_enough():
    w = _windows(train_end="2021-01-02", validate_start="2021-01-03", w_dev_end="2021-01-04",
                 holdout_start="2021-01-05", holdout_end="2021-01-06")
    assert (w.train_end, w.holdout_end) == ("2021-01-02", "2021-01-06")


# --------------------------------------------------------------------------- #
# Basis (v5): hysterese en margevloer
# --------------------------------------------------------------------------- #
def _basis(**over: Any) -> Basis:
    return Basis.model_validate({**_raw(5)["basis"], **over})


@pytest.mark.parametrize("exit_apr", [0.15, 0.20])   # enter_apr is 0,15: gelijk en erboven
def test_a_basis_exit_at_or_above_the_entry_has_no_hysteresis_and_is_refused(exit_apr):
    with pytest.raises(ValidationError, match="exit_apr moet onder enter_apr liggen"):
        _basis(exit_apr=exit_apr)


@pytest.mark.parametrize("exit_apr", [0.0499, -0.10])   # een negatieve uitstapdrempel mag
def test_a_basis_exit_below_the_entry_is_accepted(exit_apr):
    assert _basis(exit_apr=exit_apr).exit_apr == exit_apr


def test_a_basis_whose_full_notional_starts_below_the_margin_floor_is_refused():
    # Wallet bij de volle notional: 1 - 0,9 = 0,1 < margevloer 0,25 x 0,9 = 0,225.
    with pytest.raises(ValidationError, match="elke dag herbalanceren"):
        _basis(notional=0.9, margin_floor=0.25)


def test_the_margin_floor_may_equal_the_wallet_but_not_exceed_it():
    assert _basis(notional=0.5, margin_floor=1.0).notional == 0.5     # 0,5 >= 1,0 x 0,5
    with pytest.raises(ValidationError, match="elke dag herbalanceren"):
        _basis(notional=0.75, margin_floor=1.0)                        # 0,25 < 0,75


# --------------------------------------------------------------------------- #
# Backcast en zijn plek ten opzichte van W_DEV (v5)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("start", "end"), [("2020-04-01", "2020-04-01"),
                                            ("2020-06-01", "2020-04-01")])
def test_a_backcast_that_does_not_start_before_it_ends_is_refused(start, end):
    with pytest.raises(ValidationError, match=r"backcast\.start moet voor backcast\.end liggen"):
        Backcast(start=start, end=end)


def test_a_backcast_of_one_day_is_accepted():
    assert Backcast(start="2020-04-01", end="2020-04-02").end == "2020-04-02"


def _v5(**backcast: str) -> RobustBookV5Config:
    raw = _raw(5)
    raw["backcast"].update(backcast)
    return RobustBookV5Config.model_validate(raw)


@pytest.mark.parametrize("end", ["2021-01-01", "2021-02-01"])   # train_start en later
def test_a_backcast_that_reaches_into_w_dev_is_refused(end):
    with pytest.raises(ValidationError, match="de backcast moet volledig voor W_DEV liggen"):
        _v5(end=end)


def test_a_backcast_ending_the_day_before_w_dev_is_accepted():
    assert _v5(end="2020-12-31").backcast.end == "2020-12-31"


def test_the_loader_translates_a_schema_violation_into_a_config_contract_error(tmp_path: Path):
    raw = _raw(5)
    raw["backcast"]["end"] = "2021-01-01"
    path = tmp_path / "robust_book_v5.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigContractError,
                       match=r"Configuratiecontract geschonden.*RobustBookV5Config.*"
                             r"de backcast moet volledig voor W_DEV liggen"):
        robust_book_v5_config(path)


# --------------------------------------------------------------------------- #
# Leverage (v6): consistentie van het hefboomblok
# --------------------------------------------------------------------------- #
def _leverage(**over: Any) -> Leverage:
    return Leverage.model_validate({**_raw(6)["leverage"], **over})


def test_a_leverage_block_without_candidates_is_refused():
    with pytest.raises(ValidationError, match="geen hefboomkandidaten"):
        _leverage(candidates={})


def test_a_candidate_that_needs_more_gross_exposure_than_the_cap_is_refused():
    with pytest.raises(ValidationError, match="meer bruto exposure"):
        _leverage(candidates={"K9": 2.5})                  # twee benen: 5,0 > gross_cap 4,0


def test_a_candidate_whose_two_legs_exactly_fill_the_cap_is_accepted():
    assert _leverage(candidates={"K9": 2.0}).candidates == {"K9": 2.0}      # 4,0 > 4,0 is onwaar
    with pytest.raises(ValidationError, match="meer bruto exposure"):
        _leverage(candidates={"K9": 1.5}, gross_cap=2.5)                    # 3,0 > 2,5


@pytest.mark.parametrize("over", [
    {"haircut_alt_stress": 0.20},                        # gelijk aan haircut_alt
    {"haircut_alt_stress": 0.10},                        # milder dan de basis
    {"maintenance_margin_stress": 0.02},                 # gelijk aan maintenance_margin
    {"maintenance_margin_stress": 0.01},                 # milder dan de basis
])
def test_a_margin_stress_that_is_not_stricter_than_the_base_is_refused(over):
    with pytest.raises(ValidationError, match="de margestress moet strenger zijn dan de basis"):
        _leverage(**over)


def test_one_strict_margin_stress_does_not_excuse_the_other():
    # haircut-stress wel strenger, onderhoudsmarge-stress niet: het is een EN.
    with pytest.raises(ValidationError, match="de margestress moet strenger zijn"):
        _leverage(haircut_alt_stress=0.50, maintenance_margin_stress=0.02)
    with pytest.raises(ValidationError, match="de margestress moet strenger zijn"):
        _leverage(haircut_alt_stress=0.20, maintenance_margin_stress=0.05)


@pytest.mark.parametrize("over", [
    {"financing_stress_floor_apr": 0.07},                # vloer onder de basis (0,08)
    {"financing_stress_multiplier": 0.9},                # multiplier onder de basis (1,0)
])
def test_a_financing_stress_that_is_cheaper_than_the_base_is_refused(over):
    with pytest.raises(ValidationError,
                       match="de financieringsstress moet duurder zijn dan de basis"):
        _leverage(**over)


def test_a_financing_stress_equal_to_the_base_is_accepted():
    lev = _leverage(financing_stress_floor_apr=0.08, financing_stress_multiplier=1.0)
    assert (lev.financing_stress_floor_apr, lev.financing_stress_multiplier) == (0.08, 1.0)


# --------------------------------------------------------------------------- #
# Forward (v6): het vooruit-sample begint na de holdout
# --------------------------------------------------------------------------- #
def _v6(**forward: Any) -> RobustBookV6Config:
    raw = _raw(6)
    raw["forward"].update(forward)
    return RobustBookV6Config.model_validate(raw)


@pytest.mark.parametrize("start", ["2026-09-30", "2026-01-01"])   # holdout_end zelf en eerder
def test_a_forward_sample_that_starts_inside_the_holdout_is_refused(start):
    with pytest.raises(ValidationError, match="het vooruit-sample moet na de holdout beginnen"):
        _v6(start=start)


def test_a_forward_sample_starting_the_day_after_the_holdout_is_accepted():
    assert _v6(start="2026-10-01").forward == Forward(start="2026-10-01", min_months=6)


# --------------------------------------------------------------------------- #
# V7: de gefinancierde tranche en de hefboom per been
# --------------------------------------------------------------------------- #
def _v7(*, tranche: list[str] | None = None, **candidates: float) -> RobustBookV7Config:
    raw = _raw(7)
    raw["leverage"]["candidates"].update(candidates)
    if tranche is not None:
        raw["v7"]["tranche"] = tranche
    return RobustBookV7Config.model_validate(raw)


def test_tranche_candidates_without_a_leverage_rule_are_refused_and_named_sorted():
    with pytest.raises(ValidationError) as err:
        _v7(tranche=["S2_CARRY_PM_TRANCHE", "Z9_NOPE", "A1_NOPE"])
    assert "tranche-kandidaten zonder hefboomregel: ['A1_NOPE', 'Z9_NOPE']" in str(err.value)


def test_a_tranche_candidate_must_lever_each_leg_twice():
    with pytest.raises(ValidationError,
                       match=r"S2_CARRY_PM_TRANCHE: hefboom per been 1\.0, verwacht 2\.0"):
        _v7(S2_CARRY_PM_TRANCHE=1.0)


@pytest.mark.parametrize("lev", [2.0, 1.5])
def test_a_candidate_outside_the_tranche_must_lever_each_leg_once(lev):
    with pytest.raises(ValidationError,
                       match=rf"S1_CARRY_PM_SLICED: hefboom per been {lev}, verwacht 1\.0"):
        _v7(S1_CARRY_PM_SLICED=lev)


def test_a_consistent_tranche_is_accepted_and_may_name_several_candidates():
    both = _v7(tranche=["S1_CARRY_PM_SLICED", "S2_CARRY_PM_TRANCHE"], S1_CARRY_PM_SLICED=2.0)
    assert set(both.v7.tranche) == {"S1_CARRY_PM_SLICED", "S2_CARRY_PM_TRANCHE"}
    none = _v7(tranche=[], S2_CARRY_PM_TRANCHE=1.0)
    assert none.v7.tranche == ()
