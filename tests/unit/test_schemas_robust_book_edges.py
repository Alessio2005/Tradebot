"""De randen van het contract van het robuuste boek v1 (`schemas/robust_book.py`).

`test_config_contracts.py` en `test_systematic_programme.py` laden de echte
`conf/model/robust_book.yaml`; hier staat wat die geldige config NIET raakt: de drie
consistentiecontroles (vensters die strikt oplopen, een universum zonder dubbel symbool met een
kern erbinnen, lookbacks die uniek en oplopend zijn) en de veldgrenzen die het programma
vooraf vastlegt. Elke weigering komt uit de echte config door precies één waarde te verknoeien,
zodat een test alleen rood wordt om de controle die hij beweert te toetsen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from tradebot.schemas.robust_book import (
    ROBUST_BOOK_CONFIG_PATH,
    RobustBookConfig,
    Windows,
    robust_book_config,
)
from tradebot.utils.failfast import ConfigContractError

ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = ROOT / "conf" / "model" / "robust_book.yaml"

#: De vensters van de échte config: train_end en validate_start liggen precies één dag uit elkaar.
ASCENDING = ("2021-11-15", "2023-12-31", "2024-01-01", "2025-09-04", "2025-09-05")
WINDOW_KEYS = ("w_dev_start", "train_end", "validate_start", "w_dev_end", "gate_start")


def _document() -> dict[str, Any]:
    """Een verse kopie van de echte config als mapping; elke test verknoeit zijn eigen kopie."""
    return yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8"))


def _validated(doc: dict[str, Any]) -> RobustBookConfig:
    return RobustBookConfig.model_validate(doc)


def _windows(*dates: str) -> Windows:
    return Windows(**dict(zip(WINDOW_KEYS, dates, strict=True)))


def _written(tmp_path: Path, doc: dict[str, Any]) -> Path:
    path = tmp_path / "robust_book.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


class TestTheRealConfigIsTheBaseline:
    def test_the_default_path_points_at_the_repository_config(self) -> None:
        assert ROBUST_BOOK_CONFIG_PATH.resolve() == REAL_CONFIG.resolve()

    def test_the_untouched_document_validates_to_the_config_the_loader_returns(self) -> None:
        assert _validated(_document()) == robust_book_config(REAL_CONFIG)

    def test_the_real_windows_ascend_strictly_with_one_day_between_train_and_validate(self) -> None:
        w = robust_book_config(REAL_CONFIG).windows
        assert (w.w_dev_start, w.train_end, w.validate_start, w.w_dev_end, w.gate_start) == ASCENDING


class TestTheWindowsMustAscendStrictly:
    @pytest.mark.parametrize("dates", [
        pytest.param(("2021-11-15", "2021-11-14", "2024-01-01", "2025-09-04", "2025-09-05"),
                     id="train_end-before-w_dev_start"),
        pytest.param(("2021-11-15", "2021-11-15", "2024-01-01", "2025-09-04", "2025-09-05"),
                     id="train_end-equals-w_dev_start"),
        pytest.param(("2021-11-15", "2023-12-31", "2023-12-31", "2025-09-04", "2025-09-05"),
                     id="validate_start-equals-train_end"),
        pytest.param(("2021-11-15", "2023-12-31", "2024-01-01", "2023-06-01", "2025-09-05"),
                     id="w_dev_end-before-validate_start"),
        pytest.param(("2021-11-15", "2023-12-31", "2024-01-01", "2025-09-04", "2025-09-04"),
                     id="gate_start-equals-w_dev_end"),
        pytest.param(("2021-11-15", "2023-12-31", "2024-01-01", "2025-09-04", "2025-09-03"),
                     id="gate_start-before-w_dev_end"),
        pytest.param(tuple(reversed(ASCENDING)), id="fully-reversed"),
    ])
    def test_a_window_that_does_not_move_forward_is_refused(self, dates: tuple[str, ...]) -> None:
        with pytest.raises(ValidationError, match="de vensters moeten strikt oplopen"):
            _windows(*dates)

    def test_ascending_windows_are_accepted_even_when_only_a_day_apart(self) -> None:
        w = _windows("2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05")
        assert w.gate_start == "2021-01-05"

    def test_the_order_is_that_of_the_dates_not_of_the_text(self) -> None:
        """Als tekst sorteert '2021-10-1' vóór '2021-9-1'; als datum ligt september eerder."""
        w = _windows("2021-1-1", "2021-2-1", "2021-9-1", "2021-10-1", "2021-11-1")
        assert (w.validate_start, w.w_dev_end) == ("2021-9-1", "2021-10-1")
        with pytest.raises(ValidationError, match="strikt oplopen"):
            _windows("2021-1-1", "2021-2-1", "2021-10-1", "2021-9-1", "2021-11-1")

    def test_a_text_that_is_no_date_is_refused_when_the_config_is_validated(self) -> None:
        with pytest.raises(ValidationError, match="volgend jaar"):
            _windows("2021-11-15", "2023-12-31", "2024-01-01", "volgend jaar", "2025-09-05")

    def test_a_disordered_window_inside_the_whole_config_is_reported_on_the_windows_block(self) -> None:
        doc = _document()
        doc["windows"]["train_end"] = doc["windows"]["w_dev_start"]
        with pytest.raises(ValidationError) as err:
            _validated(doc)
        assert [e["loc"] for e in err.value.errors()] == [("windows",)]
        assert "strikt oplopen" in err.value.errors()[0]["msg"]

    def test_the_loader_turns_the_refusal_into_a_config_contract_error(self, tmp_path: Path) -> None:
        doc = _document()
        doc["windows"]["gate_start"] = doc["windows"]["w_dev_end"]
        with pytest.raises(ConfigContractError, match=r"windows.*strikt oplopen"):
            robust_book_config(_written(tmp_path, doc))


class TestTheUniverseMustBeConsistent:
    def test_a_duplicate_symbol_is_refused(self) -> None:
        doc = _document()
        doc["symbols"] = [*doc["symbols"], doc["symbols"][0]]
        with pytest.raises(ValidationError, match="dubbel symbool in het universum"):
            _validated(doc)

    def test_a_core_asset_outside_the_universe_is_refused(self) -> None:
        doc = _document()
        doc["core"]["assets"] = [*doc["core"]["assets"], "XRPUSDT"]
        with pytest.raises(ValidationError, match="de kern bevat een symbool buiten het universum"):
            _validated(doc)

    def test_a_core_that_is_a_strict_subset_of_the_universe_is_accepted(self) -> None:
        doc = _document()
        doc["core"]["assets"] = ["BTCUSDT"]
        assert _validated(doc).core.assets == ("BTCUSDT",)

    def test_the_loader_reports_a_foreign_core_asset_as_a_config_contract_error(self, tmp_path: Path) -> None:
        doc = _document()
        doc["core"]["assets"] = ["XRPUSDT"]
        with pytest.raises(ConfigContractError, match="buiten het universum"):
            robust_book_config(str(_written(tmp_path, doc)))


class TestTheLookbacksMustBeUniqueAndAscending:
    @pytest.mark.parametrize("lookbacks", [[5, 5, 10], [10, 5], [5, 20, 10], [160, 80, 40]])
    def test_lookbacks_that_repeat_or_descend_are_refused(self, lookbacks: list[int]) -> None:
        doc = _document()
        doc["trend"]["lookbacks"] = lookbacks
        with pytest.raises(ValidationError, match="lookbacks moeten uniek en oplopend zijn"):
            _validated(doc)

    @pytest.mark.parametrize("lookbacks", [[7], [2, 3], [5, 7, 10, 14, 20, 28, 40, 56, 80, 112, 160]])
    def test_unique_ascending_lookbacks_are_accepted(self, lookbacks: list[int]) -> None:
        doc = _document()
        doc["trend"]["lookbacks"] = lookbacks
        assert list(_validated(doc).trend.lookbacks) == lookbacks

    def test_the_lookback_grid_in_the_repository_is_unique_and_ascending(self) -> None:
        grid = list(robust_book_config(REAL_CONFIG).trend.lookbacks)
        assert grid == sorted(set(grid))


class TestTheFieldsThatArePreRegistered:
    @pytest.mark.parametrize(("section", "key", "value"), [
        ("execution", "lag_bars", 0),
        ("execution", "aum_usd", 0.0),
        ("sizing", "vol_span", 0),
        ("sizing", "gross_cap", -3.0),
        ("trend", "z_clip", 0.0),
        ("carry", "rebalance_bars", 0),
    ])
    def test_a_zero_or_negative_setting_is_refused(self, section: str, key: str, value: float) -> None:
        doc = _document()
        doc[section][key] = value
        with pytest.raises(ValidationError) as err:
            _validated(doc)
        assert [e["loc"] for e in err.value.errors()] == [(section, key)]

    def test_at_least_one_trial_is_planned_and_no_negative_prior_is_known(self) -> None:
        doc = _document()
        doc["planned_trials"] = 0
        with pytest.raises(ValidationError, match="planned_trials"):
            _validated(doc)
        doc = _document()
        doc["known_prior_trials"] = -1
        with pytest.raises(ValidationError, match="known_prior_trials"):
            _validated(doc)
        doc["known_prior_trials"] = 0
        assert _validated(doc).known_prior_trials == 0

    def test_an_unknown_key_is_refused_instead_of_ignored(self) -> None:
        doc = _document()
        doc["sizing"]["leverage_boost"] = 2.0
        with pytest.raises(ValidationError, match="leverage_boost"):
            _validated(doc)

    def test_a_missing_section_is_refused(self) -> None:
        doc = _document()
        del doc["combo"]
        with pytest.raises(ValidationError, match="combo"):
            _validated(doc)

    def test_the_validated_config_cannot_be_edited_afterwards(self) -> None:
        cfg = robust_book_config(REAL_CONFIG)
        with pytest.raises(ValidationError, match="frozen"):
            cfg.planned_trials = 99  # type: ignore[misc]
        with pytest.raises(ValidationError, match="frozen"):
            cfg.windows.gate_start = "2030-01-01"  # type: ignore[misc]
