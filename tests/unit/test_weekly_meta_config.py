"""Het contract van de wekelijkse strategie: laadt, is 1:1 per constructie, en weigert tegenstrijdigheden."""
from __future__ import annotations

import math

import pytest
import yaml

from tradebot.schemas.weekly_meta import (
    WEEKLY_META_CONFIG_PATH,
    WeeklyMetaConfig,
    weekly_meta_config,
)
from tradebot.utils.failfast import ConfigContractError


def _raw() -> dict:
    return yaml.safe_load(WEEKLY_META_CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_committed_config_loads() -> None:
    cfg = weekly_meta_config()
    assert cfg.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT")
    assert cfg.barrier_sigma == pytest.approx(math.sqrt(5.0))
    assert cfg.planned_trials == 4
    assert cfg.inner_wf_blocks == 4
    assert cfg.n_probability_bins == 5


def test_one_barrier_width_means_one_to_one_by_construction() -> None:
    fields = set(WeeklyMetaConfig.model_fields)
    assert "barrier_sigma" in fields
    assert not {"profit_target_sigma", "stop_loss_sigma"} & fields


def test_there_is_no_pseudo_posterior_knob() -> None:
    """Spec §10.4: de posterior telt gerealiseerde uitkomsten; er is geen 'kalibratiefractie' meer."""
    assert "calibration_fraction" not in WeeklyMetaConfig.model_fields


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("holdout_split_utc", "2021-06-01T00:00:00+00:00", "voor de holdout"),
        ("first_test_start_utc", "2022-01-01", "tijdzone"),
        ("k_grid", [2.0, 1.0], "oplopend"),
        ("trades_per_week_target", 9.0, "meer events"),
        ("cpcv_n_test_groups", 4, "deelbaar"),
        ("shuffle_auc_band", [0.55, 0.60], "omsluiten"),
        ("inner_wf_blocks", 1, "inner_wf_blocks"),
        ("unknown_key", 1, "unknown_key"),
    ],
)
def test_a_contradiction_is_refused(tmp_path, key, value, match) -> None:
    raw = _raw()
    raw[key] = value
    path = tmp_path / "weekly_meta.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigContractError, match=match):
        weekly_meta_config(path)
