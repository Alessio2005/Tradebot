"""Het config-contract van de options-setup: de standaard laadt, onzin wordt geweigerd."""
from __future__ import annotations

import copy

import pytest
import yaml

from tradebot.schemas.options import DEFAULT_OPTIONS_CONFIG, load_options_config
from tradebot.utils.failfast import ConfigContractError

RAW = yaml.safe_load(DEFAULT_OPTIONS_CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path, raw):
    p = tmp_path / "options.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_the_default_config_loads():
    cfg = load_options_config()
    assert cfg.universe.underlyings == ("SPY", "QQQ", "IWM", "AAPL", "MSFT")
    assert set(cfg.universe.share_eligible) == {"AAPL", "MSFT"}
    assert cfg.pricing.decision_engine == "binomial_lr"


def _mutate(path, val):
    raw = copy.deepcopy(RAW)
    node = raw
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = val
    return raw


@pytest.mark.parametrize(("path", "val", "msg"), [
    (("pricing", "unknown_key"), 1, "Extra inputs"),
    (("universe", "underlyings"), ["SPY", "SPY"], "dubbelen"),
    (("universe", "share_eligible"), ["TSLA"], "deel van underlyings"),
    (("universe", "beta_to_spy"), {"SPY": 1.0}, "beta_to_spy mist"),
    (("universe", "underlyings"), ["spy"], "pattern"),
    (("pricing", "binomial_steps"), 400, "oneven"),
    (("pricing", "iv_min"), 6.0, "iv_min"),
    (("pricing", "decision_engine"), "european", "decision_engine"),
    (("pricing", "risk_free_rate"), 4.0, "risk_free_rate"),
])
def test_invalid_configs_are_refused(tmp_path, path, val, msg):
    with pytest.raises(ConfigContractError, match=msg):
        load_options_config(_write(tmp_path, _mutate(path, val)))


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    p = tmp_path / "options.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="mapping"):
        load_options_config(p)
