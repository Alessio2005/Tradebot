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
    assert cfg.filters.min_open_interest == 500
    assert cfg.iv_metrics.ivr_lookback_days == 365
    assert cfg.iv_metrics.fallback.enabled
    assert cfg.strategies.credit_spread.enabled and not cfg.strategies.wheel.enabled
    assert not cfg.strategies.short_straddle_strangle.enabled
    assert not cfg.risk.allow_undefined_risk
    assert cfg.risk.max_open_positions == 10 and cfg.risk.pdt_guard
    assert cfg.execution.ticks.combo == 0.01 and cfg.execution.max_order_usd > 0
    assert cfg.account.capital_usd == 25_000 and cfg.schedule.scan_times == ("10:00", "13:30")
    assert cfg.paper.fill_at_concession == 0.40


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
    (("pricing", "risk_free_rate"), -0.01, "risk_free_rate"),
    (("filters", "max_spread_frac_mid"), 0.0, "max_spread_frac_mid"),
    (("filters", "min_open_interest"), -1, "min_open_interest"),
    (("iv_metrics", "low_ivr"), 50, "very_low_ivr <= low_ivr <= high_ivr"),
    (("iv_metrics", "fallback"), {"high_iv_ratio": 0.8},
     "very_low_iv_ratio <= low_iv_ratio <= high_iv_ratio"),
    (("iv_metrics", "ivr_min_observations"), 400, "ivr_min_observations <= ivr_lookback_days"),
    (("iv_metrics", "ivr_lookback_days"), 5, "ivr_lookback_days"),
    (("strategies", "credit_spread", "dte"), [45, 30], "dte: ondergrens boven bovengrens"),
    (("strategies", "iron_condor", "unknown"), 1, "Extra inputs"),
    (("strategies", "calendar_spread", "long_dte"), [35, 90], "helemaal vóór long_dte"),
    (("strategies", "short_straddle_strangle", "max_per_underlying"), 2,
     "max_per_underlying"),
    (("strategies", "long_straddle_strangle", "style"), "butterfly", "style"),
    (("schedule", "close_before_expiry_time"), "25:00", "close_before_expiry_time"),
    (("sizing", "ranking"), "sharpe", "ranking"),
    (("risk", "daily_loss_stop_entries_frac"), 0.05, "daily_loss_stop_entries_frac"),
    (("risk", "scenario_vol_points"), [], "scenario_vol_points mag niet leeg"),
    (("risk", "scenario_spot"), [-0.95], "scenario_spot"),
    (("execution", "ticks", "combo"), 0.0, "combo"),
    (("execution", "max_order_usd"), 0, "max_order_usd"),
    (("schedule", "scan_times"), [], "scan_times mag niet leeg"),
    (("schedule", "scan_times"), ["13:30", "10:00"], "oplopend"),
    (("schedule", "scan_times"), ["10:00", "10:00"], "oplopend"),
    (("schedule", "end_of_day"), "9:00", "end_of_day"),
    (("account", "capital_usd"), 0, "capital_usd"),
    (("paper", "fill_at_concession"), 1.5, "fill_at_concession"),
])
def test_invalid_configs_are_refused(tmp_path, path, val, msg):
    with pytest.raises(ConfigContractError, match=msg):
        load_options_config(_write(tmp_path, _mutate(path, val)))


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    p = tmp_path / "options.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="mapping"):
        load_options_config(p)
