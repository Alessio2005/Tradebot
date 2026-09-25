"""De vensters worden afgeleid, niet gedefinieerd — en het poortsample wordt niet gelezen.

R-11, eerlijk: `phase11_breadth_measurement.py` is vóór deze test geschreven. De
zekerheid is gehaald met een mutatie: `W_GATE` gelijkzetten aan het volle
venster maakt `test_the_windows_partition_the_full_window` rood.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.breadth import breadth_config
from tradebot.validation.phase11_breadth_measurement import measure_breadth, split_windows

ROOT = Path(__file__).resolve().parents[2]
IDX = pd.date_range("2022-01-01", periods=730, freq="D", tz="UTC")


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    market = rng.standard_normal((len(IDX), 1))
    return pd.DataFrame(0.02 * market + 0.01 * rng.standard_normal((len(IDX), 4)),
                        index=IDX, columns=list("ABCD"))


def _lock(tmp_path: Path) -> Path:
    path = tmp_path / "holdout_lock.json"
    path.write_text(json.dumps({"split_utc": "2023-07-01T00:00:00+00:00",
                                "git_sha": "test", "frozen_utc": "test",
                                "reads": []}), encoding="utf-8")
    return path


def test_the_windows_partition_the_full_window(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    windows = split_windows(_panel(), lock_path=lock)
    dev, gate, full = windows["W_DEV"], windows["W_GATE"], windows["W_FULL"]
    assert len(dev) + len(gate) == len(full)
    assert dev.index.intersection(gate.index).empty
    assert dev.index.max() < pd.Timestamp("2023-07-01", tz="UTC") <= gate.index.min()


def test_splitting_does_not_register_a_gate_read(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    split_windows(_panel(), lock_path=lock)
    assert json.loads(lock.read_text(encoding="utf-8"))["reads"] == []


def test_a_gap_in_the_full_window_is_refused(tmp_path: Path) -> None:
    panel = _panel()
    panel.iloc[3, 1] = np.nan
    with pytest.raises(DataContractError):
        split_windows(panel, lock_path=_lock(tmp_path))


def test_every_construction_is_measured_on_every_window_and_year(tmp_path: Path) -> None:
    cfg = breadth_config(ROOT / "conf" / "research" / "breadth.yaml")
    windows = split_windows(_panel(), lock_path=_lock(tmp_path))
    result = measure_breadth(windows, cfg=cfg, n_boot=40, seed=3, ci_level=0.95,
                             block_length=None)
    assert len(result["rows"]) == len(windows) * len(cfg.constructions)
    years = {row["year"] for row in result["per_year"]}
    assert years == set(IDX.year)
    assert result["first_moments_computed"] is False
    assert all("n_eff" not in row for row in result["rows"])


def test_the_signal_clock_is_measured_on_the_development_window_only(tmp_path: Path) -> None:
    from tradebot.validation.phase11_breadth_measurement import measure_signal_clock

    cfg = breadth_config(ROOT / "conf" / "research" / "breadth.yaml")
    prices = 100.0 * np.exp(_panel().cumsum())
    development = split_windows(_panel(), lock_path=_lock(tmp_path))["W_DEV"].index
    constant = pd.DataFrame(0.25, index=IDX, columns=list("ABCD"))
    rng = np.random.default_rng(22)
    blocks = pd.DataFrame(np.repeat(rng.uniform(-1, 1, (len(IDX) // 10 + 1, 4)), 10,
                                    axis=0)[: len(IDX)], index=IDX, columns=list("ABCD"))
    result = measure_signal_clock({"constant": constant, "blocks": blocks},
                                  prices=prices, usable=IDX,
                                  development_index=development, cfg=cfg,
                                  bars_per_year=365.0)
    assert result["window"] == "W_DEV"
    assert result["tracks"]["constant"]["independent_decisions_per_year"] == 0.0
    assert result["tracks"]["constant"]["unique_rows"] == 1
    assert result["tracks"]["blocks"]["n_bars"] == len(development)
    # instapbar inbegrepen: van nul naar 0,25 op vier namen is omzet 1,0
    assert result["tracks"]["constant"]["mean_turnover"] == pytest.approx(
        1.0 / len(development))


def test_the_wall_follows_its_preregistered_decision_rule(tmp_path: Path) -> None:
    from tradebot.validation.phase11_breadth_measurement import measure_wall

    cfg = breadth_config(ROOT / "conf" / "research" / "breadth.yaml")
    small = cfg.model_copy(update={"wall": cfg.wall.model_copy(update={
        "simulation_n_obs": 20_000, "simulation_ic_grid": (0.05, 0.10)})})
    windows = split_windows(_panel(), lock_path=_lock(tmp_path))
    breadth = measure_breadth(windows, cfg=small, n_boot=40, seed=3, ci_level=0.95,
                              block_length=None)
    wall = measure_wall(windows, breadth, cfg=small, bars_per_year=365.0, m_new=25,
                        dsr_target=0.95)
    assert wall["hurdles"]["W_DEV"]["dsr"] > wall["hurdles"]["W_DEV"]["t"] > 0
    for construction, block in wall["constructions"].items():
        assert construction in small.constructions
        assert len(block["table"]) == len(small.wall.horizons_bars)
        if block["formula_holds"]:
            assert block["correction"] == 1.0
        for row in block["table"]:
            for key, value in row["formula"].items():
                assert row["wall"][key] == value / block["correction"]
        first, last = block["table"][0], block["table"][-1]
        assert last["formula"]["dsr_W_DEV"] > first["formula"]["dsr_W_DEV"]
