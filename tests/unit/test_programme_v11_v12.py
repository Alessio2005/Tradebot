"""`systematic/programme_v11.py` en `programme_v12.py`: ensemble- en long/short-trend.

Synthetisch, in de tijdelijke root van de v10-tests (geen netwerk, geen bevroren artefact)."""
from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.unit.test_programme_v10 import DAYS, REPO, _prices, _root, _u1
from tradebot.systematic import programme_v10 as v10
from tradebot.systematic import programme_v11 as v11
from tradebot.systematic import programme_v12 as v12
from tradebot.utils.failfast import ConfigContractError, DataContractError


def _full_root(tmp_path: Path) -> Path:
    root = _root(tmp_path)
    for c in (v11.CONFIG, v12.CONFIG):
        shutil.copy(REPO / c, root / c)
    return root


def _funding(root: Path) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", "2026-09-30", freq="D", name="date")
    return pd.DataFrame({"binance_BTC": 1e-4, "binance_ETH": -1e-4, "kraken_BTC": np.nan},
                        index=idx)


def _bad(tmp_path: Path, src: Path, mutate, loader, match: str) -> None:
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigContractError, match=match):
        loader(path)


# --------------------------------------------------------------------------- #
def test_v11_config_and_refusals(tmp_path: Path):
    cfg = v11.load_config(REPO / v11.CONFIG, REPO)
    assert cfg.primary == "G1_ENSEMBLE_10" and cfg.signal.lookbacks == (20, 40, 80, 160, 320)

    def load(p: Path) -> v11.V11Config:
        return v11.load_config(p, REPO)
    src = REPO / v11.CONFIG
    _bad(tmp_path, src, lambda r: r.pop("primary"), load, "v11-config")
    _bad(tmp_path, src, lambda r: r.__setitem__("trials", {}), load, "Geen trials")
    _bad(tmp_path, src, lambda r: r.__setitem__("primary", "X"), load, "hoofdkandidaat")
    _bad(tmp_path, src, lambda r: r.__setitem__("rebalance_weekday", 9), load, "parameter")
    _bad(tmp_path, src, lambda r: r["trials"]["G1_ENSEMBLE_10"].__setitem__("vol_target", 2),
         load, "trial buiten")
    _bad(tmp_path, src, lambda r: r["signal"].pop("lookbacks"), load, "schema")
    (tmp_path / "l.yaml").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="v11-config"):
        load(tmp_path / "l.yaml")


def test_ensemble_score_vol_filter_and_targets():
    close = pd.DataFrame({"BTC": [1.0, 2.0, 3.0, 1.0, 5.0], "ETH": [5.0, 4.0, 3.0, 2.0, 1.0]})
    s = v11.ensemble_score(close, [1, 2])
    assert s["BTC"].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]  # pas met alle periodes
    assert s["ETH"].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    sig = pd.DataFrame({"BTC": [1.0, 1.0, 1.0, 9.0]})
    assert v11.vol_filter(sig, 3, 0.5)["BTC"].tolist() == [True, True, True, False]
    cfg = v11.load_config(REPO / v11.CONFIG, REPO)
    prices, u1 = _prices(), _u1(DAYS)
    plain = v11.sleeve_targets(prices, u1["committed"], cfg, cfg.trials["G1_ENSEMBLE_10"])
    filt = v11.sleeve_targets(prices, u1["committed"], cfg, cfg.trials["G2_ENSEMBLE_VF_10"])
    assert (plain.sum(axis=1) <= 0.40 + 1e-12).all() and plain.values.max() > 0
    assert (filt.gt(0) <= plain.gt(0)).all().all()


def test_v10_book_weekly_rhythm_only_lets_exits_through():
    cfg = v10.load_config(REPO / v10.CONFIG)
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    u1 = pd.DataFrame({"net": 0.0, "rf": 0.0, "cash_fraction": 0.9}, index=idx)
    close = pd.DataFrame({"BTC": 100.0, "ETH": 100.0}, index=idx)
    tgt = pd.DataFrame({"BTC": [0.2, 0.3, 0.0], "ETH": [0.0, 0.1, 0.0]}, index=idx)
    f = v10.book(u1, close, tgt, cfg, rebalance=np.array([True, False, False]))
    assert f["exposure"].tolist() == pytest.approx([0.2, 0.2, 0.0])


def test_v11_freeze_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    root = _full_root(tmp_path)
    with pytest.raises(DataContractError, match="freeze"):
        v11.run_programme(root)
    monkeypatch.setattr(v11, "ROOT", root)
    monkeypatch.setattr(v11, "current_git_sha", lambda: "abc1234")
    assert v11.main(["freeze"]) == 13
    with pytest.raises(DataContractError, match="al geboekt"):
        v11.freeze(root, git_sha="x")
    rec = v11.main(["run"])
    assert set(rec["trials"]) == {"G1_ENSEMBLE_10", "G2_ENSEMBLE_VF_10", "G3_ENSEMBLE_5"}
    assert json.loads((root / v11.ARTEFACT_DIR / "programme.json").read_text())["primary"] \
        == "G1_ENSEMBLE_10"
    capsys.readouterr()
    with pytest.raises(ConfigContractError, match="Gebruik"):
        v11.main([])


# --------------------------------------------------------------------------- #
def test_v12_config_and_refusals(tmp_path: Path):
    cfg = v12.load_config(REPO / v12.CONFIG, REPO)
    assert cfg.primary == "H1_SMA_LS_10" and cfg.funding_columns["BTC"] == "binance_BTC"

    def load(p: Path) -> v12.V12Config:
        return v12.load_config(p, REPO)
    src = REPO / v12.CONFIG
    _bad(tmp_path, src, lambda r: r.pop("primary"), load, "v12-config")
    _bad(tmp_path, src, lambda r: r.__setitem__("trials", {}), load, "Geen trials")
    _bad(tmp_path, src, lambda r: r["trials"]["H1_SMA_LS_10"].__setitem__("signal", "x"),
         load, "schema of bereik")
    _bad(tmp_path, src, lambda r: r["funding_columns"].pop("ETH"), load, "Funding")
    _bad(tmp_path, src, lambda r: r.__setitem__("primary", "X"), load, "hoofdkandidaat")
    _bad(tmp_path, src, lambda r: r.__setitem__("v10_config", "elders.yaml"), load,
         "andere configs")
    (tmp_path / "l.yaml").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="v12-config"):
        load(tmp_path / "l.yaml")


def test_v12_targets_are_signed_and_the_short_earns_funding():
    cfg = v12.load_config(REPO / v12.CONFIG, REPO)
    prices, u1 = _prices(), _u1(DAYS)
    for name in cfg.trials:
        t = v12.signed_targets(prices, u1["committed"], cfg, cfg.trials[name])
        assert (t.abs().sum(axis=1) <= 0.40 + 1e-12).all()
        assert (t < 0).any().any() and (t > 0).any().any()
    c10 = cfg.v11.v10
    idx = pd.date_range("2021-01-01", periods=3, freq="D")
    u1s = pd.DataFrame({"net": 0.0, "rf": 0.0, "cash_fraction": 0.9}, index=idx)
    close = pd.DataFrame({"BTC": [100.0, 90.0, 90.0], "ETH": 100.0}, index=idx)
    fund = pd.DataFrame({"BTC": 1e-3, "ETH": 0.0}, index=idx)
    tgt = pd.DataFrame({"BTC": [-0.2, -0.2, 0.1], "ETH": 0.0}, index=idx)
    f = v12.book(u1s, close, fund, tgt, c10)
    assert f["sleeve"].iloc[1] == pytest.approx(0.02)
    assert f["funding"].iloc[1] == pytest.approx(0.2 * 1e-3)
    assert f["exposure"].iloc[2] == pytest.approx(0.1)          # omgedraaid
    g = v12.book(u1s, close, fund, tgt.assign(BTC=[-0.2, -0.4, -0.4]), c10,
                 rebalance=np.array([True, False, False]))
    assert g["exposure"].iloc[2] < 0.3                          # geen herschaling buiten ritme


def test_v12_freeze_and_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    root = _full_root(tmp_path)
    monkeypatch.setattr(v12.ex, "load_funding", _funding)
    with pytest.raises(DataContractError, match="freeze"):
        v12.run_programme(root)
    monkeypatch.setattr(v12, "ROOT", root)
    monkeypatch.setattr(v12, "current_git_sha", lambda: "abc1234")
    assert v12.main(["freeze"]) == 12
    with pytest.raises(DataContractError, match="al geboekt"):
        v12.freeze(root, git_sha="x")
    rec = v12.main(["run"])
    assert set(rec["trials"]) == {"H1_SMA_LS_10", "H2_ENSEMBLE_LS_10"}
    assert 0.0 < rec["trials"]["H1_SMA_LS_10"]["short_share"] < 1.0
    capsys.readouterr()
    with pytest.raises(ConfigContractError, match="Gebruik"):
        v12.main(["x"])


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
@pytest.mark.parametrize("mod", ["programme_v11", "programme_v12"])
def test_script_guards(mod: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", [mod])
    with pytest.raises(ConfigContractError, match="Gebruik"):
        runpy.run_module(f"tradebot.systematic.{mod}", run_name="__main__")
