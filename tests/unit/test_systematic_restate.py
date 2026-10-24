"""De herberekening van de bevroren uitkomsten (`systematic/restate.py`)."""
from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.systematic import restate as rs
from tradebot.utils.failfast import DataContractError


def _write(root: Path, prog: str, cand: str, window: str, start: str, n: int, seed: int,
           mu: float = 5e-4) -> None:
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    net = np.random.default_rng(seed).standard_normal(n) * 0.003 + mu
    d = root / rs.RESEARCH / prog
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"net": net}, index=pd.Index(idx, name="asof_ts")).to_csv(
        d / f"{cand}_daily_{window}.csv")


def _tree(root: Path) -> None:
    prog, cand = rs.FLAGSHIP
    _write(root, prog, cand, "backcast", "2020-04-01", 275, 1)
    _write(root, prog, cand, "w_dev", "2021-01-01", 1642, 2)
    _write(root, prog, cand, "holdout", "2025-07-01", 457, 3, mu=-1e-4)
    _write(root, "robust_book_v1", "C2_TREND_LF", "gate", "2024-01-01", 200, 4)


def test_the_restatement_covers_every_window_and_the_flagship(tmp_path: Path):
    _tree(tmp_path)
    out = rs.restate(tmp_path)
    flag = out["books"][rs.FLAGSHIP[0]][rs.FLAGSHIP[1]]
    assert {"backcast", "w_dev", "holdout", "full_cycle", "dsr_monthly_w_dev"} <= set(flag)
    full = flag["full_cycle"]
    assert full["n_obs"] == 275 + 1642 + 457
    assert sum(full["share_of_log_pnl_by_year"].values()) == pytest.approx(1.0)
    assert set(full["rolling_12m_return"]) == {"min", "p10", "median"}
    assert "rolling_12m_return" not in flag["backcast"]
    dsr = flag["dsr_monthly_w_dev"]
    assert dsr["M_27"]["dsr"] >= dsr["M_2000"]["dsr"]
    assert dsr["M_27"]["required_annual_sharpe"] < dsr["M_2000"]["required_annual_sharpe"]
    gate_only = out["books"]["robust_book_v1"]["C2_TREND_LF"]
    assert set(gate_only) == {"gate"}
    table = out["exchange_default_" + rs.FLAGSHIP[1]]
    assert table["honest_vol"]["p_0.03"]["1_venue"] < table["honest_vol"]["p_0.03"]["3_venue"]
    json.dumps(out, allow_nan=False)


def test_the_exchange_default_formula():
    t = rs.exchange_default_table(0.10, 0.05)
    p, lgd = 0.03, 0.5
    assert t["p_0.03"]["2_venue"] == pytest.approx((0.10 - p * lgd) / math.sqrt(
        0.05 ** 2 + p * (1 - p) * lgd ** 2))


def test_the_longest_underwater_run():
    r = pd.Series([0.1, -0.05, 0.01, 0.01, 0.1, -0.01])
    assert rs._longest_underwater(r) == 3


def test_refusals(tmp_path: Path):
    with pytest.raises(DataContractError, match="Geen bevroren dagreeksen"):
        rs.restate(tmp_path)
    d = tmp_path / rs.RESEARCH / "robust_book_v9"
    d.mkdir(parents=True)
    (d / "X_daily_weird.csv").write_text("asof_ts,net\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="Onverwachte bestandsnaam"):
        rs.restate(tmp_path)
    short = pd.Series([0.01, 0.02], index=pd.date_range("2021-01-01", periods=2, tz="UTC"))
    with pytest.raises(DataContractError, match="Te weinig maanden"):
        rs.dsr_monthly(short)
    with pytest.raises(DataContractError, match="Gebruik"):
        rs.main(["extra"])


def test_main_writes_the_artefact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    _tree(tmp_path)
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    out = rs.main([])
    assert json.loads((tmp_path / rs.OUT).read_text(encoding="utf-8")) == json.loads(
        json.dumps(out))
    assert "kandidaten" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_the_module_runs_as_a_script_and_refuses_arguments(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["restate", "extra"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.systematic.restate", run_name="__main__")


def test_the_committed_restatement_matches_the_frozen_series():
    """Het artefact in de repo is wat de bevroren reeksen nu opleveren."""
    committed = json.loads((rs.ROOT / rs.OUT).read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(rs.restate(rs.ROOT), sort_keys=True))
    t2 = committed["books"]["robust_book_v8"]["T2_CARRY_TRANCHE_JUMPCAP"]["full_cycle"]
    assert t2["sharpe_iid"] == pytest.approx(5.05, abs=0.01)
    assert t2["sharpe_monthly"] == pytest.approx(1.55, abs=0.01)
