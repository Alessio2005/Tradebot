"""De eindkandidaat door de authoritative engine (`systematic/authoritative.py`).

`run_authoritative` komt ná de poortlezing: dezelfde doelgewichten als het screeningsboek, maar door
de soevereine `RiskEngine` en de `OrderRouter` van `backtest/engine.py`, met een bar vertraging.
Hier draait hij op een wegwerpmap met de synthetische markt van `test_systematic_programme_flow.py`:
een bevroren programma, de W_DEV-artefacten van een kandidaat, en een echte `read_gate` die de
poortlezing in het slot registreert. De config van het risicobudget en van de backtest komen uit de
echte `conf/`; er wordt niets uit `data/` gelezen en niets in de repo geschreven.

Wat bewezen wordt: de run weigert zonder geregistreerde poortlezing, rekent dezelfde kandidaat als
de poortlezing, legt vast dat dit WEL promotiebewijs is (anders dan het screeningsboek), en zet
het engine-resultaat naast het screeningsboek met vertraging 1 en 2 op dezelfde vensters.
"""
from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_systematic_programme_flow import (
    GATE_START,
    WINDOWS,
    flow_market,
    install,
    read_json,
    stage_gate,
    utc,
    write_root,
)
from tradebot.systematic import authoritative, programme
from tradebot.systematic.evaluate import summarize
from tradebot.utils.failfast import ConfigContractError, DataContractError

REPO = Path(__file__).resolve().parents[2]
CANDIDATE = "C2_TREND_LF"
NO_FILE = "bestaat niet"


def with_engine_config(root: Path) -> Path:
    """De engine leest het risicobudget en de backtestconfig; die zijn niet klein te maken."""
    for rel in ("conf/risk/default.yaml", "conf/backtest/default.yaml"):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, target)
    return root


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """freeze -> W_DEV-artefacten -> poortlezing -> authoritative, één keer; de toetsen lezen."""
    market = flow_market()
    base = tmp_path_factory.mktemp("authoritative")
    with pytest.MonkeyPatch.context() as mp:
        install(mp, market)
        mp.setattr(authoritative, "load_market", lambda root, cfg: market)
        root, pid = stage_gate(base, market, name=CANDIDATE)
        with_engine_config(root)
        gate = programme.read_gate(root, log=lambda _m: None)
        out = authoritative.run_authoritative(root)
    return SimpleNamespace(root=root, market=market, pid=pid, gate=gate, out=out,
                           art=root / programme.ARTEFACT_DIR)


# --------------------------------------------------------------------------- #
# Weigeringen: vóór de poortlezing is er geen authoritative run
# --------------------------------------------------------------------------- #
class TestRefusals:
    @staticmethod
    def _no_market(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        loads: list[str] = []
        monkeypatch.setattr(authoritative, "load_market",
                            lambda root, cfg: loads.append("load") or flow_market())
        return loads

    def test_without_a_gate_record_there_is_nothing_to_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads = self._no_market(monkeypatch)
        root = write_root(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"gate_read\.json"):
            authoritative.run_authoritative(root)
        assert loads == []

    def test_a_gate_record_whose_read_is_not_in_the_lock_is_refused_before_any_market_is_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads = self._no_market(monkeypatch)
        root = write_root(tmp_path)
        art = root / programme.ARTEFACT_DIR
        art.mkdir(parents=True)
        (art / "gate_read.json").write_text(json.dumps({
            "hypothesis_id": "robust_book_v1/abc/C3_VOLMAN_CORE", "candidate": "C3_VOLMAN_CORE",
            "included": []}), encoding="utf-8")
        with pytest.raises(DataContractError, match="hoort na de geregistreerde poortlezing"):
            authoritative.run_authoritative(root)
        assert loads == [] and not (art / "authoritative.json").exists()

    def test_a_read_registered_for_another_hypothesis_does_not_count(
        self, run: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loads = self._no_market(monkeypatch)
        root = tmp_path / "root"
        shutil.copytree(run.root, root)
        path = root / programme.ARTEFACT_DIR / "gate_read.json"
        path.write_text(json.dumps({**read_json(path), "hypothesis_id": "robust_book_v1/other/C1"}),
                        encoding="utf-8")
        with pytest.raises(DataContractError, match="hoort na de geregistreerde poortlezing"):
            authoritative.run_authoritative(root)
        assert loads == []

    def test_a_missing_engine_config_stops_the_run_and_leaves_no_result(
        self, run: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._no_market(monkeypatch)
        root = tmp_path / "root"
        shutil.copytree(run.root, root)
        (root / "conf/risk/default.yaml").unlink()
        (root / programme.ARTEFACT_DIR / "authoritative.json").unlink()
        with pytest.raises(ConfigContractError, match=NO_FILE):
            authoritative.run_authoritative(root)
        assert not (root / programme.ARTEFACT_DIR / "authoritative.json").exists()


# --------------------------------------------------------------------------- #
# De run
# --------------------------------------------------------------------------- #
class TestTheRun:
    def test_it_runs_the_candidate_the_gate_read_selected_and_counts_as_promotion_evidence(
        self, run: SimpleNamespace,
    ) -> None:
        out = run.out
        assert out["candidate"] == run.gate["candidate"] == CANDIDATE
        assert out["included"] == run.gate["included"] == []
        assert out["engine"] == "backtest.engine.EventDrivenEngine"
        assert out["counts_as_promotion_evidence"] is True

    def test_the_audit_names_the_risk_policy_and_the_assumed_costs(self, run: SimpleNamespace) -> None:
        out = run.out
        assert isinstance(out["risk_policy_hash"], str) and len(out["risk_policy_hash"]) >= 8
        assert out["spread_status"] == "SPREAD_ASSUMED"
        assert out["impact_status"] == "IMPACT_UNCALIBRATED"
        assert out["audit"]["risk_policy_hash"] == out["risk_policy_hash"]

    def test_the_audit_counts_the_bars_from_the_first_tradable_bar_and_shows_that_costs_were_paid(
        self, run: SimpleNamespace,
    ) -> None:
        cfg = programme.robust_book_config(run.root / "conf/model/robust_book.yaml")
        live = run.market.live(cfg.sizing.min_history_bars)
        first = live.any(axis=1).idxmax()
        audit = run.out["audit"]
        assert audit["n_bars"] == int((run.market.index >= first).sum())
        assert audit["n_orders"] > 0 and audit["n_fills"] > 0
        assert audit["total_fees"] > 0.0 and audit["final_equity"] > 0.0

    def test_the_engine_gets_the_fees_of_the_repository_one_bar_of_latency_and_the_assumed_spread(
        self, run: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[tuple[Any, ...]] = []
        real = authoritative.build_engine

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append((*args, kwargs))
            return real(*args, **kwargs)
        monkeypatch.setattr(authoritative, "build_engine", spy)
        monkeypatch.setattr(authoritative, "load_market", lambda root, cfg: run.market)
        root = tmp_path / "root"
        shutil.copytree(run.root, root)
        authoritative.run_authoritative(root)
        (_risk, _impact, venue, spread, kwargs), = seen
        assert (venue.maker_fee_bps, venue.taker_fee_bps) == (2.0, 6.0)      # FEES van de wegwerpmap
        assert venue.latency_bars == 1                                      # het besluit op t vult op t+1
        assert spread.half_spread_bps == 2.5 and spread.source == "conf/execution/fees.yaml"
        assert kwargs == {"initial_equity": 100000.0}                       # conf/backtest/default.yaml

    def test_the_result_is_written_next_to_the_gate_record_and_is_what_was_returned(
        self, run: SimpleNamespace,
    ) -> None:
        written = read_json(run.art / "authoritative.json")
        assert json.dumps(written, sort_keys=True) == json.dumps(run.out, sort_keys=True, default=float)
        assert list(written) == sorted(written)                 # sleutels gesorteerd: diff-vriendelijk

    def test_two_windows_are_reported_the_dev_window_and_the_gate_window(self, run: SimpleNamespace) -> None:
        assert set(run.out["windows"]) == {"w_dev", "gate"}
        for window in run.out["windows"].values():
            assert set(window) == {"authoritative", "screen_lag1", "screen_lag2",
                                   "per_bar_rmse_vs_screen_lag2_bp", "correlation_vs_screen_lag2",
                                   "yearly_authoritative"}
            assert set(window["authoritative"]) == {
                "sharpe", "sharpe_se", "cagr", "ann_vol", "max_drawdown", "sortino", "calmar",
                "sharpe_ci_low", "sharpe_ci_high"}
            assert set(window["screen_lag1"]) == set(window["screen_lag2"]) == {
                "sharpe", "cagr", "ann_vol", "max_drawdown"}

    def test_the_screening_book_is_the_one_the_programme_runs_at_lag_one_and_lag_two(
        self, run: SimpleNamespace,
    ) -> None:
        cfg = programme.robust_book_config(run.root / "conf/model/robust_book.yaml")
        target = programme.build_trial(CANDIDATE, run.market, cfg, included=())
        costs = programme.cost_spec(run.root, cfg)
        dev = (utc(WINDOWS["w_dev_start"]), utc(WINDOWS["w_dev_end"]))
        gate = (utc(GATE_START), run.market.index[-1])
        for lag in (1, 2):
            res = programme.run_targets(target, run.market, costs, lag=lag)
            for label, window in (("w_dev", dev), ("gate", gate)):
                expected = summarize(res.window(*window))
                got = run.out["windows"][label][f"screen_lag{lag}"]
                for key in ("sharpe", "cagr", "ann_vol", "max_drawdown"):
                    assert got[key] == pytest.approx(expected[key]), (label, lag, key)

    def test_the_engine_result_follows_the_engine_and_not_the_screening_book(
        self, run: SimpleNamespace,
    ) -> None:
        """De engine kent geen vlakke kosten maar de volledige kostenketen: zijn Sharpe is een ander
        getal dan dat van het screeningsboek, en de afwijking per bar is de gemeten RMSE."""
        for window in run.out["windows"].values():
            engine, lag2 = window["authoritative"], window["screen_lag2"]
            assert np.isfinite(engine["sharpe"]) and np.isfinite(lag2["sharpe"])
            assert engine["sharpe"] != lag2["sharpe"]
            assert window["per_bar_rmse_vs_screen_lag2_bp"] > 0.0
            assert -1.0 <= window["correlation_vs_screen_lag2"] <= 1.0
            assert engine["sharpe_ci_low"] <= engine["sharpe"] <= engine["sharpe_ci_high"]

    def test_the_yearly_breakdown_covers_the_calendar_years_of_each_window(
        self, run: SimpleNamespace,
    ) -> None:
        assert set(run.out["windows"]["w_dev"]["yearly_authoritative"]) == {"2021", "2022"}
        assert set(run.out["windows"]["gate"]["yearly_authoritative"]) == {"2022"}

    def test_a_second_run_gives_the_same_numbers(
        self, run: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(authoritative, "load_market", lambda root, cfg: run.market)
        root = tmp_path / "root"
        shutil.copytree(run.root, root)
        again = authoritative.run_authoritative(root)
        assert json.dumps(again, sort_keys=True, default=float) == json.dumps(
            run.out, sort_keys=True, default=float)


class TestTheCombinationCarriesItsSleeves:
    def test_the_sleeves_of_the_combination_are_taken_from_the_gate_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        market = flow_market()
        install(monkeypatch, market)
        monkeypatch.setattr(authoritative, "load_market", lambda root, cfg: market)
        included = ("C2_TREND_LF", "C4_CARRY_XS")
        root, _ = stage_gate(tmp_path, market, name="C5_COMBO", included=included)
        with_engine_config(root)
        gate = programme.read_gate(root, log=lambda _m: None)
        out = authoritative.run_authoritative(root)
        assert gate["included"] == list(included) and out["included"] == list(included)
        assert out["candidate"] == "C5_COMBO"


# --------------------------------------------------------------------------- #
# Het scriptpad
# --------------------------------------------------------------------------- #
@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
class TestAsAScript:
    def test_arguments_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["authoritative", "--holdout"])
        with pytest.raises(DataContractError, match="Geen argumenten"):
            runpy.run_module("tradebot.systematic.authoritative", run_name="__main__")

    def test_without_arguments_it_runs_the_repository_root_and_prints_the_result(
        self, run: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`ROOT` is een standaardwaarde die bij het laden van de module wordt vastgelegd; hier wijst
        hij naar de wegwerpmap, zodat het scriptpad nooit de echte repository raakt."""
        root = tmp_path / "root"
        shutil.copytree(run.root, root)
        install(monkeypatch, run.market)                      # programme.load_market, ook voor de import
        monkeypatch.setattr(programme, "ROOT", root)
        monkeypatch.setattr(sys, "argv", ["authoritative"])
        runpy.run_module("tradebot.systematic.authoritative", run_name="__main__")
        printed: Any = json.loads(capsys.readouterr().out)
        assert printed["candidate"] == CANDIDATE and printed["counts_as_promotion_evidence"] is True
        assert read_json(root / programme.ARTEFACT_DIR / "authoritative.json")["candidate"] == CANDIDATE


def test_the_engine_frame_is_the_net_return_with_every_other_column_zero() -> None:
    """`_frame_from_engine` geeft alleen de nettoreeks van de engine door; de kostenkolommen horen
    bij het screeningsboek en staan hier op nul, zodat `summarize` ze niet als engine-kosten leest."""
    index = pd.date_range("2022-01-01", periods=4, freq="D", tz="UTC")
    result = SimpleNamespace(returns=pd.Series([0.01, -0.02, 0.03], index=index[:3]))
    frame = authoritative._frame_from_engine(result, index)
    assert list(frame["net"]) == [0.01, -0.02, 0.03, 0.0]       # ontbrekende bar telt als nul
    assert list(frame["gross"]) == list(frame["net"])
    for column in ("funding", "fees", "spread", "slippage", "impact", "turnover",
                   "gross_leverage", "net_leverage", "n_trades"):
        assert (frame[column] == 0.0).all(), column
