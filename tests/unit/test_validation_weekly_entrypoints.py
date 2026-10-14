# tests/unit/test_validation_weekly_entrypoints.py
"""De twee entrypoints van de wekelijkse tak: de BTC-trendregel (`trend_overlay.main`) en het eindmodel
(`weekly_final_model.main`).

Beide lezen en schrijven onder de werkmap; hier is dat een `tmp_path` met een echt bevroren holdout-slot
(`freeze_holdout`) en de synthetische dagmarkt. De berekening zelf staat in `test_trend_overlay.py` en
`test_weekly_final_model.py`; wat hier getoetst wordt is de bedrading: welke sample de ontwikkelrun ziet,
dat de poort alleen met `--holdout` en dan precies eenmaal wordt gelezen, en wat er in het artefact staat.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

from tests.weekly_fixtures import synthetic_market
from tradebot.labeling.barrier_fills import round_trip_cost
from tradebot.schemas.config import ExecutionConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError
from tradebot.validation import trend_overlay, weekly_final_model
from tradebot.validation.holdout import HoldoutAlreadyUsed, freeze_holdout, gate_slice

ROOT = Path(__file__).resolve().parents[2]
N_BARS, SPLIT_BAR = 420, 330


@pytest.fixture()
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Een werkmap met de echte `conf/`, een leeg governance-map en een bevroren holdout-slot."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "conf").symlink_to(ROOT / "conf")
    (tmp_path / "artefacts" / "governance").mkdir(parents=True)
    market = synthetic_market(n=N_BARS, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    freeze_holdout(split_utc=market.grid[SPLIT_BAR].isoformat(),
                   out=tmp_path / "artefacts/governance/holdout_lock.json", git_sha="deadbee")
    return tmp_path


def _market():
    return synthetic_market(n=N_BARS, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"))


def _lock(workdir: Path) -> dict:
    return json.loads((workdir / "artefacts/governance/holdout_lock.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# trend_overlay
# --------------------------------------------------------------------------- #
class TestTrendOverlayMain:
    @pytest.fixture(autouse=True)
    def _wire(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(trend_overlay, "load_weekly_market", lambda root, symbols: _market())
        monkeypatch.setattr(trend_overlay, "current_git_sha", lambda: "beef123")
        monkeypatch.setattr(sys, "argv", ["trend_overlay"])

    def _record(self, workdir: Path) -> dict:
        return json.loads((workdir / trend_overlay.ARTEFACT).read_text(encoding="utf-8"))

    def test_without_the_flag_only_the_development_sample_is_measured(
        self, workdir: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        trend_overlay.main()
        rec = self._record(workdir)
        grid = _market().grid
        assert rec["hypothesis_id"] == trend_overlay.HYPOTHESIS_ID and rec["git_sha"] == "beef123"
        assert rec["development_window"] == [str(grid[trend_overlay.MA_BARS]), str(grid[SPLIT_BAR - 1])]
        assert "holdout" not in rec and _lock(workdir)["reads"] == []
        exec_cfg = load_config(ROOT / "conf/execution/fees.yaml", ExecutionConfig)
        assert rec["cost_per_side"] == pytest.approx(round_trip_cost(exec_cfg) / 2.0)
        assert set(rec["development"]) == {"strategy", "buy_and_hold"}
        assert set(json.loads(capsys.readouterr().out)) == {"development"}

    def test_the_flag_reads_the_gate_sample_once_and_registers_the_read(
        self, workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["trend_overlay", "--holdout"])
        trend_overlay.main()
        rec = self._record(workdir)
        grid = _market().grid
        assert rec["holdout_window"] == [str(grid[SPLIT_BAR]), str(grid[-1])]
        assert 0.0 <= rec["holdout_time_in_market"] <= 1.0
        assert rec["holdout"]["strategy"]["n_bars"] == float(N_BARS - SPLIT_BAR)
        assert [r["hypothesis_id"] for r in _lock(workdir)["reads"]] == [trend_overlay.HYPOTHESIS_ID]
        assert set(json.loads(capsys.readouterr().out)) == {"development", "holdout"}
        with pytest.raises(HoldoutAlreadyUsed):                       # een tweede lezing wordt geweigerd
            trend_overlay.main()

    @pytest.mark.filterwarnings(
        "ignore:.*found in sys.modules after import of package:RuntimeWarning")
    def test_the_module_runs_as_a_script(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zonder `conf/` in de werkmap faalt hij bij het eerste bestand dat hij leest, en dat bewijst
        dat het scriptpad `main()` aanroept."""
        (workdir / "conf").unlink()
        with pytest.raises(ConfigContractError, match="bestaat niet"):
            runpy.run_module("tradebot.validation.trend_overlay", run_name="__main__")


# --------------------------------------------------------------------------- #
# weekly_final_model
# --------------------------------------------------------------------------- #
CAMPAIGN = {"k": 1.5, "d_star": 0.4, "costs": {"round_trip": 0.0013}, "preregistration_id": "abc123",
            "verdict": {"status": "FAIL", "binding": ["overfit_probability"]}}


class TestFinalModelMain:
    @pytest.fixture(autouse=True)
    def _wire(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
        self.seen: dict = {}

        def fit(market, cfg, *, k, d_star, cost_rt):
            self.seen.update(grid_end=market.grid[-1], k=k, d_star=d_star, cost_rt=cost_rt)
            return {"kind": "fake model"}, ["feat_a", "feat_b"], 42

        monkeypatch.setattr(weekly_final_model, "weekly_meta_config",
                            lambda: SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")))
        monkeypatch.setattr(weekly_final_model, "load_weekly_market", lambda root, symbols: _market())
        monkeypatch.setattr(weekly_final_model, "fit_final_model", fit)
        monkeypatch.setattr(weekly_final_model, "current_git_sha", lambda: "beef123")
        (workdir / "artefacts/governance/weekly_meta_campaign.json").write_text(
            json.dumps(CAMPAIGN), encoding="utf-8")
        return self.seen

    def test_the_model_is_fitted_through_the_last_development_bar_and_carries_its_verdict(
        self, workdir: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        weekly_final_model.main()
        grid = _market().grid
        assert self.seen == {"grid_end": grid[SPLIT_BAR - 1], "k": 1.5, "d_star": 0.4, "cost_rt": 0.0013}
        model_file = workdir / weekly_final_model.MODEL_PATH
        saved = joblib.load(model_file)
        assert saved == {"model": {"kind": "fake model"}, "feature_columns": ["feat_a", "feat_b"]}
        card = json.loads((workdir / weekly_final_model.CARD_PATH).read_text(encoding="utf-8"))
        assert card["n_events_fitted"] == 42 and card["feature_columns"] == ["feat_a", "feat_b"]
        assert card["fitted_through"] == str(grid[SPLIT_BAR - 1]) and card["holdout_read"] is False
        assert card["model_sha256"] == hashlib.sha256(model_file.read_bytes()).hexdigest()
        assert card["preregistration_id"] == "abc123" and card["git_sha"] == "beef123"
        assert card["campaign_verdict"] == "FAIL" and card["promoted"] is False
        assert card["campaign_binding"] == ["overfit_probability"]
        assert json.loads(capsys.readouterr().out) == {
            "campaign_verdict": "FAIL", "promoted": False, "n_events_fitted": 42}
        assert _lock(workdir)["reads"] == []

    def test_only_a_pass_verdict_marks_the_model_as_promoted(self, workdir: Path) -> None:
        campaign = {**CAMPAIGN, "verdict": {"status": "PASS", "binding": []}}
        (workdir / "artefacts/governance/weekly_meta_campaign.json").write_text(
            json.dumps(campaign), encoding="utf-8")
        weekly_final_model.main()
        card = json.loads((workdir / weekly_final_model.CARD_PATH).read_text(encoding="utf-8"))
        assert card["promoted"] is True and card["campaign_binding"] == []

    def test_a_holdout_that_has_been_read_stops_the_run_before_anything_is_fitted(
        self, workdir: Path,
    ) -> None:
        gate_slice(pd.DataFrame(index=_market().grid), lock_path=workdir / "artefacts/governance/holdout_lock.json",
                   hypothesis_id="H-1")
        with pytest.raises(DataContractError, match="al gelezen"):
            weekly_final_model.main()
        assert self.seen == {} and not (workdir / weekly_final_model.MODEL_PATH).exists()

    @pytest.mark.filterwarnings(
        "ignore:.*found in sys.modules after import of package:RuntimeWarning")
    def test_the_module_runs_as_a_script(self, workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Zonder campagne-artefact weigert hij, en dat bewijst dat het scriptpad `main()` aanroept."""
        (workdir / "artefacts/governance/weekly_meta_campaign.json").unlink()
        with pytest.raises(FileNotFoundError):
            runpy.run_module("tradebot.validation.weekly_final_model", run_name="__main__")
