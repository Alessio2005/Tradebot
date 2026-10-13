# tests/unit/test_validation_edges.py
"""De validatiepaden die de hoofdtests niet raken.

Grotendeels de SERIALISATIE van oordelen: `as_dict`, `to_dict`, `as_record`.
Dat zijn geen cosmetische methoden. Een DSR, een SPA of een Sharpe die naar
een artefact gaat zonder zijn venster en zijn SE, is precies het soort getal
dat `require_sharpe_triple` moet tegenhouden -- en die poort draait IN deze
methoden. Wat hier niet wordt aangeroepen, is dus ook niet getoetst.
"""
from __future__ import annotations

import dataclasses
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.registry.preregistration import (
    freeze_preregistration,
    load_preregistration_spec,
)
from tradebot.registry.trial_counter import TrialCount
from tradebot.schemas.config import AdequacyConfig, ValidationConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.validation import weekly_campaign
from tradebot.validation.data_adequacy import PowerAnalysis, assess_meta_labeling
from tradebot.validation.dsr import dsr_from_preregistration, dsr_gate, uncertainty_note
from tradebot.validation.econometrics import SeriesDiagnostics
from tradebot.validation.feature_importance import ImportanceResult, pooled_predictions
from tradebot.validation.gate_runner import GateRun, _junit_counts
from tradebot.validation.gates import run_promotion_gates
from tradebot.validation.inference import (
    _is_positive_multiple,
    block_bootstrap_ci,
    clustered_mean,
    hac_variance_ratio,
    require_sharpe_triple,
    sharpe_difference_test,
    sharpe_with_se,
)
from tradebot.validation.sharpe_difference import jobson_korkie_memmel
from tradebot.validation.spa import SpaResult, spa_gate
from tradebot.validation.vol_metrics import (
    _ar1,
    diebold_mariano_hln,
    loss_series,
    mincer_zarnowitz,
)
from tradebot.validation.walk_forward import (
    PurgedFold,
    _purge,
    purged_walk_forward,
    verify_no_overlap,
)

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "conf/validation/default.yaml", ValidationConfig)
ADEQUACY = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
FROZEN = TrialCount(value=4, source="frozen", origin="test",
                    seed_total=0, registered_total=4)
RNG = np.random.default_rng(31)


@pytest.fixture(autouse=True)
def _fresh_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """Elke test begint met dezelfde, verse generator, zodat de data niet van
    de testvolgorde afhangt (pytest-randomly schudt die per run)."""
    monkeypatch.setattr(f"{__name__}.RNG", np.random.default_rng(31))


def _returns(n: int = 300, mu: float = 0.002) -> np.ndarray:
    return RNG.normal(mu, 0.02, n)


# --------------------------------------------------------------------------- #
# DSR, SPA en de vijf poorten
# --------------------------------------------------------------------------- #
class TestDsr:
    def test_the_record_carries_the_sharpe_triple_and_m(self) -> None:
        record = dsr_gate(_returns(), trial_count=FROZEN, config=CFG).as_dict()
        assert {"dsr", "sharpe_observed", "n_obs", "t_years"} <= set(record)
        assert record["M"] == 4 and record["M_source"] == "frozen"

    def test_the_convenience_path_reads_m_from_the_preregistration(
        self, tmp_path: Path,
    ) -> None:
        prereg = load_preregistration_spec(
            ROOT / "tests/fixtures/preregistration_example.yaml",
            data_hashes=(("crypto/ohlcv/BTCUSDT/1d", "0" * 32),),
            parameters={"alpha": {"lookback_bars": 60}, "M": 7})
        path = freeze_preregistration(prereg, git_sha="abc1234",
                                      ledger_total_at_freeze=2711, directory=tmp_path)
        returns = _returns()
        via_path = dsr_from_preregistration(returns, path, CFG)
        direct = dsr_gate(returns, trial_count=TrialCount(
            value=7, source="frozen", origin="x", seed_total=0, registered_total=7),
            config=CFG)
        assert via_path.dsr == direct.dsr
        assert via_path.as_dict()["M_source"] == "frozen"

    def test_the_uncertainty_note_is_a_fixed_text(self) -> None:
        assert uncertainty_note() == uncertainty_note()
        assert len(uncertainty_note()) > 20


class TestSpa:
    def test_an_assumption_dependent_pass_reads_as_not_significant(self) -> None:
        result = SpaResult(p_value=0.04, p_value_lower=0.01, p_value_upper=0.2,
                           best_strategy_idx=0, n_strategies=3, n_obs=300,
                           alpha=0.05, passed=False, is_assumption_dependent=True)
        assert result.verdict.startswith("ASSUMPTION-DEPENDENT")
        assert result.as_dict()["spa_verdict"] == result.verdict

    def test_a_single_strategy_may_come_as_a_flat_array(self) -> None:
        bench = _returns(mu=0.0)
        result = spa_gate(bench, bench + 0.001, config=CFG, seed=1)
        assert result.n_strategies == 1


class TestPromotionGateRecord:
    def test_the_record_and_summary_cover_every_gate(self) -> None:
        bench = RNG.normal(0.0, 0.02, 300)
        result = run_promotion_gates(
            model_id="unit", oos_returns=bench + RNG.normal(0.002, 0.02, 300),
            benchmark_returns=bench,
            competitor_returns=np.column_stack([bench + RNG.normal(0.002, 0.02, 300)]),
            trial_count=FROZEN, config=CFG, preregistration_id="prereg-abc",
            git_sha="deadbeef", config_hash="1b60cb664fbf9a2a", data_hash="d4ta",
            data_is_adequate=True, adequacy_reason="voldoende",
            lookahead_suite_passed=True, lookahead_reason="groen", spa_seed=3)
        record = result.as_dict()
        assert record["model_id"] == "unit"
        assert "dsr" in record and "spa_p_value" in record
        summary = result.summary()
        assert summary.splitlines()[0] == f"unit: {result.verdict}"
        assert len(summary.splitlines()) == 1 + len(result.gates)


# --------------------------------------------------------------------------- #
# Inferentie
# --------------------------------------------------------------------------- #
class TestInference:
    def test_a_record_without_a_sharpe_needs_no_triple(self) -> None:
        assert require_sharpe_triple({"auc": 0.6}, where="test") is None

    def test_hac_ratio_edges(self) -> None:
        assert hac_variance_ratio(np.zeros(10), 3) == 1.0
        assert np.isfinite(hac_variance_ratio(RNG.normal(size=5), 20))

    def test_every_estimate_serialises_with_its_window(self) -> None:
        a, b = _returns(), _returns(mu=0.0)
        estimate = sharpe_with_se(a, bars_per_year=52).to_dict()
        assert {"sharpe", "se", "n_obs", "t_years"} <= set(estimate)
        diff = sharpe_difference_test(a, b, bars_per_year=52, seed=1, n_boot=50)
        assert "delta_sharpe" in diff.to_dict()
        ci = block_bootstrap_ci(a, bars_per_year=52, seed=1, n_boot=50)
        assert ci.to_dict()["sharpe"] == ci.point

    def test_the_mean_statistic_has_no_sharpe_field(self) -> None:
        ci = block_bootstrap_ci(_returns(), statistic="mean", bars_per_year=52,
                                seed=1, n_boot=50)
        record = ci.to_dict()
        assert "sharpe" not in record
        assert record["statistic"] == "mean"

    def test_more_lags_than_observations_stop_at_the_sample(self) -> None:
        a, b = _returns(40), _returns(40, mu=0.0)
        diff = sharpe_difference_test(a, b, bars_per_year=52, seed=1, n_boot=20,
                                      nw_lags=60)
        assert np.isfinite(diff.se)

    def test_a_zero_vector_is_no_positive_multiple(self) -> None:
        assert _is_positive_multiple(np.zeros(5), np.ones(5)) is False

    def test_clustering_on_columns(self) -> None:
        frame = pd.DataFrame(RNG.normal(0.001, 0.01, size=(60, 4)))
        by_columns = clustered_mean(frame, cluster_axis="columns")
        by_index = clustered_mean(frame.T, cluster_axis="index")
        assert by_columns.mean == pytest.approx(by_index.mean)


# --------------------------------------------------------------------------- #
# Volatiliteitsmetrieken en Sharpe-verschil
# --------------------------------------------------------------------------- #
class TestVolMetrics:
    @staticmethod
    def _series():
        sigma2 = np.abs(RNG.normal(1e-4, 2e-5, 300)) + 1e-6
        proxy = sigma2 * RNG.normal(0.0, 1.0, 300) ** 2 + 1e-8
        return proxy, sigma2

    def test_every_result_serialises(self) -> None:
        proxy, sigma2 = self._series()
        a = loss_series(proxy, sigma2, loss="qlike", model="truthful")
        b = loss_series(proxy, sigma2 * 1.5, loss="qlike", model="over")
        assert a.as_record()["model"] == "truthful"
        assert mincer_zarnowitz(proxy, sigma2, model="m").as_record()["model"] == "m"
        dm = diebold_mariano_hln(a, b, horizon=1)
        assert dm.as_record()["model_a"] == "truthful"

    def test_the_better_model_reads_the_sign_of_the_differential(self) -> None:
        proxy, sigma2 = self._series()
        dm = diebold_mariano_hln(
            loss_series(proxy, sigma2, loss="qlike", model="a"),
            loss_series(proxy, sigma2 * 1.5, loss="qlike", model="b"), horizon=1)
        assert dataclasses.replace(dm, mean_loss_differential=-1.0).better == "a"
        assert dataclasses.replace(dm, mean_loss_differential=1.0).better == "b"
        assert dataclasses.replace(dm, mean_loss_differential=0.0).better == "gelijk"

    def test_the_ar1_of_a_constant_differential_is_zero(self) -> None:
        assert _ar1(np.zeros(8)) == 0.0


def test_the_memmel_result_serialises() -> None:
    result = jobson_korkie_memmel(_returns(), _returns(mu=0.0), bars_per_year=52,
                                  alpha=0.05)
    assert set(result.as_record()) >= {"sharpe_a", "sharpe_b", "p_value"}


# --------------------------------------------------------------------------- #
# Kleine resultaattypen
# --------------------------------------------------------------------------- #
def test_the_stationarity_verdict_covers_all_four_combinations() -> None:
    def _diag(adf: bool, kpss: bool) -> SeriesDiagnostics:
        out = SimpleNamespace
        return SeriesDiagnostics("x", out(rejected=adf), out(rejected=kpss),  # type: ignore[arg-type]
                                 out(), out(), out())  # type: ignore[arg-type]

    assert _diag(True, False).stationarity_verdict.startswith("stationair")
    assert _diag(False, True).stationarity_verdict.startswith("eenheidswortel")
    assert _diag(True, True).stationarity_verdict.startswith("tegenstrijdig")
    assert _diag(False, False).stationarity_verdict.startswith("onbeslist")


def test_importance_and_pooled_predictions() -> None:
    result = ImportanceResult(method="mdi", names=("a", "b"),
                              mean=np.array([0.2, 0.8]), std=np.array([0.1, 0.1]),
                              per_fold=np.array([[0.2, 0.8]]))
    assert [f["name"] for f in result.as_record()["features"]] == ["b", "a"]
    fold = SimpleNamespace(target=np.array([1, 0]), probability=np.array([0.7, 0.2]),
                           uniqueness=np.array([1.0, 0.5]), row_index=np.array([3, 4]))
    pooled = pooled_predictions([fold, fold])
    assert list(pooled["row_index"]) == [3, 4, 3, 4]


class TestGateRunner:
    def test_a_run_renders_with_and_without_counts(self) -> None:
        run = GateRun(name="lookahead", command=("pytest",), returncode=0,
                      passed=True, reason="groen", counts={"passed": 6})
        assert run.as_dict()["command"] == ["pytest"]
        assert run.render().startswith("[PASS] lookahead  (6 passed, 0 skipped)")
        bare = dataclasses.replace(run, passed=False, counts={})
        assert bare.render().startswith("[FAIL] lookahead\n")

    def test_missing_junit_output_has_no_counts(self, tmp_path: Path) -> None:
        assert _junit_counts(tmp_path / "absent.xml") == {}


class TestDataAdequacy:
    def test_too_few_events_names_the_reason(self) -> None:
        verdict = assess_meta_labeling([10] * 6, [0.5] * 6, ADEQUACY)
        assert verdict.adequate is False
        assert "kleinste fold" in verdict.shortfall
        assert verdict.as_record()["adequate"] is False

    def test_an_expected_effect_of_zero_is_infinitely_underpowered(self) -> None:
        power = PowerAnalysis(test="dm", minimum_detectable_effect=0.1,
                              expected_effect=0.0, unit="sharpe", n_effective=50.0)
        assert power.power_deficit == float("inf")
        assert power.as_record()["informative"] is False


class TestPurgedWalkForward:
    def test_a_fold_serialises_its_purity(self) -> None:
        fold = PurgedFold(fold_id=0, train_idx=np.arange(10), test_idx=np.arange(15, 20),
                          n_purged=2, embargo_bars=5, label_horizon_bars=3)
        record = fold.as_dict()
        assert record["train_end"] == 9 and record["test_start"] == 15

    def test_empty_sides_are_left_alone(self) -> None:
        train, n = _purge(np.array([], dtype=int), np.arange(5), 3)
        assert train.size == 0 and n == 0
        empty = PurgedFold(fold_id=0, train_idx=np.array([], dtype=int),
                           test_idx=np.arange(3), n_purged=0, embargo_bars=1,
                           label_horizon_bars=1)
        verify_no_overlap([empty])

    def test_a_handmade_config_without_room_for_training_is_refused(self) -> None:
        """De docstring belooft deze controle voor een handgemaakt configobject;
        `ValidationConfig` zelf laat dit niet toe."""
        cfg = SimpleNamespace(embargo_bars=10, label_horizon_bars=5,
                              train_bars=5, test_bars=20)
        with pytest.raises(DataContractError, match="nul bruikbare folds"):
            list(purged_walk_forward(100, cfg))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Het entrypoint van de wekelijkse campagne
# --------------------------------------------------------------------------- #
class TestWeeklyCampaignEntrypoint:
    def test_main_writes_the_artefact_and_prints_the_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        governance = tmp_path / "artefacts/governance"
        governance.mkdir(parents=True)
        (governance / "preregistration_abc.json").write_text(
            '{"preregistration_id": "abc"}', encoding="utf-8")
        grid = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        market = SimpleNamespace(grid=grid, source_hashes={"x": "h"},
                                 truncate=lambda end: SimpleNamespace(grid=grid))
        verdict = SimpleNamespace(as_dict=lambda: {"status": "UNPROVEN"})
        result = SimpleNamespace(record={"k": 1.0}, values={"auc": 0.5},
                                 verdict=verdict)
        monkeypatch.chdir(tmp_path)
        module = weekly_campaign
        monkeypatch.setattr(module, "weekly_meta_config",
                            lambda: SimpleNamespace(symbols=("BTC",)))
        monkeypatch.setattr(module, "require_preregistration",
                            lambda pid, directory: SimpleNamespace(stop_criteria=()))
        monkeypatch.setattr(module, "load_weekly_market", lambda root, symbols: market)
        monkeypatch.setattr(module, "development_slice",
                            lambda frame, lock_path: frame)
        monkeypatch.setattr(module, "frozen_trial_count", lambda path: FROZEN)
        monkeypatch.setattr(module, "current_git_sha", lambda: "beef123")
        monkeypatch.setattr(module, "run_campaign_on_market",
                            lambda *a, **kw: result)
        (tmp_path / "conf").symlink_to(ROOT / "conf")
        (tmp_path / Path(module.ARTEFACT).parent).mkdir(parents=True, exist_ok=True)
        module.main()
        written = (tmp_path / module.ARTEFACT).read_text(encoding="utf-8")
        assert '"preregistration_id": "abc"' in written
        assert '"git_sha": "beef123"' in written
        assert "UNPROVEN" in capsys.readouterr().out

    @pytest.mark.filterwarnings(
        "ignore:.*found in sys.modules after import of package:RuntimeWarning")
    def test_the_module_runs_as_a_script(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Zonder bevroren pre-registratie weigert hij, en dat is het punt."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["weekly_campaign"])
        with pytest.raises(DataContractError, match="precies één bevroren"):
            runpy.run_module("tradebot.validation.weekly_campaign",
                             run_name="__main__")


def test_a_single_name_is_not_deflated() -> None:
    """Eén naam is geen paneel: geen breedte om voor te corrigeren."""
    single = clustered_mean(pd.DataFrame({"BTC": RNG.normal(0.001, 0.01, 60)}))
    assert single.n_effective == 1.0 and single.rho_bar == 0.0
    assert single.neff_factor == 1.0
