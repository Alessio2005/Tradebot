"""De promotiepoort en zijn onderdelen — Phase 7/8 Stage B.

Elke test hier bewaakt een manier waarop de poort stilzwijgend te ruim kan
worden. Dat is een andere opgave dan bewijzen dat hij op de gelukkige weg werkt:
een poort die alleen is getest op input waarvan hij hoort te slagen, is niet
getest.

De negatieve controles staan in `TestTheGateCanActuallyRefuse` en in
`tests/killgates/test_gate_cannot_be_bypassed.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tradebot.registry.trial_counter import (
    M_UNCERTAINTY_NOTE,
    TrialCount,
    frozen_trial_count,
    live_trial_count,
    reconstruction_provenance,
)
from tradebot.schemas.config import ValidationConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError
from tradebot.validation.dsr import MIN_OBS_FOR_DSR, dsr_gate
from tradebot.validation.gates import (
    PROMOTION_GATE_NAMES,
    GateVerdict,
    run_promotion_gates,
)
from tradebot.validation.spa import spa_gate
from tradebot.validation.walk_forward import (
    purged_walk_forward,
    verify_no_overlap,
)

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"


@pytest.fixture(scope="module")
def cfg() -> ValidationConfig:
    return load_config(CONF / "validation" / "default.yaml", ValidationConfig)


@pytest.fixture
def frozen_m() -> TrialCount:
    return TrialCount(value=2776, source="frozen", origin="prereg-test",
                      seed_total=2363, registered_total=413)


def _returns(n: int, mu: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, 0.02, n)


# --------------------------------------------------------------------------- #
# M — de trial-teller
# --------------------------------------------------------------------------- #
def _booked_ledger(tmp_path: Path) -> Path:
    """Een ledger met geboekte trials. De gecommitte ledger is sinds de
    herstart van 2026-09-26 leeg (M = 0, en dat is geen bruikbare M), dus het
    mechanisme wordt op een eigen ledger getoetst."""
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(json.dumps({
        "version": 1, "seed_total": 0,
        "seed_note": "testledger, begonnen op nul",
        "entries": [{"unit": "a", "n_trials": 3}, {"unit": "b", "n_trials": 2}],
    }), encoding="utf-8")
    return path


class TestTrialCounter:
    def test_live_count_matches_the_ledger(self, tmp_path: Path) -> None:
        tc = live_trial_count(_booked_ledger(tmp_path))
        assert tc.value == tc.seed_total + tc.registered_total == 5
        assert tc.source == "live"

    def test_a_live_count_is_not_reproducible(self, tmp_path: Path) -> None:
        """De kern van de Phase 3-les: live M mag geen meting dragen."""
        assert live_trial_count(_booked_ledger(tmp_path)).is_reproducible is False

    def test_m_below_two_is_refused(self) -> None:
        """Bailey-Lopez de Prado is niet gedefinieerd onder twee trials."""
        with pytest.raises(DataContractError, match="onbruikbaar voor een DSR"):
            TrialCount(value=1, source="frozen", origin="x",
                       seed_total=1, registered_total=0)

    def test_parts_must_add_up(self) -> None:
        """Een M waarvan de delen niet optellen, is niet reconstrueerbaar."""
        with pytest.raises(DataContractError, match="niet de som van"):
            TrialCount(value=100, source="frozen", origin="x",
                       seed_total=50, registered_total=10)

    def test_uncertainty_travels_with_the_number(self, frozen_m: TrialCount) -> None:
        assert frozen_m.as_dict()["M_uncertainty"] == M_UNCERTAINTY_NOTE

    def test_provenance_quotes_the_reconstruction(self, tmp_path: Path) -> None:
        prov = reconstruction_provenance(_booked_ledger(tmp_path))
        assert prov["seed_total"] == 0
        assert prov["seed_note"] == "testledger, begonnen op nul"

    def test_missing_preregistration_crashes(self, tmp_path) -> None:
        with pytest.raises(ConfigContractError, match="bestaat niet"):
            frozen_trial_count(tmp_path / "nope.json")

    def test_preregistration_without_m_crashes(self, tmp_path) -> None:
        """Achteraf een M kiezen is de vrijheidsgraad die de DSR wegneemt."""
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps({"content": {"parameters": {}}}), encoding="utf-8")
        with pytest.raises(DataContractError, match="bevat geen M"):
            frozen_trial_count(p)

    def test_frozen_count_is_read_from_the_artefact(self, tmp_path) -> None:
        p = tmp_path / "prereg.json"
        p.write_text(json.dumps({
            "preregistration_id": "abc123",
            "content": {"parameters": {
                "M": 2776, "M_seed_total": 2363, "M_registered_total": 413}},
        }), encoding="utf-8")
        tc = frozen_trial_count(p)
        assert tc.value == 2776
        assert tc.is_reproducible is True


# --------------------------------------------------------------------------- #
# DSR
# --------------------------------------------------------------------------- #
class TestDsrGate:
    def test_live_m_is_refused_for_a_reported_result(
        self, cfg: ValidationConfig, tmp_path: Path
    ) -> None:
        """B2: de DSR weigert te draaien zonder eerlijke, bevroren M."""
        with pytest.raises(DataContractError, match="LIVE trial-count"):
            dsr_gate(_returns(200, 0.001, 1),
                     trial_count=live_trial_count(_booked_ledger(tmp_path)),
                     config=cfg)

    def test_a_bare_integer_is_not_an_m(self, cfg: ValidationConfig) -> None:
        with pytest.raises(DataContractError, match="vereist een TrialCount"):
            dsr_gate(_returns(200, 0.001, 1),
                     trial_count=2776, config=cfg)  # type: ignore[arg-type]

    def test_too_few_observations_crashes(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        with pytest.raises(DataContractError, match="minimaal"):
            dsr_gate(_returns(MIN_OBS_FOR_DSR - 1, 0.001, 1),
                     trial_count=frozen_m, config=cfg)

    def test_non_finite_input_crashes(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        arr = _returns(200, 0.001, 1)
        arr[7] = np.nan
        with pytest.raises(DataContractError, match="niet-eindige"):
            dsr_gate(arr, trial_count=frozen_m, config=cfg)

    def test_constant_series_crashes_instead_of_returning_zero(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """Een niet-gedefinieerde Sharpe is een datadefect, geen nulresultaat."""
        with pytest.raises(DataContractError, match="constante returnreeks"):
            dsr_gate(np.full(200, 0.001), trial_count=frozen_m, config=cfg)

    def test_a_larger_m_never_makes_the_dsr_easier(self, cfg: ValidationConfig) -> None:
        """De monotonie waarop de hele correctie rust."""
        arr = _returns(400, 0.004, 5)
        small = TrialCount(value=10, source="frozen", origin="x",
                           seed_total=0, registered_total=10)
        large = TrialCount(value=100_000, source="frozen", origin="x",
                           seed_total=0, registered_total=100_000)
        assert (dsr_gate(arr, trial_count=large, config=cfg).dsr
                <= dsr_gate(arr, trial_count=small, config=cfg).dsr)

    def test_skew_and_kurtosis_are_measured_not_assumed(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """De onderliggende functie defaultet naar Gaussisch; crypto is dat niet."""
        rng = np.random.default_rng(4)
        skewed = rng.standard_t(df=3, size=500) * 0.01 + 0.001
        r = dsr_gate(skewed, trial_count=frozen_m, config=cfg)
        assert r.kurtosis != 3.0
        assert r.skew != 0.0

    def test_marginal_results_are_read_as_not_significant(
        self, cfg: ValidationConfig
    ) -> None:
        """M is een ondergrens; een randgeval mag niet als PASS gelden."""
        from tradebot.validation.dsr import DsrResult
        r = DsrResult(dsr=1.0 - cfg.dsr_alpha, sharpe_observed=0.1, n_obs=100,
                      trial_count=TrialCount(2776, "frozen", "x", 2363, 413),
                      alpha=cfg.dsr_alpha, skew=0.0, kurtosis=3.0,
                      passed=False, is_marginal=True,
                      # Fase 10 stap 4A: het oordeel draagt sindsdien ook de
                      # herkomst van V[{SR_m}] en het venster (§6 en §10).
                      sr_variance=1.0 / 100, approximation="normal",
                      bars_per_year=365.0, t_years=100 / 365.0)
        assert "NIET significant" in r.verdict


# --------------------------------------------------------------------------- #
# SPA — inclusief de correctie op Hansen's drie p-waarden
# --------------------------------------------------------------------------- #
class TestSpaGate:
    """De POORT. De kern zelf — Hansen's drie p-waarden en de correctie erop —
    staat in `tests/unit/test_spa_hansen.py`; hem hier nog eens toetsen zou twee
    plekken opleveren die uit elkaar kunnen lopen."""

    def test_gate_refuses_when_only_the_liberal_variant_passes(
        self, cfg: ValidationConfig
    ) -> None:
        """Een uitslag die afhangt van hoe de verliezers meetellen, telt niet."""
        rng = np.random.default_rng(11)
        bench = rng.normal(0.0, 0.02, 400)
        s = np.column_stack([
            bench + rng.normal(0.0015, 0.02, 400),
            bench + rng.normal(-0.0002, 0.02, 400),
            bench + rng.normal(-0.05, 0.02, 400),
        ])
        r = spa_gate(bench, s, config=cfg, seed=1)
        if r.is_assumption_dependent:
            assert r.passed is False
            assert "NIET significant" in r.verdict

    def test_length_mismatch_crashes(self, cfg: ValidationConfig) -> None:
        with pytest.raises(DataContractError, match="verschillende lengte"):
            spa_gate(_returns(300, 0.0, 1),
                     _returns(250, 0.001, 2).reshape(-1, 1), config=cfg)

    def test_non_finite_crashes(self, cfg: ValidationConfig) -> None:
        b = _returns(300, 0.0, 1)
        s = _returns(300, 0.001, 2).reshape(-1, 1)
        s[3, 0] = np.inf
        with pytest.raises(DataContractError, match="niet-eindige"):
            spa_gate(b, s, config=cfg)


# --------------------------------------------------------------------------- #
# Purged walk-forward
# --------------------------------------------------------------------------- #
class TestPurgedWalkForward:
    def test_no_train_label_reaches_into_the_test_window(
        self, cfg: ValidationConfig
    ) -> None:
        folds = list(purged_walk_forward(2000, cfg))
        assert folds
        verify_no_overlap(folds)          # crasht bij overlap

    def test_zero_embargo_is_refused(self, cfg: ValidationConfig) -> None:
        bad = cfg.model_copy(update={"embargo_bars": 0, "label_horizon_bars": 1})
        with pytest.raises(ConfigContractError, match="geen purged walk-forward"):
            list(purged_walk_forward(2000, bad))

    def test_embargo_shorter_than_the_horizon_is_refused(self) -> None:
        """Het schema dwingt dit al af; deze test bewaakt een handgemaakte config."""
        with pytest.raises((ConfigContractError, Exception)):
            bad = ValidationConfig(label_horizon_bars=10, embargo_bars=2)
            list(purged_walk_forward(2000, bad))

    def test_too_little_data_crashes_instead_of_yielding_nothing(
        self, cfg: ValidationConfig
    ) -> None:
        """Nul folds stilzwijgend teruggeven laat elke aggregatie leeg middelen."""
        with pytest.raises(DataContractError, match="nul bruikbare folds"):
            list(purged_walk_forward(10, cfg))

    def test_embargo_already_covers_the_horizon_so_purging_finds_nothing(
        self, cfg: ValidationConfig
    ) -> None:
        """Purging en embargo overlappen — en dat HOORT zo.

        Bij `embargo_bars >= label_horizon_bars` heeft de embargo de lekkende
        events al verwijderd vóórdat de purge kijkt. Dat de purge dan nul events
        vindt is het bewijs dat de embargo zijn werk doet, niet dat de purge
        stuk is.

        De eerste versie van deze test verwachtte het omgekeerde en werd rood.
        De verwachting was fout, niet de code.
        """
        long_horizon = cfg.model_copy(
            update={"label_horizon_bars": 20, "embargo_bars": 25})
        folds = list(purged_walk_forward(2000, long_horizon))
        assert folds
        assert all(f.n_purged == 0 for f in folds)
        verify_no_overlap(folds)

    def test_the_purge_itself_removes_leaking_events(self) -> None:
        """De purge direct getoetst, zonder de embargo die hem overbodig maakt.

        Zonder deze test bewijst de suite alleen dat de embargo werkt, en zou
        `_purge` stilzwijgend een no-op kunnen zijn — de tweede verdedigingslinie
        die niets doet.
        """
        from tradebot.validation.walk_forward import _purge

        train = np.arange(0, 100)
        test = np.arange(100, 150)
        kept, n_purged = _purge(train, test, horizon=10)
        assert n_purged == 10, "events 90..99 hebben labels die de testset raken"
        assert int(kept.max()) == 89

    def test_verify_no_overlap_can_actually_fail(self, cfg: ValidationConfig) -> None:
        """Negatieve controle op de zelfcontrole van de splitter."""
        from tradebot.validation.walk_forward import PurgedFold

        leaking = PurgedFold(
            fold_id=0,
            train_idx=np.arange(0, 100),
            test_idx=np.arange(95, 150),      # overlapt met train
            n_purged=0, embargo_bars=5, label_horizon_bars=1,
        )
        with pytest.raises(DataContractError, match="lookahead"):
            verify_no_overlap([leaking])


# --------------------------------------------------------------------------- #
# De poort zelf
# --------------------------------------------------------------------------- #
def _gate_kwargs(cfg: ValidationConfig, m: TrialCount, **over):
    rng = np.random.default_rng(21)
    bench = rng.normal(0.0, 0.02, 300)
    base = dict(
        model_id="unit-under-test",
        oos_returns=bench + rng.normal(0.002, 0.02, 300),
        benchmark_returns=bench,
        competitor_returns=np.column_stack([
            bench + rng.normal(0.002, 0.02, 300),
            bench + rng.normal(-0.001, 0.02, 300)]),
        trial_count=m, config=cfg,
        preregistration_id="prereg-abc", git_sha="deadbeef",
        config_hash="1b60cb664fbf9a2a", data_hash="d4ta",
        data_is_adequate=True, adequacy_reason="voldoende",
        lookahead_suite_passed=True, lookahead_reason="6/6 groen",
        spa_seed=3,
    )
    base.update(over)
    return base


class TestPromotionGate:
    def test_result_is_frozen(self, cfg: ValidationConfig, frozen_m: TrialCount) -> None:
        """Er is geen manier om een False in een True te veranderen."""
        import dataclasses
        r = run_promotion_gates(**_gate_kwargs(cfg, frozen_m))
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.verdict = GateVerdict.PROMOTED     # type: ignore[misc]

    def test_all_five_gates_are_reported(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        r = run_promotion_gates(**_gate_kwargs(cfg, frozen_m))
        assert set(r.gates) == set(PROMOTION_GATE_NAMES)
        assert all(r.reasons[n] for n in PROMOTION_GATE_NAMES)

    def test_missing_provenance_crashes(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """B8: een oordeel zonder resolvable herkomst is invalide."""
        for field_name in ("git_sha", "config_hash", "data_hash"):
            with pytest.raises(DataContractError, match="herkomstveld"):
                run_promotion_gates(**_gate_kwargs(cfg, frozen_m, **{field_name: ""}))


class TestTheGateCanActuallyRefuse:
    """De negatieve controles. Een poort die nooit weigert, is geen poort."""

    def test_no_preregistration_crashes(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """Phase 2 exit-criterium 6: een gate-run zonder pre-registratie-ID
        CRASHT, hij weigert niet.

        Weigeren zou betekenen dat DSR en SPA alsnog draaien en hun p-waarden in
        het artefact belanden — p-waarden over een hypothese die pas na de meting
        is geformuleerd. Dat is precies het getal dat later wordt geciteerd.
        """
        with pytest.raises(DataContractError, match="zonder preregistration_id"):
            run_promotion_gates(**_gate_kwargs(cfg, frozen_m,
                                               preregistration_id=""))

    def test_failed_lookahead_blocks_promotion(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        r = run_promotion_gates(**_gate_kwargs(
            cfg, frozen_m, lookahead_suite_passed=False,
            lookahead_reason="test_truncation_invariance FAILED"))
        assert r.passed is False
        assert "lookahead_suite" in r.failed_gates

    def test_inadequate_data_yields_unproven_not_falsified(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """No-go 13: een model dat de Data Adequacy Gate niet haalt, is NIET
        gefalsificeerd. Onbewezen is niet hetzelfde als bewezen slecht."""
        r = run_promotion_gates(**_gate_kwargs(
            cfg, frozen_m, data_is_adequate=False,
            adequacy_reason="0,00 % 5m-dekking tegen een eis van 80 %"))
        assert r.verdict == GateVerdict.UNPROVEN
        assert r.verdict != GateVerdict.FALSIFIED
        assert r.dsr_result is None, (
            "DSR is gedraaid op ontoereikende data; die p-waarde wordt later "
            "geciteerd alsof hij iets betekent")

    def test_pure_noise_does_not_get_promoted(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """De belangrijkste negatieve controle: ruis moet worden geweigerd."""
        rng = np.random.default_rng(99)
        bench = rng.normal(0.0, 0.02, 400)
        noise = rng.normal(0.0, 0.02, 400)
        r = run_promotion_gates(**_gate_kwargs(
            cfg, frozen_m,
            oos_returns=noise, benchmark_returns=bench,
            competitor_returns=np.column_stack([noise, rng.normal(0, 0.02, 400)])))
        assert r.passed is False, (
            f"zuivere ruis is gepromoveerd met M={frozen_m.value}: {r.summary()}")

    def test_a_gate_result_carries_both_cost_labels(
        self, cfg: ValidationConfig, frozen_m: TrialCount
    ) -> None:
        """No-go 14: geen promotieclaim zonder beide kostenlabels."""
        r = run_promotion_gates(**_gate_kwargs(cfg, frozen_m))
        assert "IMPACT_UNCALIBRATED" in r.cost_labels
        assert "SPREAD_ASSUMED" in r.cost_labels
