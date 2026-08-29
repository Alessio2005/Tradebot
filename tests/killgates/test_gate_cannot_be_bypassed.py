"""DE NEGATIEVE CONTROLE OP DE POORT ZELF — Stage B-6.

    Laat de poort een model met `close.shift(-1)` door, dan is Stage B niet
    afgerond, ongeacht hoe groen de rest is.

Elke andere test in dit project bewaakt een stuk CODE. Dit bestand bewaakt de
BEWIJSVOERING. De vraag is niet of `run_promotion_gates` correct rekent, maar of
er een weg omheen bestaat — een argument, een default, een tweede entrypoint,
een object dat je met de hand kunt bouwen.

WAT HIER IS GEMETEN, EN WAAROM HET ONGEMAKKELIJK IS
===================================================
Een model met perfecte vooruitblik heeft een Sharpe van boven de 10. Het HAALT
de DSR moeiteloos, ook na correctie voor `M = 2776`, en het haalt SPA. Dat is
geen tekortkoming van die toetsen: zij zijn ontworpen om te corrigeren voor het
AANTAL geprobeerde varianten, niet om een lek te vinden. Een lek maakt een model
niet verdacht — het maakt hem goed.

`TestTheStatisticsAloneWouldPromoteIt` meet dat expliciet, met cijfers erbij.
Dat is de scherpste zin die dit bestand te melden heeft:

    **De lookahead-suite is de ENIGE poort die een lekkend model tegenhoudt.
    De statistiek helpt daar niet bij en zal dat nooit doen.**

Daarom is `research_gates.yml` blokkerend en niet adviserend, daarom is
`lookahead_suite_passed` een verplicht argument zonder default, en daarom is de
uitkomst van de suite een MEETRESULTAAT dat de poort krijgt aangeleverd in
plaats van een vlag die de aanroeper zet.

Ref: `fase_7_8_consolidatie_productie.md` §B-6, exit criterium B7; audit §17.1.
"""
from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from tests.lookahead.d1_harness import (
    ROOT,
    FutureLeakingFeature,
    assert_bit_identical,
    certified_frame,
    cut_points,
    requires_store,
)
from tradebot.registry.catalog import ModelCatalog, ModelRecord
from tradebot.registry.lifecycle import LifecycleState, ModelLifecycle, TransitionEvidence
from tradebot.registry.promotion import promote
from tradebot.registry.trial_counter import TrialCount, live_trial_count
from tradebot.schemas.config import ValidationConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.validation.dsr import dsr_gate
from tradebot.validation.gates import (
    PROMOTION_GATE_NAMES,
    GateResult,
    GateVerdict,
    run_promotion_gates,
)

pytestmark = pytest.mark.killgate

N_BARS = 600
#: De bevroren `M` uit de Phase 6-pre-registraties. Bewust de ECHTE waarde: een
#: kleinere M zou de DSR soepeler maken en de meting hieronder minder scherp.
FROZEN_M = TrialCount(value=2776, source="frozen", origin="phase6-prereg",
                      seed_total=2363, registered_total=413)


@pytest.fixture(scope="module")
def cfg() -> ValidationConfig:
    return load_config(ROOT / "conf" / "validation" / "default.yaml",
                       ValidationConfig)


@pytest.fixture(scope="module")
def leaking_returns() -> dict[str, np.ndarray]:
    """Een model dat de volgende bar ziet, en een eerlijk model ernaast.

    `perfect` is `close.shift(-1)` in returnvorm: het neemt elke bar de goede
    kant. Dat is precies het model dat exit-criterium B7 noemt.
    """
    rng = np.random.default_rng(20260829)
    market = rng.normal(0.0002, 0.02, N_BARS)
    return {
        "market": market,
        # shift(-1): de beslissing op t gebruikt de return van t+1.
        "perfect": np.abs(market),
        "honest": rng.normal(0.0, 0.02, N_BARS),
    }


def _gate(cfg: ValidationConfig, returns: np.ndarray, **over):
    rng = np.random.default_rng(4)
    bench = rng.normal(0.0, 0.02, N_BARS)
    base = dict(
        model_id="leaking-reference-model",
        oos_returns=returns,
        benchmark_returns=bench,
        competitor_returns=np.column_stack([returns, bench + rng.normal(0, 0.02, N_BARS)]),
        trial_count=FROZEN_M, config=cfg,
        preregistration_id="prereg-killgate", git_sha="deadbeef",
        config_hash="1b60cb664fbf9a2a", data_hash="d4ta",
        data_is_adequate=True, adequacy_reason="voldoende",
        lookahead_suite_passed=False,
        lookahead_reason="test_truncation_invariance FAILED op close.shift(-1)",
        spa_seed=1,
    )
    base.update(over)
    return run_promotion_gates(**base)


# =========================================================================== #
# 1. Het criterium zelf: het shift(-1)-model wordt geweigerd
# =========================================================================== #
class TestTheLeakingModelIsRefused:
    def test_the_gate_refuses_it(self, cfg, leaking_returns) -> None:
        result = _gate(cfg, leaking_returns["perfect"])
        assert result.passed is False, (
            f"HET LEKKENDE MODEL IS GEPROMOVEERD. Stage B is niet afgerond, "
            f"ongeacht hoe groen de rest staat.\n{result.summary()}")
        assert "lookahead_suite" in result.failed_gates
        assert result.verdict == GateVerdict.FALSIFIED

    def test_it_cannot_reach_candidate(self, cfg, leaking_returns) -> None:
        """De state machine weigert hem ook, en al één stap eerder."""
        result = _gate(cfg, leaking_returns["perfect"])
        lc = ModelLifecycle(model_id="leaking-reference-model")
        with pytest.raises(DataContractError, match="GROENE"):
            lc.transition(LifecycleState.TESTED, TransitionEvidence(
                git_sha="deadbeef", config_hash="1b60cb664fbf9a2a",
                data_hash="d4ta", preregistration_id="prereg-killgate",
                lookahead_suite_passed=False))
        assert lc.state == LifecycleState.REGISTERED
        assert result.passed is False

    def test_it_cannot_reach_prod(self, cfg, leaking_returns, tmp_path) -> None:
        """En de deployment-as weigert hem, met dezelfde uitslag als bewijs."""
        catalog = ModelCatalog(tmp_path / "catalog.json")
        catalog.register(ModelRecord(
            symbol="BTCUSDT", side="LONG", version=1, stage="staging",
            artifact_path="m.joblib", git_sha="abc", dvc_hash="d",
            feature_hash="f",
            metrics={"sharpe": 12.0, "max_dd": 0.01, "n_obs": N_BARS},
            extra={"engine": "event_driven"}))
        with pytest.raises(DataContractError, match="niet is geslaagd"):
            promote(catalog, "BTCUSDT", "LONG", "prod",
                    gate_result=_gate(cfg, leaking_returns["perfect"]))


# =========================================================================== #
# 2. De ongemakkelijke meting
# =========================================================================== #
class TestTheStatisticsAloneWouldPromoteIt:
    """De reden dat `research_gates.yml` blokkerend moet zijn.

    Zou iemand `lookahead_suite_passed=True` doorgeven voor dit model — door de
    suite over te slaan, door hem niet in CI te draaien, of door de uitkomst met
    de hand te zetten — dan komt het lekkende model er GEWOON doorheen.
    """

    def test_perfect_foresight_sails_through_the_dsr(
        self, cfg, leaking_returns
    ) -> None:
        r = dsr_gate(leaking_returns["perfect"],
                     trial_count=FROZEN_M, config=cfg)
        assert r.passed is True, (
            "de DSR weigerde het perfecte-vooruitblikmodel. Dat zou prettig "
            "zijn, maar het is niet wat de toets doet, en erop rekenen is "
            "gevaarlijk.")
        assert r.sharpe_observed > 0.5, (
            f"Sharpe van het lekkende model: {r.sharpe_observed:.3f} — te laag "
            f"om de stelling van dit bestand te dragen")

    def test_the_gate_promotes_it_once_the_lookahead_result_is_falsified(
        self, cfg, leaking_returns
    ) -> None:
        """DE KERNMETING VAN DIT BESTAND.

        Exact hetzelfde lekkende model, één argument anders. Slaagt hij, dan is
        aangetoond dat de lookahead-suite de enige poort is die hem tegenhoudt —
        en dus dat die suite blokkerend in CI moet draaien in plaats van
        adviserend.
        """
        result = _gate(cfg, leaking_returns["perfect"],
                       lookahead_suite_passed=True,
                       lookahead_reason="GELOGEN — de suite is niet gedraaid")
        assert result.passed is True, (
            "het lekkende model wordt ook zonder de lookahead-suite geweigerd. "
            "Dat is goed nieuws, maar controleer WAAROM: valt hij op DSR of SPA, "
            "dan is het lek per ongeluk gevangen en niet door een poort die "
            "ontworpen is om hem te vinden.")

    def test_the_honest_model_is_refused_by_the_statistics(
        self, cfg, leaking_returns
    ) -> None:
        """De tegenhanger. Zonder deze test zou de meting hierboven ook kunnen
        betekenen dat de poort alles doorlaat."""
        result = _gate(cfg, leaking_returns["honest"],
                       lookahead_suite_passed=True,
                       lookahead_reason="6/6 groen")
        assert result.passed is False, (
            f"zuivere ruis is gepromoveerd bij M={FROZEN_M.value}: "
            f"{result.summary()}")


# =========================================================================== #
# 3. De lookahead-suite vindt het lek wel degelijk
# =========================================================================== #
@requires_store
class TestTheLookaheadSuiteCatchesWhatTheStatisticsMiss:
    def test_gate_one_goes_red_on_the_shift_minus_one_model(self) -> None:
        """Dit is de meting die `lookahead_suite_passed=False` moet opleveren.

        Hij draait de ECHTE D-1-poort 1 op het ECHTE lekkende referentiemodel,
        zodat de keten van meting naar oordeel hier compleet is en niet met een
        handmatig gezette vlag begint.
        """
        source = certified_frame("BTCUSDT")
        leaker = FutureLeakingFeature()
        full = leaker.transform(source).values
        caught = 0
        for cut in cut_points(source):
            truncated = leaker.transform(source.truncate(cut)).values
            with pytest.raises(AssertionError):
                assert_bit_identical(
                    truncated, full, what="killgate/shift_minus_one", cut=cut)
            caught += 1
        assert caught == 3


# =========================================================================== #
# 4. Er is geen weg omheen
# =========================================================================== #
class TestThereIsNoBypass:
    """Elke test hier bewaakt een MANIER om de poort te omzeilen, niet een
    berekening. Een poort die je kunt overslaan door een argument te zetten, is
    geen poort."""

    @pytest.mark.parametrize("forbidden", ["force", "override", "warn_only",
                                           "skip_gates", "ignore_lookahead"])
    def test_no_escape_hatch_in_the_signature(self, forbidden: str) -> None:
        params = inspect.signature(run_promotion_gates).parameters
        assert forbidden not in params, (
            f"`run_promotion_gates` heeft een `{forbidden}`-argument. Zo'n "
            f"argument bestaat om op de dag van de deadline te worden gebruikt.")

    def test_every_gate_input_is_mandatory(self) -> None:
        """Een default is een stille aanname. Alleen de bootstrap-seed mag er
        een hebben, want die verandert geen oordeel."""
        params = inspect.signature(run_promotion_gates).parameters
        with_default = {
            name for name, p in params.items()
            if p.default is not inspect.Parameter.empty
        }
        assert with_default <= {"spa_seed"}, (
            f"deze gate-argumenten hebben een default en kunnen dus stil worden "
            f"overgeslagen: {sorted(with_default - {'spa_seed'})}")

    def test_the_verdict_cannot_be_edited_afterwards(
        self, cfg, leaking_returns
    ) -> None:
        result = _gate(cfg, leaking_returns["perfect"])
        for field in ("verdict", "gates", "model_id"):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(result, field, "PROMOTED")

    def test_a_hand_built_pass_with_a_failed_gate_crashes(self) -> None:
        """Het object zelf mag niet in een tegenstrijdige vorm bestaan."""
        gates = dict.fromkeys(PROMOTION_GATE_NAMES, True)
        gates["lookahead_suite"] = False
        with pytest.raises(DataContractError, match="verdict=PROMOTED terwijl"):
            GateResult(
                model_id="forged", verdict=GateVerdict.PROMOTED, gates=gates,
                reasons=dict.fromkeys(PROMOTION_GATE_NAMES, ""),
                trial_count=FROZEN_M, dsr_result=None, spa_result=None,
                git_sha="x", config_hash="y", data_hash="z",
                preregistration_id="p")

    def test_a_verdict_without_provenance_crashes(self) -> None:
        """Audit §26: een oordeel zonder resolvable herkomst is INVALIDE."""
        gates = dict.fromkeys(PROMOTION_GATE_NAMES, True)
        with pytest.raises(DataContractError, match="herkomstveld"):
            GateResult(
                model_id="forged", verdict=GateVerdict.PROMOTED, gates=gates,
                reasons=dict.fromkeys(PROMOTION_GATE_NAMES, ""),
                trial_count=FROZEN_M, dsr_result=None, spa_result=None,
                git_sha="", config_hash="y", data_hash="z",
                preregistration_id="p")

    def test_a_live_m_cannot_carry_a_reported_dsr(self, cfg) -> None:
        """Een live `M` groeit; hetzelfde model zou dan morgen een ander oordeel
        krijgen op dezelfde data."""
        rng = np.random.default_rng(9)
        with pytest.raises(DataContractError, match="LIVE trial-count"):
            dsr_gate(rng.normal(0.001, 0.02, 200),
                     trial_count=live_trial_count(), config=cfg)

    def test_inadequate_data_never_yields_a_promotion(
        self, cfg, leaking_returns
    ) -> None:
        """No-go 13 van de andere kant: UNPROVEN is nooit een pass."""
        result = _gate(cfg, leaking_returns["perfect"],
                       lookahead_suite_passed=True,
                       lookahead_reason="6/6 groen",
                       data_is_adequate=False,
                       adequacy_reason="0,00 % 5m-dekking tegen een eis van 80 %")
        assert result.verdict == GateVerdict.UNPROVEN
        assert result.passed is False
        assert result.dsr_result is None and result.spa_result is None
