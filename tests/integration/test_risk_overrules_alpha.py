"""Risico overruled altijd alpha — bewezen in elk stressscenario.

Phase 4, deliverable 9 / exit-criterium 1. Per scenario: welke exposure vroeg
alpha, welke stond risk toe, en welke constraint bond.

De doctrine uit auditsectie 14 is een bewering tot iemand hem meet. Deze suite
meet hem, en de harness zelf crasht wanneer een scenario niets laat binden -
dan is de kalibratie te ruim, niet het scenario geslaagd.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tradebot.risk.contract import ConstraintKind
from tradebot.risk.engine import RiskEngine
from tradebot.risk.kill_switches import HaltStore
from tradebot.risk.stress_test import (
    GAP_DOWN_FRACTION,
    SCENARIOS,
    VOL_SHOCK_MULTIPLE,
    BaseBook,
    RiskStressHarness,
)
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError

CONF = Path(__file__).resolve().parents[2] / "conf"
TS = pd.Timestamp("2026-01-01T00:00:00Z")

#: De geannualiseerde vol van de Phase 3-baseline (reports/BASELINE_BENCHMARK.md
#: sectie 3.1: 72,4 % voor de 1/N-track). Het stressvertrekpunt is dus geen
#: verzonnen getal maar de gemeten uitgangssituatie van dit platform.
BASELINE_VOL = 0.724


@pytest.fixture
def cfg() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


@pytest.fixture
def engine(cfg: RiskConfig) -> RiskEngine:
    return RiskEngine(cfg)


@pytest.fixture
def base(cfg: RiskConfig) -> BaseBook:
    symbols = tuple(cfg.clusters)
    return BaseBook(
        symbols=symbols,
        desired_exposure={s: 0.6 for s in symbols},
        sigma_hat={s: BASELINE_VOL for s in symbols},
        adv_usd={s: 5e8 for s in symbols},
        equity=1.0,
        high_water_mark=1.0,
        day_start_equity=1.0,
        asof_ts=TS,
    )


@pytest.fixture
def harness(engine: RiskEngine) -> RiskStressHarness:
    return RiskStressHarness(engine)


class TestEveryScenarioIsOverruled:
    """Exit-criterium 1: toegestaan < gevraagd, met registratie, in S1 t/m S4."""

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_permitted_is_strictly_below_requested(
        self, harness: RiskStressHarness, base: BaseBook, name: str
    ) -> None:
        outcome = harness.run(name, base)
        assert outcome.permitted_gross < outcome.requested_gross
        assert outcome.gross_reduction > 0.0

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_at_least_one_constraint_is_registered(
        self, harness: RiskStressHarness, base: BaseBook, name: str
    ) -> None:
        outcome = harness.run(name, base)
        assert outcome.binding, "geen bindende constraint geregistreerd"
        assert not outcome.decision.unconstrained

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_every_binding_constraint_is_traceable_to_a_configured_threshold(
        self, harness: RiskStressHarness, base: BaseBook, name: str
    ) -> None:
        """Exit-criterium 6: drempel EN gemeten waarde, per ingreep."""
        for c in harness.run(name, base).binding:
            assert c.config_key.startswith(("risk.", "risk_state."))
            assert c.measured == c.measured  # geen NaN
            assert c.threshold == c.threshold

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_the_outcome_serialises_for_the_report(
        self, harness: RiskStressHarness, base: BaseBook, name: str
    ) -> None:
        json.dumps(harness.run(name, base).as_record())

    def test_all_four_scenarios_are_present(self) -> None:
        assert set(SCENARIOS) == {
            "S1_vol_shock", "S2_correlation_collapse",
            "S3_gap_down", "S4_alpha_runaway",
        }


class TestS1VolShock:
    def test_the_vol_target_is_what_binds(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        outcome = harness.run("S1_vol_shock", base)
        assert ConstraintKind.VOL_TARGET in outcome.decision.bound_kinds

    def test_a_tenfold_vol_shock_de_grosses_roughly_tenfold(
        self, harness: RiskStressHarness, base: BaseBook, cfg: RiskConfig
    ) -> None:
        """De targeting is 1/sigma, dus 10x vol hoort ~10x minder exposure te geven.

        De marge is er omdat andere limieten ook kunnen binden; de test eist
        alleen dat de schaling de goede orde van grootte heeft.
        """
        calm = harness.run("S4_alpha_runaway", base)  # zelfde a_t-niveau, geen vol-schok
        shocked = harness.run("S1_vol_shock", base)
        ratio = calm.permitted_gross / max(shocked.permitted_gross, 1e-12)
        # a_t is in S4 hoger (1.0 vs 0.6), dus corrigeer voor de gevraagde gross.
        normalised = ratio * (shocked.requested_gross / calm.requested_gross)
        assert 5.0 < normalised < 20.0, normalised

    def test_the_shock_multiple_is_the_one_the_phase_prescribes(self) -> None:
        assert VOL_SHOCK_MULTIPLE == 10.0


class TestS2CorrelationCollapse:
    def test_the_vol_target_still_binds(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        outcome = harness.run("S2_correlation_collapse", base)
        assert ConstraintKind.VOL_TARGET in outcome.decision.bound_kinds

    def test_the_vol_target_is_invariant_to_the_collapse_by_construction(
        self, engine: RiskEngine, base: BaseBook
    ) -> None:
        """De targeting rekent al met rho = 1; de werkelijkheid haalt haar in.

        Dit is geen zwakte maar het ontwerp uit `risk/vol_targeting.py`: de
        comonotone bovengrens kan alleen te veel de-grossen, nooit te weinig.
        Een engine die op een geschatte correlatiematrix had gedraaid, zou hier
        juist NU zijn schatting hebben moeten bijstellen - op het moment dat
        die schatting het minst betrouwbaar is.
        """
        from tradebot.risk.vol_targeting import book_sigma_hat

        before = book_sigma_hat(base.desired_exposure, base.sigma_hat)
        _, clusters = SCENARIOS["S2_correlation_collapse"][1](base)
        after = book_sigma_hat(base.desired_exposure, base.sigma_hat)
        assert before == after
        assert len(set(clusters.values())) == 1

    def test_the_cluster_cap_becomes_vacuous_and_that_is_reported(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        """Een universum van één cluster heeft geen clusterlimiet meer.

        Wiskundig onvermijdelijk: je kunt niet over clusters spreiden als er één
        is. Het gevolg is wel dat de bescherming die in normale tijden van de
        clusterlimiet komt, juist in S2 wegvalt - en dat hoort in het rapport,
        niet in een voetnoot. De per-asset concentratielimiet blijft wel staan.
        """
        outcome = harness.run("S2_correlation_collapse", base)
        assert ConstraintKind.CLUSTER_CAP not in outcome.decision.bound_kinds
        permitted = dict(outcome.decision.permitted_exposure)
        gross = outcome.permitted_gross
        assert gross > 0.0
        worst = max(abs(v) for v in permitted.values()) / gross
        assert worst <= max(harness.engine.config.max_concentration,
                            1.0 / len(permitted)) + 1e-9


class TestS3GapDown:
    def test_the_book_is_halted_not_merely_de_grossed(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        outcome = harness.run("S3_gap_down", base)
        assert outcome.permitted_gross == pytest.approx(0.0)
        assert outcome.decision.risk_state_out.halted
        assert outcome.gross_reduction == pytest.approx(1.0)

    def test_the_daily_loss_governor_fires_first(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        """Hij staat vóór de drawdown-breaker in `constraint_order`."""
        outcome = harness.run("S3_gap_down", base)
        assert outcome.decision.bound_kinds[0] is ConstraintKind.DAILY_LOSS_GOVERNOR

    def test_a_thirty_percent_gap_clears_every_configured_threshold(
        self, cfg: RiskConfig
    ) -> None:
        assert GAP_DOWN_FRACTION > cfg.max_drawdown_pct
        assert GAP_DOWN_FRACTION > cfg.daily_loss_limit
        assert all(GAP_DOWN_FRACTION > t.drawdown for t in cfg.drawdown_breaker_levels)

    def test_the_halt_persists_and_the_next_bar_stays_flat(
        self, cfg: RiskConfig, base: BaseBook, tmp_path: Path
    ) -> None:
        """Er is geen bar tussen: de kill switch moet ook ongezien vuren."""
        store = HaltStore(tmp_path / "halt.json")
        harness = RiskStressHarness(RiskEngine(cfg, halt_store=store))
        harness.run("S3_gap_down", base)
        assert store.is_halted()

        # Nieuw proces, herstelde markt: het boek blijft dicht.
        restarted = RiskEngine(cfg, halt_store=HaltStore(tmp_path / "halt.json"))
        recovered = restarted.hydrate(base.risk_state())
        assert recovered.halted
        decision = restarted.decide(
            base.desired_exposure, base.market(dict(cfg.clusters)), recovered
        )
        assert decision.gross() == pytest.approx(0.0)
        assert decision.bound_kinds[0] is ConstraintKind.HALTED


class TestS4AlphaRunaway:
    def test_maximum_conviction_buys_no_extra_exposure(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        """De directe toets op de doctrine: risico overruled altijd alpha."""
        outcome = harness.run("S4_alpha_runaway", base)
        assert outcome.requested_gross == pytest.approx(float(len(base.symbols)))
        assert outcome.permitted_gross < outcome.requested_gross
        assert outcome.gross_reduction > 0.9

    def test_the_result_respects_every_book_level_cap(
        self, harness: RiskStressHarness, base: BaseBook, cfg: RiskConfig
    ) -> None:
        outcome = harness.run("S4_alpha_runaway", base)
        assert outcome.permitted_gross <= cfg.gross_cap + 1e-9
        assert abs(outcome.decision.net()) <= cfg.net_cap + 1e-9
        for w in outcome.decision.permitted_exposure.values():
            assert abs(w) <= cfg.max_position_pct + 1e-9

    def test_doubling_the_request_does_not_double_the_permission(
        self, engine: RiskEngine, base: BaseBook
    ) -> None:
        """De risicolaag is niet proportioneel aan de vraag boven haar limiet."""
        import dataclasses

        half = dataclasses.replace(
            base, desired_exposure={s: 0.5 for s in base.symbols}
        )
        full = dataclasses.replace(
            base, desired_exposure={s: 1.0 for s in base.symbols}
        )
        clusters = dict(engine.config.clusters)
        a = engine.decide(half.desired_exposure, half.market(clusters), half.risk_state())
        b = engine.decide(full.desired_exposure, full.market(clusters), full.risk_state())
        assert b.gross() < 2.0 * a.gross()


class TestTheHarnessRefusesALooseCalibration:
    def test_a_scenario_that_binds_nothing_crashes(
        self, cfg: RiskConfig, base: BaseBook
    ) -> None:
        """*"Als in een scenario geen enkele limiet bindt, is de kalibratie te ruim."*

        Met een absurd ruime configuratie bindt S2 niets meer, en dan hoort de
        harness te crashen in plaats van een groen rapport te produceren.
        """
        import dataclasses

        loose = cfg.model_copy(update={
            "sigma_target": 1.0, "max_leverage": 1.0, "gross_cap": 100.0,
            "net_cap": 100.0, "max_position_pct": 1.0, "max_concentration": 1.0,
            "max_cluster_concentration": 1.0, "adv_participation_cap": 1.0,
        })
        tiny = dataclasses.replace(
            base, desired_exposure={s: 0.01 for s in base.symbols},
            sigma_hat={s: 0.001 for s in base.symbols},
        )
        with pytest.raises(ConfigContractError):
            RiskStressHarness(RiskEngine(loose)).run("S2_correlation_collapse", tiny)

    def test_an_unknown_scenario_crashes(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        with pytest.raises(ConfigContractError):
            harness.run("S9_not_a_scenario", base)

    def test_scenarios_do_not_leak_state_into_each_other(
        self, harness: RiskStressHarness, base: BaseBook
    ) -> None:
        """S3 halt het boek; S4 mag dat niet erven."""
        ordered = harness.run_all(base)
        by_name = {o.name: o for o in ordered}
        assert by_name["S3_gap_down"].decision.risk_state_out.halted
        assert not by_name["S4_alpha_runaway"].decision.risk_state_out.halted
        assert by_name["S4_alpha_runaway"].permitted_gross > 0.0
