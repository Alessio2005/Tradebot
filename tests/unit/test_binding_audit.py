# tests/unit/test_binding_audit.py
"""Welke limiet bindt, en bindt de vol-target nog? Fase 11, stap 3.

De ladder telt dat de soevereine laag op 1.742 van 1.743 bars ingrijpt, maar
niet WELKE limiet dat doet. Die vraag heeft drie formuleringen gekregen (DI-27,
DI-30, failure 1 van fase 11 §3.1) en geen antwoord. Deze tests leggen de
instrumenten vast waarmee hij wordt gemeten:

* `RiskEngine.trace` geeft de keten STAP VOOR STAP terug, en is aantoonbaar
  dezelfde keten als `decide` -- geen tweede implementatie (R-3);
* `step_effects` meet per stap of het boek werkelijk is veranderd, los van wat
  het auditspoor beweert, en telt waar die twee uiteenlopen;
* `policy_from_registry` reconstrueert het VERVALLEN beleid uit het register
  en weigert een reconstructie die niet op haar hash uitkomt;
* `first_divergent_step` wijst aan waar een uniforme factor de ketenuitkomst
  laat afwijken -- de vraag van stap 3.4, als meting in plaats van redenering.
"""
from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.risk.binding_audit import (
    TracingRiskEngine,
    audit_layers,
    first_divergent_step,
    policy_from_registry,
    step_effects,
    summarise_traces,
)
from tradebot.risk.contract import MarketState, RiskState
from tradebot.risk.engine import ChainStep, RiskEngine, risk_config_hash
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

RISK = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)
LAPSED = "1b60cb664fbf9a2a"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
REGISTRY = ROOT / "artefacts/governance/risk_config_registry.json"


def _market(sigma: float = 0.6, adv: float = 5.0e8) -> MarketState:
    return MarketState(
        asof_ts=pd.Timestamp("2024-01-02", tz="UTC"),
        sigma_hat={s: sigma for s in SYMBOLS},
        adv_usd={s: adv for s in SYMBOLS},
        cluster={s: "crypto_perp" for s in SYMBOLS},
    )


def _state(equity: float = 100_000.0) -> RiskState:
    return RiskState(equity=equity, high_water_mark=equity,
                     day_start_equity=equity)


def _book(a: float, b: float) -> dict[str, float]:
    return {"BTCUSDT": a, "ETHUSDT": b}


class TestTheEngineHasOneChain:
    """`trace` is geen reconstructie van `decide` maar dezelfde lus."""

    def test_one_step_per_constraint_in_the_configured_order(self) -> None:
        steps, _ = RiskEngine(RISK).trace(_book(1.0, -0.5), _market(), _state())
        assert [s.name for s in steps] == list(RISK.constraint_order)

    def test_each_step_starts_where_the_previous_one_ended(self) -> None:
        steps, _ = RiskEngine(RISK).trace(_book(1.0, -0.5), _market(), _state())
        assert steps[0].exposure_before == _book(1.0, -0.5)
        for prev, nxt in pairwise(steps):
            assert nxt.exposure_before == prev.exposure_after

    def test_trace_and_decide_are_the_same_chain(self) -> None:
        engine = RiskEngine(RISK)
        steps, _ = engine.trace(_book(1.0, -0.5), _market(), _state())
        decision = engine.decide(_book(1.0, -0.5), _market(), _state())
        assert dict(steps[-1].exposure_after) == dict(decision.permitted_exposure)
        assert tuple(b for s in steps for b in s.bound) == \
            decision.binding_constraints

    def test_trace_refuses_what_decide_refuses(self) -> None:
        with pytest.raises(DataContractError):
            RiskEngine(RISK).trace(_book(float("nan"), 0.0), _market(), _state())

    def test_the_tracing_engine_decides_bit_identically(self) -> None:
        """Meten mag het besluit niet veranderen."""
        plain = RiskEngine(RISK).decide(_book(0.7, 0.3), _market(), _state())
        tracer = TracingRiskEngine(RISK)
        traced = tracer.decide(_book(0.7, 0.3), _market(), _state())
        assert traced == plain
        assert len(tracer.records) == 1
        assert tracer.records[0].asof_ts == _market().asof_ts


class TestStepEffects:
    def test_a_binding_vol_target_shrinks_the_book_by_its_ratio(self) -> None:
        """sigma_boek = (1 + 1) * 0,6 = 1,2; de schaal is 0,20 / 1,2 = 1/6."""
        steps, _ = RiskEngine(RISK).trace(_book(1.0, 1.0), _market(), _state())
        vol = {e.name: e for e in step_effects(steps)}["vol_target"]
        assert vol.changed and vol.recorded
        assert vol.shrink == pytest.approx(1.0 - (0.20 / 1.2), rel=1e-12)

    def test_a_step_that_does_nothing_is_neither_changed_nor_recorded(self) -> None:
        steps, _ = RiskEngine(RISK).trace(_book(1.0, 1.0), _market(), _state())
        halted = {e.name: e for e in step_effects(steps)}["halted"]
        assert not halted.changed and not halted.recorded
        assert halted.shrink == 0.0

    def test_a_record_without_a_change_is_counted_as_a_disagreement(self) -> None:
        """DE NEGATIEVE CONTROLE op de kruiscontrole. Een spoor dat een
        ingreep meldt die het boek niet veranderde, moet zichtbaar worden --
        anders bewijst `n_disagree == 0` op de echte data niets."""
        real, _ = RiskEngine(RISK).trace(_book(1.0, 1.0), _market(), _state())
        vol = next(s for s in real if s.name == "vol_target")
        forged = ChainStep(name="gross_cap", exposure_before=vol.exposure_after,
                           exposure_after=vol.exposure_after, bound=vol.bound)
        effect = step_effects([forged])[0]
        assert effect.recorded and not effect.changed
        summary = summarise_traces([_record(forged)], RISK)
        assert summary["steps"]["gross_cap"]["n_disagree"] == 1


class TestTheVolTarget:
    """Stap 3.3. `w_t = min(1, min(max_leverage, sigma_target / sigma_boek))`."""

    def test_max_leverage_cannot_bind_in_the_sovereign_layer(self) -> None:
        """sigma_boek = 0,1: de verhouding 0,20/0,1 = 2 ligt ONDER max_leverage
        (4,0), dus volgens de formule uit de prompt 'bindt de vol-target'. Maar
        de laag verkleint alleen, en 2 > 1: het boek blijft ongemoeid. Een
        `max_leverage` >= 1 kan in deze laag niets doen."""
        book = _book(0.1 / 1.2, 0.1 / 1.2)
        tracer = TracingRiskEngine(RISK)
        tracer.decide(book, _market(), _state())
        vt = summarise_traces(tracer.records, RISK)["vol_target"]
        assert vt["n_ratio_below_max_leverage"] == 1
        assert vt["n_ratio_below_one"] == 0
        vol = {e.name: e for e in step_effects(tracer.records[0].steps)}["vol_target"]
        assert not vol.changed

    def test_the_largest_ratio_is_reported(self) -> None:
        """REGEL V geldt op bar t voor een factor c precies wanneer
        c >= sigma_target / sigma_boek(t). De grootste verhouding over de run
        is dus de kleinste c waarvoor de regel op ELKE bar houdt."""
        tracer = TracingRiskEngine(RISK)
        tracer.decide(_book(1.0, 1.0), _market(), _state())
        tracer.decide(_book(0.5, 0.5), _market(), _state())
        vt = summarise_traces(tracer.records, RISK)["vol_target"]
        assert vt["ratio_max"] == pytest.approx(0.20 / 0.6, rel=1e-12)


class TestThePolicyFromTheRegister:
    def test_the_lapsed_policy_is_reconstructed_exactly(self) -> None:
        cfg = policy_from_registry(LAPSED)
        assert risk_config_hash(cfg) == LAPSED
        assert cfg.sigma_target == pytest.approx(0.08)

    def test_a_tampered_register_is_refused(self, tmp_path: Path) -> None:
        """DE NEGATIEVE CONTROLE. Eén drempel in de opgeslagen configuratie
        veranderen levert een config die niet meer op haar hash uitkomt."""
        doc = json.loads(REGISTRY.read_text(encoding="utf-8"))
        for entry in doc["entries"]:
            if entry["config_hash"] == LAPSED:
                entry["config"]["sigma_target"] = 0.09
        forged = tmp_path / "risk_config_registry.json"
        forged.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(DataContractError, match=LAPSED):
            policy_from_registry(LAPSED, forged)

    def test_an_unregistered_hash_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="0123456789abcdef"):
            policy_from_registry("0123456789abcdef")


class TestTheUniformFactor:
    """Stap 3.4. Een uniforme factor `c` hoort na de vol-target weg te vallen.
    Waar dat niet zo is, wijst `first_divergent_step` de stap aan."""

    def test_invariant_while_the_vol_target_binds_both_books(self) -> None:
        engine = RiskEngine(RISK)
        full, _ = engine.trace(_book(1.0, 1.0), _market(), _state())
        half, _ = engine.trace(_book(0.5, 0.5), _market(), _state())
        assert first_divergent_step(full, half, 0.5) is None

    def test_the_clamp_breaks_invariance_on_a_small_book(self) -> None:
        """sigma_boek = 0,3 bindt onder 0,20; gehalveerd is het 0,15 en bindt
        het NIET meer. Het gehalveerde boek wordt niet teruggeschaald."""
        engine = RiskEngine(RISK)
        full, _ = engine.trace(_book(0.25, 0.25), _market(), _state())
        half, _ = engine.trace(_book(0.125, 0.125), _market(), _state())
        assert first_divergent_step(full, half, 0.5) == "vol_target"

    def test_the_same_book_stays_invariant_under_the_lapsed_policy(self) -> None:
        """Dezelfde twee boeken onder 0,08: 0,15 > 0,08, dus beide binden."""
        engine = RiskEngine(policy_from_registry(LAPSED))
        full, _ = engine.trace(_book(0.25, 0.25), _market(), _state())
        half, _ = engine.trace(_book(0.125, 0.125), _market(), _state())
        assert first_divergent_step(full, half, 0.5) is None


class TestTheSummary:
    def test_it_counts_per_step_and_reports_both_books(self) -> None:
        tracer = TracingRiskEngine(RISK)
        tracer.decide(_book(1.0, 1.0), _market(), _state())
        tracer.decide(_book(0.1 / 1.2, 0.1 / 1.2), _market(), _state())
        summary = summarise_traces(tracer.records, RISK)
        assert summary["n_decisions"] == 2
        assert summary["steps"]["vol_target"]["n_changed"] == 1
        assert summary["steps"]["vol_target"]["share_changed"] == pytest.approx(0.5)
        assert summary["steps"]["halted"]["n_changed"] == 0
        assert summary["mean_gross_desired"] == pytest.approx(
            (2.0 + 0.2 / 1.2) / 2.0)
        assert summary["risk_policy_hash"] == risk_config_hash(RISK)


class TestTheLadderIsMeasuredNotRebuilt:
    """`run_all_layers(engine_factory=...)`: de audit meet de ladderlussen zelf.

    Zonder deze haak zou de audit L1 en L3 moeten nabouwen, en dan meet hij
    een tweede implementatie van wat de ladder doet (R-3)."""

    @staticmethod
    def _run(factory=None, risk_cfg=RISK):
        import numpy as np

        from tradebot.backtest.phase5_baseline import run_all_layers
        from tradebot.execution.impact_model import ImpactParams, ImpactStatus
        from tradebot.execution.order_router import (
            SpreadModel,
            SpreadStatus,
            VenueSpec,
        )

        n = 60
        index = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC",
                              name="asof_ts")
        rng = np.random.default_rng(4)
        prices = pd.DataFrame({s: 100.0 * np.exp(np.cumsum(
            rng.normal(0.0, 0.02, n))) for s in SYMBOLS}, index=index)
        sigma = pd.DataFrame(0.6, index=index, columns=list(SYMBOLS))
        weights = pd.DataFrame(rng.uniform(-1.0, 1.0, (n, len(SYMBOLS))),
                               index=index, columns=list(SYMBOLS))
        flat = pd.DataFrame(0.0, index=index, columns=list(SYMBOLS))
        kwargs = {} if factory is None else {"engine_factory": factory}
        return run_all_layers(
            "synthetic", weights, prices, sigma, sigma / np.sqrt(365.0),
            flat + 5.0e8, flat + 5.0e7, flat, dict(RISK.clusters),
            risk_cfg=risk_cfg,
            impact=ImpactParams(
                eta=2.9919, kappa_d=0.6720,
                status=ImpactStatus.IMPACT_UNCALIBRATED, method="test",
                data_hash="test", sample_size=1, period_start="2021-01-01",
                period_end="2021-12-31", instruments=("BTCUSDT",),
                eta_ci_low=0.0, eta_ci_high=10.0),
            venue=VenueSpec(maker_fee_bps=1.0, taker_fee_bps=5.5,
                            funding_cap_abs=0.02, min_notional=10.0,
                            latency_bars=1),
            spread=SpreadModel(half_spread_bps=1.0,
                               status=SpreadStatus.SPREAD_ASSUMED,
                               source="test"),
            initial_equity=100_000.0, cost_per_side=6.5e-4,
            bars_per_year=365.0, **kwargs)

    def test_the_factory_is_asked_once_for_l1_and_once_for_l3(self) -> None:
        tracers: dict[str, TracingRiskEngine] = {}

        def factory(layer: str, cfg: RiskConfig) -> TracingRiskEngine:
            tracers[layer] = TracingRiskEngine(cfg)
            return tracers[layer]

        self._run(factory)
        assert sorted(tracers) == ["L1_sovereign", "L3_execution"]
        assert len(tracers["L1_sovereign"].records) == 60
        assert 0 < len(tracers["L3_execution"].records) <= 60

    def test_tracing_leaves_every_ladder_number_unchanged(self) -> None:
        plain = [r.as_record() for r in self._run()]
        traced = [r.as_record() for r in self._run(
            lambda _layer, cfg: TracingRiskEngine(cfg))]
        assert json.dumps(plain, sort_keys=True, default=str) == \
            json.dumps(traced, sort_keys=True, default=str)

    def test_audit_layers_reports_both_layers_with_their_ladder_numbers(
        self,
    ) -> None:
        """De compositie die de app gebruikt: per laag de keten-samenvatting
        plus de laddergetallen van DEZELFDE run, zodat die twee niet uit
        verschillende runs kunnen komen."""
        out = audit_layers(self._run, RISK)
        assert sorted(out) == ["L1_sovereign", "L3_execution"]
        assert out["L1_sovereign"]["n_decisions"] == 60
        assert "mean_gross" in out["L1_sovereign"]["ladder"]
        assert "n_sovereign_clipped" in out["L3_execution"]["ladder"]
        assert out["L3_execution"]["risk_policy_hash"] == risk_config_hash(RISK)

    def test_an_engine_under_another_policy_is_refused(self) -> None:
        """DE NEGATIEVE CONTROLE. De router meldt `risk_cfg`'s hash; een engine
        die onder een ander beleid beslist, zou een besluit onder de verkeerde
        hash laten reizen -- precies het defect van fase 11."""
        with pytest.raises(DataContractError, match="ander beleid"):
            self._run(lambda _layer, _cfg: RiskEngine(policy_from_registry(LAPSED)))


class TestEverySharpeInTheLadderCarriesItsUncertainty:
    """Fase 11 stap 4, R-8 en MEASUREMENT_CONTRACT §10. `phase5_revaluation.json`
    droeg Sharpes zonder SE en zonder het drietal `(n_obs, bars_per_year,
    t_years)`; §10 zegt dat de serializer zo'n Sharpe MOET weigeren."""

    def test_each_row_carries_the_triple_and_both_standard_errors(self) -> None:
        from tradebot.validation.inference import require_sharpe_triple
        for row in TestTheLadderIsMeasuredNotRebuilt._run():
            record = row.as_record()
            require_sharpe_triple(record, where=f"{row.track}/{row.layer}")
            assert record["net_sharpe_se"] > 0.0
            assert record["gross_sharpe_se"] > 0.0

    def test_the_se_comes_from_the_one_implementation(self) -> None:
        """R-3: de SE is die van `validation/inference.py::sharpe_with_se`,
        en de Sharpe ernaast is dezelfde Sharpe als die de SE bij hoort."""
        from tradebot.validation.inference import sharpe_with_se
        for row in TestTheLadderIsMeasuredNotRebuilt._run():
            est = sharpe_with_se(row.returns, bars_per_year=365.0)
            record = row.as_record()
            assert record["net_sharpe_se"] == est.se
            assert record["net_sharpe"] == pytest.approx(est.sharpe, rel=1e-12)
            assert record["n_obs"] == est.n_obs


class TestTwoLaddersSideBySide:
    """Stap 4.3: per cel het verschil, met de SE van elke Sharpe ernaast (R-8),
    uit `sharpe_difference_test` en `sharpe_with_se` (R-3)."""

    @staticmethod
    def _both():
        lapsed = policy_from_registry(LAPSED)
        current = TestTheLadderIsMeasuredNotRebuilt._run()
        old = TestTheLadderIsMeasuredNotRebuilt._run(
            lambda _layer, _cfg: RiskEngine(lapsed), risk_cfg=lapsed)
        return current, old

    def test_every_cell_is_compared_with_its_uncertainty(self) -> None:
        from tradebot.backtest.phase5_baseline import compare_ladders
        from tradebot.validation.inference import require_sharpe_triple
        current, old = self._both()
        cells = compare_ladders(current, old, bars_per_year=365.0)
        assert {(c["track"], c["layer"]) for c in cells} == \
            {(r.track, r.layer) for r in current}
        for cell in cells:
            require_sharpe_triple(cell, where=f"{cell['track']}/{cell['layer']}")
            assert cell["se_current"] > 0.0 and cell["se_lapsed"] > 0.0

    def test_the_layers_before_the_risk_layer_do_not_move(self) -> None:
        """L0 kent geen risicolaag; L1 en L2 zijn onder twee beleidsregels een
        positief veelvoud van elkaar zolang de vol-target op beide bindt. Het
        Sharpe-verschil is daar exact nul -- en dat moet de vergelijking ook
        zeggen, niet een ruisgetal."""
        from tradebot.backtest.phase5_baseline import compare_ladders
        current, old = self._both()
        by = {c["layer"]: c for c in compare_ladders(current, old, bars_per_year=365.0)}
        assert by["L0_vectorized"]["delta_sharpe"] == 0.0

    def test_a_cell_without_its_counterpart_is_refused(self) -> None:
        from tradebot.backtest.phase5_baseline import compare_ladders
        current, old = self._both()
        with pytest.raises(DataContractError, match="tegenhanger"):
            compare_ladders(current, old[:-1], bars_per_year=365.0)


class TestTheRederivationBlock:
    """Stap 4.1/4.2: een her-afleiding onder een exogeen gewijzigd beleid kost
    nul trials, en het artefact zegt dat zelf, met de reden erbij."""

    def test_it_names_what_it_supersedes_and_costs_no_trial(
        self, tmp_path: Path
    ) -> None:
        from tradebot.backtest.phase5_baseline import rederivation_block
        old = tmp_path / "artefacts" / "baseline" / "ladder.json"
        old.parent.mkdir(parents=True)
        old.write_text("{}", encoding="utf-8")
        block = rederivation_block(
            "artefacts/baseline/ladder.json", risk_config_hash(RISK),
            root=tmp_path)
        assert block["supersedes"] == "artefacts/baseline/ladder.json"
        assert block["trials"] == 0
        assert block["risk_policy_hash"] == risk_config_hash(RISK)
        assert "R-2" in block["reason"]

    def test_it_refuses_to_supersede_what_does_not_exist(self) -> None:
        """Een `supersedes` die nergens naar wijst, is een herkomst die niet te
        volgen is."""
        from tradebot.backtest.phase5_baseline import rederivation_block
        with pytest.raises(DataContractError, match="bestaat niet"):
            rederivation_block("artefacts/baseline/nope.json",
                               risk_config_hash(RISK), root=ROOT)


def _record(step: ChainStep):
    """Een `TraceRecord` rond één losse stap, voor de negatieve controle."""
    from tradebot.risk.binding_audit import TraceRecord
    return TraceRecord(asof_ts=_market().asof_ts, steps=(step,),
                       sigma_hat=dict(_market().sigma_hat))
