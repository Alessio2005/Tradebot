"""Een risicoconfiguratie is geen hypothese — fase-opdracht §6 en §18.

DE INVARIANT
------------
    M_before = hypothesis_ledger.total_n_hypotheses()
    register_risk_config(...)
    M_after  = hypothesis_ledger.total_n_hypotheses()
    M_before == M_after

WAAROM DIT ERTOE DOET
---------------------
`total_n_hypotheses()` sommeert `n_trials` over de ledger, en dat getal is de
`M` waarmee de Deflated Sharpe Ratio deflateert. Elke rij die daar bij komt,
verlaagt de DSR van ELK eerder resultaat dat met een live `M` is gemeten.

Een risicoconfiguratie voorspelt niets en wordt tegen geen nulhypothese
getoetst. Haar daar neerzetten zou de Phase 3-baseline statistisch slechter
maken omdat iemand een limiet heeft opgeschreven. Phase 5 wijzigt de
clusterlabels en dus de risicoconfiguratie; zonder deze tests is niet
aantoonbaar dat die wijziging de historische DSR-baselines ongemoeid laat.

Deze suite werkt uitsluitend op TIJDELIJKE registers. Een eerdere versie van de
clustertest schreef naar `artefacts/governance/` en liet daar een rij met
`git_sha="test"` achter - een test die een governance-artefact vervuilt,
ondermijnt de auditbaarheid die hij hoort te bewaken.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry
from tradebot.registry.risk_registry import RiskConfigRegistry
from tradebot.risk.engine import RiskEngine, risk_config_hash
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"
GOV = ROOT / "artefacts" / "governance"


@pytest.fixture(scope="module")
def cfg() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


@pytest.fixture
def ledger(tmp_path: Path) -> HypothesisLedger:
    """Een tijdelijke ledger met een gezaaide beginstand.

    `HypothesisLedger` weigert een ontbrekend bestand aan te maken - de seed
    wordt handmatig gezet (Wave 20, stap 0.1) zodat een reconstructie altijd
    een bewuste handeling is. Die eigenschap wordt hier gerespecteerd.
    """
    path = tmp_path / "hypothesis_ledger.json"
    path.write_text(
        json.dumps({"version": 1, "seed_total": 2_711, "entries": []}),
        encoding="utf-8")
    led = HypothesisLedger(path)
    led.append(LedgerEntry(
        wave=1, unit="seed_unit", market="crypto", config_hash="seed",
        n_trials=7, result="archived",
        git_sha="deadbeef", data_hash="d4ta", preregistration_id="prereg-seed"))
    return led


@pytest.fixture
def registry(tmp_path: Path) -> RiskConfigRegistry:
    return RiskConfigRegistry(tmp_path / "risk_config_registry.json")


def _register(registry: RiskConfigRegistry, cfg: RiskConfig, *,
              notes: str = "test") -> bool:
    engine = RiskEngine(cfg)
    return registry.register(
        config_hash=engine.config_hash, git_sha="0000000",
        config=cfg.model_dump(mode="json"), audit_header=engine.audit_header(),
        notes=notes)


# =========================================================================== #
# 1. De invariant zelf
# =========================================================================== #
class TestMDoesNotMove:
    def test_registering_a_risk_config_leaves_m_unchanged(
        self, ledger: HypothesisLedger, registry: RiskConfigRegistry,
        cfg: RiskConfig,
    ) -> None:
        before = ledger.total_n_hypotheses()
        assert _register(registry, cfg)
        assert ledger.total_n_hypotheses() == before

    def test_registering_many_variants_leaves_m_unchanged(
        self, ledger: HypothesisLedger, registry: RiskConfigRegistry,
        cfg: RiskConfig,
    ) -> None:
        """Ook een sweep over risicodrempels telt niet als hypothesen."""
        before = ledger.total_n_hypotheses()
        for gross in (0.5, 1.0, 1.5, 2.0, 2.5):
            _register(registry, cfg.model_copy(update={"gross_cap": gross}),
                      notes=f"gross_cap={gross}")
        assert len(registry.hashes()) == 5
        assert ledger.total_n_hypotheses() == before

    def test_a_real_hypothesis_does_move_m(
        self, ledger: HypothesisLedger
    ) -> None:
        """De teller werkt wél; hij telt alleen geen risicoconfiguraties.

        Zonder deze test zou een kapotte ledger die altijd hetzelfde getal
        teruggeeft, de suite groen maken.
        """
        before = ledger.total_n_hypotheses()
        ledger.append(LedgerEntry(
            wave=2, unit="a_real_unit", market="crypto", config_hash="x",
            n_trials=3, result="falsified",
            git_sha="deadbeef", data_hash="d4ta",
            preregistration_id="prereg-real"))
        assert ledger.total_n_hypotheses() == before + 3


# =========================================================================== #
# 2. De DSR verandert dus niet
# =========================================================================== #
class TestDsrIsUnaffected:
    def test_the_deflated_sharpe_is_identical_before_and_after(
        self, ledger: HypothesisLedger, registry: RiskConfigRegistry,
        cfg: RiskConfig,
    ) -> None:
        """Fase-opdracht §6 punt 3: 'DSR `M` verandert niet'."""
        from tradebot.backtest.metrics import deflated_sharpe

        args = {"sr_observed": 0.10, "n_obs": 2_000}
        before = deflated_sharpe(n_trials=ledger.total_n_hypotheses(), **args)
        _register(registry, cfg)
        after = deflated_sharpe(n_trials=ledger.total_n_hypotheses(), **args)
        assert after == before

    def test_the_phase_5_relabelling_did_not_move_the_live_ledger(self) -> None:
        """Op het ECHTE artefact: `M` is niet gewijzigd door deze fase.

        Read-only. De Phase 3-bevriezing legde `M = 2.715` vast en de baseline
        citeert dat getal; als de clusterwijziging de ledger had geraakt, zou
        die DSR nu anders zijn.
        """
        ledger = HypothesisLedger(GOV / "hypothesis_ledger.json")
        entries = ledger.entries()
        risk_related = [
            e for e in entries
            if "risk" in str(e.get("unit", "")).lower()
            or "cluster" in str(e.get("notes", "")).lower()
        ]
        assert not risk_related, (
            f"de hypothese-ledger bevat risicoconfiguratie-rijen: {risk_related}"
        )


# =========================================================================== #
# 3. De twee registers zijn gescheiden
# =========================================================================== #
class TestTheRegistersAreSeparate:
    def test_the_risk_registry_has_no_trial_counter(self) -> None:
        """Structureel: er is geen veld dat naar `M` kan lekken."""
        assert not hasattr(RiskConfigRegistry, "total_n_hypotheses")
        assert not hasattr(RiskConfigRegistry, "n_trials")

    def test_a_risk_entry_carries_no_n_trials_field(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        _register(registry, cfg)
        doc = json.loads(registry.path.read_text(encoding="utf-8"))
        for entry in doc["entries"]:
            assert "n_trials" not in entry
            assert "result" not in entry

    def test_registration_is_idempotent(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        """Twee keer dezelfde config levert geen tweede rij.

        Zonder idempotentie zou herhaald draaien van `apps/run_stress.py` een
        groeiend register opleveren en de indruk wekken dat er iets veranderde.
        """
        assert _register(registry, cfg) is True
        assert _register(registry, cfg) is False
        doc = json.loads(registry.path.read_text(encoding="utf-8"))
        assert len(doc["entries"]) == 1

    def test_a_changed_limit_produces_a_new_hash_and_a_new_row(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        _register(registry, cfg)
        tighter = cfg.model_copy(update={"gross_cap": 1.25})
        assert risk_config_hash(tighter) != risk_config_hash(cfg)
        assert _register(registry, tighter) is True
        assert len(registry.hashes()) == 2


# =========================================================================== #
# 4. Het echte register is schoon en compleet
# =========================================================================== #
class TestTheLiveRegistry:
    def test_the_current_config_is_registered(self, cfg: RiskConfig) -> None:
        live = RiskConfigRegistry(GOV / "risk_config_registry.json")
        assert risk_config_hash(cfg) in live.hashes()

    def test_every_row_carries_a_real_commit_sha(self) -> None:
        doc = json.loads(
            (GOV / "risk_config_registry.json").read_text(encoding="utf-8"))
        for entry in doc["entries"]:
            sha = str(entry["git_sha"])
            assert sha and sha not in ("test", "unknown", "0000000"), (
                f"registry-rij zonder echte git_sha: {entry['config_hash']}"
            )

    def test_every_row_carries_the_full_config(self) -> None:
        """Een hash zonder inhoud bewijst dat er iets veranderde, niet wat."""
        doc = json.loads(
            (GOV / "risk_config_registry.json").read_text(encoding="utf-8"))
        for entry in doc["entries"]:
            assert entry["config"]
            assert "clusters" in entry["config"]
            assert "constraint_order" in entry["config"]

    def test_the_phase_4_and_phase_5_policies_are_both_recorded(self) -> None:
        """De historie is compleet: je kunt beide regimes terugvinden."""
        doc = json.loads(
            (GOV / "risk_config_registry.json").read_text(encoding="utf-8"))
        clusters = [set(e["config"]["clusters"].values()) for e in doc["entries"]]
        assert {"crypto_l1", "crypto_oracle"} in clusters, (
            "de Phase 4-labeling is niet meer terug te vinden"
        )
        assert {"crypto_perp"} in clusters, (
            "de Phase 5-labeling staat niet in het register"
        )
