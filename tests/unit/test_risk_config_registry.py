"""Het risicoconfiguratie-register — auditbaar, append-only, en NIET de ledger.

Phase 4, stap 11 / exit-criterium 7.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.registry.hypothesis_ledger import HypothesisLedger
from tradebot.registry.risk_registry import RiskConfigRegistry
from tradebot.risk.engine import RiskEngine, risk_config_hash
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

CONF = Path(__file__).resolve().parents[2] / "conf"
GOV = Path(__file__).resolve().parents[2] / "artefacts" / "governance"


@pytest.fixture
def cfg() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


@pytest.fixture
def registry(tmp_path: Path) -> RiskConfigRegistry:
    return RiskConfigRegistry(tmp_path / "risk_config_registry.json")


def _register(registry: RiskConfigRegistry, cfg: RiskConfig, sha: str = "abc1234") -> bool:
    engine = RiskEngine(cfg)
    return registry.register(
        config_hash=engine.config_hash, git_sha=sha,
        config=cfg.model_dump(mode="json"), audit_header=engine.audit_header(),
    )


class TestRegistration:
    def test_a_fresh_registry_is_empty_without_a_file(
        self, registry: RiskConfigRegistry
    ) -> None:
        assert registry.hashes() == set()
        assert not registry.path.exists()

    def test_registering_stores_the_hash_and_the_full_config(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        assert _register(registry, cfg) is True
        entry = registry.load()["entries"][0]
        assert entry["config_hash"] == risk_config_hash(cfg)
        assert entry["git_sha"] == "abc1234"
        # De inhoud moet mee: een hash bewijst dat er iets veranderde, niet wat.
        assert entry["config"]["gross_cap"] == pytest.approx(cfg.gross_cap)
        assert entry["config"]["sigma_target"] == pytest.approx(cfg.sigma_target)
        assert entry["audit_header"]["config_hash"] == risk_config_hash(cfg)

    def test_registering_the_same_hash_twice_is_a_no_op(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        assert _register(registry, cfg) is True
        assert _register(registry, cfg) is False
        assert len(registry.load()["entries"]) == 1

    def test_a_changed_threshold_registers_as_a_new_entry(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        _register(registry, cfg)
        _register(registry, cfg.model_copy(update={"gross_cap": 1.2}))
        assert len(registry.load()["entries"]) == 2
        assert len(registry.hashes()) == 2

    def test_registration_without_a_hash_or_sha_crashes(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        for bad in ({"config_hash": ""}, {"git_sha": ""}):
            kwargs = {
                "config_hash": "h", "git_sha": "s",
                "config": {}, "audit_header": {}, **bad,
            }
            with pytest.raises(DataContractError):
                registry.register(**kwargs)  # type: ignore[arg-type]

    def test_a_corrupt_register_crashes_rather_than_reading_as_empty(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "risk_config_registry.json"
        path.write_text('{"schema": "x"}', encoding="utf-8")
        with pytest.raises(DataContractError):
            RiskConfigRegistry(path).load()

    def test_the_write_is_atomic_and_leaves_no_temp_files(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        _register(registry, cfg)
        assert not list(registry.path.parent.glob("*.tmp"))
        json.loads(registry.path.read_text(encoding="utf-8"))


class TestItIsDeliberatelyNotTheHypothesisLedger:
    def test_registering_a_risk_config_does_not_inflate_M(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        """De DSR deflateert met M; een limiet opschrijven is geen trial.

        Zou de risicoconfiguratie in `hypothesis_ledger.json` landen, dan zou de
        Deflated Sharpe van de Phase 3-baseline dalen omdat iemand een drempel
        heeft vastgelegd. Deze test bewaakt die scheiding.
        """
        ledger = HypothesisLedger(GOV / "hypothesis_ledger.json")
        before = ledger.total_n_hypotheses()
        _register(registry, cfg)
        assert ledger.total_n_hypotheses() == before

    def test_the_registry_has_no_n_trials_field_at_all(
        self, registry: RiskConfigRegistry, cfg: RiskConfig
    ) -> None:
        _register(registry, cfg)
        assert "n_trials" not in registry.load()["entries"][0]


class TestTheShippedConfigIsRegistered:
    def test_the_live_registry_contains_the_current_config_hash(
        self, cfg: RiskConfig
    ) -> None:
        """Exit-criterium 7: elke limiet gedekt door een `config_hash`."""
        registry = RiskConfigRegistry(GOV / "risk_config_registry.json")
        assert risk_config_hash(cfg) in registry.hashes(), (
            "de geleverde risicoconfiguratie staat niet in het register; "
            "draai apps/run_stress.py"
        )
