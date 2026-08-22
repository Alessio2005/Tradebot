"""Phase 0, deliverable 10 - configuratiecontracten.

Bewijst dat:
  * elke YAML in `conf/` laadt onder zijn Pydantic-schema;
  * een geinjecteerde ONBEKENDE sleutel een `ConfigContractError` triggert;
  * config na laden onveranderlijk is;
  * de bindende constanten uit het auditdocument daadwerkelijk in `conf/` staan
    en niet als literal in `src/`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tradebot.schemas.config import (
    AlphaConfig,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    RiskConfig,
    StrictModel,
    ValidationConfig,
    VolatilityConfig,
    load_config,
    validate_mapping,
)
from tradebot.utils.failfast import ConfigContractError

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"

# Elke YAML die een gevalideerd domein beschrijft, met zijn schema.
DOMAIN_FILES: list[tuple[str, type[StrictModel]]] = [
    ("data/default.yaml", DataConfig),
    ("model/volatility.yaml", VolatilityConfig),
    ("model/alpha.yaml", AlphaConfig),
    ("risk/default.yaml", RiskConfig),
    ("execution/fees.yaml", ExecutionConfig),
    ("backtest/default.yaml", BacktestConfig),
    ("validation/default.yaml", ValidationConfig),
]

ALL_MODELS = [m for _, m in DOMAIN_FILES]


class TestEveryConfigLoads:
    @pytest.mark.parametrize("rel,model", DOMAIN_FILES, ids=[r for r, _ in DOMAIN_FILES])
    def test_loads_under_its_schema(self, rel: str, model: type[StrictModel]) -> None:
        cfg = load_config(CONF / rel, model)
        assert isinstance(cfg, model)

    @pytest.mark.parametrize("rel,model", DOMAIN_FILES, ids=[r for r, _ in DOMAIN_FILES])
    def test_file_exists(self, rel: str, model: type[StrictModel]) -> None:
        assert (CONF / rel).is_file(), f"{rel} ontbreekt in conf/"


class TestUnknownKeyIsRejected:
    """Een typo in conf/ mag nooit stilzwijgend een default laten winnen."""

    @pytest.mark.parametrize("rel,model", DOMAIN_FILES, ids=[r for r, _ in DOMAIN_FILES])
    def test_injected_unknown_key_raises(
        self, rel: str, model: type[StrictModel], tmp_path: Path
    ) -> None:
        raw = yaml.safe_load((CONF / rel).read_text(encoding="utf-8"))
        payload = {k: v for k, v in raw.items() if k not in ("defaults", "_self_")}
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            key = next(iter(payload))
            payload[key]["definitely_not_a_real_setting"] = 123
        else:
            payload["definitely_not_a_real_setting"] = 123

        bad = tmp_path / "bad.yaml"
        bad.write_text(yaml.safe_dump(payload), encoding="utf-8")

        with pytest.raises(ConfigContractError, match="definitely_not_a_real_setting"):
            load_config(bad, model)

    @pytest.mark.parametrize("model", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
    def test_extra_forbid_is_set(self, model: type[StrictModel]) -> None:
        assert model.model_config.get("extra") == "forbid"

    @pytest.mark.parametrize("model", ALL_MODELS, ids=[m.__name__ for m in ALL_MODELS])
    def test_frozen_is_set(self, model: type[StrictModel]) -> None:
        assert model.model_config.get("frozen") is True


class TestImmutability:
    def test_loaded_config_cannot_be_mutated(self) -> None:
        from pydantic import ValidationError

        cfg = load_config(CONF / "model/volatility.yaml", VolatilityConfig)
        with pytest.raises(ValidationError, match="frozen"):
            cfg.ewma_lambda = 0.5  # type: ignore[misc]


class TestErrorTranslation:
    def test_missing_file_raises_config_contract_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigContractError, match="bestaat niet"):
            load_config(tmp_path / "nope.yaml", RiskConfig)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ConfigContractError, match="leeg"):
            load_config(p, RiskConfig)

    def test_out_of_range_value_raises(self) -> None:
        with pytest.raises(ConfigContractError):
            validate_mapping(VolatilityConfig, {"ewma_lambda": 1.5})

    def test_pydantic_error_is_translated_not_leaked(self) -> None:
        """ValidationError mag niet naar de aanroeper lekken."""
        from pydantic import ValidationError

        try:
            validate_mapping(RiskConfig, {"max_position_pct": 99.0})
        except ConfigContractError:
            pass
        except ValidationError:  # pragma: no cover
            pytest.fail("ruwe pydantic.ValidationError lekte naar de aanroeper")


class TestBindingConstantsLiveInConf:
    """De constanten die het auditdocument bindend maakt, staan in conf/."""

    def test_ewma_lambda_is_094(self) -> None:
        """Audit sectie 9.1 / 22: RiskMetrics lambda = 0.94 is de Level-1 baseline."""
        cfg = load_config(CONF / "model/volatility.yaml", VolatilityConfig)
        assert cfg.ewma_lambda == pytest.approx(0.94)

    def test_estimator_is_level_one_only(self) -> None:
        """GARCH is Level 2 en is in Phase 3 verboden."""
        cfg = load_config(CONF / "model/volatility.yaml", VolatilityConfig)
        assert cfg.estimator == "ewma"

    def test_walk_forward_is_the_only_scheme(self) -> None:
        """Audit sectie 17.1: Purged Walk-Forward is de enige toegestane CV."""
        cfg = load_config(CONF / "validation/default.yaml", ValidationConfig)
        assert cfg.scheme == "purged_walk_forward"

    def test_cpcv_and_pbo_have_no_gate_authority(self) -> None:
        """OPTIONAL-diagnostiek mag nooit een gate zijn."""
        cfg = load_config(CONF / "validation/default.yaml", ValidationConfig)
        assert cfg.cpcv_is_gate is False
        assert cfg.pbo_is_gate is False

    def test_cpcv_gate_cannot_be_switched_on(self) -> None:
        """Het schema pint de vlag; conf kan hem niet omzetten."""
        with pytest.raises(ConfigContractError):
            validate_mapping(ValidationConfig, {"cpcv_is_gate": True})

    def test_embargo_must_cover_label_horizon(self) -> None:
        """embargo < horizon laat overlappende labels van test naar train lekken."""
        with pytest.raises(ConfigContractError, match="embargo"):
            validate_mapping(
                ValidationConfig, {"label_horizon_bars": 10, "embargo_bars": 2}
            )

    def test_vectorized_backtest_is_not_promotion_evidence(self) -> None:
        """Audit sectie 16.1."""
        cfg = load_config(CONF / "backtest/default.yaml", BacktestConfig)
        if cfg.engine == "vectorized":
            assert cfg.counts_as_promotion_evidence is False

    def test_allocator_is_baseline_or_production_default(self) -> None:
        """HRP/Markowitz/BL zijn in Phase 3 niet toegestaan (sectie 13.1)."""
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        assert cfg.allocator in ("equal_weight", "risk_parity")

    def test_cost_assumption_is_flagged_provisional(self) -> None:
        """De eta-kalibratie volgt pas in Phase 5; dat moet zichtbaar zijn."""
        cfg = load_config(CONF / "execution/fees.yaml", ExecutionConfig)
        assert cfg.cost_assumption_is_provisional is True

    def test_funding_interval_matches_venue(self) -> None:
        """Bybit perp funding settelt elke 8 uur."""
        cfg = load_config(CONF / "data/default.yaml", DataConfig)
        assert cfg.venue == "bybit"
        assert cfg.funding_interval_hours == 8

    def test_gap_policy_never_interpolates(self) -> None:
        cfg = load_config(CONF / "data/default.yaml", DataConfig)
        assert cfg.gap_policy in ("reject", "register")


class TestUniverseConsistency:
    def test_symbols_match_conf_symbols_directory(self) -> None:
        cfg = load_config(CONF / "data/default.yaml", DataConfig)
        on_disk = {p.stem for p in (CONF / "symbols").glob("*.yaml")}
        assert set(cfg.symbols) == on_disk, (
            "conf/data/default.yaml::symbols wijkt af van conf/symbols/*.yaml"
        )
