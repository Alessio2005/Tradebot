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
from typing import ClassVar

import pytest
import yaml

from tradebot.schemas.config import (
    AlphaConfig,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    FeatureConfig,
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
    ("features/default.yaml", FeatureConfig),
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


class TestRiskConfigCarriesNoAlphaParameters:
    """Exit-criterium 2 (Phase 4): `risk/` bevat nul alpha-parameters.

    Vóór Phase 4 stond `min_signal_confidence` in RiskConfig en werd hij ook
    daadwerkelijk gezet in conf/risk/default.yaml. Een risicolaag die een
    signaaldrempel kent, kan positiegrootte koppelen aan modelovertuiging -
    precies de bypass die audit sectie 14 uitsluit.
    """

    FORBIDDEN: ClassVar[set[str]] = {
        "min_signal_confidence", "max_funding_cost_bps_day",
        "expected_alpha", "alpha", "edge", "mu", "signal_threshold",
        "confidence", "conviction", "model", "strategy", "sharpe",
    }

    def test_schema_has_no_alpha_field(self) -> None:
        leaked = set(RiskConfig.model_fields) & self.FORBIDDEN
        assert not leaked, f"alpha-parameter in RiskConfig: {sorted(leaked)}"

    def test_conf_risk_yaml_has_no_alpha_key(self) -> None:
        import yaml

        raw = yaml.safe_load((CONF / "risk/default.yaml").read_text(encoding="utf-8"))
        leaked = set(raw["risk"]) & self.FORBIDDEN
        assert not leaked, f"alpha-parameter in conf/risk/default.yaml: {sorted(leaked)}"


class TestRiskBudgetIsComplete:
    """Deliverable 11: elke Phase 4-limiet komt uit conf/risk/, niet uit code."""

    REQUIRED = (
        "sigma_target", "max_leverage", "max_concentration", "gross_cap",
        "net_cap", "drawdown_breaker_levels", "daily_loss_limit",
        "adv_participation_cap", "constraint_order",
    )

    @pytest.mark.parametrize("key", REQUIRED)
    def test_key_is_present_and_set(self, key: str) -> None:
        import yaml

        raw = yaml.safe_load((CONF / "risk/default.yaml").read_text(encoding="utf-8"))
        assert key in raw["risk"], f"{key} ontbreekt in conf/risk/default.yaml"

    def test_the_own_capital_mandate_values_are_configured(self) -> None:
        """`docs/RISK_MANDATE.md` §2 — de eigen-kapitaalwaarden staan in conf/.

        VOORGESCHIEDENIS. Deze test heette
        `test_strictest_of_the_conflicting_values_was_chosen` en eiste
        `gross_cap == 1.5` en `max_drawdown_pct == 0.08`: de fase-4-regel
        "strengste wint" (entanglement map §5) bovenop een propfirm-contract dat
        de drawdownlijn dicteerde. Er wordt niet meer met propfirms gewerkt, dus
        die grond is weg en de test toetste een mandaat dat niet meer bestaat.

        Zij toetst nu het NIEUWE mandaat, met dezelfde strengheid. Wijzigt
        iemand een van deze getallen zonder `docs/RISK_MANDATE.md` bij te
        werken, dan wordt dit rood — dat is het punt van deze test.
        """
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        assert cfg.gross_cap == pytest.approx(4.0)
        assert cfg.max_leverage == pytest.approx(4.0)
        assert cfg.sigma_target == pytest.approx(0.20)
        assert cfg.net_cap == pytest.approx(2.0)
        assert cfg.max_drawdown_pct == pytest.approx(0.25)
        assert cfg.daily_loss_limit == pytest.approx(0.10)

    def test_the_market_fact_limits_were_not_widened(self) -> None:
        """RISK_MANDATE §1 soort C: liquiditeit is geen risicobereidheid.

        `adv_participation_cap` verruimen is niet moediger worden; het is de
        backtest laten rekenen met fills die niet bestaan. Omdat
        `conf/execution/impact.yaml` op IMPACT_UNCALIBRATED staat, vervalst het
        de kostenkant van ELKE meting — inclusief de 16,3-57,8 bps break-even
        waaraan elke kandidaat wordt getoetst. Deze cap mag niet meeliften op
        een mandaatwijziging die over eigen kapitaal gaat.
        """
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        assert cfg.adv_participation_cap == pytest.approx(0.01)

    def test_the_halt_still_trips_on_the_gap_down_scenario(self) -> None:
        """RISK_MANDATE §2.1: 0.30 is het plafond van `max_drawdown_pct`.

        `risk/stress_test.py` schokt de equity met GAP_DOWN_FRACTION en
        `tests/integration/test_risk_overrules_alpha.py` eist dat die schok de
        halt tript. Zet iemand de ruinelijn op of boven die schok, dan wordt het
        S3-scenario non-bindend en toetst de stress-suite niets meer — zonder
        dat er ergens een test rood wordt. Deze test is die test.
        """
        from tradebot.risk.stress_test import GAP_DOWN_FRACTION

        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        assert cfg.max_drawdown_pct < GAP_DOWN_FRACTION, (
            f"max_drawdown_pct {cfg.max_drawdown_pct} ligt op of boven "
            f"GAP_DOWN_FRACTION {GAP_DOWN_FRACTION}; het S3-gap-down-scenario "
            "tript de halt dan niet meer en de stress-suite toetst niets"
        )

    def test_one_day_cannot_exceed_the_ruin_line(self) -> None:
        """RISK_MANDATE §7 regel 5. Een daglimiet boven de harde halt is dood.

        De Daily Loss Governor staat vóór de drawdown-breaker in
        `constraint_order`. Stond hij ruimer dan de halt, dan zou de halt altijd
        eerst binden en zou de governor nooit iets doen.
        """
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        assert cfg.daily_loss_limit < cfg.max_drawdown_pct

    def test_the_gross_cap_still_binds_on_this_universe(self) -> None:
        """RISK_MANDATE §3. Een limiet die niet bindt is documentatie.

        `max_position_pct` clampt per symbool; op zes symbolen is de hoogst
        haalbare gross dus `6 x max_position_pct`. Ligt die onder `gross_cap`,
        dan kan de gross-cap per constructie nooit meer binden.
        """
        import yaml

        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        raw = yaml.safe_load((CONF / "data/default.yaml").read_text(encoding="utf-8"))
        n_symbols = len(raw["data"]["symbols"])
        reachable_gross = n_symbols * cfg.max_position_pct
        assert reachable_gross > cfg.gross_cap, (
            f"{n_symbols} symbolen x max_position_pct {cfg.max_position_pct} = "
            f"{reachable_gross} <= gross_cap {cfg.gross_cap}; de gross-cap kan "
            "niet meer binden"
        )

    def test_a_position_may_live_long_enough_to_earn_back_its_costs(self) -> None:
        """RISK_MANDATE §4.1 — de limiet die kandidaat B onmeetbaar maakte.

        Funding carry op ETH is ~1,95 bps/dag (`reports/diag_funding_short_edge.csv`,
        all-bucket) tegen 13,0 bps VASTE kosten per round trip
        (`conf/execution/fees.yaml`). Een positie die eerder gedwongen sluit dan
        het break-evenpunt, kan die kosten per constructie niet terugverdienen —
        en dan meet een negatief resultaat de config en niet de markt.
        """
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        fixed_cost_bps = 13.0
        eth_carry_bps_per_day = 1.95
        break_even_days = fixed_cost_bps / eth_carry_bps_per_day  # ~6.7
        assert cfg.max_position_age_h / 24.0 > break_even_days, (
            f"max_position_age_h {cfg.max_position_age_h}h sluit de positie na "
            f"{cfg.max_position_age_h / 24.0:.1f} dagen, terwijl de carry "
            f"{break_even_days:.1f} dagen nodig heeft om alleen de vaste "
            f"{fixed_cost_bps} bps terug te verdienen"
        )

    def test_degrossing_tiers_stay_below_the_hard_halt(self) -> None:
        """Een trap boven de eindlimiet is dode code."""
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        assert cfg.drawdown_breaker_levels, "geen getrapte de-grossing geconfigureerd"
        assert all(t.drawdown < cfg.max_drawdown_pct for t in cfg.drawdown_breaker_levels)

    def test_monotone_tiers_are_enforced(self) -> None:
        with pytest.raises(ConfigContractError):
            validate_mapping(RiskConfig, {"drawdown_breaker_levels": [
                {"drawdown": 0.06, "gross_multiplier": 0.25},
                {"drawdown": 0.04, "gross_multiplier": 0.50},
            ]})

    def test_a_deeper_drawdown_may_not_loosen_the_multiplier(self) -> None:
        with pytest.raises(ConfigContractError):
            validate_mapping(RiskConfig, {"drawdown_breaker_levels": [
                {"drawdown": 0.04, "gross_multiplier": 0.25},
                {"drawdown": 0.06, "gross_multiplier": 0.50},
            ]})


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
