"""De soevereine RiskEngine — compositie, auditspoor en puurheid.

Phase 4, stap 6. Exit-criteria 1, 6 en 7 hangen aan deze suite:
elke ingreep geregistreerd, elke drempel uit `conf/risk/`, en een besluit dat
uitsluitend van zijn drie argumenten afhangt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tradebot.risk.contract import ConstraintKind, MarketState, RiskState
from tradebot.risk.engine import KNOWN_CONSTRAINTS, RiskEngine, risk_config_hash
from tradebot.risk.kill_switches import HaltStore
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError

CONF = Path(__file__).resolve().parents[2] / "conf"
TS = pd.Timestamp("2026-01-01T00:00:00Z")


@pytest.fixture
def cfg() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


@pytest.fixture
def engine(cfg: RiskConfig) -> RiskEngine:
    return RiskEngine(cfg)


@pytest.fixture
def symbols(cfg: RiskConfig) -> list[str]:
    return list(cfg.clusters)


def market(symbols: list[str], sigma: float = 0.72, adv: float = 5e8) -> MarketState:
    return MarketState(
        asof_ts=TS,
        sigma_hat={s: sigma for s in symbols},
        adv_usd={s: adv for s in symbols},
    )


def state(equity: float = 1.0, hwm: float = 1.0, day: float = 1.0) -> RiskState:
    return RiskState(equity=equity, high_water_mark=hwm, day_start_equity=day)


class TestConstraintOrderIsConfiguration:
    def test_the_shipped_order_names_every_implemented_limit(self, cfg: RiskConfig) -> None:
        assert set(cfg.constraint_order) == KNOWN_CONSTRAINTS

    def test_the_book_level_caps_come_last(self, cfg: RiskConfig) -> None:
        """Sectie 6: alleen dan geldt hun garantie onvoorwaardelijk."""
        order = list(cfg.constraint_order)
        assert order[-2:] == ["gross_cap", "net_cap"]

    def test_halted_comes_first(self, cfg: RiskConfig) -> None:
        assert list(cfg.constraint_order)[0] == "halted"

    def test_an_unknown_limit_name_crashes(self, cfg: RiskConfig) -> None:
        bad = cfg.model_copy(update={"constraint_order": (*cfg.constraint_order, "typo")})
        with pytest.raises(ConfigContractError):
            RiskEngine(bad)

    def test_an_omitted_limit_crashes_rather_than_being_skipped(
        self, cfg: RiskConfig
    ) -> None:
        """Een limiet weglaten is een beslissing, geen omissie."""
        trimmed = tuple(c for c in cfg.constraint_order if c != "gross_cap")
        with pytest.raises(ConfigContractError):
            RiskEngine(cfg.model_copy(update={"constraint_order": trimmed}))

    def test_an_empty_order_crashes(self, cfg: RiskConfig) -> None:
        with pytest.raises(ConfigContractError):
            RiskEngine(cfg.model_copy(update={"constraint_order": ()}))

    def test_a_duplicate_entry_crashes(self, cfg: RiskConfig) -> None:
        dupe = (*cfg.constraint_order, "gross_cap")
        with pytest.raises(ConfigContractError):
            RiskEngine(cfg.model_copy(update={"constraint_order": dupe}))


class TestTheDecisionIsPure:
    def test_repeated_calls_are_bit_identical(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        args = ({s: 1.0 for s in symbols}, market(symbols), state())
        first = engine.decide(*args)
        second = engine.decide(*args)
        assert dict(first.permitted_exposure) == dict(second.permitted_exposure)
        assert first.as_record() == second.as_record()

    def test_two_engines_on_the_same_config_agree(
        self, cfg: RiskConfig, symbols: list[str]
    ) -> None:
        a = RiskEngine(cfg).decide({s: 0.7 for s in symbols}, market(symbols), state())
        b = RiskEngine(cfg).decide({s: 0.7 for s in symbols}, market(symbols), state())
        assert a.as_record() == b.as_record()

    def test_the_engine_holds_no_state_between_calls(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        """Een tussenliggende crisisbar mag de volgende bar niet beinvloeden."""
        calm = ({s: 0.2 for s in symbols}, market(symbols, sigma=0.05), state())
        before = engine.decide(*calm)
        engine.decide({s: 1.0 for s in symbols}, market(symbols, sigma=5.0),
                      state(equity=0.95))
        after = engine.decide(*calm)
        assert before.as_record() == after.as_record()

    def test_the_input_mappings_are_not_mutated(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        desired = {s: 1.0 for s in symbols}
        engine.decide(desired, market(symbols), state())
        assert desired == {s: 1.0 for s in symbols}


class TestTheAuditTrailIsComplete:
    def test_every_intervention_names_its_config_key(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        d = engine.decide({s: 1.0 for s in symbols}, market(symbols), state())
        assert d.binding_constraints
        for c in d.binding_constraints:
            assert c.config_key.startswith(("risk.", "risk_state.")), c.config_key
            assert c.threshold == c.threshold  # geen NaN
            assert abs(c.exposure_after) <= abs(c.exposure_before) + 1e-12

    def test_the_record_is_json_serialisable(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        d = engine.decide({s: 1.0 for s in symbols}, market(symbols), state())
        json.dumps(d.as_record())

    def test_an_untouched_book_is_explicitly_marked_unconstrained(
        self, engine: RiskEngine
    ) -> None:
        """Deliverable 4: nooit `a_t` doorgeven zonder registratie."""
        d = engine.decide(
            {"BTCUSDT": 0.1, "LINKUSDT": -0.1},
            market(["BTCUSDT", "LINKUSDT"], sigma=0.001),
            state(),
        )
        assert d.unconstrained
        assert d.binding_constraints == ()
        assert dict(d.permitted_exposure) == {"BTCUSDT": 0.1, "LINKUSDT": -0.1}

    def test_a_constrained_book_is_never_marked_unconstrained(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        d = engine.decide({s: 1.0 for s in symbols}, market(symbols), state())
        assert not d.unconstrained
        assert d.gross() < len(symbols)

    def test_the_decision_carries_the_config_hash(
        self, engine: RiskEngine, cfg: RiskConfig, symbols: list[str]
    ) -> None:
        d = engine.decide({s: 1.0 for s in symbols}, market(symbols), state())
        assert d.config_hash == risk_config_hash(cfg)
        assert len(d.config_hash) == 16

    def test_a_changed_threshold_changes_the_hash(self, cfg: RiskConfig) -> None:
        assert risk_config_hash(cfg) != risk_config_hash(
            cfg.model_copy(update={"gross_cap": 1.4})
        )


class TestPostConditionsHold:
    @pytest.mark.parametrize("sigma", [0.01, 0.1, 0.5, 0.72, 3.0])
    @pytest.mark.parametrize("a_t", [1.0, 0.4, -1.0])
    def test_every_configured_limit_holds_on_the_result(
        self, engine: RiskEngine, cfg: RiskConfig, symbols: list[str],
        sigma: float, a_t: float,
    ) -> None:
        d = engine.decide({s: a_t for s in symbols}, market(symbols, sigma=sigma), state())
        assert d.gross() <= cfg.gross_cap + 1e-9
        assert abs(d.net()) <= cfg.net_cap + 1e-9
        for w in d.permitted_exposure.values():
            assert abs(w) <= cfg.max_position_pct + 1e-9

    def test_the_engine_never_grows_exposure(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        desired = {s: (0.3 if i % 2 else -0.3) for i, s in enumerate(symbols)}
        d = engine.decide(desired, market(symbols, sigma=0.001), state())
        for s, w in desired.items():
            assert abs(d.permitted_exposure[s]) <= abs(w) + 1e-9

    def test_out_of_contract_alpha_crashes(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        with pytest.raises(DataContractError):
            engine.decide({symbols[0]: 1.5}, market(symbols), state())

    def test_a_missing_sigma_hat_crashes_the_decision(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        partial = MarketState(
            asof_ts=TS,
            sigma_hat={s: 0.5 for s in symbols[:-1]},
            adv_usd={s: 5e8 for s in symbols},
        )
        with pytest.raises(DataContractError):
            engine.decide({s: 0.5 for s in symbols}, partial, state())


class TestKillSwitchesRunFirst:
    def test_a_halted_book_gets_zero_exposure(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        halted = RiskState(
            equity=1.0, high_water_mark=1.0, day_start_equity=1.0,
            halted=True, halt_reason="eerder gehalt",
        )
        d = engine.decide({s: 1.0 for s in symbols}, market(symbols), halted)
        assert d.gross() == pytest.approx(0.0)
        assert d.bound_kinds[0] is ConstraintKind.HALTED

    def test_the_daily_loss_governor_halts_and_persists(
        self, cfg: RiskConfig, symbols: list[str], tmp_path: Path
    ) -> None:
        store = HaltStore(tmp_path / "halt.json")
        engine = RiskEngine(cfg, halt_store=store)
        d = engine.decide(
            {s: 1.0 for s in symbols}, market(symbols),
            state(equity=0.96, hwm=1.0, day=1.0),
        )
        assert d.gross() == pytest.approx(0.0)
        assert d.risk_state_out.halted
        assert store.is_halted()
        assert ConstraintKind.DAILY_LOSS_GOVERNOR in d.bound_kinds

    def test_the_drawdown_breaker_de_grosses_before_it_halts(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        d = engine.decide(
            {s: 1.0 for s in symbols}, market(symbols, sigma=0.001),
            state(equity=0.95, hwm=1.0, day=0.96),
        )
        assert ConstraintKind.DRAWDOWN_BREAKER in d.bound_kinds
        assert not d.risk_state_out.halted
        assert d.gross() > 0.0

    def test_hydrate_restores_a_halt_after_a_restart(
        self, cfg: RiskConfig, symbols: list[str], tmp_path: Path
    ) -> None:
        store = HaltStore(tmp_path / "halt.json")
        RiskEngine(cfg, halt_store=store).decide(
            {s: 1.0 for s in symbols}, market(symbols), state(equity=0.96),
        )
        # Nieuw proces: verse state, zelfde store.
        restarted = RiskEngine(cfg, halt_store=HaltStore(tmp_path / "halt.json"))
        fresh = restarted.hydrate(state())
        assert fresh.halted
        d = restarted.decide({s: 1.0 for s in symbols}, market(symbols), fresh)
        assert d.gross() == pytest.approx(0.0)

    def test_the_store_is_a_side_channel_and_does_not_change_the_decision(
        self, cfg: RiskConfig, symbols: list[str], tmp_path: Path
    ) -> None:
        """Puurheid: dezelfde inputs, dezelfde output, met of zonder store."""
        args = ({s: 1.0 for s in symbols}, market(symbols), state(equity=0.96))
        without = RiskEngine(cfg).decide(*args)
        with_store = RiskEngine(
            cfg, halt_store=HaltStore(tmp_path / "halt.json")
        ).decide(*args)
        assert dict(without.permitted_exposure) == dict(with_store.permitted_exposure)
        assert [c.as_record() for c in without.binding_constraints] == [
            c.as_record() for c in with_store.binding_constraints
        ]


class TestNoAlphaKnowledgeInTheDecisionPath:
    def test_the_engine_exposes_no_alpha_entry_point(self, engine: RiskEngine) -> None:
        """E3/E4/E8: Kelly, meta-labels en de HMM zijn niet meegeport."""
        api = {n for n in dir(engine) if not n.startswith("__")}
        forbidden = {
            "set_hmm_detector", "update_cvar_estimates", "kelly_fraction",
            "expected_alpha", "meta_label", "set_alpha", "conviction",
        }
        assert not (api & forbidden), sorted(api & forbidden)

    def test_the_module_imports_no_alpha_bearing_layer(self) -> None:
        """Op de AST, niet op de tekst — een docstring die HMM NOEMT is prima.

        Wat niet mag, is een import. `risk/kelly.py` (expected_alpha),
        `alpha/factor_alpha.py` en `risk/hmm_regime.py` staan alle drie buiten
        het besluitpad; deze test houdt dat zo.
        """
        import ast

        src = (
            Path(__file__).resolve().parents[2]
            / "src" / "tradebot" / "risk" / "engine.py"
        ).read_text(encoding="utf-8")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)

        forbidden = {"kelly", "factor_alpha", "hmm_regime", "alpha", "portfolio"}
        for module in imported:
            tail = {p for p in module.split(".") if p}
            assert not (tail & forbidden), f"engine.py importeert uit {module}"

    def test_the_audit_header_reports_every_binding_threshold(
        self, engine: RiskEngine, cfg: RiskConfig
    ) -> None:
        header = engine.audit_header(TS)
        assert header["config_hash"] == risk_config_hash(cfg)
        assert header["sigma_target"] == pytest.approx(cfg.sigma_target)
        assert header["gross_cap"] == pytest.approx(cfg.gross_cap)
        assert header["constraint_order"] == list(cfg.constraint_order)
        json.dumps(header)
