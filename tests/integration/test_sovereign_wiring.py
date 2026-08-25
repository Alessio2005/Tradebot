"""De harde poort: is de soevereine risicolaag daadwerkelijk overal bedraad?

Fase-opdracht §16 somt negen eigenschappen op die moeten worden AANGETOOND, niet
afgesproken. Elke test hieronder draagt het nummer waar hij bij hoort.

De reden dat deze suite bestaat naast `tests/unit/test_risk_engine.py`: die
bewijst dat de engine correct beslist. Deze bewijst dat er geen enkele manier is
om hem te omzeilen. Dat zijn verschillende eigenschappen, en Phase 4 had alleen
de eerste - `reports/phase5_sovereign_wiring_audit.md` telde elf productiepaden
die een order bereikten zonder de laag ooit aan te roepen.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

from tradebot.backtest.engine import EventDrivenEngine
from tradebot.execution.order_router import Order, OrderRouter
from tradebot.risk.contract import MarketState, RiskDecision, RiskState
from tradebot.risk.engine import RiskEngine
from tradebot.utils.failfast import ConfigContractError, DataContractError

from ._engine_fixtures import (
    INITIAL_EQUITY,
    impact_params,
    make_engine,
    make_replay,
    make_router,
    risk_config,
    spread_model,
    venue_spec,
)

pytestmark = pytest.mark.integration

SRC = Path(__file__).resolve().parents[2] / "src" / "tradebot"


@pytest.fixture(scope="module")
def replay():
    return make_replay(n_bars=50)


@pytest.fixture(scope="module")
def result(replay):
    return make_engine().run(replay.slices, replay.exposures)


# =========================================================================== #
# §16.1 — portfolio target generation vereist sovereign approval
# §16.2 — backtest order generation vereist sovereign approval
# =========================================================================== #
class TestOrdersRequireApproval:
    def test_build_orders_has_no_overload_without_a_decision(self) -> None:
        """Er is geen handtekening waarmee je een order maakt zonder besluit."""
        params = list(inspect.signature(OrderRouter.build_orders).parameters)
        assert params[1] == "decision", (
            "RiskDecision is niet het eerste inhoudelijke argument; er zou een "
            "pad kunnen bestaan dat hem weglaat"
        )
        assert inspect.signature(
            OrderRouter.build_orders).parameters["decision"].default \
            is inspect.Parameter.empty, "decision heeft een default gekregen"

    def test_passing_something_that_is_not_a_decision_fails(self) -> None:
        router = make_router(RiskEngine(risk_config()))
        with pytest.raises(ConfigContractError):
            router.build_orders(
                {"BTCUSDT": 0.1},  # type: ignore[arg-type]
                equity=INITIAL_EQUITY, marks={"BTCUSDT": 100.0},
                current_qty={}, ts_decision=pd.Timestamp("2024-01-01", tz="UTC"),
                ts_earliest_fill=pd.Timestamp("2024-01-02", tz="UTC"),
            )

    def test_the_engine_produced_a_decision_for_every_order_bar(
        self, result
    ) -> None:
        assert len(result.decisions) > 0
        assert all(isinstance(d, RiskDecision) for d in result.decisions)

    def test_every_order_carries_the_policy_hash(self, result) -> None:
        assert all(r.order.risk_config_hash == result.risk_policy_hash
                   for r in result.reports)


# =========================================================================== #
# §16.3 — lokale risk constraints beslissen niet zelfstandig
# §16.4 — alle risk-relevant limits komen uit de sovereign policy
# =========================================================================== #
class TestNoSecondRiskRegime:
    def test_the_participation_cap_comes_from_the_sovereign_policy(self) -> None:
        """Niet uit de venue: hoeveel van het volume je mag zijn, is risico."""
        source = (SRC / "execution" / "order_router.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        venue_fields = {
            t.target.id  # type: ignore[union-attr]
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "VenueSpec"
            for t in node.body if isinstance(t, ast.AnnAssign)
            and isinstance(t.target, ast.Name)
        }
        assert "max_participation" not in venue_fields
        assert "adv_participation_cap" not in venue_fields

    def test_the_engine_takes_the_participation_cap_from_risk_config(self) -> None:
        cfg = risk_config()
        engine = make_engine(cfg)
        assert engine._participation_cap == pytest.approx(  # noqa: SLF001
            cfg.adv_participation_cap)

    def test_the_venue_only_carries_mechanics(self) -> None:
        """Alles in VenueSpec beschrijft de exchange, niet ons risicobudget."""
        venue = venue_spec()
        mechanical = {"maker_fee_bps", "taker_fee_bps", "funding_cap_abs",
                      "min_notional", "latency_bars"}
        assert set(type(venue).__dataclass_fields__) == mechanical


# =========================================================================== #
# §16.5 — policy version/hash wordt doorgegeven
# §16.8 — een oude lokale limiet kan de sovereign policy niet overrulen
# =========================================================================== #
class TestPolicyHashPropagation:
    def test_a_router_built_on_another_policy_refuses_the_decision(self) -> None:
        """§17: 'policy hash mismatch -> FAIL'."""
        cfg = risk_config()
        engine = RiskEngine(cfg)
        other = RiskEngine(cfg.model_copy(update={"gross_cap": 1.25}))
        assert engine.config_hash != other.config_hash

        router = make_router(other)
        decision = engine.decide(
            {"BTCUSDT": 1.0},
            MarketState(asof_ts=pd.Timestamp("2024-01-01", tz="UTC"),
                        sigma_hat={"BTCUSDT": 0.6}, adv_usd={"BTCUSDT": 1e9},
                        cluster={"BTCUSDT": "crypto_perp"}),
            RiskState(equity=INITIAL_EQUITY, high_water_mark=INITIAL_EQUITY,
                      day_start_equity=INITIAL_EQUITY),
        )
        with pytest.raises(ConfigContractError):
            router.build_orders(
                decision, equity=INITIAL_EQUITY, marks={"BTCUSDT": 100.0},
                current_qty={}, ts_decision=pd.Timestamp("2024-01-01", tz="UTC"),
                ts_earliest_fill=pd.Timestamp("2024-01-02", tz="UTC"),
            )

    def test_the_engine_refuses_a_mismatched_router_at_construction(self) -> None:
        cfg = risk_config()
        engine = RiskEngine(cfg)
        other_router = make_router(
            RiskEngine(cfg.model_copy(update={"net_cap": 0.5})))
        with pytest.raises(ConfigContractError):
            EventDrivenEngine(risk_engine=engine, router=other_router,
                              initial_equity=INITIAL_EQUITY)

    def test_the_result_carries_the_policy_identifier(self, result) -> None:
        assert result.risk_policy_hash
        assert result.as_record()["risk_policy_hash"] == result.risk_policy_hash


# =========================================================================== #
# §16.6 — risk decisions zijn deterministisch reproduceerbaar
# =========================================================================== #
class TestDeterminism:
    def test_two_identical_runs_are_bit_identical(self, replay) -> None:
        a = make_engine().run(replay.slices, replay.exposures)
        b = make_engine().run(replay.slices, replay.exposures)
        assert a.equity_curve.equals(b.equity_curve)
        assert [d.as_record() for d in a.decisions] == \
               [d.as_record() for d in b.decisions]
        assert [r.as_record() for r in a.reports] == \
               [r.as_record() for r in b.reports]


# =========================================================================== #
# §16.7 — een gewijzigde sovereign limiet werkt door in de backtest output
# =========================================================================== #
class TestALimitChangeReachesTheOutput:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            # Waarden die op DEZE opstelling daadwerkelijk binden. Dat is geen
            # detail: de vol-targeting schaalt het boek al terug naar een gross
            # van ~0,05, en een `gross_cap` van 0,5 of een `max_position_pct`
            # van 0,05 raakt dat nooit. Een test die met zulke waarden groen
            # wordt, bewijst niets - en een test die er ROOD mee wordt, meet de
            # verkeerde eigenschap. De limiet moet strenger zijn dan wat de
            # vol-target al oplegt om iets te kunnen aantonen.
            ("sigma_target", 0.04),
            ("max_leverage", 0.01),
            ("gross_cap", 0.02),
            ("max_position_pct", 0.005),
        ],
    )
    def test_tightening_a_limit_changes_the_equity_curve(
        self, replay, field: str, value: float
    ) -> None:
        """Zonder deze test kan de laag bedraad LIJKEN en niets doen.

        Dit is de eigenschap die het verschil maakt tussen 'de engine roept
        decide() aan' en 'de engine gehoorzaamt decide()'.
        """
        base = make_engine().run(replay.slices, replay.exposures)
        tight_cfg = risk_config().model_copy(update={field: value})
        tight_engine = RiskEngine(tight_cfg)
        tight = EventDrivenEngine(
            risk_engine=tight_engine, router=make_router(tight_engine),
            initial_equity=INITIAL_EQUITY,
        ).run(replay.slices, replay.exposures)

        assert tight.risk_policy_hash != base.risk_policy_hash
        assert not tight.equity_curve.equals(base.equity_curve), (
            f"een strengere {field} verandert het resultaat niet; de limiet "
            f"bereikt de output niet"
        )
        base_gross = sum(abs(v) for v in base.decisions[-1].permitted_exposure.values())
        tight_gross = sum(
            abs(v) for v in tight.decisions[-1].permitted_exposure.values())
        assert tight_gross <= base_gross + 1e-12

    def test_a_looser_limit_lets_more_exposure_through(self, replay) -> None:
        loose_cfg = risk_config().model_copy(update={"sigma_target": 0.16})
        loose_engine = RiskEngine(loose_cfg)
        loose = EventDrivenEngine(
            risk_engine=loose_engine, router=make_router(loose_engine),
            initial_equity=INITIAL_EQUITY,
        ).run(replay.slices, replay.exposures)
        base = make_engine().run(replay.slices, replay.exposures)
        assert sum(abs(v) for v in loose.decisions[-1].permitted_exposure.values()) \
            > sum(abs(v) for v in base.decisions[-1].permitted_exposure.values())


# =========================================================================== #
# §16.9 — een ontbrekende sovereign policy veroorzaakt een harde failure
# =========================================================================== #
class TestMissingPolicyIsFatal:
    def test_the_engine_cannot_be_built_without_a_risk_engine(self) -> None:
        with pytest.raises(ConfigContractError):
            EventDrivenEngine(risk_engine=None, router=make_router(  # type: ignore[arg-type]
                RiskEngine(risk_config())), initial_equity=INITIAL_EQUITY)

    def test_the_router_cannot_be_built_without_a_policy_hash(self) -> None:
        with pytest.raises(ConfigContractError):
            OrderRouter(venue=venue_spec(), spread=spread_model(),
                        impact=impact_params(), risk_config_hash="")

    def test_the_router_cannot_be_built_without_an_impact_parameter(self) -> None:
        with pytest.raises(ConfigContractError):
            OrderRouter(venue=venue_spec(), spread=spread_model(),
                        impact=None, risk_config_hash="abc")  # type: ignore[arg-type]

    def test_an_engine_without_a_router_is_a_hard_failure(self) -> None:
        with pytest.raises(ConfigContractError):
            EventDrivenEngine(risk_engine=RiskEngine(risk_config()),
                              router=None,  # type: ignore[arg-type]
                              initial_equity=INITIAL_EQUITY)


# =========================================================================== #
# Repository-level statische audit — fase-opdracht §16, laatste alinea
# =========================================================================== #
#: Namen die een LOKALE risicolimiet aanduiden. Een module in de authoritative
#: keten die er een definieert, heeft een tweede risicoregime.
_LOCAL_LIMIT_NAMES = frozenset({
    "max_leverage", "max_gross_leverage", "max_position_pct", "max_weight",
    "max_concentration", "max_cluster_concentration", "gross_cap", "net_cap",
    "max_drawdown_pct", "daily_loss_limit", "adv_participation_cap",
    "max_portfolio_leverage", "max_notional_per_symbol", "max_gross_notional",
    "sigma_target", "target_vol", "vol_target", "per_asset_cap",
})

#: De modules die na Phase 5 op het authoritative pad liggen. Zij mogen geen
#: eigen limiet DEFINIEREN; hem uit `RiskConfig` LEZEN mag wel.
_AUTHORITATIVE_PATH = (
    "backtest/engine.py",
    "backtest/accounting.py",
    "execution/order_router.py",
    "execution/context.py",
    "execution/impact_model.py",
)


def _assigned_names(path: Path) -> set[str]:
    """Namen die in deze module een waarde krijgen toegewezen."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.arg) and node.arg in _LOCAL_LIMIT_NAMES:
            names.add(node.arg)
    return names


@pytest.mark.parametrize("relpath", _AUTHORITATIVE_PATH)
def test_no_module_on_the_authoritative_path_defines_its_own_limit(
    relpath: str,
) -> None:
    """Statische audit tegen bekende lokale limiet-patronen.

    Deze test is de reden dat een toekomstige wijziging de laag niet stilletjes
    opnieuw kan omzeilen. `reports/phase5_sovereign_wiring_audit.md` §4 laat
    zien dat vier van de zeven lokale limieten RUIMER waren dan de soevereine
    policy; zoiets ontstaat niet met opzet maar door toevoeging.
    """
    offenders = _assigned_names(SRC / relpath) & _LOCAL_LIMIT_NAMES
    assert not offenders, (
        f"{relpath} definieert risicolimieten die uit de soevereine policy "
        f"horen te komen: {sorted(offenders)}"
    )


def test_the_static_audit_actually_detects_a_violation(tmp_path: Path) -> None:
    """De audit is echt en niet decoratief."""
    bad = tmp_path / "bad.py"
    bad.write_text("max_leverage = 2.0\ngross_cap = 4.0\n", encoding="utf-8")
    assert _assigned_names(bad) & _LOCAL_LIMIT_NAMES == {"max_leverage",
                                                        "gross_cap"}


def test_the_authoritative_path_list_is_not_empty_and_exists() -> None:
    """Een lijst die per ongeluk leeg raakt, maakt de audit stilzwijgend groen."""
    assert len(_AUTHORITATIVE_PATH) >= 5
    for relpath in _AUTHORITATIVE_PATH:
        assert (SRC / relpath).is_file(), f"{relpath} bestaat niet meer"


# =========================================================================== #
# Overige negatieve tests uit §17
# =========================================================================== #
class TestNegativeCases:
    def test_an_order_that_could_fill_on_its_own_decision_bar_is_rejected(
        self,
    ) -> None:
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        with pytest.raises(DataContractError):
            Order(symbol="BTCUSDT", ts_decision=ts, ts_earliest_fill=ts,
                  target_qty_delta=1.0, risk_config_hash="abc")

    def test_an_order_without_a_policy_hash_is_rejected(self) -> None:
        with pytest.raises(DataContractError):
            Order(symbol="BTCUSDT",
                  ts_decision=pd.Timestamp("2024-01-01", tz="UTC"),
                  ts_earliest_fill=pd.Timestamp("2024-01-02", tz="UTC"),
                  target_qty_delta=1.0, risk_config_hash="")

    def test_a_missing_fee_schedule_is_a_hard_failure(self) -> None:
        """§17: 'ontbrekende fee schedule -> FAIL'."""
        from tradebot.execution.order_router import VenueSpec

        with pytest.raises(ConfigContractError):
            VenueSpec(maker_fee_bps=float("nan"), taker_fee_bps=5.5,
                      funding_cap_abs=0.02, min_notional=10.0, latency_bars=1)

    def test_zero_latency_is_rejected_as_lookahead(self) -> None:
        from tradebot.execution.order_router import VenueSpec

        with pytest.raises(ConfigContractError):
            VenueSpec(maker_fee_bps=2.0, taker_fee_bps=5.5, funding_cap_abs=0.02,
                      min_notional=10.0, latency_bars=0)

    def test_a_zero_spread_is_rejected_as_a_mid_fill(self) -> None:
        """§12: 'een fill tegen mid zonder bewijs is niet toegestaan'."""
        from tradebot.execution.order_router import SpreadModel, SpreadStatus

        with pytest.raises(ConfigContractError):
            SpreadModel(half_spread_bps=0.0, status=SpreadStatus.SPREAD_ASSUMED,
                        source="test")

    def test_a_run_shorter_than_two_bars_is_a_hard_failure(self) -> None:
        replay = make_replay(n_bars=2)
        engine = make_engine()
        with pytest.raises(DataContractError):
            engine.run(replay.slices[:1], replay.exposures)
