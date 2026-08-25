"""`portfolio/constraints.py` is geen risicoautoriteit meer.

Fase-opdracht §4.2 en exit-criterium 11. De opdracht laat drie uitwegen en geen
vierde: verplaats naar sovereign, maak een pure adapter, of classificeer
expliciet als niet-risk executie-mechanica. Deze suite toetst dat elke
constraint in het bestand er precies een van die drie heeft gekregen.

DE BEVINDING DIE DIT MOEST OPLOSSEN
------------------------------------
`reports/phase5_sovereign_wiring_audit.md` §4: `PortfolioConstraints.max_leverage`
stond op **1.00** terwijl `risk.gross_cap` op **1.50** staat. Twee limieten met
dezelfde betekenis en verschillende waarden; welke bond, hing af van welk pad je
draaide. Dat is de klasse fout die niet met een comment op te lossen is.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from tradebot.portfolio.constraints import (
    CONSTRAINT_ORDER,
    PortfolioConstraints,
    apply_constraints,
    compute_turnover,
)
from tradebot.risk.engine import risk_config_hash
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def risk() -> RiskConfig:
    return load_config(ROOT / "conf/risk/default.yaml", RiskConfig)


@pytest.fixture(scope="module")
def sovereign(risk: RiskConfig) -> PortfolioConstraints:
    return PortfolioConstraints.from_risk_config(risk)


# =========================================================================== #
# 1. De sovereign velden zijn adapters, geen kopieën
# =========================================================================== #
class TestTheSovereignFieldsAreAdapters:
    def test_max_weight_comes_from_max_concentration(
        self, sovereign: PortfolioConstraints, risk: RiskConfig
    ) -> None:
        assert sovereign.max_weight == pytest.approx(risk.max_concentration)

    def test_max_leverage_comes_from_gross_cap(
        self, sovereign: PortfolioConstraints, risk: RiskConfig
    ) -> None:
        assert sovereign.max_leverage == pytest.approx(risk.gross_cap)

    def test_they_track_a_change_in_the_sovereign_policy(
        self, risk: RiskConfig
    ) -> None:
        """Een adapter die niet meebeweegt, is een kopie met extra stappen."""
        tightened = risk.model_copy(
            update={"max_concentration": 0.20, "gross_cap": 0.75})
        adapted = PortfolioConstraints.from_risk_config(tightened)
        assert adapted.max_weight == pytest.approx(0.20)
        assert adapted.max_leverage == pytest.approx(0.75)
        assert adapted.risk_config_hash == risk_config_hash(tightened)

    def test_the_set_carries_the_policy_hash(
        self, sovereign: PortfolioConstraints, risk: RiskConfig
    ) -> None:
        assert sovereign.risk_config_hash == risk_config_hash(risk)
        assert sovereign.has_sovereign_provenance

    def test_the_old_divergence_is_gone(
        self, sovereign: PortfolioConstraints, risk: RiskConfig
    ) -> None:
        """Vóór Phase 5: max_leverage 1.00 tegen risk.gross_cap 1.50."""
        assert sovereign.max_leverage == pytest.approx(risk.gross_cap)
        assert sovereign.max_leverage != 1.00 or risk.gross_cap == 1.00


# =========================================================================== #
# 2. Er is geen default meer die stilzwijgend een limiet zet
# =========================================================================== #
class TestNoSilentDefaults:
    def test_the_risk_fields_have_no_defaults(self) -> None:
        params = inspect.signature(PortfolioConstraints).parameters
        for name in ("max_weight", "max_leverage"):
            assert params[name].default is inspect.Parameter.empty, (
                f"{name} heeft een default gekregen; dan bestaat er weer een "
                f"tweede risicopolicy voor wie het argument oversla at"
            )

    def test_constructing_without_them_fails(self) -> None:
        with pytest.raises(TypeError):
            PortfolioConstraints()  # type: ignore[call-arg]

    def test_a_hand_built_set_has_no_provenance(self) -> None:
        manual = PortfolioConstraints(max_weight=0.9, max_leverage=9.0)
        assert not manual.has_sovereign_provenance

    def test_production_paths_refuse_a_hand_built_set(self) -> None:
        """`enforce_provenance=True` is wat de bypass onmogelijk maakt."""
        manual = PortfolioConstraints(max_weight=0.9, max_leverage=9.0)
        raw = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        with pytest.raises(ConfigContractError):
            apply_constraints(raw, manual, enforce_provenance=True)

    def test_a_sovereign_set_passes_the_same_gate(
        self, sovereign: PortfolioConstraints
    ) -> None:
        raw = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": 0.5})
        out = apply_constraints(raw, sovereign, enforce_provenance=True)
        assert out.sum() == pytest.approx(1.0, abs=1e-9)

    def test_the_optimizer_requires_explicit_constraints(self) -> None:
        from tradebot.portfolio.optimizer import optimize

        params = inspect.signature(optimize).parameters
        assert params["constraints"].default is inspect.Parameter.empty

    def test_the_live_controller_requires_explicit_constraints(self) -> None:
        """Wiring audit C5: L13 koos zijn eigen `max_weight=0.40`."""
        from tradebot.live.portfolio_controller import PortfolioControllerConfig

        params = inspect.signature(PortfolioControllerConfig).parameters
        assert params["constraints"].default is inspect.Parameter.empty
        with pytest.raises(ConfigContractError):
            PortfolioControllerConfig(constraints=None)  # type: ignore[arg-type]


# =========================================================================== #
# 3. De executie-only constraints blijven, expliciet geclassificeerd
# =========================================================================== #
class TestExecutionOnlyConstraintsStay:
    def test_min_weight_and_max_turnover_are_not_sovereign_fields(
        self, sovereign: PortfolioConstraints
    ) -> None:
        assert set(sovereign.sovereign_fields) == {"max_weight", "max_leverage"}
        assert "min_weight" not in sovereign.sovereign_fields
        assert "max_turnover" not in sovereign.sovereign_fields

    def test_the_dust_floor_only_ever_reduces_exposure(self) -> None:
        """Daarom is hij geen bypass: hij kan de soevereine cap niet oprekken."""
        cfg = PortfolioConstraints(max_weight=0.5, max_leverage=1.0,
                                   min_weight=0.10)
        raw = pd.Series({"A": 0.45, "B": 0.45, "C": 0.05, "D": 0.05})
        out = apply_constraints(raw, cfg)
        assert out["C"] == pytest.approx(0.0)
        assert out["D"] == pytest.approx(0.0)
        assert out.max() <= cfg.max_weight + 1e-9

    def test_the_turnover_cap_only_ever_reduces_trading(self) -> None:
        cfg = PortfolioConstraints(max_weight=1.0, max_leverage=1.0,
                                   max_turnover=0.10)
        current = pd.Series({"A": 1.0, "B": 0.0})
        target = pd.Series({"A": 0.0, "B": 1.0})
        out = apply_constraints(target, cfg, current)
        assert compute_turnover(out, current) <= 0.10 + 1e-6

    def test_turnover_is_a_measurement_not_a_limit(self) -> None:
        old = pd.Series({"A": 0.5, "B": 0.5})
        new = pd.Series({"A": 0.3, "B": 0.7})
        assert compute_turnover(new, old) == pytest.approx(0.2)


# =========================================================================== #
# 4. De toepassingsvolgorde is expliciet
# =========================================================================== #
class TestTheOrderIsExplicit:
    def test_the_order_is_declared(self) -> None:
        """Wiring audit D5 classificeerde de impliciete volgorde als BUG.

        `risk/engine.py` crasht op een limiet die niet in `constraint_order`
        staat, om exact deze reden: een volgorde die je alleen uit de
        codevolgorde kunt aflezen, verandert bij een refactor zonder dat iemand
        een beslissing neemt.
        """
        assert CONSTRAINT_ORDER == (
            "concentration", "dust", "renormalise", "leverage", "turnover")

    def test_the_sovereign_caps_hold_after_everything(
        self, sovereign: PortfolioConstraints
    ) -> None:
        raw = pd.Series({"BTCUSDT": 0.95, "ETHUSDT": 0.03, "SOLUSDT": 0.02})
        out = apply_constraints(raw, sovereign, enforce_provenance=True)
        assert out.max() <= sovereign.max_weight + 1e-9
        assert out.sum() <= sovereign.max_leverage + 1e-9


# =========================================================================== #
# 5. Contractvalidatie
# =========================================================================== #
class TestContractValidation:
    @pytest.mark.parametrize(("field", "value"), [
        ("max_weight", 0.0), ("max_weight", -0.1), ("max_weight", float("nan")),
        ("max_leverage", 0.0), ("max_leverage", float("inf")),
    ])
    def test_invalid_sovereign_values_are_rejected(
        self, field: str, value: float
    ) -> None:
        kwargs = {"max_weight": 0.4, "max_leverage": 1.5, field: value}
        with pytest.raises(ConfigContractError):
            PortfolioConstraints(**kwargs)  # type: ignore[arg-type]

    def test_a_dust_floor_above_the_cap_is_rejected(self) -> None:
        with pytest.raises(ConfigContractError):
            PortfolioConstraints(max_weight=0.2, max_leverage=1.0,
                                 min_weight=0.3)

    def test_an_out_of_range_turnover_cap_is_rejected(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ConfigContractError):
                PortfolioConstraints(max_weight=0.4, max_leverage=1.5,
                                     max_turnover=bad)

    def test_the_set_is_frozen(self, sovereign: PortfolioConstraints) -> None:
        with pytest.raises((AttributeError, TypeError)):
            sovereign.max_weight = 0.99  # type: ignore[misc]
