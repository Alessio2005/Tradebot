# tests/unit/test_live_limits_are_sovereign.py
"""De live-executiecontroller stelt geen eigen limieten. Stage D, C3 en C4.

WAT HIER WORDT AFGEDWONGEN, EN WAAROM HET GEEN STIJLKWESTIE IS
===============================================================
`live/execution_controller.py` droeg tot Stage D twee eigen limietvelden:

    self._max_notional_per_symbol: float = float("inf")
    self._max_gross_notional:      float = float("inf")

met de opmerking *"set via attributes after init"*. Gemeten op 2026-09-01:
**niets in de hele boom zette ze ooit**. `_check_position_limits` vergeleek dus
elk order met oneindig en kon per constructie niet vuren — een poort die eruitzag
als een limiet, in de enige laag waar een ontbrekende limiet echt geld kost.

Dat is geen "default die je moet overschrijven" maar precies de
*stilzwijgend degraderen naar geen limiet*-modus die audit §23 verbiedt, en het
is no-go 7 van de fase: *"Een limiet in `live/` defaultet naar oneindig, of
ontbreekt zonder te crashen."*

De reparatie volgt AD-1: de soevereine laag is de ENIGE die een limiet stelt. De
live-controller rekent er daarom geen eigen limiet meer bij uit — hij
**verifieert** het doelboek tegen `conf/risk/default.yaml` en crasht wanneer het
daarbuiten valt. Verifiëren en niet cappen is opzet: cappen zou een tweede
sizing-implementatie zijn (no-go 6), en een doelboek dat de policy overschrijdt
betekent dat er stroomopwaarts iets stuk is.

Ref: `reports/phase7_divergence_map.md` §4; fase-opdracht Stage D-2 (C3, C4);
`risk/limits.py::_require_fraction`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.live.execution_controller import (
    ExecutionController,
    ExecutionControllerConfig,
)
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError

RISK = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
PRICES = {"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0}
EQUITY = 1_000_000.0


def _controller(risk: RiskConfig = RISK) -> ExecutionController:
    return ExecutionController(ExecutionControllerConfig(
        risk=risk, git_sha="deadbeef", model_version="test",
        # De inertiepoorten staan uit: deze tests gaan over limieten, en een
        # order die op de inertiefilter sneuvelt zou de limiettoets nooit halen.
        min_notional_per_trade=0.0, min_weight_change=0.0,
        max_weight_change=1.0))


class TestThereIsNoInfiniteLimit:
    def test_the_config_cannot_be_built_without_the_risk_policy(self) -> None:
        """Geen `risk` betekent geen controller, niet een controller zonder
        limiet."""
        with pytest.raises(TypeError):
            ExecutionControllerConfig()  # type: ignore[call-arg]

    def test_no_attribute_on_the_controller_is_infinite(self) -> None:
        controller = _controller()
        infinite = [
            name for name, value in vars(controller).items()
            if isinstance(value, float) and value == float("inf")
        ]
        assert not infinite, (
            f"Deze velden staan op oneindig: {infinite}. Een limiet die naar "
            f"oneindig defaultet is no-go 7.")

    def test_the_limits_are_the_ones_from_conf(self) -> None:
        controller = _controller()
        assert controller.max_position_pct == RISK.max_position_pct
        assert controller.gross_cap == RISK.gross_cap


class TestTheGateActuallyFires:
    """De negatieve controle: de poort moet rood KUNNEN worden, en groen."""

    def test_a_book_inside_the_policy_produces_orders(self) -> None:
        controller = _controller()
        orders = controller.size_orders(
            target_weights=pd.Series({"BTCUSDT": 0.20, "ETHUSDT": 0.15}),
            current_weights={}, prices=PRICES, equity=EQUITY)
        assert orders, "Een boek binnen de policy hoort gewoon te handelen."

    def test_a_position_above_max_position_pct_crashes(self) -> None:
        controller = _controller()
        too_big = RISK.max_position_pct + 0.05
        with pytest.raises(ConfigContractError) as exc:
            controller.size_orders(
                target_weights=pd.Series({"BTCUSDT": too_big}),
                current_weights={}, prices=PRICES, equity=EQUITY)
        assert "risk.max_position_pct" in str(exc.value)

    def test_a_book_above_gross_cap_crashes(self) -> None:
        controller = _controller()
        # Elk been binnen de per-asset-cap, samen boven de gross cap.
        per_leg = RISK.max_position_pct
        n = int(RISK.gross_cap / per_leg) + 2
        weights = {f"SYM{i}": per_leg for i in range(n)}
        with pytest.raises(ConfigContractError) as exc:
            controller.size_orders(
                target_weights=pd.Series(weights), current_weights={},
                prices={s: 100.0 for s in weights}, equity=EQUITY)
        assert "risk.gross_cap" in str(exc.value)

    def test_a_short_position_is_capped_on_its_absolute_size(self) -> None:
        """`|w|`, niet `w`. Een short van -0,40 is even groot als een long."""
        controller = _controller()
        with pytest.raises(ConfigContractError):
            controller.size_orders(
                target_weights=pd.Series(
                    {"BTCUSDT": -(RISK.max_position_pct + 0.05)}),
                current_weights={}, prices=PRICES, equity=EQUITY)


class TestTheLimitFollowsTheConfigAndNotTheCode:
    def test_a_stricter_policy_refuses_what_the_looser_one_allowed(self) -> None:
        """Verlaag de limiet in `conf` en de live-poort beweegt mee.

        Dit is de eigenschap die de twee onafhankelijke kopieën uit
        `reports/phase7_divergence_map.md` §4 juist NIET hadden.
        """
        weights = pd.Series({"BTCUSDT": 0.20})
        assert _controller().size_orders(
            target_weights=weights, current_weights={}, prices=PRICES,
            equity=EQUITY)

        strict = RISK.model_copy(update={"max_position_pct": 0.10})
        with pytest.raises(ConfigContractError):
            _controller(strict).size_orders(
                target_weights=weights, current_weights={}, prices=PRICES,
                equity=EQUITY)


def _confidence_identifiers(path: Path) -> list[str]:
    """Elke plek waar `min_confidence` als CODE voorkomt.

    Bewust op de AST en niet op de tekst: een moduledocstring die uitlegt
    waarom de parameter is verwijderd, moet hem kunnen noemen. Een test die
    het documenteren van een besluit verbiedt, dwingt af dat besluiten
    ongedocumenteerd blijven.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "min_confidence":
            found.append(f"{path.name}:{node.lineno} naam")
        elif isinstance(node, ast.Attribute) and node.attr == "min_confidence":
            found.append(f"{path.name}:{node.lineno} attribuut")
        elif isinstance(node, ast.arg) and node.arg == "min_confidence":
            found.append(f"{path.name}:{node.lineno} parameter")
        elif isinstance(node, ast.keyword) and node.arg == "min_confidence":
            found.append(f"{path.name}:{node.lineno} keyword-argument")
    return found


class TestModelConfidenceIsGoneFromLive:
    """Audit §14 sluit modelvertrouwen als sizingparameter uit (no-go 8)."""

    def test_no_module_in_live_uses_min_confidence_as_code(self) -> None:
        offenders = [
            hit
            for path in sorted((ROOT / "src" / "tradebot" / "live").glob("*.py"))
            for hit in _confidence_identifiers(path)
        ]
        assert not offenders, (
            f"`min_confidence` leeft nog als code in: {offenders}. Audit §14 "
            f"sluit modelvertrouwen als sizingparameter uit; no-go 8.")

    def test_the_detector_itself_can_go_red(self, tmp_path: Path) -> None:
        """Negatieve controle op de detector.

        Zonder deze test zou een kapotte AST-wandeling de twee tests hierboven
        permanent groen houden -- precies de klasse fout die no-go 18 beschrijft.
        """
        offender = tmp_path / "offender.py"
        offender.write_text(
            "class C:\n"
            "    def __init__(self, min_confidence: float = 0.55) -> None:\n"
            "        self.min_confidence = min_confidence\n",
            encoding="utf-8")
        assert _confidence_identifiers(offender)

    def test_a_docstring_mentioning_it_is_not_a_violation(
            self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text(
            '"""min_confidence bestaat hier niet meer; zie audit 14."""\n',
            encoding="utf-8")
        assert not _confidence_identifiers(clean)


# ── C1: de breaker-drempels komen uit de policy, niet uit code ───────────────

#: De velden van `CircuitBreakerConfig` die een tegenhanger in `RiskConfig`
#: hebben. `feed_timeout_sec` en `model_hash_mismatch` staan er BEWUST niet in:
#: dat zijn operationele condities van de live-loop zonder equivalent in de
#: backtest, en `from_risk_config` laat ze daarom als expliciete override toe.
_POLICY_OWNED_CB_FIELDS = frozenset({
    "max_drawdown_pct", "max_daily_loss_pct", "max_position_age_h",
})


def _hardcoded_breaker_thresholds(path: Path) -> list[str]:
    """Vindt `CircuitBreakerConfig(max_drawdown_pct=0.08, ...)` als code.

    Een directe constructie met een policy-veld erin is een TWEEDE kopie van een
    limiet die al in `conf/risk/default.yaml` staat. Alleen
    `CircuitBreakerConfig.from_risk_config(...)` is toegestaan; die leest de
    policy en kan dus niet uit de pas gaan lopen.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # `from_risk_config` is juist de goede weg; die heeft een Attribute-func.
        if not (isinstance(func, ast.Name) and func.id == "CircuitBreakerConfig"):
            continue
        for kw in node.keywords:
            if kw.arg in _POLICY_OWNED_CB_FIELDS:
                found.append(f"{path.name}:{node.lineno} {kw.arg}")
    return found


class TestTheBreakerThresholdsAreNotDuplicatedInCode:
    """STAGE D, C1 -- `reports/phase7_divergence_map.md` §4.

    `apps/live_paper_trader.py` bouwde zijn eigen `CircuitBreakerConfig` met
    `max_drawdown_pct=0.08`, `max_daily_loss_pct=0.03` en
    `max_position_age_h=48` als losse getallen. Ze waren op dat moment gelijk
    aan `conf/risk/default.yaml`, en niets hield ze gelijk: wie de policy
    aanscherpte, liet deze app stilzwijgend op de oude limieten doorhandelen.
    `live/engine.py` was hier al voor gerepareerd; de app niet.
    """

    def test_no_live_module_hardcodes_one_either(self) -> None:
        offenders = [
            hit
            for path in sorted((ROOT / "src" / "tradebot" / "live").glob("*.py"))
            if path.name != "circuit_breaker.py"   # de dataclass zelf
            for hit in _hardcoded_breaker_thresholds(path)
        ]
        assert not offenders, f"Drempel als code in: {offenders}"

    def test_the_detector_itself_can_go_red(self, tmp_path: Path) -> None:
        """Negatieve controle; zonder deze blijft het bovenstaande vanzelf groen."""
        offender = tmp_path / "offender.py"
        offender.write_text(
            "cfg = CircuitBreakerConfig(max_drawdown_pct=0.08,\n"
            "                           feed_timeout_sec=120)\n",
            encoding="utf-8")
        assert _hardcoded_breaker_thresholds(offender)

    def test_from_risk_config_is_not_flagged(self, tmp_path: Path) -> None:
        """De goede weg mag niet als overtreding gelden."""
        clean = tmp_path / "clean.py"
        clean.write_text(
            "cfg = CircuitBreakerConfig.from_risk_config(risk,\n"
            "                                           feed_timeout_sec=120)\n",
            encoding="utf-8")
        assert not _hardcoded_breaker_thresholds(clean)

    def test_a_live_only_override_is_not_flagged(self, tmp_path: Path) -> None:
        """`feed_timeout_sec` hoort de policy NIET toe; die mag hier staan."""
        clean = tmp_path / "clean.py"
        clean.write_text(
            "cfg = CircuitBreakerConfig(feed_timeout_sec=120)\n", encoding="utf-8")
        assert not _hardcoded_breaker_thresholds(clean)
