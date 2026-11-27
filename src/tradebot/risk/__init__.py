# src/tradebot/risk/__init__.py
"""L7 — de soevereine risicolaag.

Phase 4 maakt `RiskEngine` het enige besluitpad: hij neemt `a_t` plus een
gemeten marktstaat en geeft de toegestane exposure terug met een volledig
machineleesbaar auditspoor. Zie `docs/RISK_CONTRACT.md`.

Wat hier NIET meer woont: `factor_alpha.py` (verhuisd naar `alpha/`) en
`portfolio.py` (gesplitst naar `portfolio/covariance.py` en
`portfolio/legacy_sizing.py`). Beide waren instanties van de alpha/risk-
verstrengeling uit auditsectie 24.

Daarnaast: Kelly-sizing, positielimieten, drawdown, VaR, factor risk, stress
testing en liquiditeitsrisico."""
from __future__ import annotations

# Lui geladen (PEP 562, opruim-PR 2026-11-27): `import tradebot.risk.x` laadt
# niet meer elke submodule, en dus niet meer zware deps (pandera, sklearn, hmmlearn) voor wie
# maar één module nodig heeft (de options-dienst). `from tradebot.risk import Naam`
# werkt ongewijzigd: `__getattr__` importeert de submodule bij het eerste gebruik.
# Phase 0: `Regime` bestaat TWEE keer in dit pakket en betekent iets volledig
# anders. daily_loss_governor.Regime is een PROPFIRM-accountregime
# (CHALLENGE/FUNDED); hmm_regime.Regime is een MARKTREGIME (BEAR/FLAT/BULL).
# Beide werden ongealiast geimporteerd, waardoor `tradebot.risk.Regime`
# stilzwijgend het marktregime was en de propfirm-variant onbereikbaar. De
# accountregime-variant is nu expliciet AccountRegime; `Regime` blijft het
# marktregime, zoals het feitelijk al oploste.
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .beta_hedge import compute_btc_hedge_size, compute_rolling_betas
    from .contract import BindingConstraint, ConstraintKind, MarketState, RiskDecision, RiskState
    from .daily_loss_governor import (
        GovernorAction,
        GovernorDecision,
        PropfirmGovernor,
        PropfirmLimits,
        RegimeConfig,
        target_gross_multiplier,
    )
    from .daily_loss_governor import Regime as AccountRegime
    from .drawdown import BreakerState, DrawdownBreaker, DrawdownConfig, compute_current_drawdown
    from .engine import RiskEngine
    from .factor_risk import FactorExposure, FactorRiskModel, compute_factor_risk
    from .hmm_regime import HMMRegimeDetector, Regime
    from .kelly import gap_risk_kelly_size, kelly_fraction, meta_label_kelly
    from .kill_switches import HaltStore
    from .liquidity_risk import (
        LiquidityRiskAssessment,
        assess_liquidity_risk,
        liquidity_adjusted_var,
    )
    from .position_limits import PositionLimits, PositionViolation, check_position_limits
    from .stress_test import StressResult, StressScenario, StressTestSuite
    from .var import historical_cvar, historical_var, rolling_cvar, rolling_var, stress_test_var

#: publieke naam -> (submodule, naam in die submodule)
_LAZY: dict[str, tuple[str, str]] = {
    "compute_btc_hedge_size": ("beta_hedge", "compute_btc_hedge_size"),
    "compute_rolling_betas": ("beta_hedge", "compute_rolling_betas"),
    "BindingConstraint": ("contract", "BindingConstraint"),
    "ConstraintKind": ("contract", "ConstraintKind"),
    "MarketState": ("contract", "MarketState"),
    "RiskDecision": ("contract", "RiskDecision"),
    "RiskState": ("contract", "RiskState"),
    "GovernorAction": ("daily_loss_governor", "GovernorAction"),
    "GovernorDecision": ("daily_loss_governor", "GovernorDecision"),
    "PropfirmGovernor": ("daily_loss_governor", "PropfirmGovernor"),
    "PropfirmLimits": ("daily_loss_governor", "PropfirmLimits"),
    "RegimeConfig": ("daily_loss_governor", "RegimeConfig"),
    "target_gross_multiplier": ("daily_loss_governor", "target_gross_multiplier"),
    "AccountRegime": ("daily_loss_governor", "Regime"),
    "BreakerState": ("drawdown", "BreakerState"),
    "DrawdownBreaker": ("drawdown", "DrawdownBreaker"),
    "DrawdownConfig": ("drawdown", "DrawdownConfig"),
    "compute_current_drawdown": ("drawdown", "compute_current_drawdown"),
    "RiskEngine": ("engine", "RiskEngine"),
    "FactorExposure": ("factor_risk", "FactorExposure"),
    "FactorRiskModel": ("factor_risk", "FactorRiskModel"),
    "compute_factor_risk": ("factor_risk", "compute_factor_risk"),
    "HMMRegimeDetector": ("hmm_regime", "HMMRegimeDetector"),
    "Regime": ("hmm_regime", "Regime"),
    "gap_risk_kelly_size": ("kelly", "gap_risk_kelly_size"),
    "kelly_fraction": ("kelly", "kelly_fraction"),
    "meta_label_kelly": ("kelly", "meta_label_kelly"),
    "HaltStore": ("kill_switches", "HaltStore"),
    "LiquidityRiskAssessment": ("liquidity_risk", "LiquidityRiskAssessment"),
    "assess_liquidity_risk": ("liquidity_risk", "assess_liquidity_risk"),
    "liquidity_adjusted_var": ("liquidity_risk", "liquidity_adjusted_var"),
    "PositionLimits": ("position_limits", "PositionLimits"),
    "PositionViolation": ("position_limits", "PositionViolation"),
    "check_position_limits": ("position_limits", "check_position_limits"),
    "StressResult": ("stress_test", "StressResult"),
    "StressScenario": ("stress_test", "StressScenario"),
    "StressTestSuite": ("stress_test", "StressTestSuite"),
    "historical_cvar": ("var", "historical_cvar"),
    "historical_var": ("var", "historical_var"),
    "rolling_cvar": ("var", "rolling_cvar"),
    "rolling_var": ("var", "rolling_var"),
    "stress_test_var": ("var", "stress_test_var"),
}

__all__ = [
    # L7 contract + soevereine engine (Phase 4)
    "BindingConstraint",
    "ConstraintKind",
    "HaltStore",
    "MarketState",
    "RiskDecision",
    "RiskEngine",
    "RiskState",
    # kelly
    "kelly_fraction",
    "gap_risk_kelly_size",
    "meta_label_kelly",
    # position limits
    "PositionLimits",
    "PositionViolation",
    "check_position_limits",
    # drawdown
    "BreakerState",
    "DrawdownConfig",
    "DrawdownBreaker",
    "compute_current_drawdown",
    # propfirm governor (daily-loss + static-DD + regime/vol-target)
    "GovernorAction",
    "GovernorDecision",
    "PropfirmGovernor",
    "PropfirmLimits",
    "AccountRegime",
    "RegimeConfig",
    "target_gross_multiplier",
    # var
    "historical_var",
    "historical_cvar",
    "rolling_var",
    "rolling_cvar",
    "stress_test_var",
    # factor risk (Wave 7)
    "FactorExposure",
    "FactorRiskModel",
    "compute_factor_risk",
    # stress test (Wave 7)
    "StressScenario",
    "StressResult",
    "StressTestSuite",
    # liquidity risk (Wave 7)
    "LiquidityRiskAssessment",
    "assess_liquidity_risk",
    "liquidity_adjusted_var",
    # beta hedge (v3 T1.2)
    "compute_rolling_betas",
    "compute_btc_hedge_size",
    # hmm regime (v3 T1.4)
    "HMMRegimeDetector",
    "Regime",
]


def __getattr__(name: str) -> Any:
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})
