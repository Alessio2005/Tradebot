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

from .beta_hedge import compute_btc_hedge_size, compute_rolling_betas
from .contract import (
    BindingConstraint,
    ConstraintKind,
    MarketState,
    RiskDecision,
    RiskState,
)
from .engine import RiskEngine
from .kill_switches import HaltStore

# Phase 0: `Regime` bestaat TWEE keer in dit pakket en betekent iets volledig
# anders. daily_loss_governor.Regime is een PROPFIRM-accountregime
# (CHALLENGE/FUNDED); hmm_regime.Regime is een MARKTREGIME (BEAR/FLAT/BULL).
# Beide werden ongealiast geimporteerd, waardoor `tradebot.risk.Regime`
# stilzwijgend het marktregime was en de propfirm-variant onbereikbaar. De
# accountregime-variant is nu expliciet AccountRegime; `Regime` blijft het
# marktregime, zoals het feitelijk al oploste.
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
from .factor_risk import FactorExposure, FactorRiskModel, compute_factor_risk
from .hmm_regime import HMMRegimeDetector, Regime
from .kelly import gap_risk_kelly_size, kelly_fraction, meta_label_kelly
from .liquidity_risk import LiquidityRiskAssessment, assess_liquidity_risk, liquidity_adjusted_var
from .position_limits import PositionLimits, PositionViolation, check_position_limits
from .stress_test import StressResult, StressScenario, StressTestSuite
from .var import historical_cvar, historical_var, rolling_cvar, rolling_var, stress_test_var

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
