# src/tradebot/risk/__init__.py
"""Risk sub-package — portfolio risk manager, Kelly sizing, position limits, drawdown, VaR,
factor risk, stress testing, and liquidity risk."""
from __future__ import annotations

from .daily_loss_governor import (
    GovernorAction,
    GovernorDecision,
    PropfirmGovernor,
    PropfirmLimits,
    Regime,
    RegimeConfig,
    target_gross_multiplier,
)
from .drawdown import BreakerState, DrawdownBreaker, DrawdownConfig, compute_current_drawdown
from .factor_alpha import G4_FACTORSETS, FactorAlphaResult, factor_residual_alpha
from .factor_risk import FactorExposure, FactorRiskModel, compute_factor_risk
from .kelly import gap_risk_kelly_size, kelly_fraction, meta_label_kelly
from .liquidity_risk import LiquidityRiskAssessment, assess_liquidity_risk, liquidity_adjusted_var
from .portfolio import PortfolioRiskManager, RiskState, SizingDecision, effective_n_assets
from .position_limits import PositionLimits, PositionViolation, check_position_limits
from .stress_test import StressResult, StressScenario, StressTestSuite
from .beta_hedge import compute_btc_hedge_size, compute_rolling_betas
from .hmm_regime import HMMRegimeDetector, Regime
from .var import historical_cvar, historical_var, rolling_cvar, rolling_var, stress_test_var

__all__ = [
    # portfolio
    "PortfolioRiskManager",
    "RiskState",
    "SizingDecision",
    "effective_n_assets",
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
    "Regime",
    "RegimeConfig",
    "target_gross_multiplier",
    # var
    "historical_var",
    "historical_cvar",
    "rolling_var",
    "rolling_cvar",
    "stress_test_var",
    # factor alpha / G4 lab (Wave 20)
    "G4_FACTORSETS",
    "FactorAlphaResult",
    "factor_residual_alpha",
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
