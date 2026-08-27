# src/tradebot/alpha/__init__.py
"""Alpha signal library — production signals + IC-weighted combiner.

Signals:
  TSMomentum         — 12-1 time-series momentum (Moskowitz 2012)
  CSMomentum         — Cross-sectional rank momentum (calendar time)
  CSMVolumeClockSignal — CSM on volume-clock macro-bars (T2.1)
  FundingCarry       — Funding-rate carry (crypto perpetual)
  OUMeanReversion    — OU mean-reversion via MLE
  KalmanOUMeanReversion — OU with Kalman half-life tracking + Judge gate (T2.2)
  OFISignal          — Order-flow imbalance (Cont et al. 2014)
  MacroRegimeOverlay — Macro regime filter
  ICWeightedCombiner — IC-DAMP weighted signal ensemble
"""
from __future__ import annotations

from .base import AlphaSignal, SignalResult
from .carry import FundingCarry
from .combination import ICWeightedCombiner
from .csm_volume_clock import (
    CSMVolumeClockSignal,
    compute_csm_volume_clock_signals,
    rank_normalize_cross_section,
)
from .factor_alpha import (
    G4_FACTORSETS,
    FactorAlphaResult,
    factor_residual_alpha,
)
from .kalman_ou import KalmanOUMeanReversion
from .macro_regime import MacroRegimeOverlay, RegimeLabel, RegimeState
from .mean_reversion import OUMeanReversion
from .microstructure import OFISignal
from .momentum import CSMomentum, TSMomentum
from .research_harness import HarnessResult, run_signal_harness

__all__ = [
    # protocol + result
    "AlphaSignal",
    "SignalResult",
    # factor lab / G4 (Phase 4: verplaatst uit risk/)
    "G4_FACTORSETS",
    "FactorAlphaResult",
    "factor_residual_alpha",
    # signals — calendar time
    "TSMomentum",
    "CSMomentum",
    "FundingCarry",
    "OUMeanReversion",
    "OFISignal",
    "MacroRegimeOverlay",
    # signals — Tier 2 (volume-clock + Kalman)
    "CSMVolumeClockSignal",
    "compute_csm_volume_clock_signals",
    "rank_normalize_cross_section",
    "KalmanOUMeanReversion",
    # regime
    "RegimeLabel",
    "RegimeState",
    # combination
    "ICWeightedCombiner",
    # research
    "HarnessResult",
    "run_signal_harness",
]
