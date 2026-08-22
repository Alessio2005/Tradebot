# src/tradebot/train/stack.py
"""QuantArchitectStack — bundle of all live-loop stateful components.

Extracted from ``quant_architect.py``.

Lets the orchestrator instantiate one object and pass it around without nine
separate imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .calibration import PathSpecificPlattCalibrator
from .quant_arch import SymmetricQuantileScaler
from .reward import KalmanImpactObserver, NetAlphaReward
from .schema_guard import EntropyGate, FeatureSchemaGuard
from .thompson import LedoitWolfThompsonSampler

logger = logging.getLogger("train.stack")

# Re-import the availability flag without re-importing the module body
try:
    from ..execution import market_impact as _mi  # noqa: F401
    _MARKET_IMPACT_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    _MARKET_IMPACT_AVAILABLE = False


@dataclass
class QuantArchitectStack:
    """Bundle of all live-loop stateful components.

    Example::

        >>> stack = QuantArchitectStack.default()
        >>> stack.schema_guard.stamp(feature_names)
        >>> # ... later in the live-loop:
        >>> stack.schema_guard.check(runtime_feature_names)
        >>> gate_decision = stack.entropy_gate.evaluate([prob_loss, prob_win])
        >>> if not gate_decision.pass_through:
        ...     return  # NEUTRAL — no trade
        >>> # Bandit update with net-alpha:
        >>> reward = stack.net_alpha.compute(...).reward
        >>> stack.eta_observer.update(realised_slip, sigma, q, v)
    """

    sym_scaler: SymmetricQuantileScaler = field(default_factory=SymmetricQuantileScaler)
    platt: PathSpecificPlattCalibrator = field(default_factory=PathSpecificPlattCalibrator)
    schema_guard: FeatureSchemaGuard = field(default_factory=FeatureSchemaGuard)
    entropy_gate: EntropyGate = field(default_factory=EntropyGate)
    eta_observer: KalmanImpactObserver = field(default_factory=KalmanImpactObserver)
    # Late-binding fields that depend on market_impact runtime availability:
    lw_thompson: LedoitWolfThompsonSampler | None = None
    net_alpha: NetAlphaReward | None = None

    @classmethod
    def default(
        cls,
        v_thompson: float = 1.0,
        eta_init: float = 0.142,
        fee_bps: float = 4.0,
    ) -> QuantArchitectStack:
        """Construct with sensible defaults; soft-fails if market_impact is absent."""
        stack = cls()
        if _MARKET_IMPACT_AVAILABLE:
            stack.lw_thompson = LedoitWolfThompsonSampler(v=v_thompson)
            stack.net_alpha = NetAlphaReward(eta=eta_init, fee_bps=fee_bps)
        else:  # pragma: no cover
            logger.warning(
                "market_impact not importable — LedoitWolfTS and NetAlphaReward "
                "remain None. Standalone scaler/Platt/Gate/Schema still work."
            )
        return stack


__all__ = ["QuantArchitectStack"]
