# src/tradebot/risk/drawdown.py
"""Drawdown circuit-breaker with hysteresis.

The circuit-breaker has three states:
    OPEN   : normal trading; drawdown < ``trigger_threshold``
    TRIPPED: DD >= trigger_threshold; sizing halted or scaled by ``scale_factor``
    RESUME : DD falls below ``resume_threshold`` → return to OPEN

Hysteresis (trigger > resume) prevents rapid on/off flapping when equity
bounces near a single threshold.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BreakerState",
    "DrawdownBreaker",
    "DrawdownConfig",
    "compute_current_drawdown",
]


class BreakerState(Enum):
    """Circuit-breaker state machine."""

    OPEN = "OPEN"        # normal — no drawdown concern
    TRIPPED = "TRIPPED"  # DD exceeded trigger — sizing halted/scaled
    RESUME = "RESUME"    # transitional: DD recovered, waiting for full open


@dataclass
class DrawdownConfig:
    """Drawdown circuit-breaker parameters.

    Attributes
    ----------
    trigger_threshold : drawdown at which the breaker trips (e.g. 0.15 = 15 %).
    resume_threshold : drawdown at which the breaker resets (e.g. 0.08 = 8 %).
    scale_factor : leverage multiplier when TRIPPED (0 = full halt, 0.5 = half size).
    lookback_bars : rolling window for peak equity (0 = use all-time high).
    """

    trigger_threshold: float = 0.15
    resume_threshold: float = 0.08
    scale_factor: float = 0.0
    lookback_bars: int = 0

    def __post_init__(self) -> None:
        if self.resume_threshold >= self.trigger_threshold:
            raise ValueError(
                f"resume_threshold ({self.resume_threshold}) must be < "
                f"trigger_threshold ({self.trigger_threshold})."
            )


def compute_current_drawdown(equity: np.ndarray, lookback_bars: int = 0) -> float:
    """Compute drawdown from peak within a rolling window (or all-time if lookback_bars=0).

    Parameters
    ----------
    equity : equity curve array (must be > 0).
    lookback_bars : window size; 0 = all-time high.

    Returns
    -------
    float : current drawdown as a positive fraction in [0, 1].
    """
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size == 0:
        return 0.0
    current = float(eq[-1])
    window = eq if lookback_bars <= 0 else eq[max(0, len(eq) - lookback_bars):]
    peak = float(np.max(window))
    if peak <= 1e-12:
        return 0.0
    return max(0.0, (peak - current) / peak)


class DrawdownBreaker:
    """Stateful drawdown circuit-breaker.

    Usage
    -----
    >>> breaker = DrawdownBreaker(DrawdownConfig(trigger_threshold=0.15))
    >>> scale = breaker.update(equity_array)  # returns sizing multiplier
    """

    def __init__(self, config: DrawdownConfig) -> None:
        self.config = config
        self._state: BreakerState = BreakerState.OPEN
        self._trips: int = 0

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def trips(self) -> int:
        return self._trips

    def update(self, equity: np.ndarray) -> float:
        """Update state and return the sizing multiplier for this bar.

        Parameters
        ----------
        equity : equity curve up to and including the current bar.

        Returns
        -------
        float : sizing multiplier in [0, 1].
            1.0 = full size (OPEN)
            scale_factor = halted/scaled (TRIPPED)
        """
        dd = compute_current_drawdown(equity, self.config.lookback_bars)

        if self._state == BreakerState.OPEN:
            if dd >= self.config.trigger_threshold:
                self._state = BreakerState.TRIPPED
                self._trips += 1
                logger.warning(
                    "DrawdownBreaker TRIPPED: dd=%.4f >= trigger=%.4f (trip #%d).",
                    dd, self.config.trigger_threshold, self._trips,
                )
            return 1.0

        if self._state == BreakerState.TRIPPED:
            if dd <= self.config.resume_threshold:
                self._state = BreakerState.OPEN
                logger.info(
                    "DrawdownBreaker OPEN: dd=%.4f <= resume=%.4f.",
                    dd, self.config.resume_threshold,
                )
                return 1.0
            return float(self.config.scale_factor)

        # Should never reach here; treat as OPEN.
        return 1.0

    def reset(self) -> None:
        """Force the breaker back to OPEN (e.g. after a new CPCV fold starts)."""
        self._state = BreakerState.OPEN
        logger.info("DrawdownBreaker manually reset to OPEN.")
