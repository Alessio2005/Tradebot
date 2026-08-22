"""Live Sharpe degradation monitor — Wave 19.

Compares rolling live Sharpe to backtest Sharpe baseline.
Trips CircuitBreaker when degradation exceeds 2σ.
"""
from __future__ import annotations
import logging
import math
from collections import deque
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..live.circuit_breaker import CircuitBreaker

import numpy as np

logger = logging.getLogger(__name__)


class LiveSharpeMonitor:
    """Monitor live Sharpe vs backtest baseline (Wave 19).

    Alert + halt when: live_sharpe < backtest_sharpe - 2 * sigma_threshold
    """

    def __init__(
        self,
        backtest_sharpe: float,
        backtest_sharpe_std: float,
        circuit_breaker: Optional["CircuitBreaker"] = None,
        window: int = 252,
        sigma_halt_threshold: float = 2.0,
        min_trades_before_halt: int = 20,
    ) -> None:
        self.backtest_sharpe = backtest_sharpe
        self.backtest_sharpe_std = max(float(backtest_sharpe_std), 0.01)
        self.circuit_breaker = circuit_breaker
        self.window = window
        self.sigma_halt_threshold = sigma_halt_threshold
        self.min_trades = min_trades_before_halt
        self._returns: deque[float] = deque(maxlen=window)
        self._n_trades: int = 0

    def record_trade_return(self, ret: float) -> Optional[dict]:
        """Add one trade return and check for Sharpe degradation."""
        self._returns.append(float(ret))
        self._n_trades += 1

        if self._n_trades < self.min_trades or len(self._returns) < 10:
            return None

        arr = np.array(self._returns, dtype=np.float64)
        live_sharpe = self._compute_sharpe(arr)

        degradation_sigmas = (
            (self.backtest_sharpe - live_sharpe) / self.backtest_sharpe_std
        )

        result = {
            "live_sharpe": live_sharpe,
            "backtest_sharpe": self.backtest_sharpe,
            "degradation_sigmas": degradation_sigmas,
            "n_trades": self._n_trades,
        }

        if degradation_sigmas >= self.sigma_halt_threshold:
            msg = (
                f"LIVE SHARPE DEGRADATION: live={live_sharpe:.3f} "
                f"backtest={self.backtest_sharpe:.3f} "
                f"({degradation_sigmas:.1f}σ below baseline). HALTING."
            )
            logger.critical(msg)
            if self.circuit_breaker is not None:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(
                            self.circuit_breaker.trip(f"live_sharpe_degradation:{degradation_sigmas:.1f}sigma")
                        )
                except Exception as exc:
                    logger.error("Failed to trip CB via async: %s. Trip manually.", exc)
            result["halt_triggered"] = True

        return result

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
        if len(returns) < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std <= 0:
            return 0.0
        return float(np.mean(returns) / std * math.sqrt(annualization))
