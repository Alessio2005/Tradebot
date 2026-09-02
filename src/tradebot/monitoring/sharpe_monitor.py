"""Live Sharpe degradation monitor — Wave 19.

Compares rolling live Sharpe to backtest Sharpe baseline.
Trips CircuitBreaker when degradation exceeds 2σ.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING

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
        circuit_breaker: CircuitBreaker | None = None,
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

    def record_trade_return(self, ret: float) -> dict | None:
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
                # `CircuitBreaker.trip` is SYNCHROON. Hier stond een
                # `asyncio.create_task(...)` eromheen, en die constructie kon
                # per definitie niet werken: er bestond helemaal geen `trip` op
                # de breaker (alleen het private `_trip`), dus de aanroep wierp
                # een `AttributeError` nog voordat `create_task` een coroutine
                # te zien kreeg -- de HALT kwam nooit en `check()` crashte in
                # plaats daarvan. Een schakelaar die het boek sluit hoort ook
                # niet op een event-loop te wachten: hij gaat nu direct om.
                self.circuit_breaker.trip(
                    f"live_sharpe_degradation:{degradation_sigmas:.1f}sigma",
                    measured=float(degradation_sigmas),
                    threshold=float(self.sigma_halt_threshold),
                    config_key="live.sigma_halt_threshold",
                )
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
