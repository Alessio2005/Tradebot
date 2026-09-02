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

        result = {
            "live_sharpe": live_sharpe,
            "backtest_sharpe": self.backtest_sharpe,
            "n_trades": self._n_trades,
        }

        # GEEN OORDEEL is niet hetzelfde als een SLECHT oordeel.
        # `_compute_sharpe` gaf 0,0 terug wanneer de Sharpe niet bestaat -- bij
        # nulvariantie. Dat getal is hier niet neutraal maar het SLECHTST
        # denkbare: met een baseline van 1,5 en sigma 0,25 leest een Sharpe van
        # 0,0 als 6 sigma degradatie, en de monitor sloot het boek. Een reeks
        # van 25 winstgevende trades van +1 % heeft variantie nul en werd zo
        # afgestraft als een ramp; een boek dat stilstaat (louter
        # nulrendementen) net zo goed. De poort vuurde dus juist in de gevallen
        # waarin hij niets had gemeten. Spiegelbeeld van de fout in
        # `vol_forecast_monitor` -- daar las "geen oordeel" als groen, hier als
        # rood -- en van de twee is deze de duurdere, want hij handelt.
        if not math.isfinite(live_sharpe):
            result["degradation_sigmas"] = float("nan")
            result["conclusive"] = False
            result["halt_triggered"] = False
            logger.warning(
                "Live Sharpe is niet gedefinieerd over %d trades (variantie "
                "nul). Geen oordeel, en dus GEEN halt.", self._n_trades,
            )
            return result

        degradation_sigmas = (
            (self.backtest_sharpe - live_sharpe) / self.backtest_sharpe_std
        )
        result["degradation_sigmas"] = degradation_sigmas
        result["conclusive"] = True

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
        """De live Sharpe, of NaN wanneer hij niet bestaat.

        NaN en niet 0,0. Een Sharpe is een verhouding tot de spreiding; is die
        spreiding nul, dan is de verhouding ongedefinieerd en niet "nul". De
        aanroeper moet dat onderscheid kunnen maken, want 0,0 is hier geen
        neutrale waarde maar de waarde die de grootste degradatie oplevert --
        zie `record_trade_return`.
        """
        if len(returns) < 2:
            return float("nan")
        std = float(np.std(returns, ddof=1))
        if not math.isfinite(std):
            return float("nan")
        # De ruisvloer, AFGELEID en niet gekozen. Een toets op `std <= 0.0`
        # is niet genoeg: n identieke waarden geven in exacte rekenkunde
        # precies nul, maar in float64 een opgetelde afrondingsfout. GEMETEN:
        # 25 rendementen van -0,02 leveren `std ~ 1e-18` op, en daarmee een
        # Sharpe van -8,9e16 en een "degradatie" van 3,6e17 sigma. Dat getal is
        # finiet, dus een `isfinite`-poort laat het door -- en het is ook nog
        # eens de kant op die haltert. Alles onder n * eps * schaal is
        # rekenruis en geen spreiding; dezelfde afleiding als de exactheids-
        # tolerantie in `vol_forecast_monitor._mz_verdict_is_bias`.
        scale = float(np.max(np.abs(returns)))
        noise_floor = len(returns) * float(np.finfo(np.float64).eps) * scale
        if std <= noise_floor:
            return float("nan")
        return float(np.mean(returns) / std * math.sqrt(annualization))
