"""Alpha Decay Tracker — Wave 18.

Tracks Information Coefficient (IC) rolling decay per alpha signal.
Alerts when IC decays below threshold (signal is dead).
"""
from __future__ import annotations
import logging
from collections import deque
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlphaDecayTracker:
    """Rolling IC tracker per alpha signal (Wave 18).

    Monitors signal decay by computing rolling Spearman IC between
    signal values and subsequent returns.

    Alert when:
        - Rolling IC < ic_threshold (signal is flat/dead)
        - IC trend (rolling regression) is significantly negative
    """

    def __init__(
        self,
        window: int = 252,
        ic_threshold: float = 0.02,
        alert_callback=None,
        horizon_bars: int = 1,
    ) -> None:
        """Parameters
        ----------
        window : int
            Rolling window for IC computation.
        ic_threshold : float
            Below this |IC| level, the decay alert fires.
        alert_callback : callable
            Optional callback ``fn(msg: str)`` for external alerting.
        horizon_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-4): forecast horizon in bars.  The caller
            MUST invoke :meth:`update` once per bar with ``(signal_t, return_t)``.
            We internally buffer ``horizon_bars`` observations so the IC is
            computed between ``signal[t]`` and ``realized_return[t+horizon_bars]``.
            This prevents accidental same-bar IC where a caller passes
            ``return_t`` already aligned with ``signal_t``.
        """
        self.window = window
        self.ic_threshold = ic_threshold
        self.alert_callback = alert_callback
        # CHIEF AUDIT 2026-05-23 (P-4): horizon alignment between signal and
        # forward return.  Buffer is used to defer IC computation until enough
        # future returns are observed.
        self.horizon_bars = max(1, int(horizon_bars))
        # Pending (signal, ts) waiting for their matching forward return.
        # Each entry is a tuple (signal_value, bars_remaining).
        self._pending: deque[list] = deque()
        self._signals: deque[float] = deque(maxlen=window)
        self._returns: deque[float] = deque(maxlen=window)
        self._ic_history: list[float] = []

    def update(self, signal_value: float, realized_return: float) -> Optional[float]:
        """Add one (signal, return) observation and compute rolling IC.

        CHIEF AUDIT 2026-05-23 (P-4): the ``realized_return`` argument is the
        return realised on the CURRENT bar.  The signal value is buffered for
        ``horizon_bars`` bars and only then paired with the realised return on
        that future bar.  Callers MUST invoke update() per bar (including bars
        where the signal is 0 or stale) to keep the alignment consistent.
        """
        # Decrement countdown on all pending signals; the matured ones (counter
        # ≤ 0) are paired with the CURRENT realised return.
        new_pending: deque[list] = deque()
        matured_signal: Optional[float] = None
        for entry in self._pending:
            entry[1] -= 1
            if entry[1] <= 0 and matured_signal is None:
                # Match this signal against the current bar's return.
                matured_signal = entry[0]
            else:
                new_pending.append(entry)
        self._pending = new_pending

        # Queue the new signal observation for its forward-return match.
        self._pending.append([float(signal_value), self.horizon_bars])

        if matured_signal is None:
            return None

        self._signals.append(float(matured_signal))
        self._returns.append(float(realized_return))

        if len(self._signals) < 20:
            return None

        from scipy import stats as _stats
        sig_arr = np.array(self._signals, dtype=np.float64)
        ret_arr = np.array(self._returns, dtype=np.float64)

        if np.std(sig_arr) <= 0 or np.std(ret_arr) <= 0:
            return None

        ic, pval = _stats.spearmanr(sig_arr, ret_arr)
        ic = float(ic)
        self._ic_history.append(ic)

        if abs(ic) < self.ic_threshold:
            msg = (
                f"Alpha decay alert: rolling IC={ic:.4f} below threshold "
                f"{self.ic_threshold:.4f} (window={len(self._signals)})"
            )
            logger.warning(msg)
            if self.alert_callback is not None:
                self.alert_callback(msg)

        return ic

    def ic_trend(self, lookback: int = 52) -> Optional[float]:
        """Linear trend of IC over last lookback observations (negative = decaying)."""
        h = self._ic_history[-lookback:]
        if len(h) < 10:
            return None
        x = np.arange(len(h), dtype=np.float64)
        y = np.array(h, dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    def summary(self) -> dict[str, float]:
        """Return IC summary statistics."""
        if not self._ic_history:
            return {"mean_ic": 0.0, "std_ic": 0.0, "latest_ic": 0.0}
        h = np.array(self._ic_history, dtype=np.float64)
        return {
            "mean_ic": float(np.mean(h)),
            "std_ic": float(np.std(h, ddof=1)) if len(h) > 1 else 0.0,
            "latest_ic": float(h[-1]),
            "ic_trend_52w": self.ic_trend(52) or 0.0,
        }
