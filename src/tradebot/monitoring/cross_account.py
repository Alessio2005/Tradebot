"""monitoring/cross_account.py — Cross-account correlation monitor.

The propfirm plan runs one strategy per account precisely so that a blow-up on
one account does not drag the others.  That only holds if the accounts' daily
P&L stays uncorrelated.  If two accounts drift into the same exposure (e.g.
both net-short crypto beta), a single bad day can trip the 5 % daily limit on
BOTH at once — defeating the diversification.

This monitor tracks each account's recent returns, computes pairwise rolling
correlation, and — for any pair exceeding ``corr_threshold`` — throttles the
LOWER-Sharpe account of the pair, so capital stays on the better book while
the redundant exposure is cut.

Stateless-per-call API: feed it the latest per-account return each bar; ask it
for the current throttle multipliers.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CrossAccountConfig", "CrossAccountMonitor"]


@dataclass(frozen=True)
class CrossAccountConfig:
    """Parameters for the cross-account correlation throttle.

    corr_window : rolling window (bars) for the correlation/Sharpe estimate.
    corr_threshold : pairwise correlation above which a pair is "too coupled".
    throttle_factor : sizing multiplier applied to the throttled account
        (0.5 = half size; 0 = flat).
    min_obs : minimum paired observations before any throttle is applied.
    """

    corr_window: int = 20
    corr_threshold: float = 0.40
    throttle_factor: float = 0.50
    min_obs: int = 10

    def __post_init__(self) -> None:
        if not (0.0 <= self.throttle_factor <= 1.0):
            raise ValueError("throttle_factor must be in [0, 1].")
        if not (0.0 < self.corr_threshold <= 1.0):
            raise ValueError("corr_threshold must be in (0, 1].")
        if self.min_obs < 2 or self.corr_window < self.min_obs:
            raise ValueError("require corr_window >= min_obs >= 2.")


@dataclass
class CrossAccountMonitor:
    """Rolling cross-account correlation throttle."""

    config: CrossAccountConfig = field(default_factory=CrossAccountConfig)
    _returns: dict[str, deque[float]] = field(default_factory=dict, init=False)

    def update(self, account_returns: dict[str, float]) -> None:
        """Append the latest per-account return (one value per account)."""
        w = self.config.corr_window
        for acct, r in account_returns.items():
            buf = self._returns.get(acct)
            if buf is None:
                buf = deque(maxlen=w)
                self._returns[acct] = buf
            buf.append(float(r))

    def _sharpe(self, buf: deque[float]) -> float:
        arr = np.asarray(buf, dtype=np.float64)
        if arr.size < 2:
            return 0.0
        sd = arr.std(ddof=1)
        if sd <= 1e-12:
            return 0.0
        return float(arr.mean() / sd)

    def throttles(self) -> dict[str, float]:
        """Return the sizing multiplier per account (default 1.0).

        For every account pair whose recent correlation exceeds the threshold,
        the lower-Sharpe member is throttled.  An account throttled by ANY pair
        gets ``throttle_factor`` (the minimum, if implicated in several pairs).
        """
        cfg = self.config
        accts = list(self._returns.keys())
        mult: dict[str, float] = {a: 1.0 for a in accts}

        for i in range(len(accts)):
            for j in range(i + 1, len(accts)):
                a, b = accts[i], accts[j]
                ra = np.asarray(self._returns[a], dtype=np.float64)
                rb = np.asarray(self._returns[b], dtype=np.float64)
                n = min(ra.size, rb.size)
                if n < cfg.min_obs:
                    continue
                ra, rb = ra[-n:], rb[-n:]
                if ra.std() <= 1e-12 or rb.std() <= 1e-12:
                    continue
                corr = float(np.corrcoef(ra, rb)[0, 1])
                if corr <= cfg.corr_threshold:
                    continue
                # Throttle the lower-Sharpe account of this coupled pair.
                loser = a if self._sharpe(self._returns[a]) <= self._sharpe(self._returns[b]) else b
                mult[loser] = min(mult[loser], cfg.throttle_factor)
                logger.warning(
                    "Cross-account corr(%s,%s)=%.2f > %.2f → throttling %s to %.2f.",
                    a, b, corr, cfg.corr_threshold, loser, cfg.throttle_factor,
                )
        return mult
