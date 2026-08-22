"""hmm_regime.py — 3-state HMM regime detector (v3 T1.4).

Bear(0) / Flat(1) / Bull(2) op BTC+ETH macro bars.
In Bear: LONG-posities automatisch 50% reduceren.
Valt terug op EMA-based regime als hmmlearn niet geïnstalleerd is.

Referentie: CHIEF_MASTER_PLAN v3 §3 T1.4; Hamilton (1989).
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    logger.warning("hmmlearn niet beschikbaar — EMA-fallback actief. pip install hmmlearn")


class Regime(IntEnum):
    BEAR = 0
    FLAT = 1
    BULL = 2


_LONG_CAP_BEAR = 0.5


def _ema_regime_labels(returns: pd.Series, fast: int = 20, slow: int = 100) -> np.ndarray:
    """EMA-based regime fallback: Bull/Flat/Bear op basis van fast/slow EMA ratio."""
    fast_ema = returns.ewm(span=fast, adjust=False).mean()
    slow_ema = returns.ewm(span=slow, adjust=False).mean()
    ratio = fast_ema / slow_ema.replace(0, np.nan).fillna(1.0)
    labels = np.full(len(returns), int(Regime.FLAT), dtype=int)
    labels[ratio > 1.02] = int(Regime.BULL)
    labels[ratio < 0.98] = int(Regime.BEAR)
    return labels


class HMMRegimeDetector:
    """3-state Gaussisch HMM regime detector (Bear/Flat/Bull).

    Traint op dagelijkse returns + rolling volatiliteit.
    Geeft LONG-cap factor: 0.5 in Bear, 1.0 anders.
    """

    def __init__(self, n_iter: int = 100, random_state: int = 42) -> None:
        self._model: Optional[object] = None
        self._state_map: dict[int, int] = {}
        self._fitted = False
        self._n_iter = n_iter
        self._random_state = random_state

    def _features(self, returns: pd.Series) -> np.ndarray:
        vol = returns.rolling(5, min_periods=2).std().fillna(returns.std())
        X = np.column_stack([returns.values, vol.values])
        return X[np.isfinite(X).all(axis=1)]

    def fit(self, returns: pd.Series) -> "HMMRegimeDetector":
        X = self._features(returns)
        if len(X) < 50:
            logger.warning("Onvoldoende data voor HMM (%d obs).", len(X))
            return self

        if _HMM_AVAILABLE:
            try:
                model = GaussianHMM(
                    n_components=3, covariance_type="full",
                    n_iter=self._n_iter, random_state=self._random_state, tol=1e-4,
                )
                model.fit(X)
                means = model.means_[:, 0]
                order = np.argsort(means)
                self._state_map = {int(order[0]): int(Regime.BEAR), int(order[1]): int(Regime.FLAT), int(order[2]): int(Regime.BULL)}
                self._model = model
                self._fitted = True
                logger.info("HMM getraind op %d obs. State-map: %s", len(X), {k: Regime(v).name for k, v in self._state_map.items()})
            except Exception as exc:
                logger.error("HMM training mislukt: %s", exc)
        return self

    def predict(self, returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        X = self._features(returns)
        if not X.size:
            return np.array([int(Regime.FLAT)]), np.ones((1, 3)) / 3

        if self._fitted and _HMM_AVAILABLE and self._model is not None:
            try:
                raw = self._model.predict(X)
                labels = np.array([self._state_map.get(int(s), int(Regime.FLAT)) for s in raw])
                probs = self._model.predict_proba(X)
                return labels, probs
            except Exception as exc:
                logger.error("HMM predict fout: %s", exc)

        labels = _ema_regime_labels(returns)
        probs = np.zeros((len(labels), 3))
        for i, lbl in enumerate(labels):
            probs[i, int(lbl)] = 1.0
        return labels, probs

    def get_long_cap(self, regime: int) -> float:
        return _LONG_CAP_BEAR if Regime(regime) == Regime.BEAR else 1.0

    def current_regime(self, recent_returns: pd.Series) -> Regime:
        labels, _ = self.predict(recent_returns)
        return Regime(int(labels[-1]))
