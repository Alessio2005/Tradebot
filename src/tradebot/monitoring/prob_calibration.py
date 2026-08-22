# src/tradebot/monitoring/prob_calibration.py
"""Probability calibration monitoring — Expected Calibration Error (ECE).

Per the blueprint (§8.4): ECE > 0.05 per (sym, side) → alert.

ECE measures the gap between predicted probabilities and observed
frequencies.  A perfectly calibrated model has ECE = 0.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CalibrationResult", "expected_calibration_error", "reliability_diagram_data"]

ECE_ALERT_THRESHOLD: float = 0.05


@dataclass
class CalibrationResult:
    """ECE result for a (symbol, side) model."""
    symbol: str
    side: str
    ece: float
    n_samples: int
    n_bins: int
    is_alert: bool

    def __str__(self) -> str:
        flag = " [ALERT]" if self.is_alert else ""
        return f"CalibrationResult({self.symbol}/{self.side}: ECE={self.ece:.4f}{flag})"


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error.

    ECE = Σ_b (|B_b| / n) · |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    probs : predicted probabilities in [0, 1], shape (n,).
    labels : binary labels in {0, 1}, shape (n,).
    n_bins : number of equal-width bins.

    Returns
    -------
    float : ECE in [0, 1].
    """
    p = np.asarray(probs, dtype=np.float64).flatten()
    y = np.asarray(labels, dtype=np.float64).flatten()
    if p.size < 2:
        return float("nan")
    n = p.size

    bins = np.linspace(0.0, 1.0 + 1e-12, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        acc = float(np.mean(y[mask]))
        conf = float(np.mean(p[mask]))
        ece_val += (mask.sum() / n) * abs(acc - conf)

    return float(ece_val)


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Data for a reliability (calibration) diagram.

    Returns
    -------
    dict with keys "bin_centers", "accuracy", "confidence", "counts".
    """
    p = np.asarray(probs, dtype=np.float64).flatten()
    y = np.asarray(labels, dtype=np.float64).flatten()

    bins = np.linspace(0.0, 1.0 + 1e-12, n_bins + 1)
    centers, accuracy, confidence, counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi)
        centers.append(float((lo + hi) / 2))
        if mask.any():
            accuracy.append(float(np.mean(y[mask])))
            confidence.append(float(np.mean(p[mask])))
            counts.append(int(mask.sum()))
        else:
            accuracy.append(float("nan"))
            confidence.append(float("nan"))
            counts.append(0)

    return {"bin_centers": centers, "accuracy": accuracy, "confidence": confidence, "counts": counts}
