"""HAR-RV (Heterogeneous Autoregressive Realized Volatility) — Corsi 2009.

Long-memory volatility model that empirically outperforms EWMA on crypto.
Wave 18 microstructure upgrade.

CHIEF AUDIT 2026-05-23 (P-12) — CPCV USAGE CONTRACT
====================================================
``har_rv_fit`` is an OLS regression on the FULL input series.  If the caller
passes the entire backtest dataset, the β coefficients are fit using bars from
EVERY future fold — the resulting in-sample forecasts contain look-ahead.

CORRECT USAGE inside a CPCV / walk-forward loop:
    - Slice ``realized_var`` to the current TRAIN fold only, OR
    - Pass ``fit_indices`` explicitly so β is estimated on the TRAIN rows
      while forecasts are produced for ALL rows (test bars are then valid
      because their β did not see them).

A runtime warning fires when the input series exceeds 5000 bars without
``fit_indices`` being supplied — this is a strong heuristic for the
"someone passed the whole dataset" mistake.
"""
from __future__ import annotations

import logging
import warnings
from typing import NamedTuple, Optional

import numpy as np

from ..utils.failfast import DataContractError

logger = logging.getLogger(__name__)


class HARRVResult(NamedTuple):
    """HAR-RV fit results."""
    c: float       # intercept
    beta_d: float  # daily RV coefficient
    beta_w: float  # weekly RV (5-bar) coefficient
    beta_m: float  # monthly RV (21-bar) coefficient
    forecast: np.ndarray  # 1-step-ahead forecasts (same length as input)
    r_squared: float


def har_rv_fit(
    realized_var: np.ndarray,
    daily_window: int = 1,
    weekly_window: int = 5,
    monthly_window: int = 21,
    fit_indices: Optional[np.ndarray] = None,
) -> HARRVResult:
    """Fit HAR-RV model to realized variance series (Corsi 2009).

    Model: RV_t = c + β_d * RV_{t-1} + β_w * RV̄_{t-1:t-5} + β_m * RV̄_{t-1:t-21} + ε_t

    Args:
        realized_var : 1D array of realized variance (squared returns or RV estimator).
        daily_window : lag for daily component (default 1).
        weekly_window: lags for weekly avg (default 5).
        monthly_window: lags for monthly avg (default 21).
        fit_indices  : CHIEF AUDIT 2026-05-23 (P-12) — integer indices into
                       ``realized_var`` that define the TRAIN slice on which β
                       is estimated.  Forecasts are still produced for ALL
                       rows of ``realized_var`` (so the test rows have a
                       prediction).  ``None`` keeps the legacy global-fit
                       behaviour but triggers a warning for large inputs.

    Returns:
        HARRVResult with fitted coefficients and 1-step forecasts.
    """
    rv = np.asarray(realized_var, dtype=np.float64)
    n = len(rv)
    min_obs = monthly_window + 2
    if n < min_obs:
        raise ValueError(f"HAR-RV needs at least {min_obs} observations, got {n}")

    # CHIEF AUDIT 2026-05-23 (P-12): warn loudly when the caller passes a
    # large series without specifying ``fit_indices``.  A 5000-bar threshold
    # is a heuristic that catches the "passed the whole CPCV dataset"
    # mistake without spamming legitimate per-fold callers (TRAIN fold is
    # typically smaller).
    if fit_indices is None and n > 5000:
        warnings.warn(
            f"har_rv_fit called on {n} bars without fit_indices — β will be "
            "estimated using EVERY bar, including future folds.  Inside a "
            "CPCV / walk-forward loop you MUST pass fit_indices=<train_idx> "
            "or slice realized_var to the train fold beforehand.",
            UserWarning,
            stacklevel=2,
        )

    # Construct feature matrix
    start = monthly_window
    y = rv[start:]
    T = len(y)

    # Daily: RV_{t-1}
    rv_d = rv[start - daily_window : start - daily_window + T]

    # Weekly: rolling mean over [t-weekly_window, t-1]
    rv_w = np.array([
        np.mean(rv[max(0, start + i - weekly_window) : start + i])
        for i in range(T)
    ])

    # Monthly: rolling mean over [t-monthly_window, t-1]
    rv_m = np.array([
        np.mean(rv[max(0, start + i - monthly_window) : start + i])
        for i in range(T)
    ])

    X = np.column_stack([np.ones(T), rv_d, rv_w, rv_m])

    # CHIEF AUDIT 2026-05-23 (P-12): fit β on ONLY the TRAIN rows when
    # fit_indices is provided.  Forecasts are then produced for the FULL
    # series so test bars receive an unbiased β-prediction.
    if fit_indices is not None:
        fit_idx_arr = np.asarray(fit_indices, dtype=np.int64)
        # Convert original-array indices into X/y row indices (X is shifted
        # forward by ``start`` bars).
        local_idx = fit_idx_arr[fit_idx_arr >= start] - start
        local_idx = local_idx[local_idx < T]
        if local_idx.size < 5:
            logger.warning(
                "HAR-RV fit_indices yielded %d usable rows (<5) — falling "
                "back to full-series fit.", local_idx.size,
            )
            X_fit, y_fit = X, y
        else:
            X_fit, y_fit = X[local_idx], y[local_idx]
    else:
        X_fit, y_fit = X, y

    # OLS fit
    # Phase 0: `except LinAlgError: beta = [mean(y), 0, 0, 0]` degradeerde het
    # HAR-RV-model (Level 2) naar een CONSTANTE gemiddelde-voorspelling
    # (Level 0) - met beta_d = beta_w = beta_m = 0 verdwijnt de volledige
    # heterogene-autoregressiestructuur. Elke QLIKE-vergelijking tegen EWMA zou
    # dan feitelijk EWMA-vs-constante zijn.
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise DataContractError(
            "HAR-RV kleinste-kwadratenfit singulier. Er wordt NIET "
            "teruggevallen op een constante gemiddelde-voorspelling."
        ) from exc

    c, beta_d, beta_w, beta_m = float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])

    forecasts_in_sample = X @ beta

    # Pad with NaN for the warm-up period
    full_forecast = np.full(n, np.nan, dtype=np.float64)
    full_forecast[start:] = forecasts_in_sample

    # R-squared (on the FIT rows so the score reflects the train-slice fit)
    fit_pred = X_fit @ beta
    ss_res = float(np.sum((y_fit - fit_pred)**2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return HARRVResult(
        c=c, beta_d=beta_d, beta_w=beta_w, beta_m=beta_m,
        forecast=full_forecast, r_squared=r2,
    )


def har_rv_feature(
    realized_var: np.ndarray,
    daily_window: int = 1,
    weekly_window: int = 5,
    monthly_window: int = 21,
) -> np.ndarray:
    """Return HAR-RV 1-step-ahead forecast as a feature series.

    Suitable for use as a volatility feature in the ML pipeline.
    Returns NaN for the first monthly_window bars (warm-up).
    """
    result = har_rv_fit(realized_var, daily_window, weekly_window, monthly_window)
    return result.forecast
