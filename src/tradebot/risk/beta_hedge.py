"""beta_hedge.py — Dynamische bèta-neutraliteit via BTC-perp hedge (v3 T1.2).

Berekent rolling Huber-robuuste bèta van altcoins t.o.v. BTC en genereert
compenserende hedge-positie om netto portfolio-bèta binnen [-0.1, +0.1] te houden.

Referentie: CHIEF_MASTER_PLAN v3 §3 T1.2; Huber (1981) robuuste regressie.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError

logger = logging.getLogger(__name__)

_BETA_BAND = (-0.1, 0.1)
_ROLLING_DAYS = 30
_MIN_OBS = 20


def _huber_beta(y: np.ndarray, x: np.ndarray) -> float:
    """Huber-robuuste bèta via iteratieve gewogen OLS (IRLS, delta=1.345*MAD)."""
    mask = np.isfinite(y) & np.isfinite(x)
    y_c, x_c = y[mask], x[mask]
    if len(y_c) < _MIN_OBS:
        return np.nan

    residuals = y_c - np.mean(y_c)
    mad = max(np.median(np.abs(residuals - np.median(residuals))), 1e-10)
    delta = 1.345 * mad
    weights = np.ones(len(y_c))

    for _ in range(10):
        X_mat = np.column_stack([np.ones(len(x_c)), x_c])
        W = np.diag(weights)
        try:
            coefs, _, _, _ = np.linalg.lstsq(X_mat.T @ W @ X_mat, X_mat.T @ W @ y_c, rcond=None)
            beta_hat = float(coefs[1])
        except np.linalg.LinAlgError as exc:
            # Phase 0: dit gaf een gewone OLS-beta terug in plaats van de
            # ROBUUSTE Huber-beta. Precies bij de outliers waarvoor Huber is
            # gekozen, viel de schatter dus terug op de variant die daar
            # gevoelig voor is - en de hedge-ratio heette nog steeds "Huber".
            raise DataContractError(
                "Huber-IRLS regressie singulier. Er wordt NIET teruggevallen op "
                "een niet-robuuste OLS-beta onder de naam Huber."
            ) from exc

        resid = y_c - X_mat @ coefs
        abs_resid = np.abs(resid).clip(min=1e-10)
        weights = np.where(abs_resid <= delta, 1.0, delta / abs_resid)

    return beta_hat


def compute_rolling_betas(
    returns: pd.DataFrame,
    btc_col: str = "BTCUSDT",
    window_days: int = _ROLLING_DAYS,
    bar_hours: float = 1.0,
) -> pd.DataFrame:
    """Rolling Huber-bèta van elke altcoin t.o.v. BTC (causal, shift(0) — punt t gebruikt [t-w, t-1])."""
    if btc_col not in returns.columns:
        logger.warning("BTC-kolom '%s' niet gevonden — bèta = 1.0 voor alles.", btc_col)
        return pd.DataFrame(1.0, index=returns.index, columns=returns.columns)

    window_bars = int(window_days * 24 / bar_hours)
    btc = returns[btc_col].values
    betas = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for col in returns.columns:
        if col == btc_col:
            betas[col] = 1.0
            continue
        alt = returns[col].values
        beta_arr = np.full(len(alt), np.nan)
        for i in range(window_bars, len(alt)):
            beta_arr[i] = _huber_beta(alt[i - window_bars: i], btc[i - window_bars: i])
        betas[col] = beta_arr

    return betas


def compute_portfolio_beta(positions: Dict[str, float], betas: Dict[str, float]) -> float:
    """β_p = Σ(w_i × β_i)."""
    return sum(pos * betas.get(sym, 1.0) for sym, pos in positions.items() if np.isfinite(betas.get(sym, np.nan)))


def compute_btc_hedge_size(
    positions: Dict[str, float],
    betas: Dict[str, float],
    beta_band: tuple = _BETA_BAND,
) -> float:
    """Bereken BTC-hedge om β_p binnen [-0.1, +0.1] te brengen. Negatief = short BTC."""
    beta_p = compute_portfolio_beta(positions, betas)
    lo, hi = beta_band
    if lo <= beta_p <= hi:
        return 0.0
    hedge = -beta_p
    logger.info("BTC hedge: beta_p=%.3f -> hedge=%.3f", beta_p, hedge)
    return float(hedge)
