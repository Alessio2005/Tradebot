# src/tradebot/portfolio/covariance.py
"""L8 covariantie- en correlatiehelpers — allocatie-input, geen risicolimiet.

Phase 4, stap 7. Verhuisd uit `risk/portfolio.py`. Een covariantiematrix is een
SCHATTINGSOBJECT met een eigen venster, eigen shrinkage en eigen faalmodi. Dat
hoort in de allocatielaag, niet in de soevereine limietlaag: L7 mag niet
afhankelijk zijn van de kwaliteit van een correlatieschatting die in een crisis
juist het slechtst is. `risk/vol_targeting.py` gebruikt daarom de comonotone
bovengrens (rho = 1) en heeft deze module niet nodig.

Ref: `reports/phase4_entanglement_map.md` sectie 6.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..execution.market_impact import ledoit_wolf_shrunk_corr

logger = logging.getLogger(__name__)

__all__ = ["effective_n_assets"]


def _safe_corr(
    returns: np.ndarray,
    min_obs: int = 20,
    ewma_lambda: float = 0.94,
    *,
    use_shrinkage: bool = True,
    shrinkage_target: str = "const_corr",
) -> np.ndarray:
    """Correlatie-matrix met EWMA-gewogen covariantie (RiskMetrics λ=0.94).

    AUDIT-FIX (N22 — pairwise complete observations):
      Pairwise complete observaties worden gehandhaafd: elk asset-paar gebruikt
      alleen de gemeenschappelijke niet-NaN bars.

    AUDIT-FIX (Round 3 — EWMA covariance, λ=0.94):
      Vervangt rolling Pearson (gelijke gewichten over venster) door exponentieel
      gewogen covariantie (RiskMetrics standaard). Voordelen:
        • Reageert 5-10× sneller op regime-omslag (crypto panic, flash-crash);
        • Lagere asymptotische overschatting van correlatie na reset;
        • Stabiel voor sparse NaN-perioden (gewicht valt snel weg).
      λ=0.94 ≡ half-life ≈ 11 bars (standaard Bybit 15-min bars).
      Zet ewma_lambda=1.0 om terug te vallen op het uniforme Pearson-gedrag.

    Args:
        returns:      shape (T, N). Mag NaN bevatten.
        min_obs:      minimum aantal geldige observaties per paar (NaN-guard).
        ewma_lambda:  vervalconstante 0 < λ ≤ 1 (0.94 = RiskMetrics default).

    Returns:
        (N, N) correlatie-matrix in [-1, 1]. Identity als T te klein.
    """
    if returns.ndim != 2 or returns.shape[0] < min_obs or returns.shape[1] == 0:
        n = returns.shape[1] if returns.ndim == 2 else 0
        return np.eye(max(n, 1))

    # AUDIT-FIX (Issue 6 — Ledoit-Wolf shrinkage):
    #   Voor multi-asset (N ≥ 2) en wanneer market_impact-module geladen is,
    #   schakelen we standaard over op LW-shrunk corr.  Dit dempt de
    #   correlatie-explosie die EWMA pas reproduceert na ~λ-decay-bars
    #   (te traag bij flash-crashes).
    if use_shrinkage and returns.shape[1] >= 2:
        corr_lw, _delta = ledoit_wolf_shrunk_corr(
            returns,
            target=shrinkage_target,
            min_obs=min_obs,
        )
        return corr_lw

    lam = float(np.clip(ewma_lambda, 0.0, 1.0))

    n_t, n_assets = returns.shape

    if lam >= 1.0 - 1e-9:
        # Fallback: gelijke gewichten = origineel Pearson pairwise gedrag (N22)
        df = pd.DataFrame(returns)
        corr_df = df.corr(method="pearson", min_periods=min_obs).fillna(0.0)
        corr = corr_df.to_numpy(dtype=np.float64)
        if corr.ndim == 0:
            return np.array([[1.0]])
        corr = np.where(np.isfinite(corr), corr, 0.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    # ── EWMA gewogen covariantie ─────────────────────────────────────────
    # Gewichten: w_t ∝ λ^(T-1-t), nieuwste bar krijgt gewicht λ^0 = 1.
    # Normalisering zodat Σw=1 → gewogen covariantie is schaalvrij.
    raw_w: np.ndarray = np.power(
        lam, np.arange(n_t - 1, -1, -1, dtype=np.float64)
    )  # shape (T,): [λ^(T-1), λ^(T-2), ..., λ^0]
    raw_w /= raw_w.sum()  # normaliseer

    # NaN-masker: behandel NaN-bars per asset als missing weight.
    # Pairwise: cov[i,j] gebruikt alleen bars waar BEIDE assets geldig zijn.
    valid = np.isfinite(returns)  # (T, N) bool

    cov = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in range(n_assets):
        for j in range(i, n_assets):
            mask_ij = valid[:, i] & valid[:, j]
            if mask_ij.sum() < min_obs:
                # Minder dan min_obs gemeenschappelijke obs → behandel als ongecorr.
                cov[i, j] = 0.0
                cov[j, i] = 0.0
                if i == j:
                    cov[i, i] = 1.0
                continue
            w_ij = raw_w * mask_ij.astype(np.float64)
            w_sum = w_ij.sum()
            if w_sum < 1e-15:
                cov[i, j] = 0.0
                cov[j, i] = 0.0
                if i == j:
                    cov[i, i] = 1.0
                continue
            w_ij_norm = w_ij / w_sum
            mu_i = float(np.dot(w_ij_norm, returns[:, i]))
            mu_j = float(np.dot(w_ij_norm, returns[:, j]))
            r_i = returns[:, i] - mu_i
            r_j = returns[:, j] - mu_j
            c_ij = float(np.dot(w_ij_norm, r_i * r_j))
            cov[i, j] = c_ij
            cov[j, i] = c_ij

    # Converteer covariantie → correlatie
    std_arr = np.sqrt(np.diag(cov))
    std_safe = np.where(std_arr > 1e-9, std_arr, 1.0)
    corr = cov / np.outer(std_safe, std_safe)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _avg_offdiag(corr: np.ndarray) -> float:
    """Gemiddelde van de off-diagonal elementen (de echte cross-asset correlatie)."""
    n = corr.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    vals = corr[mask]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals)))


def _dual_window_ewma_corr(
    returns_df: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 500,
    blend_weight: float = 0.3,
) -> np.ndarray:
    """Dual-window EWMA correlation: fast reacts to regime shifts (Wave 16).

    Blends fast (50-bar) and slow (500-bar) EWMA correlations.
    In crisis: fast dominates; in calm: slow stabilizes.
    """
    fast_cov = returns_df.ewm(span=fast_window, adjust=False).cov().iloc[-len(returns_df.columns):]
    slow_cov = returns_df.ewm(span=slow_window, adjust=False).cov().iloc[-len(returns_df.columns):]

    def cov_to_corr(cov_df: pd.DataFrame) -> np.ndarray:
        cov = cov_df.values
        std = np.sqrt(np.diag(cov))
        std = np.where(std > 0, std, 1.0)
        return cov / np.outer(std, std)

    fast_corr = cov_to_corr(fast_cov)
    slow_corr = cov_to_corr(slow_cov)
    blended = blend_weight * fast_corr + (1.0 - blend_weight) * slow_corr
    return blended


def effective_n_assets(corr: np.ndarray) -> float:
    """Effectief aantal onafhankelijke assets (Bouchaud-style).

    N_eff = (Σ λᵢ)² / Σ λᵢ²  waarbij λᵢ de eigenvalues van corr zijn.
    Bij identity (geen correlatie): N_eff = N. Bij volledig gecorreleerd: N_eff → 1.
    """
    if corr.shape[0] < 1:
        return 1.0
    c = 0.5 * (corr + corr.T)
    evals = np.linalg.eigvalsh(c)
    evals = np.maximum(evals, 1e-10)
    s1 = float(evals.sum())
    s2 = float((evals ** 2).sum())
    if s2 <= 0:
        return float(corr.shape[0])
    return s1 * s1 / s2
