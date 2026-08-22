# src/tradebot/train/thompson.py
"""Ledoit-Wolf Thompson Sampler — shrunken covariance before TS draw.

Extracted from ``quant_architect.py`` (audit item III.5).

Uses :func:`~execution.market_impact.ledoit_wolf_shrunk_cov` via a soft-import
so this module can be loaded even when ``market_impact`` is not installed.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("train.thompson")

# Phase 0: `..execution.market_impact` is een INTERNE module binnen dit pakket en
# kan niet legitiem ontbreken. De try/except zette de vlag _MARKET_IMPACT_AVAILABLE
# op False, waarna het model stilzwijgend zonder de betreffende correctie draaide.
from ..execution.market_impact import ledoit_wolf_shrunk_cov


class LedoitWolfThompsonSampler:
    """Thompson Sampling with Ledoit-Wolf shrinkage on the bandit covariance (audit III.5).

    Problem: in :class:`~train.ensemble.ContextualBanditEnsemble` the per-arm
    posterior covariance ``v² · B⁻¹`` is ill-conditioned for correlated features
    (BTC/ETH) → TS produces extreme exploration choices.

    Solution: apply Ledoit-Wolf shrinkage to ``B⁻¹`` before the multivariate-
    normal draw. Results in a convex combination::

        Σ̂_shrunk = δ · F + (1 − δ) · v²·B⁻¹

    where ``F`` is a structured target (default ``identity · avg_var``). ``δ`` is
    estimated by :func:`~execution.market_impact.ledoit_wolf_shrunk_cov`.

    Args:
        v:        Thompson-Sampling temperature (same as LinUCB).
        target:   "const_corr" or "identity" — see ledoit_wolf_shrunk_cov.
        rng:      ``np.random.Generator`` for reproducible draws.

    Methods:
        sample_theta(theta_hat, B_inv) → (sampled_theta, applied_delta)
    """

    def __init__(
        self,
        v: float = 1.0,
        target: str = "identity",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.v: float = float(v)
        self.target: str = str(target)
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def sample_theta(
        self,
        theta_hat: np.ndarray,
        B_inv: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Draw one θ̃ ∼ N(θ̂, Σ̂_shrunk).

        Args:
            theta_hat: posterior mean, shape (d,).
            B_inv:     posterior precision-inverse, shape (d, d).

        Returns:
            (sampled_theta, applied_delta) — applied_delta ∈ [0, 1].
            On singular B_inv falls back to ``theta_hat``.
        """
        mu = np.asarray(theta_hat, dtype=np.float64).flatten()
        cov = (self.v ** 2) * np.asarray(B_inv, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)

        d = mu.size
        if d == 0:
            return mu, 0.0

        # ledoit_wolf_shrunk_cov expects a returns matrix; we generate a synthetic
        # R from the current cov via Cholesky+iid-normals so the LW formula can
        # estimate a correct δ. This is conceptually a "synthetic observation" step
        # — equivalent to directly applying LW shrinkage on the given cov.
        # Jitter raised from 1e-9 → 1e-6: correlated features (BTC/ETH) produce
        # near-rank-deficient B_inv; 1e-9 is insufficient to prevent Cholesky
        # failure and the resulting exploration-death fallback to θ̂.
        try:
            L = np.linalg.cholesky(cov + 1e-6 * np.eye(d))
        except np.linalg.LinAlgError:
            # Eigenvalue floor as last resort: clip negatives and rebuild.
            evals, evecs = np.linalg.eigh(cov)
            evals_clipped = np.maximum(evals, 1e-6)
            cov_pd = evecs @ np.diag(evals_clipped) @ evecs.T
            try:
                L = np.linalg.cholesky(cov_pd)
            except np.linalg.LinAlgError:
                logger.debug("LedoitWolfTS: singular cov even after eigh → fallback θ̂.")
                return mu, 1.0

        T_synth = max(4 * d, 30)
        z = self._rng.standard_normal(size=(T_synth, d))
        R_synth = z @ L.T  # (T, d) with cov ≈ cov

        try:
            cov_shrunk, delta = ledoit_wolf_shrunk_cov(
                R_synth, target=self.target, min_obs=30
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("LW shrinkage failed (%s) — using raw cov.", exc)
            cov_shrunk = cov
            delta = 0.0

        cov_shrunk = 0.5 * (cov_shrunk + cov_shrunk.T)
        try:
            sampled = self._rng.multivariate_normal(mu, cov_shrunk, check_valid="ignore")
        except (np.linalg.LinAlgError, ValueError):
            logger.debug("LedoitWolfTS: MVN sampling singular → fallback θ̂.")
            return mu, float(delta)
        return np.asarray(sampled, dtype=np.float64), float(delta)


__all__ = ["LedoitWolfThompsonSampler"]
