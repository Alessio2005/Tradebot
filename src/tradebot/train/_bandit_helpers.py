# src/tradebot/train/_bandit_helpers.py
"""Helper classes for the ContextualBandit ensemble.

Split from ensemble.py to keep each module under the 800-LOC CI limit.
"""
from __future__ import annotations

import numpy as np


class _RandomFourierFeatures:
    """Random Fourier Features (Rahimi & Recht, 2007) — RBF-kernel benadering.

    KernelUCB exact = O(t^2) geheugen/tijd → niet werkbaar online.
    RFF mapt context x ∈ R^d  →  φ(x) ∈ R^D zodat <φ(x), φ(y)> ≈ k_RBF(x, y; σ).
    Bandit-infrastructuur (Sherman-Morrison + Cholesky) blijft lineair in D.

    sigma  = RBF-kernel bandwidth (grotere σ = vlakkere kernel)
    out_dim= aantal Fourier-features (typisch 64..512)
    """

    def __init__(self, in_dim: int, out_dim: int, sigma: float = 1.0, seed: int = 0) -> None:
        self.in_dim: int = int(in_dim)
        self.out_dim: int = int(out_dim)
        self.sigma: float = float(sigma) if sigma > 0 else 1.0
        self.seed: int = int(seed)
        rng = np.random.default_rng(self.seed)
        # ω ~ N(0, σ^-2 · I) zodat <φ(x), φ(y)> = exp(-||x-y||^2 / (2σ^2))
        self.W: np.ndarray = rng.standard_normal(size=(self.in_dim, self.out_dim)) / self.sigma
        self.b: np.ndarray = rng.uniform(0.0, 2.0 * np.pi, size=self.out_dim)
        self.scale: float = float(np.sqrt(2.0 / self.out_dim))

    def transform(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64).flatten()
        if v.size != self.in_dim:
            # Pad/truncate defensief — voorkomt crash bij dim-shift na lazy init.
            tmp = np.zeros(self.in_dim, dtype=np.float64)
            tmp[: min(self.in_dim, v.size)] = v[: self.in_dim]
            v = tmp
        return self.scale * np.cos(v @ self.W + self.b)

class _RewardClipper:
    """Per-arm running mean/variance via EMA + hard ±k·σ clip (Huber-equivalent).

    Voorkomt dat een flash-crash reward de θ-vector van de bandit opblaast.
    EMA i.p.v. Welford zodat het regimewissels volgt en niet alle history bewaart.
    """

    def __init__(self, n_arms: int, sigma_clip: float = 3.0, alpha: float = 0.01) -> None:
        self.n_arms: int = max(1, int(n_arms))
        self.sigma_clip: float = float(sigma_clip)
        self.alpha: float = float(alpha)
        self.mean: np.ndarray = np.zeros(self.n_arms, dtype=np.float64)
        self.var: np.ndarray = np.ones(self.n_arms, dtype=np.float64)
        self.warmup: np.ndarray = np.zeros(self.n_arms, dtype=np.int64)
        self.warmup_steps: int = 50

    def resize(self, n_arms: int) -> None:
        if n_arms == self.n_arms:
            return
        self.n_arms = max(1, int(n_arms))
        self.mean = np.zeros(self.n_arms, dtype=np.float64)
        self.var = np.ones(self.n_arms, dtype=np.float64)
        self.warmup = np.zeros(self.n_arms, dtype=np.int64)

    def clip(self, arm: int, reward: float) -> float:
        if not (0 <= arm < self.n_arms):
            return float(reward)
        r = float(reward)
        if not np.isfinite(r):
            return 0.0
        m_prev = self.mean[arm]
        self.mean[arm] = (1.0 - self.alpha) * m_prev + self.alpha * r
        d = r - self.mean[arm]
        self.var[arm] = (1.0 - self.alpha) * self.var[arm] + self.alpha * (d * d)
        self.warmup[arm] += 1
        if self.warmup[arm] < self.warmup_steps:
            return float(np.clip(r, -10.0, 10.0))
        sigma = float(np.sqrt(max(self.var[arm], 1e-12)))
        lo = self.mean[arm] - self.sigma_clip * sigma
        hi = self.mean[arm] + self.sigma_clip * sigma
        return float(np.clip(r, lo, hi))
from ._bandit_helpers import _RandomFourierFeatures  # noqa: F401

