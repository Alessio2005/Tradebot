# src/tradebot/train/seeded.py
"""Deterministic-seed utilities for reproducible CPCV training runs.

Every fold is trained with an independent seed derived from the global seed
and the fold index:

    fold_seed = global_seed * 1_000 + fold_id          (mod 2^31)

This guarantees:
  - Bit-for-bit reproducibility per fold when run in isolation.
  - No seed collisions between folds (different fold_id → different seed).
  - No cross-run contamination (global_seed in the formula differentiates runs).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SeedConfig", "derive_fold_seed", "seed_everything"]


@dataclass
class SeedConfig:
    """Seed configuration for a full CPCV run.

    Attributes
    ----------
    global_seed : int
        Master seed.  Used to derive per-fold seeds and to seed top-level
        random state before feature engineering.
    n_folds : int
        Total number of CPCV folds (used for pre-validation only).
    extra_seeds : dict[str, int]
        Optional named seeds for secondary stochastic components
        (e.g. "optuna": 42, "bootstrap": 99).
    """

    global_seed: int = 42
    n_folds: int = 10
    extra_seeds: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.global_seed < 0:
            raise ValueError(f"global_seed must be non-negative, got {self.global_seed}.")
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {self.n_folds}.")


def seed_everything(seed: int) -> None:
    """Seed all random number generators used by the training stack.

    Sets:
      - Python ``random`` module
      - ``numpy.random`` (legacy RandomState)
      - ``numpy`` default_rng is stateless (no global state to set)
      - ``torch`` CPU + CUDA (indien geinstalleerd)        — CHIEF AUDIT P-8
      - ``cupy`` (indien geinstalleerd)                    — CHIEF AUDIT P-8

    CatBoost seeds are passed per-call via ``random_seed`` in params.
    Numba JIT functions use the Numba random state (seeded via numpy).

    CHIEF AUDIT 2026-05-23 (P-8): GPU-determinisme. Zonder torch/cupy
    seeds blijft elke GPU-pad (cuDNN, kernel-launch volgorde) niet-
    deterministisch en zijn runs niet bit-identiek reproduceerbaar.
    De try/except ImportError zorgt dat de utility ook werkt als torch
    of cupy niet beschikbaar zijn.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)

    # CHIEF AUDIT 2026-05-23 (P-8): torch CPU+CUDA + deterministisch cuDNN.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # CHIEF AUDIT 2026-05-23 (P-8): cupy (RAPIDS / GPU-numpy) RNG.
    try:
        import cupy as cp
        cp.random.seed(seed)
    except ImportError:
        pass


def derive_fold_seed(global_seed: int, fold_id: int) -> int:
    """Derive a deterministic seed for a specific CPCV fold.

    Formula:  fold_seed = (global_seed * 1_000 + fold_id) % (2**31)

    Parameters
    ----------
    global_seed : int
        Master seed (>= 0).
    fold_id : int
        Zero-based fold index.

    Returns
    -------
    int
        Positive seed in [0, 2^31).
    """
    return (global_seed * 1_000 + fold_id) % (2 ** 31)
