"""blocks.py — Feature block operations: stack micro/meso/macro arrays.

Extracted from train_regime.py line 1495 (stack_feats).

Public API:
  stack_feats(x1, x4, xd) -> np.ndarray
    Horizontally stacks non-None, non-empty feature blocks.
    Handles 1D → 2D promotion automatically.

Strangler-fig: train_regime.py imports from here; local definition
removed once equivalence test passes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stack_feats(
    x1: np.ndarray | None,
    x4: np.ndarray | None,
    xd: np.ndarray | None,
) -> np.ndarray:
    """Horizontally stack non-empty feature blocks (micro, meso, macro).

    Each block is promoted to 2D if 1D before hstack.  Returns an empty
    array when all inputs are None or empty — callers must guard on size.
    """
    parts = []
    for x in [x1, x4, xd]:
        if x is not None and x.size > 0:
            if x.ndim == 1:
                x = x.reshape(-1, 1)  # noqa: PLW2901
            parts.append(x)
    return np.hstack(parts) if parts else np.array([])


def split_feature_blocks(
    df: pd.DataFrame,
    feat_map: dict,
) -> tuple:
    """Split a feature DataFrame into (X_micro, X_meso, X_macro) numpy arrays.

    Parameters
    ----------
    df:
        DataFrame whose columns include the feature names listed in feat_map.
    feat_map:
        Dict with keys 'micro', 'meso', 'macro' mapping to lists of column
        names (as returned by ``pipeline.derive_feature_map``).

    Returns
    -------
    (X_micro, X_meso, X_macro) each as float64 ndarray, shape (n, k).
    Missing tiers (empty lists) are returned as empty arrays of shape (n, 0).
    """

    n = len(df)
    blocks = []
    for tier in ("micro", "meso", "macro"):
        cols = [c for c in feat_map.get(tier, []) if c in df.columns]
        if cols:
            blocks.append(df[cols].to_numpy(dtype=np.float64))
        else:
            blocks.append(np.empty((n, 0), dtype=np.float64))
    return tuple(blocks)
