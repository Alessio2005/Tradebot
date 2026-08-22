"""mda.py — Causal Mean Decrease Accuracy (AFML §8.2 + blok-shuffling).

In plaats van willekeurige feature-shuffling (wat tijdsafhankelijkheid vernietigt),
passen we blok-shuffling toe: shuffle blokken van opeenvolgende bars, zodat de
temporele structuur deels bewaard blijft maar de cross-bar informatie vernietigd wordt.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _block_shuffle(
    arr: np.ndarray,
    block_size: int = 20,
    rng: Optional[np.random.Generator] = None,
    embargo_bars: int = 0,
) -> np.ndarray:
    """Shuffle array in blokken van block_size (tijdsstructuur deels bewaard).

    CHIEF AUDIT 2026-05-23 (P-10): ``embargo_bars`` — number of bars to zero
    out between consecutive shuffled blocks.  Without an embargo, FFD (or any
    fractional-differentiation) memory from the end of block_i can leak into
    the start of the next shuffled block_{i+1}, biasing permutation importance
    upward for memory-rich features.  The embargo bars are set to NaN so the
    downstream model ignores them (or, for non-NaN-aware models, zeroed).

    Defaults to ``embargo_bars=0`` to preserve legacy behaviour.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(arr)
    n_blocks = max(1, n // block_size)
    block_indices = np.array_split(np.arange(n), n_blocks)
    rng.shuffle(block_indices)
    shuffled_idx = np.concatenate(block_indices)
    out = arr[shuffled_idx]
    # CHIEF AUDIT 2026-05-23 (P-10): apply embargo by NaN-ing the first
    # ``embargo_bars`` samples of each shuffled block (except the very first).
    if embargo_bars > 0 and len(block_indices) > 1:
        out = out.astype(np.float64, copy=True)
        offset = len(block_indices[0])
        for blk in block_indices[1:]:
            blk_len = len(blk)
            emb = min(embargo_bars, blk_len)
            if emb > 0:
                # Phase 0: de fallback zette het embargo-blok op 0 wanneer de
                # dtype geen NaN toestond. Een nul is een GELDIGE observatie
                # ("geen beweging"), geen ontbrekende - dat verschuift de
                # MDA-nulverdeling en dus elke feature-importance eruit volgt.
                # `out` is hierboven al naar float64 gecast, dus NaN past altijd.
                out[offset:offset + emb] = np.nan
            offset += blk_len
    return out


def causal_mda(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    block_size: int = 20,
    n_repeats: int = 5,
    random_seed: int = 42,
    embargo_bars: int = 0,
) -> pd.DataFrame:
    """Bereken Causal MDA via blok-shuffling op OOS-data.

    Features met negatieve MDA (shuffling verbetert resultaat) worden
    verwijderd — ze voegen ruis toe aan het model.

    Args:
        model:         Getraind classifier met predict_proba methode.
        X:             OOS feature matrix (n, d).
        y:             OOS labels (n,).
        feature_names: Lijst van feature-namen (len d).
        block_size:    Blokgrootte voor shuffling (standaard 20 bars).
        n_repeats:     Aantal shuffle-herhalingen per feature.
        random_seed:   Seed voor reproduceerbaarheid.
        embargo_bars:  CHIEF AUDIT 2026-05-23 (P-10): aantal bars dat als
                       embargo tussen geshuffelde blokken wordt geNaN'd zodat
                       FFD-memory niet over de blokgrens leakt (default 0 =
                       legacy gedrag).

    Returns:
        DataFrame met kolommen: feature, mda_mean, mda_std, t_stat, drop.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_seed)

    base_probs = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, base_probs)

    results = []
    for feat_idx, feat_name in enumerate(feature_names):
        auc_drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[:, feat_idx] = _block_shuffle(
                X[:, feat_idx], block_size, rng, embargo_bars=embargo_bars,
            )
            shuf_probs = model.predict_proba(X_shuffled)[:, 1]
            shuf_auc = roc_auc_score(y, shuf_probs)
            auc_drops.append(base_auc - shuf_auc)  # positief = feature helpt

        if not auc_drops:
            continue

        mda_mean = float(np.mean(auc_drops))
        mda_std  = float(np.std(auc_drops, ddof=1)) if len(auc_drops) > 1 else 0.0
        t_stat   = mda_mean / max(mda_std / np.sqrt(len(auc_drops)), 1e-10)
        results.append({
            "feature": feat_name,
            "mda_mean": mda_mean,
            "mda_std":  mda_std,
            "t_stat":   t_stat,
            "drop":     t_stat < 2.0,  # verwijder als t-stat < 2 (niet significant)
        })

    df = pd.DataFrame(results).sort_values("mda_mean", ascending=False)
    n_drop = df["drop"].sum()
    logger.info("Causal MDA voltooid: %d/%d features worden verwijderd (t-stat < 2.0)", n_drop, len(df))
    return df


def filter_by_mda(
    X: pd.DataFrame,
    mda_result: pd.DataFrame,
) -> pd.DataFrame:
    """Filter feature matrix op basis van MDA resultaten.

    Verwijder features waarbij drop=True (t-stat < 2.0 of negatieve MDA).
    """
    to_drop = mda_result.loc[mda_result["drop"], "feature"].tolist()
    to_keep = [c for c in X.columns if c not in to_drop]
    logger.info("MDA filter: %d features behouden, %d verwijderd", len(to_keep), len(to_drop))
    return X[to_keep]
