"""pipeline.py — Adapter around legacy ``features.FeaturePipeline``.

Extracted from train_regime.py lines 3607-3814 (the per-symbol feature build
section inside train_pipeline).

Why a wrapper?
  ``FeaturePipeline.transform`` (legacy ``features.py:739``) returns a 4-tuple
  ``(X_micro, X_meso, X_macro, df_merged)`` whose semantics are opaque without
  context. This adapter:
    1. Names the unpacking explicitly.
    2. Derives the canonical feature_map (micro/meso/macro buckets) from
       ``df_merged.columns`` using the same suffix-rules as train_regime.py:3773-3794.
    3. Persists the fitted PCA orthogonalizers for downstream determinism
       (Stage 3 must reproduce the exact PCA projection seen during Stage 2).

Strangler-fig: once ``features.py`` is fully decomposed into
``tradebot.bars.*`` + ``tradebot.features.{ta,orthogonalize}``, this module
becomes the single import surface for Stage 1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Feature-map derivation (canonical rules from train_regime.py:3773-3794)
# =============================================================================

def derive_feature_map(df_merged: pd.DataFrame) -> dict[str, list[str]]:
    """Bucket ``feat_*`` columns into micro / meso / macro by suffix/prefix.

    Rules (mirror train_regime.py exactly):
      • ``feat_macro_*`` or ``*_macro``                 → macro
      • ``*_meso``                                       → meso
      • everything else starting with ``feat_``         → micro
    """
    all_cols: list[str] = df_merged.columns.tolist()

    feats_micro = [
        c for c in all_cols
        if c.startswith("feat_")
        and not c.endswith("_meso")
        and not c.endswith("_macro")
        and not c.startswith("feat_macro_")
    ]
    feats_meso = [c for c in all_cols if c.startswith("feat_") and c.endswith("_meso")]
    feats_macro = [
        c for c in all_cols
        if c.startswith("feat_")
        and (c.endswith("_macro") or c.startswith("feat_macro_"))
    ]

    return {
        "micro": sorted(feats_micro),
        "meso":  sorted(feats_meso),
        "macro": sorted(feats_macro),
    }


# =============================================================================
# Pipeline driver
# =============================================================================

def build_features(
    cfg: DictConfig,
    sym: str,
    df_micro: pd.DataFrame,
    orthogonalizer_save_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Run FeaturePipeline.transform and return ``(df_merged, feat_map)``.

    Parameters
    ----------
    cfg:
        Hydra DictConfig (already merged with per-symbol overrides).
    sym:
        Trading-pair symbol (e.g. "BTCUSDT").
    df_micro:
        Raw intraday bars from ``data.ingestion.ingest_raw``.
    orthogonalizer_save_dir:
        If provided, the fitted orthogonalizers are persisted as
        ``{dir}/orthogonalizer_{tier}_{sym}.pkl`` (mirrors train_regime.py:3801-3814).
        REQUIRED for Stage 3 reproducibility.

    Returns
    -------
    df_merged: pd.DataFrame
        Bars + micro/meso/macro features with DatetimeIndex.
    feat_map: dict[str, list[str]]
        Canonical feature_map with micro/meso/macro buckets.
    """
    from .regime import FeaturePipeline

    logger.info("[%s] Running FeaturePipeline.transform...", sym)

    engine = FeaturePipeline(cfg, sym)
    result = engine.transform(df_micro)

    # Defensive unpack: the contract is (X_micro, X_meso, X_macro, df_merged).
    if not (isinstance(result, tuple) and len(result) == 4):
        raise RuntimeError(
            f"[{sym}] FeaturePipeline.transform returned {type(result).__name__} "
            f"(len={len(result) if hasattr(result, '__len__') else 'N/A'}); "
            "expected 4-tuple (X_micro, X_meso, X_macro, df_merged)."
        )

    _X_micro, _X_meso, _X_macro, df_merged = result

    if df_merged is None or len(df_merged) == 0:
        raise ValueError(
            f"[{sym}] FeaturePipeline returned empty df_merged. "
            "Check bar configuration (min_bars, timeframe, burn-in)."
        )

    # Derive the canonical feature_map (suffix-rules).
    feat_map = derive_feature_map(df_merged)

    # ── Persist fitted orthogonalizers for Stage 3 determinism ────────────────
    # Mirrors train_regime.py:3801-3814 naming convention.
    if orthogonalizer_save_dir is not None:
        orthogonalizer_save_dir.mkdir(parents=True, exist_ok=True)
        for tier in ("micro", "meso", "macro"):
            attr_name = f"orthogonalizer_{tier}"
            orth = getattr(engine, attr_name, None)
            if orth is None:
                continue
            if not getattr(orth, "is_fitted", False):
                logger.debug("[%s] Orthogonalizer '%s' not fitted — skipping persist.", sym, tier)
                continue
            out_path = orthogonalizer_save_dir / f"orthogonalizer_{tier}_{sym}.pkl"
            joblib.dump(orth, out_path)
            logger.info(
                "[%s] Orthogonalizer '%s' saved → %s (%d components).",
                sym, tier, out_path.name,
                len(getattr(orth, "output_names", [])),
            )

    logger.info(
        "[%s] FeaturePipeline complete: %d bars, %d cols (%d micro / %d meso / %d macro).",
        sym, len(df_merged), df_merged.shape[1],
        len(feat_map["micro"]), len(feat_map["meso"]), len(feat_map["macro"]),
    )
    return df_merged, feat_map


# =============================================================================
# Re-loadable orthogonalizers (for Stage 3 inference)
# =============================================================================

def load_orthogonalizers(
    sym: str,
    orthogonalizer_dir: Path,
) -> dict[str, Any]:
    """Load previously saved orthogonalizers for downstream stages.

    Returns ``{'micro': obj, 'meso': obj, 'macro': obj}`` with missing tiers
    set to None.  Matches the file naming used by ``build_features``.
    """
    out: dict[str, Any] = {"micro": None, "meso": None, "macro": None}
    for tier in out:
        path = orthogonalizer_dir / f"orthogonalizer_{tier}_{sym}.pkl"
        if path.exists():
            out[tier] = joblib.load(path)
    return out
