"""feature_selection.py — Per-asset feature selection pipeline.

Runs SFI (Single Feature Importance) + Causal MDA + CFI (Clustered Feature
Importance) per asset, outputs selected feature lists to:
    artefacts/{SYM}/selected_features.json

Usage::

    python apps/feature_selection.py

    # Or via Hydra for per-symbol override:
    python apps/feature_selection.py training.training_universe=[AVAXUSDT]

The FeaturePipeline in train_cpcv.py checks for ``selected_features.json``
and restricts the feature matrix to the listed features when the file exists.
This reduces overfitting from irrelevant features and speeds up training.

Algorithm:
  1. Load OOS probs + feature matrix from artefacts (requires Stage 3 to
     have run at least once to produce baseline OOS probs).
  2. SFI: for each feature, compute OOS AUC using PURGED time-series splits
     (``selection.sfi._purged_timeseries_splits``) + CatBoost. Remove features
     with AUC < ``min_auc_sfi`` (default 0.52).
  3. Causal MDA: block-shuffle each remaining feature (preserving temporal
     autocorrelation) and measure AUC degradation.  Remove features with
     t-stat of degradation < ``min_tstat_mda`` (default 2.0).
  4. CFI: use clustered feature importance (mean shap per cluster) to further
     prune near-redundant features within clusters.  Threshold: keep top
     ``cfi_keep_frac`` fraction of feature clusters (default 0.7 = top 70%).
  5. Save intersection of surviving features to JSON.

References:
  AFML §8 (Feature Importance): SFI, MDA, CFI
  López de Prado (2020) "Machine Learning for Asset Managers" §6

Notes:
  • When artefacts are absent (first run), the script logs a warning and exits
    gracefully — rerun after Stage 3 produces baseline OOS probs.
  • Backward-compatible: if ``selected_features.json`` is not present, the
    FeaturePipeline uses all features (no change to current behavior).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tradebot.schemas.config import ValidationConfig, load_config
from tradebot.selection.mda import filter_by_mda
from tradebot.selection.sfi import rank_features_by_sfi
from tradebot.validation.walk_forward import purged_walk_forward

logger = logging.getLogger(__name__)


# =============================================================================
# CFI helpers
# =============================================================================

def _compute_cfi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_clusters: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Clustered Feature Importance via CatBoost + hierarchical clustering.

    Groups features by Spearman correlation, then computes mean feature
    importance per cluster.  Returns {feature_name: cluster_mean_importance}.

    Falls back to unclustered SHAP importance if scipy / sklearn are missing.
    """
    try:
        import scipy.spatial.distance as ssd
        from catboost import CatBoostClassifier
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.stats import spearmanr
    except ImportError as e:
        logger.warning("CFI: missing dependency (%s) — using raw importance.", e)
        return _raw_feature_importance(X, y, feature_names, seed)

    p = X.shape[1]
    if p < 2:
        return {fn: 1.0 for fn in feature_names}

    # ── Step 1: Hierarchical Spearman clustering ──────────────────────────────
    try:
        rho_result = spearmanr(X, axis=0, nan_policy="omit")
        rho = np.asarray(getattr(rho_result, "correlation", rho_result[0]), dtype=np.float64)
        if rho.ndim == 0:
            rho = np.array([[1.0, float(rho)], [float(rho), 1.0]])
        rho = np.nan_to_num(rho, nan=0.0)
        np.fill_diagonal(rho, 1.0)
        dist = 1.0 - np.abs(rho)
        dist = np.clip(0.5 * (dist + dist.T), 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        cond = ssd.squareform(dist, checks=False)
        Z = linkage(cond, method="ward")
        n_c = min(n_clusters, p)
        cluster_ids = fcluster(Z, t=n_c, criterion="maxclust")
    except Exception as exc:
        logger.warning("CFI: clustering failed (%s) — using raw importance.", exc)
        return _raw_feature_importance(X, y, feature_names, seed)

    # ── Step 2: CatBoost importance per fold ─────────────────────────────────
    # PHASE 7/8 STAGE B-5: dit gebruikte `TimeSeriesSplit(n_splits=5)`.
    #
    # `TimeSeriesSplit` respecteert de tijdsvolgorde en ziet er daarom correct
    # uit, maar hij kent geen purge en geen embargo: het label van de laatste H
    # trainbars loopt het testvenster in. Voor feature-SELECTIE is dat niet
    # onschuldig — de gekozen featureset stroomt door naar modellen die wél
    # worden gepromoveerd, en een lek stroomapwaarts is niet te repareren met
    # een strengere gate stroomafwaarts.
    #
    # De embargo en de labelhorizon komen nu uit `conf/validation/default.yaml`
    # en niet uit deze aanroepcode; twee runs met een andere embargo zijn anders
    # niet vergelijkbaar zonder de aanroep ernaast te leggen.
    validation_cfg = load_config(
        _ROOT / "conf" / "validation" / "default.yaml", ValidationConfig)
    importances_list: list[np.ndarray] = []

    for fold in purged_walk_forward(len(X), validation_cfg):
        tr_idx = fold.train_idx
        if len(np.unique(y[tr_idx])) < 2:
            continue
        try:
            cb = CatBoostClassifier(
                iterations=200, depth=4, learning_rate=0.05,
                loss_function="Logloss", verbose=False, random_seed=seed,
                allow_writing_files=False,
            )
            cb.fit(X[tr_idx], y[tr_idx])
            importances_list.append(cb.get_feature_importance())
        except Exception as exc:
            logger.warning("CFI: fold training failed (%s).", exc)

    if not importances_list:
        return {fn: 1.0 for fn in feature_names}

    avg_importance = np.mean(importances_list, axis=0)

    # ── Step 3: Cluster-mean importance ──────────────────────────────────────
    cluster_mean: dict[int, float] = {}
    for c in np.unique(cluster_ids):
        members = np.where(cluster_ids == c)[0]
        cluster_mean[int(c)] = float(np.mean(avg_importance[members]))

    cfi: dict[str, float] = {}
    for i, fn in enumerate(feature_names):
        cfi[fn] = cluster_mean.get(int(cluster_ids[i]), 0.0)
    return cfi


def _raw_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> dict[str, float]:
    """Fallback: plain CatBoost feature importance without clustering."""
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.05,
            loss_function="Logloss", verbose=False, random_seed=seed,
            allow_writing_files=False,
        )
        cb.fit(X, y)
        imp = cb.get_feature_importance()
        return {fn: float(v) for fn, v in zip(feature_names, imp)}
    except Exception as exc:
        logger.warning("Raw importance failed (%s) — returning uniform.", exc)
        return {fn: 1.0 for fn in feature_names}


# =============================================================================
# Per-symbol selection pipeline
# =============================================================================

def run_feature_selection(
    sym: str,
    artefacts_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    min_auc_sfi: float = 0.52,
    min_tstat_mda: float = 2.0,
    cfi_keep_frac: float = 0.70,
    seed: int = 42,
) -> list[str]:
    """Full SFI + MDA + CFI feature selection for one asset-side.

    Parameters
    ----------
    sym :
        Symbol name (for logging).
    artefacts_dir :
        Artefacts root (used to save ``selected_features.json``).
    X :
        Feature matrix (n_events, n_features), float32.
    y :
        Binary labels (n_events,), int.
    feature_names :
        List of feature column names (len == X.shape[1]).
    min_auc_sfi :
        Minimum OOS AUC for a feature to survive SFI.
    min_tstat_mda :
        Minimum t-stat for a feature to survive Causal MDA.
    cfi_keep_frac :
        Fraction of features to keep based on CFI ranking.
    seed :
        Random seed.

    Returns
    -------
    List[str] of selected feature names.
    """
    logger.info("[%s] Feature selection: %d features, %d samples.", sym, len(feature_names), len(y))

    if len(feature_names) == 0 or len(y) < 50:
        logger.warning("[%s] Too few features/samples — keeping all.", sym)
        return feature_names

    if np.unique(y).size < 2:
        logger.warning("[%s] Degenerate labels — keeping all features.", sym)
        return feature_names

    # ── Stage 1: SFI ─────────────────────────────────────────────────────────
    logger.info("[%s] Stage 1: SFI (min_auc=%.3f) ...", sym, min_auc_sfi)
    try:
        sfi_features = rank_features_by_sfi(
            X=X, y=y, feature_names=feature_names,
            min_auc=min_auc_sfi, n_splits=5, seed=seed,
        )
        logger.info("[%s] SFI: %d -> %d features.", sym, len(feature_names), len(sfi_features))
    except Exception as exc:
        logger.warning("[%s] SFI failed (%s) — using all features.", sym, exc)
        sfi_features = feature_names

    if not sfi_features:
        logger.warning("[%s] SFI removed all features — keeping all.", sym)
        return feature_names

    # Restrict X to SFI survivors
    sfi_mask = [fn in set(sfi_features) for fn in feature_names]
    X_sfi = X[:, sfi_mask]
    sfi_names = [fn for fn, keep in zip(feature_names, sfi_mask) if keep]

    # ── Stage 2: Causal MDA ───────────────────────────────────────────────────
    logger.info("[%s] Stage 2: Causal MDA (min_tstat=%.1f) ...", sym, min_tstat_mda)
    try:
        mda_features = filter_by_mda(
            X=X_sfi, y=y, feature_names=sfi_names,
            min_tstat=min_tstat_mda, n_repeats=10, seed=seed,
        )
        logger.info("[%s] MDA: %d -> %d features.", sym, len(sfi_names), len(mda_features))
    except Exception as exc:
        logger.warning("[%s] MDA failed (%s) — using SFI features.", sym, exc)
        mda_features = sfi_names

    if not mda_features:
        logger.warning("[%s] MDA removed all features — keeping SFI features.", sym)
        mda_features = sfi_names

    # Restrict X to MDA survivors
    mda_mask = [fn in set(mda_features) for fn in sfi_names]
    X_mda = X_sfi[:, mda_mask]
    mda_names = [fn for fn, keep in zip(sfi_names, mda_mask) if keep]

    # ── Stage 3: CFI ─────────────────────────────────────────────────────────
    logger.info("[%s] Stage 3: CFI (keep_frac=%.2f) ...", sym, cfi_keep_frac)
    try:
        cfi_scores = _compute_cfi(X_mda, y, mda_names, seed=seed)
        # Keep top cfi_keep_frac by cluster-mean importance
        threshold_count = max(1, int(len(mda_names) * cfi_keep_frac))
        sorted_feats = sorted(cfi_scores, key=cfi_scores.get, reverse=True)  # type: ignore[arg-type]
        cfi_features = sorted_feats[:threshold_count]
        logger.info("[%s] CFI: %d -> %d features.", sym, len(mda_names), len(cfi_features))
    except Exception as exc:
        logger.warning("[%s] CFI failed (%s) — using MDA features.", sym, exc)
        cfi_features = mda_names

    if not cfi_features:
        logger.warning("[%s] CFI removed all features — keeping MDA features.", sym)
        cfi_features = mda_names

    logger.info(
        "[%s] Feature selection complete: %d → %d (%.1f%% reduction).",
        sym, len(feature_names), len(cfi_features),
        100.0 * (1.0 - len(cfi_features) / max(len(feature_names), 1)),
    )
    return list(cfi_features)


# =============================================================================
# Load feature matrix from artefacts
# =============================================================================

def _load_features_and_labels(
    sym: str,
    side: str,
    artefacts_dir: Path,
) -> tuple | None:
    """Load OOS probs and feature matrix from artefacts.

    Returns (X, y, feature_names) or None if artefacts are missing.
    """
    # Load OOS probs (= labels for selection purposes)
    probs_path = artefacts_dir / "oos_probs" / f"{sym}_{side}.parquet"
    if not probs_path.exists():
        logger.warning("[%s/%s] OOS probs not found: %s — run Stage 3 first.", sym, side, probs_path)
        return None

    df_probs = pd.read_parquet(probs_path)
    if "prob" not in df_probs.columns:
        logger.warning("[%s/%s] Column 'prob' missing from OOS probs.", sym, side)
        return None

    # Load feature parquet
    feat_path = artefacts_dir / "features" / f"{sym}_features.parquet"
    if not feat_path.exists():
        logger.warning("[%s/%s] Feature parquet not found: %s", sym, side, feat_path)
        return None

    df_feat = pd.read_parquet(feat_path)
    feat_cols = [c for c in df_feat.columns if c.startswith("feat_")]
    if not feat_cols:
        logger.warning("[%s/%s] No feat_* columns in feature parquet.", sym, side)
        return None

    # Align on common index
    common_idx = df_probs.index.intersection(df_feat.index)
    if len(common_idx) < 50:
        logger.warning("[%s/%s] Only %d common events — insufficient.", sym, side, len(common_idx))
        return None

    X = df_feat.loc[common_idx, feat_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X)

    # Binary labels: prob > 0.5 → positive class (meta-label)
    probs = df_probs.loc[common_idx, "prob"].values
    y = (probs > 0.5).astype(np.int32)

    # Also load label-based y from label parquet if available
    lbl_path = artefacts_dir / "labels" / f"{sym}_{side}_labels.parquet"
    if lbl_path.exists():
        df_lbl = pd.read_parquet(lbl_path)
        if "label" in df_lbl.columns:
            lbl_idx = df_lbl.index.intersection(common_idx)
            if len(lbl_idx) > len(common_idx) * 0.5:
                y = df_lbl.loc[common_idx, "label"].fillna(0).to_numpy(dtype=np.int32)

    return X, y, feat_cols


# =============================================================================
# Main entry
# =============================================================================

@hydra.main(config_path="../conf", config_name="conf_config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Run feature selection for all assets in training_universe."""
    artefacts_dir = Path(OmegaConf.select(cfg, "machine.artefacts_dir", default="artefacts"))
    universe = list(OmegaConf.select(cfg, "training.training_universe", default=[]))
    seed = int(OmegaConf.select(cfg, "training.random_seed", default=42))

    min_auc_sfi    = float(OmegaConf.select(cfg, "feature_selection.min_auc_sfi",    default=0.52))
    min_tstat_mda  = float(OmegaConf.select(cfg, "feature_selection.min_tstat_mda",  default=2.0))
    cfi_keep_frac  = float(OmegaConf.select(cfg, "feature_selection.cfi_keep_frac",  default=0.70))

    logger.info(
        "Feature selection: universe=%s, min_auc_sfi=%.3f, min_tstat_mda=%.1f, cfi_keep_frac=%.2f",
        universe, min_auc_sfi, min_tstat_mda, cfi_keep_frac,
    )

    for sym in universe:
        sym_artefacts = artefacts_dir / sym
        sym_artefacts.mkdir(parents=True, exist_ok=True)
        out_path = sym_artefacts / "selected_features.json"

        # Combine LONG + SHORT events for a joint feature selection
        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        feat_names_union: list[str] | None = None

        for side in ("LONG", "SHORT"):
            result = _load_features_and_labels(sym, side, artefacts_dir)
            if result is None:
                continue
            X, y, feat_cols = result
            all_X.append(X)
            all_y.append(y)
            if feat_names_union is None:
                feat_names_union = feat_cols
            else:
                # Intersect feature names across sides
                feat_names_union = [f for f in feat_names_union if f in feat_cols]

        if not all_X or feat_names_union is None:
            logger.warning("[%s] No artefacts available — skipping.", sym)
            continue

        # Stack and restrict to common feature columns
        X_all = np.vstack([x[:, [feat_cols.index(f) for f in feat_names_union if f in feat_cols]] for x in all_X])
        y_all = np.concatenate(all_y)

        selected = run_feature_selection(
            sym=sym,
            artefacts_dir=artefacts_dir,
            X=X_all,
            y=y_all,
            feature_names=feat_names_union,
            min_auc_sfi=min_auc_sfi,
            min_tstat_mda=min_tstat_mda,
            cfi_keep_frac=cfi_keep_frac,
            seed=seed,
        )

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"symbol": sym, "selected_features": selected, "n_selected": len(selected)}, fh, indent=2)

        logger.info("[%s] Selected %d features -> %s", sym, len(selected), out_path)

    logger.info("Feature selection complete for %d assets.", len(universe))


if __name__ == "__main__":
    main()
