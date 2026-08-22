"""features/cfi.py — Clustered Feature Importance (CFI) + filter.

Extracted from train_regime.py lines 3118-3351.
Bit-identical to the monolith.  Public API:

  compute_clustered_feature_importance(model, X, y, feature_names, ...)
      → (pd.DataFrame of importance scores, linkage_matrix)

  filter_feature_map(cfi_df, original_feat_map, mode, param)
      → filtered feat_map dict {micro: [...], meso: [...], macro: [...]}

Algorithm (AFML ch. 8 — CFI with Structural Break filter):
  1. Spearman corr → Ward linkage → feature clusters.
  2. Chronological regime split (n_regimes=4, purge_bars gap).
  3. Cluster-level permutation importance PER regime.
  4. Structural stability: features predictive in <50% of regimes or with
     negative mean importance → zero score (Chow-test proxy).
  5. Sharpe-like penalty on variance across regimes.

Strangler-fig: once train_regime.py imports from here and the equivalence
test passes, delete lines 3118-3351 in train_regime.py.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss

from ..utils.failfast import DataContractError

logger = logging.getLogger(__name__)


# =============================================================================
# CLUSTERED FEATURE IMPORTANCE
# =============================================================================

def compute_clustered_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    cluster_threshold: float = 0.5,
    scoring_metric: str = "log_loss",
    n_regimes: int = 4,
    purge_bars: int = 50,
    sample_weight: np.ndarray | None = None,
    n_perms: int = 1,
    rng_seed: int | None = None,
    train_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Clustered Feature Importance with structural stability tests.

    Splits the dataset into temporal regimes and measures each cluster's
    importance across all regimes.  Features predictive in a minority of
    regimes are penalised to zero (structural break filter).

    Parameters
    ----------
    model             : Fitted classifier with predict / predict_proba.
    X                 : Feature matrix (n_samples, n_features), float64.
    y                 : Label vector (n_samples,), int.
    feature_names     : Column names matching X.shape[1].
    cluster_threshold : Ward linkage distance threshold for fcluster.
    scoring_metric    : 'log_loss' (default) or 'accuracy'.
    n_regimes         : Number of chronological regimes for stability test.
    purge_bars        : Gap between consecutive regimes to prevent leakage.
    sample_weight     : AFML §8.4 sample uniqueness weights of length
                        ``n_samples``. Used in the scoring metric so
                        overlapping observations don't dominate. ``None``
                        keeps uniform weights (legacy behaviour).
                        AUDIT C-1.
    n_perms           : Number of permutation replicates per cluster
                        (AUDIT C-2). Importance is averaged across all
                        replicates to dampen single-shuffle variance.
                        Default 1 = legacy behaviour; recommended ≥ 20 in
                        production runs.
    rng_seed          : Optional deterministic seed for the permutation
                        sequence. ``None`` uses a fresh RNG per call.
    train_indices     : CHIEF AUDIT 2026-05-23 (P-8): integer indices of the
                        rows that form the TRAIN fold of the current CPCV
                        split.  When supplied, the Spearman correlation that
                        drives the feature clustering is computed on
                        ``X[train_indices]`` only — preventing test-fold
                        information from leaking into the cluster topology.
                        Default ``None`` preserves the legacy global-fit
                        behaviour for backward compatibility.

    Returns
    -------
    imp_df      : pd.DataFrame indexed by feature name.  Columns:
                  cluster_id, importance (adjusted), raw_mean_imp,
                  stability_ratio, n_members.
    link_mat    : SciPy linkage matrix (for dendrogram visualisation).
    """
    rng = np.random.default_rng(rng_seed)
    n_perms = int(max(1, n_perms))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape[0] != X.shape[0]:
            raise ValueError(
                f"sample_weight length {sample_weight.shape[0]} "
                f"!= X rows {X.shape[0]}"
            )
    n_features = X.shape[1]

    # ── 1. Validate feature names ─────────────────────────────────────────────
    if len(feature_names) != n_features:
        logger.warning(
            "CFI: feature_names length %d ≠ X.shape[1] %d — using dummies.",
            len(feature_names), n_features,
        )
        feature_names = [f"feat_{i}" for i in range(n_features)]

    # ── 2. Spearman correlation → Ward linkage → clusters ─────────────────────
    # CHIEF AUDIT 2026-05-23 (P-8): when ``train_indices`` is provided the
    # clustering correlation matrix uses ONLY training-fold rows.  This is the
    # causally correct variant for CPCV (cluster topology computed on the same
    # data the model was trained on, evaluated on held-out folds).
    if train_indices is not None:
        train_idx_arr = np.asarray(train_indices, dtype=np.int64)
        X_for_clustering = X[train_idx_arr]
        if X_for_clustering.shape[0] < 5:
            logger.warning(
                "CFI: train_indices yielded %d rows (< 5) — falling back to "
                "full-X clustering.", X_for_clustering.shape[0],
            )
            X_for_clustering = X
    else:
        X_for_clustering = X
    corr_matrix, _ = spearmanr(X_for_clustering, axis=0)
    corr_matrix = np.nan_to_num(np.asarray(corr_matrix), nan=0.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    dist_matrix = np.sqrt(2.0 * (1.0 - corr_matrix))
    dist_vec    = squareform(dist_matrix, checks=False)
    link_mat    = hierarchy.linkage(dist_vec, method="ward")
    cluster_ids = hierarchy.fcluster(
        link_mat, t=cluster_threshold, criterion="distance"
    )

    clusters: dict[int, list[int]] = {}
    for feat_idx, clust_id in enumerate(cluster_ids):
        clusters.setdefault(int(clust_id), []).append(feat_idx)

    logger.info(
        "CFI: %d clusters from %d features (threshold=%.2f).",
        len(clusters), n_features, cluster_threshold,
    )

    # ── 3. Chronological regime split with purging ────────────────────────────
    n_samples      = len(X)
    raw_regime_size = n_samples // n_regimes
    regimes: list[tuple[int, int]] = []

    for i in range(n_regimes):
        start = i * raw_regime_size
        end   = n_samples if i == n_regimes - 1 else (i + 1) * raw_regime_size
        safe_start = start + purge_bars if i > 0 else start
        if safe_start < end:
            regimes.append((safe_start, end))

    logger.info(
        "CFI: Structural break tests across %d purged temporal regimes.",
        len(regimes),
    )

    # ── 4. Baseline score per regime (weighted, AUDIT C-1) ────────────────────
    _metric = scoring_metric  # may be updated to 'accuracy' if model has no proba

    def _score(y_true: np.ndarray, X_in: np.ndarray, w: np.ndarray | None) -> float:
        """Compute regime score; falls back to accuracy if log_loss fails."""
        nonlocal _metric
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_in)
            if _metric == "log_loss":
                # Phase 0: `except ValueError: _metric = "accuracy"` wisselde de
                # scoringsmetriek permanent om, midden in de
                # feature-importance-berekening. De importances voor en na de
                # wissel zijn dan niet vergelijkbaar, terwijl het rapport een
                # enkele metriek noemt.
                try:
                    return float(log_loss(y_true, proba, sample_weight=w))
                except ValueError as exc:
                    raise DataContractError(
                        "log_loss faalde tijdens clustered feature importance; "
                        "er wordt niet stilzwijgend op accuracy overgeschakeld."
                    ) from exc
            return float(
                accuracy_score(y_true, model.predict(X_in), sample_weight=w)
            )
        return float(
            accuracy_score(y_true, model.predict(X_in), sample_weight=w)
        )

    base_scores: list[float] = []
    regime_weights: list[np.ndarray | None] = []
    for r_start, r_end in regimes:
        X_reg = X[r_start:r_end]
        y_reg = y[r_start:r_end]
        w_reg = (
            sample_weight[r_start:r_end] if sample_weight is not None else None
        )
        regime_weights.append(w_reg)
        base_scores.append(_score(y_reg, X_reg, w_reg))

    # ── 5. Clustered permutation (n_perms replicates, AUDIT C-2) ──────────────
    importances: dict[str, dict[str, Any]] = {}
    X_shuffled = X.copy()

    for clust_id, feat_indices in clusters.items():
        originals = [X_shuffled[:, i].copy() for i in feat_indices]

        # ── n_perms replicates: average regime-impact across permutations.
        per_replicate_regime_impacts = np.zeros((n_perms, len(regimes)), dtype=np.float64)
        for replicate in range(n_perms):
            perm_idx = rng.permutation(n_samples)
            for i in feat_indices:
                X_shuffled[:, i] = X[perm_idx, i]   # source = original X to keep
                                                    # replicates independent

            for r_idx, (r_start, r_end) in enumerate(regimes):
                X_reg_shuf = X_shuffled[r_start:r_end]
                y_reg      = y[r_start:r_end]
                base_s     = base_scores[r_idx]
                new_s      = _score(y_reg, X_reg_shuf, regime_weights[r_idx])

                # log_loss: higher = worse (positive impact = shuffle degrades)
                imp = (
                    (new_s - base_s)
                    if _metric == "log_loss"
                    else (base_s - new_s)
                )
                per_replicate_regime_impacts[replicate, r_idx] = imp

        # Average across replicates BEFORE regime aggregation: this is the
        # variance-reduced per-regime impact.
        regime_impacts = per_replicate_regime_impacts.mean(axis=0).tolist()

        # ── Structural stability (Chow-test proxy) ───────────────────────────
        mean_imp = float(np.mean(regime_impacts))
        std_imp  = float(np.std(regime_impacts))

        positive_regimes  = sum(1 for imp in regime_impacts if imp > 0)
        stability_ratio   = positive_regimes / max(len(regimes), 1)

        if stability_ratio <= 0.5 or mean_imp <= 0:
            adjusted_imp = 0.0
        else:
            adjusted_imp = max(0.0, mean_imp - 0.5 * std_imp)

        # Replicate-level standard error (informative for diagnostics).
        replicate_mean_impacts = per_replicate_regime_impacts.mean(axis=1)
        replicate_se = float(replicate_mean_impacts.std(ddof=1)) if n_perms > 1 else float("nan")

        for i in feat_indices:
            importances[feature_names[i]] = {
                "cluster_id":      int(clust_id),
                "importance":      adjusted_imp,
                "raw_mean_imp":    mean_imp,
                "stability_ratio": stability_ratio,
                "n_members":       len(feat_indices),
                "n_perms":         n_perms,
                "replicate_se":    replicate_se,
            }

        # Restore
        for i, org in zip(feat_indices, originals):
            X_shuffled[:, i] = org

    # ── 6. Results ────────────────────────────────────────────────────────────
    imp_df = pd.DataFrame.from_dict(importances, orient="index")
    imp_df = imp_df.sort_values("importance", ascending=False)

    n_destroyed = len(imp_df[(imp_df["raw_mean_imp"] > 0) & (imp_df["importance"] == 0)])
    if n_destroyed > 0:
        logger.info(
            "CFI Structural Break Filter: destroyed %d features due to "
            "cross-regime instability.",
            n_destroyed,
        )

    return imp_df, link_mat


# =============================================================================
# FEATURE MAP FILTER
# =============================================================================

def filter_feature_map(
    cfi_df: pd.DataFrame,
    original_feat_map: dict[str, list[str]],
    mode: str = "cumulative",
    param: float = 0.95,
) -> dict[str, list[str]]:
    """Select best features based on cumulative importance or top-N.

    Parameters
    ----------
    cfi_df             : DataFrame with CFI scores (output of
                         compute_clustered_feature_importance).
    original_feat_map  : Original mapping ``{micro: [...], meso: [...],
                         macro: [...]}``.
    mode               : ``'cumulative'`` (keep X% of total signal) or
                         ``'top_n'`` (keep top X features).
    param              : Threshold: 0.95 → 95% signal retention;
                         or integer-like N for top_n mode.

    Returns
    -------
    Dict with keys ``micro``, ``meso``, ``macro`` each containing the
    filtered feature name list.
    """
    cfi_sorted = cfi_df.sort_values("importance", ascending=False).copy()
    cfi_sorted = cfi_sorted[cfi_sorted["importance"] > 0]

    selected_features: set = set()

    if mode == "top_n":
        top_n = int(param)
        selected_features = set(cfi_sorted.head(top_n).index)
        logger.info("CFI filter: TOP %d features selected.", top_n)

    elif mode == "cumulative":
        total_imp = cfi_sorted["importance"].sum()
        if total_imp <= 0:
            logger.warning(
                "CFI filter: total importance == 0 — returning all features."
            )
            return original_feat_map

        cfi_sorted["cum_imp"]  = cfi_sorted["importance"].cumsum()
        cfi_sorted["cum_perc"] = cfi_sorted["cum_imp"] / total_imp

        cutoff_mask = cfi_sorted["cum_perc"] <= param
        last_idx    = int(cutoff_mask.sum())
        # Always keep at least top 3 to avoid over-aggressive filtering
        if last_idx < 3:
            last_idx = min(3, len(cfi_sorted))

        selected_features = set(cfi_sorted.iloc[:last_idx].index)
        explained = (
            cfi_sorted.iloc[:last_idx]["importance"].sum() / total_imp
        )
        logger.info(
            "CFI filter: CUMULATIVE (%.0f%%). Selected %d features "
            "explaining %.1f%% of importance.",
            param * 100, len(selected_features), explained * 100,
        )
    else:
        raise ValueError(
            f"Unknown filter mode '{mode}'. Use 'cumulative' or 'top_n'."
        )

    new_map: dict[str, list[str]] = {"micro": [], "meso": [], "macro": []}
    for tier in ("micro", "meso", "macro"):
        for f in original_feat_map.get(tier, []):
            if f in selected_features:
                new_map[tier].append(f)

    return new_map


# Backward-compat short alias used by features/__init__.py and downstream code.
clustered_feature_importance = compute_clustered_feature_importance
