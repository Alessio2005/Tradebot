"""sfi.py — Single Feature Importance (AFML §8.1).

Evalueer de out-of-sample AUC van elke feature individueel.
Geeft een non-lineaire baseline voor feature ranking.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Phase 0: catboost is een harde dependency (pyproject.toml). De try/except
# zette _CB_OK=False, waarna single_feature_importance stilzwijgend een lege
# importance-tabel opleverde en elke feature dus even belangrijk leek.
from catboost import CatBoostClassifier, Pool

logger = logging.getLogger(__name__)


def _purged_timeseries_splits(
    n_samples: int,
    n_splits: int,
    purge_bars: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """CHIEF AUDIT 2026-05-23 (P-11): Manual purged time-series CV.

    ``sklearn.model_selection.TimeSeriesSplit`` has no purge/embargo so train
    samples ending at ``fold_end - 1`` and test samples starting at ``fold_end``
    overlap in feature/label memory (AFML §7.2).  This generator returns
    train/test index arrays with a configurable gap of ``purge_bars`` between
    the end of the train fold and the start of the test fold.

    Layout (one fold):
        train = [0, fold_end)
        gap   = [fold_end, fold_end + purge_bars)   ← discarded
        test  = [fold_end + purge_bars,
                 fold_end + purge_bars + test_size)
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if n_samples < (n_splits + 1) * 2:
        return splits
    # Equal-sized test folds across the held-out tail.
    fold_size = max(1, n_samples // (n_splits + 1))
    for k in range(1, n_splits + 1):
        fold_end = k * fold_size
        test_start = fold_end + purge_bars
        test_end = min(test_start + fold_size, n_samples)
        if test_start >= n_samples or test_end - test_start < 2:
            continue
        train_idx = np.arange(0, fold_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue
        splits.append((train_idx, test_idx))
    return splits


def single_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: int = 5,
    random_seed: int = 42,
    purge_bars: int = 10,
) -> pd.Series:
    """Bereken OOS AUC voor elke feature afzonderlijk.

    Traint een simpel CatBoost model (depth=3, 100 iteraties) per feature
    met een tijdsgebaseerde train/test split.

    Args:
        X:          Feature DataFrame (index = timestamps, kolommen = features).
        y:          Binaire labels (n,).
        cv_splits:  Aantal tijdgebaseerde splits voor OOS evaluatie.
        random_seed: Reproduceerbaarheids-seed.
        purge_bars: CHIEF AUDIT 2026-05-23 (P-11): aantal bars purge tussen
                    train- en test-fold (default 10).  Aanbevolen:
                    ``max(10, geschatte_horizon)`` om memory-overlap te elimineren.

    Returns:
        pd.Series met AUC per feature (index = feature namen), gesorteerd aflopend.
    """
    from sklearn.metrics import roc_auc_score

    # CHIEF AUDIT 2026-05-23 (P-11): vervang sklearn TimeSeriesSplit (geen
    # purge/embargo) door de manuele variant hierboven die wel een purge
    # gat aanhoudt tussen train en test, conform AFML §7.2.
    n_samples = len(X)
    splits = _purged_timeseries_splits(n_samples, cv_splits, max(0, int(purge_bars)))
    feature_aucs: dict[str, float] = {}

    for feat in X.columns:
        x_feat = X[[feat]].values
        aucs = []
        for train_idx, test_idx in splits:
            X_tr, X_te = x_feat[train_idx], x_feat[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # Vervang NaN met mediaan (feature-specifiek)
            col_median = np.nanmedian(X_tr)
            X_tr = np.where(np.isnan(X_tr), col_median, X_tr)
            X_te = np.where(np.isnan(X_te), col_median, X_te)

            model = CatBoostClassifier(
                iterations=100, depth=3, random_seed=random_seed,
                verbose=False, allow_writing_files=False,
                loss_function="Logloss",
            )
            model.fit(Pool(X_tr, label=y_tr))
            probs = model.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, probs))

        feature_aucs[feat] = float(np.mean(aucs)) if aucs else 0.5

    result = pd.Series(feature_aucs).sort_values(ascending=False)
    logger.info("SFI voltooid: top-5 features = %s", result.head(5).to_dict())
    return result


def rank_features_by_sfi(
    X: pd.DataFrame,
    y: np.ndarray,
    min_auc: float = 0.52,
    **kwargs,
) -> list[str]:
    """Retourneer features met SFI-AUC > min_auc, gesorteerd aflopend.

    Args:
        min_auc: Minimum AUC-drempel (0.5 = random; 0.52 = zwak signaal).
    """
    sfi = single_feature_importance(X, y, **kwargs)
    selected = sfi[sfi > min_auc].index.tolist()
    logger.info("SFI selectie: %d/%d features boven %.2f AUC", len(selected), len(sfi), min_auc)
    return selected
