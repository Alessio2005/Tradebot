"""meta_train.py — Stage B Meta-Labeling Judge trainer (v3 T0.3).

Traint een Secondary Judge CatBoost model per (symbol, side) op basis van:
  - OOS Primary Scout probabilities (anti-lekkage: uitsluitend OOS probs)
  - Originele features + tijdsfeatures
  - Meta-labels: y=1 als (bruto_return * richting - cost_bps/10000) > 0

Referentie: AFML §3.7 (Meta-Labeling), CHIEF_MASTER_PLAN v3 §2 T0.3.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Phase 0: catboost is een harde dependency. De except-tak schakelde de
# Judge-training in zijn geheel uit met alleen een warning.
from catboost import CatBoostClassifier, Pool


def build_judge_features(
    X_primary: pd.DataFrame,
    oos_probs: pd.Series,
) -> pd.DataFrame:
    """Bouw Judge feature-matrix: originele features + P_primary + tijdsfeatures.

    P_primary is al OOS — geen lekkage. Tijdsfeatures zijn cyclisch gecodeerd
    (stationair).

    CPCV-SAFE: Label-based .loc op een non-unique DatetimeIndex (CPCV hergebruikt
    timestamps over folds) geeft inconsistente resultaten:
      - X_primary.loc[unique_ts] expandeert naar N_oos rijen (alle fold-kopieën)
      - oos_probs.loc[unique_ts] geeft slechts N_unique waarden terug
    Fix: positional .values/.iloc — inputs zijn al uitgelijnd vanuit train_cpcv.py.
    """
    n = min(len(X_primary), len(oos_probs))
    if n < 20:
        raise ValueError(
            f"Onvoldoende data voor Judge feature build (X={len(X_primary)}, "
            f"probs={len(oos_probs)})."
        )

    X = X_primary.iloc[:n].copy()
    X["prob_primary"] = oos_probs.values[:n]

    idx = X.index
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["dow_sin"]  = np.sin(2 * np.pi * idx.dayofweek / 7)
    X["dow_cos"]  = np.cos(2 * np.pi * idx.dayofweek / 7)

    return X


def make_judge_labels(
    gross_returns: pd.Series,
    directions: pd.Series,
    cost_bps: float = 2.0,
) -> pd.Series:
    """Bouw binaire Judge-labels: y=1 als netto alpha > drempel.

    y_meta = 1  als  r_t * sgn(f_t) - cost_bps/10000 > 0
    y_meta = 0  anders

    Args:
        gross_returns: Bruto returns per event (index = timestamps).
        directions:    Signaalrichting {-1, +1} per event (index = timestamps).
        cost_bps:      Minimum netto winst drempel in basispunten.

    CPCV-SAFE: positional alignment (zie build_judge_features).
    """
    n = min(len(gross_returns), len(directions))
    r = gross_returns.values[:n]
    d = directions.values[:n]
    threshold = cost_bps / 10_000.0
    net_alpha = r * d - threshold
    y_arr = (net_alpha > 0).astype(np.int32)
    y_meta = pd.Series(y_arr, index=gross_returns.index[:n])
    logger.info(
        "Judge labels: n=%d, pos=%.1f%% (drempel=%.1fbps)",
        len(y_meta), 100.0 * y_arr.mean(), cost_bps,
    )
    return y_meta


def train_judge(
    sym: str,
    side: str,
    X_primary: pd.DataFrame,
    oos_probs: pd.Series,
    gross_returns: pd.Series,
    directions: pd.Series,
    artefacts_dir: Path,
    cost_bps: float = 2.0,
    judge_iterations: int = 300,
    judge_depth: int = 5,
    judge_l2: float = 5.0,
    random_seed: int = 42,
) -> Optional[pd.Series]:
    """Train Secondary Judge en persisteer model + OOS probs.

    Returns:
        OOS Judge probabilities (pd.Series), of None bij falen.
    """
    if not _CATBOOST_OK:
        logger.error("CatBoost niet beschikbaar — Judge overgeslagen.")
        return None

    try:
        X_judge = build_judge_features(X_primary, oos_probs)
    except ValueError as exc:
        logger.error("[%s/%s] Judge feature build mislukt: %s", sym, side, exc)
        return None

    y_judge = make_judge_labels(gross_returns, directions, cost_bps)

    # Positional alignment — both X_judge and y_judge use the same CPCV-OOS index.
    n_j = min(len(X_judge), len(y_judge))
    if n_j < 30:
        logger.warning("[%s/%s] Onvoldoende Judge events (%d).", sym, side, n_j)
        return None

    X_j = X_judge.values[:n_j].astype(np.float32)
    y_j = y_judge.values[:n_j].astype(np.int32)

    # Verwijder NaN-rijen
    nan_mask = np.isnan(X_j).any(axis=1)
    X_j = X_j[~nan_mask]
    y_j = y_j[~nan_mask]
    clean_idx = X_judge.index[:n_j][~nan_mask]

    if len(y_j) < 30:
        logger.warning("[%s/%s] Te weinig schone events na NaN-filter (%d).", sym, side, len(y_j))
        return None

    pos_rate = y_j.mean()
    if not (0.05 <= pos_rate <= 0.95):
        logger.warning("[%s/%s] Scheef label-distributie (pos_rate=%.2f).", sym, side, pos_rate)

    model = CatBoostClassifier(
        iterations=judge_iterations,
        depth=judge_depth,
        l2_leaf_reg=judge_l2,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(Pool(X_j, label=y_j))

    model_dir = artefacts_dir / "judge_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{sym}_{side}_judge.cbm"
    model.save_model(str(model_path))
    logger.info("[%s/%s] Judge opgeslagen: %s", sym, side, model_path)

    probs_arr = model.predict_proba(X_j)[:, 1]
    oos_judge = pd.Series(probs_arr, index=clean_idx, name="prob_judge")

    probs_dir = artefacts_dir / "oos_probs"
    probs_dir.mkdir(parents=True, exist_ok=True)
    out_path = probs_dir / f"{sym}_{side}_judge.parquet"
    oos_judge.to_frame().to_parquet(out_path)
    logger.info("[%s/%s] Judge OOS probs: %s (n=%d, pos=%.1f%%)", sym, side, out_path, len(oos_judge), 100 * pos_rate)

    return oos_judge
