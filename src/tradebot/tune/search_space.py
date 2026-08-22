"""search_space.py — CatBoost + bandit hyperparameter search space for Optuna.

Extracted from train_regime.py lines 1673-1708 (inside optuna_objective_binary).

Separating the search space definition from the objective body allows:
  1. Independent unit-tests (no CPCV loop needed).
  2. Easy comparison / extension of the search space without touching
     the 770-line objective function.
  3. Reproducible space definition: identical inputs + seed → identical params.

Design notes:
  • CatBoost params follow the SB-CLASS-WEIGHTS-FIX rationale: auto_class_weights
    is intentionally omitted.  Class imbalance is handled via per-fold sample
    weights, not CatBoost's internal balancer, to avoid double-counting with
    the uniqueness weights (AUDIT-FIX Round 3).
  • GAMMA-FIX: bandit_gamma searched on log-scale (0.970 – 0.9995) to properly
    cover the half-life range [≈23, ≈1385] trades.  The legacy default of
    gamma=0.995 (HL≈138) was too slow for crypto regime detection.
"""
from __future__ import annotations

from typing import Any

import optuna


def sample_catboost_params(
    trial: optuna.Trial,
    horizon_keys: list[Any],
    min_conf_high: float = 0.65,
) -> tuple[int, float, float, dict[str, Any]]:
    """Sample a complete CatBoost + bandit configuration from the trial.

    Parameters
    ----------
    trial:
        Active Optuna trial.
    horizon_keys:
        List of valid label-horizon values (keys of target_store dict in the
        objective).  Horizon is sampled categorically over this list.
    min_conf_high:
        Upper bound for min_conf search.  LONG: 0.65 (default).
        SHORT: typically 0.42 (from conf/symbols/{sym}.yaml:min_conf_short_max).
        SHORT models have average calibrated probabilities ~0.20-0.30, so the
        standard 0.65 upper bound leads to zero signals and silent under-trading.

    Returns
    -------
    selected_h: int
        The sampled label horizon (bars).
    min_conf: float
        Minimum confidence threshold for trade entry [0.35, min_conf_high].
    bandit_gamma: float
        Contextual bandit decay factor, log-scale [0.970, 0.9995].
    cb_params: dict
        CatBoost model kwargs ready for ``catboost.CatBoostClassifier(**cb_params)``.
    """
    selected_h: int    = trial.suggest_categorical("horizon", horizon_keys)
    # SHORT-CONF-FIX: per-asset bovengrens (0.42 SHORT vs 0.65 LONG)
    min_conf: float    = trial.suggest_float("min_conf", 0.35, min_conf_high)

    # GAMMA-FIX: log-scale sweep for correct half-life coverage.
    bandit_gamma: float = trial.suggest_float("bandit_gamma", 0.970, 0.9995, log=True)

    cb_params: dict[str, Any] = {
        "iterations":          trial.suggest_int("iterations", 600, 1500),
        "learning_rate":       trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
        "depth":               trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 2.0, 15.0),
        "random_strength":     trial.suggest_float("random_strength", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 50, 200),
        # bandit_gamma stored for downstream ContextualBanditEnsemble construction.
        "bandit_gamma":        bandit_gamma,
        # Fixed architectural choices — not tuned.
        "loss_function":       "Logloss",
        "eval_metric":         "Logloss",
        "task_type":           "CPU",
        "verbose":             False,
        "allow_writing_files": False,
        "border_count":        128,
        # auto_class_weights intentionally omitted — see SB-CLASS-WEIGHTS-FIX.
    }

    return selected_h, min_conf, bandit_gamma, cb_params


# Backward-compat alias (tune/__init__.py public API)
build_search_space = sample_catboost_params
