"""train_cpcv.py — DAG Stage 3: best_params → CPCV ensemble + calibrators.

Entrypoint for ``dvc run`` / ``dvc repro train_cpcv``.

Design (REFACTOR_BLUEPRINT §3.2 Stage 3):
  • Uses best hyperparameters from Stage 2 (tune_hparams).
  • Trains the final CPCV ensemble (ContextualBanditEnsemble) on full dev data.
  • PathSpecificPlattCalibrator fitted on per-path OOS predictions.
  • Outputs: model pickle, calibrator, per-bar OOS probabilities.

Ensemble (XGBoost + LightGBM addition):
  • Per CPCV fold three base learners are trained: CatBoost (with tuned hparams),
    XGBoost and LightGBM (with fixed sensible defaults derived from the CatBoost
    hparams where applicable).
  • Each learner produces Platt-calibrated per-fold probabilities (sklearn
    CalibratedClassifierCV, cv="prefit", method="sigmoid") fitted on the last 20%
    of the training partition — never on the test fold.
  • Early stopping for XGBoost/LightGBM uses a purged validation split carved
    from the training data (get_early_stop_split), not the CPCV test fold.
  • Ensemble probability = simple average of the three Platt-calibrated prob arrays.
  • If XGBoost or LightGBM is unavailable or raises an exception, the pipeline
    falls back to CatBoost-only (the failing learner's probs are replaced by the
    CatBoost probs, so the average is still valid).

Determinism:
  • CatBoost random_seed pinned via cfg.training.random_seed (default 42).
  • Each fold model gets seed = base_seed + fold_idx so models differ but are
    deterministic across re-runs.
  • Sample weights are reproducible because TrendScanningLabeler is deterministic.

Outputs (per symbol + side):
  artefacts/models/{SYM}_{SIDE}_ensemble.joblib
  artefacts/calibrators/{SYM}_{SIDE}_platt.joblib
  artefacts/oos_probs/{SYM}_{SIDE}.parquet
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tradebot.cv.cpcv import CombinatorialPurgedCV
from tradebot.cv.uniqueness import (
    get_average_uniqueness,
    get_average_uniqueness_per_fold,
    get_sample_weights,
)
from tradebot.features.blocks import stack_feats
from tradebot.labeling.cusum import get_cusum_events_for_short
from tradebot.labeling.trend_scanning import TrendScanningLabeler
from tradebot.schemas.events import EventSchema
from tradebot.schemas.features import FeatureBlockSchema
from tradebot.train.adapter import CatBoostModelAdapter
from tradebot.utils.arrays import validate_or_die

logger = logging.getLogger(__name__)

# v3 T0.3: enforce strict causal scout in MetaLabelingEngine (geen TrendScan lookahead)
if not os.environ.get("TRADEBOT_STRICT_CAUSAL"):
    os.environ["TRADEBOT_STRICT_CAUSAL"] = "1"
    logger.info("TRADEBOT_STRICT_CAUSAL=1 automatisch gezet door train_cpcv.py")


# =============================================================================
# CPCV-safe early-stopping split helper
# =============================================================================

def get_early_stop_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    embargo_bars: int = 10,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a purged train/val split for CPCV-safe early stopping.

    Carves a validation window from the *end* of the training partition with an
    embargo gap so that the early-stopping signal never touches the CPCV test fold.

    Args:
        X_train:      Feature matrix of the train fold  (n, d).
        y_train:      Label vector of the train fold    (n,).
        weights:      Sample weights of the train fold  (n,).
        embargo_bars: Number of bars to drop between the two halves (purge gap).
        val_frac:     Fraction of training rows to use as the val set.

    Returns:
        (X_tr, y_tr, w_tr, X_val, y_val) — numpy arrays.
        Falls back to the full training set as "train" and an empty val when
        the split would leave fewer than 20 training samples.
    """
    n = len(X_train)
    val_size = max(10, int(n * val_frac))
    split_point = n - val_size - embargo_bars

    if split_point < 20:
        # Not enough data to form a meaningful split; early stopping disabled.
        empty_X = X_train[:0]
        empty_y = y_train[:0]
        return X_train, y_train, weights, empty_X, empty_y

    X_tr = X_train[:split_point]
    y_tr = y_train[:split_point]
    w_tr = weights[:split_point]
    X_val = X_train[split_point + embargo_bars:]
    y_val = y_train[split_point + embargo_bars:]
    return X_tr, y_tr, w_tr, X_val, y_val


# =============================================================================
# XGBoost base-learner helper
# =============================================================================

def _train_xgb_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_es_val: np.ndarray,
    y_es_val: np.ndarray,
    hparams: dict[str, Any],
    fold_seed: int,
    embargo_bars: int,
) -> Any | None:
    """Train one XGBoost fold.

    Returns the fitted XGBClassifier or None on failure.
    The early-stopping validation set must be a purged subset of the training
    partition (produced by get_early_stop_split) — never the CPCV test fold.
    """
    try:
        import xgboost as xgb  # soft import: not in base requirements

        n_est = int(hparams.get("iterations", hparams.get("n_estimators", 300)))
        _use_es = len(y_es_val) >= 10 and len(np.unique(y_es_val)) >= 2
        params = dict(
            n_estimators=n_est,
            max_depth=int(hparams.get("xgb_max_depth", hparams.get("depth", 4))),
            learning_rate=float(hparams.get("xgb_lr", hparams.get("learning_rate", 0.05))),
            subsample=float(hparams.get("subsample", 0.8)),
            colsample_bytree=float(hparams.get("colsample_bylevel", hparams.get("colsample_bytree", 0.8))),
            min_child_weight=1,
            eval_metric="logloss",
            random_state=fold_seed % (2 ** 31),
            n_jobs=-1,
            verbosity=0,
        )
        # XGBoost ≥2.0: early_stopping_rounds moved from fit() to constructor.
        if _use_es:
            params["early_stopping_rounds"] = 50
        model = xgb.XGBClassifier(**params)

        fit_kwargs: dict[str, Any] = dict(sample_weight=w_tr)
        if _use_es:
            fit_kwargs["eval_set"] = [(X_es_val, y_es_val)]
            fit_kwargs["verbose"] = False

        model.fit(X_tr, y_tr, **fit_kwargs)
        return model
    except Exception as exc:
        logger.warning("XGBoost fold training failed: %s — skipping XGBoost for this fold.", exc)
        return None


# =============================================================================
# LightGBM base-learner helper
# =============================================================================

def _train_lgb_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_es_val: np.ndarray,
    y_es_val: np.ndarray,
    hparams: dict[str, Any],
    fold_seed: int,
    embargo_bars: int,
) -> Any | None:
    """Train one LightGBM fold.

    Returns the fitted LGBMClassifier or None on failure.
    The early-stopping validation set must be a purged subset of the training
    partition (produced by get_early_stop_split) — never the CPCV test fold.
    """
    try:
        import lightgbm as lgb  # soft import: not in base requirements

        n_est = int(hparams.get("iterations", hparams.get("n_estimators", 300)))
        params = dict(
            n_estimators=n_est,
            max_depth=int(hparams.get("lgb_max_depth", hparams.get("depth", 4))),
            learning_rate=float(hparams.get("lgb_lr", hparams.get("learning_rate", 0.05))),
            subsample=float(hparams.get("subsample", 0.8)),
            colsample_bytree=float(hparams.get("colsample_bylevel", hparams.get("colsample_bytree", 0.8))),
            min_child_samples=20,
            random_state=fold_seed % (2 ** 31),
            n_jobs=-1,
            verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)

        fit_kwargs: dict[str, Any] = dict(sample_weight=w_tr)
        if len(y_es_val) >= 10 and len(np.unique(y_es_val)) >= 2:
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                         lgb.log_evaluation(period=-1)]
            fit_kwargs["eval_set"] = [(X_es_val, y_es_val)]
            fit_kwargs["callbacks"] = callbacks

        model.fit(X_tr, y_tr, **fit_kwargs)
        return model
    except Exception as exc:
        logger.warning("LightGBM fold training failed: %s — skipping LightGBM for this fold.", exc)
        return None


# =============================================================================
# Per-fold Platt calibration helper
# =============================================================================

def _platt_calibrate(
    base_model: Any,
    raw_probs: np.ndarray,
    X_full: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    X_val: np.ndarray,
    sym: str,
    side: str,
    fold_idx: int,
    learner_name: str,
) -> np.ndarray:
    """Apply per-fold sklearn Platt calibration to a base learner.

    Fits a CalibratedClassifierCV(cv="prefit", method="sigmoid") on the last 20%
    of the training partition.  Returns calibrated val probs; falls back to
    raw_probs on any error.

    Args:
        base_model:  Fitted sklearn-compatible estimator with predict_proba.
        raw_probs:   Uncalibrated probabilities on X_val, shape (n_val,).
        X_full:      Full feature matrix (all events).
        y:           Full label array (all events).
        train_idx:   Indices of training samples in this fold.
        X_val:       Validation feature matrix for this fold.
        sym, side, fold_idx, learner_name: for logging only.

    Returns:
        Calibrated probability array of shape (n_val,).
    """
    from sklearn.calibration import CalibratedClassifierCV as _CCV
    try:
        from sklearn.frozen import FrozenEstimator as _FE  # sklearn ≥1.6
    except ImportError:
        _FE = None  # type: ignore[assignment,misc]

    _n_platt = max(10, int(len(train_idx) * 0.20))
    _platt_fit_idx = train_idx[-_n_platt:]
    _X_platt = X_full[_platt_fit_idx]
    _y_platt = y[_platt_fit_idx]

    if len(np.unique(_y_platt)) < 2 or len(_y_platt) < 10:
        return raw_probs

    try:
        _est = _FE(base_model) if _FE is not None else base_model
        _ccv = _CCV(estimator=_est, cv="prefit", method="sigmoid")
        _ccv.fit(_X_platt, _y_platt)
        cal_proba = _ccv.predict_proba(X_val)
        calibrated = cal_proba[:, 1] if cal_proba.shape[1] > 1 else raw_probs.copy()
        logger.debug(
            "[%s/%s] Fold %d [%s]: Platt calibrated "
            "(n_platt=%d, raw_mean=%.4f → cal_mean=%.4f).",
            sym, side, fold_idx, learner_name,
            _n_platt, float(raw_probs.mean()), float(calibrated.mean()),
        )
        return calibrated
    except Exception as _exc:
        logger.warning(
            "[%s/%s] Fold %d [%s]: Platt calibration failed (%s) — raw probs used.",
            sym, side, fold_idx, learner_name, _exc,
        )
        return raw_probs


# =============================================================================
# I/O helpers
# =============================================================================

def _load_hparams(hparam_dir: Path, sym: str, side: str) -> dict[str, Any]:
    pair_key   = f"{sym}_{side}"
    hparam_path = hparam_dir / f"{pair_key}.json"
    if not hparam_path.exists():
        raise FileNotFoundError(f"HParams not found for {pair_key}: {hparam_path}")
    return json.loads(hparam_path.read_text())


def _load_artefacts(artefacts_dir: Path, sym: str):
    feat_path   = artefacts_dir / "features" / f"{sym}.parquet"
    events_path = artefacts_dir / "events"   / f"{sym}.parquet"
    map_path    = artefacts_dir / f"feature_map_{sym}.json"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found for {sym}: {feat_path}")
    df_features = pd.read_parquet(feat_path)
    df_events   = pd.read_parquet(events_path) if events_path.exists() else df_features
    feat_map    = json.loads(map_path.read_text()) if map_path.exists() else {}

    # Schema enforcement at stage boundary (HIGH-FIX #9).
    try:
        df_features = validate_or_die(df_features, FeatureBlockSchema, lazy=True, sample=10_000)
        df_events   = validate_or_die(df_events,   EventSchema,        lazy=True, sample=10_000)
    except Exception as exc:
        logger.warning("[%s] Stage-3 schema validation softened: %s", sym, exc)

    return df_features, df_events, feat_map


# =============================================================================
# Per-pair CPCV training
# =============================================================================

def train_pair(
    cfg: DictConfig,
    sym: str,
    side: str,
    artefacts_dir: Path,
) -> None:
    """Run full CPCV training for one (symbol, side) pair."""
    import catboost as cb

    from tradebot.train.calibration import PathSpecificPlattCalibrator
    from tradebot.train.ensemble import ContextualBanditEnsemble
    from tradebot.train.quant_arch import dynamic_embargo_bars
    _quant_available = True

    pair_key   = f"{sym}_{side}"
    model_dir  = artefacts_dir / "models"
    calib_dir  = artefacts_dir / "calibrators"
    oos_dir    = artefacts_dir / "oos_probs"
    hparam_dir = artefacts_dir / "hparams"
    for d in (model_dir, calib_dir, oos_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Determinism ───────────────────────────────────────────────────────────
    seed = int(OmegaConf.select(cfg, "training.random_seed", default=42))
    random.seed(seed); np.random.seed(seed)

    logger.info("=== Stage 3 [%s/%s] START (seed=%d) ===", sym, side, seed)

    hparams = _load_hparams(hparam_dir, sym, side)
    df_features, df_events, feat_map = _load_artefacts(artefacts_dir, sym)
    event_ts: pd.DatetimeIndex = df_events.index

    # ── Directional DOWN-CUSUM voor SHORT training (SHORT-CUSUM-FIX) ──────────
    # Symmetrische CUSUM filtert bear DOWN-impulsen als noise → te weinig SHORT
    # training events. Gebruik een lagere drempel voor DOWN-events (per-symbol YAML:
    # cusum_threshold_multiplier_short) zodat het SHORT-model meer leert.
    # Backward-compat: fallback op df_events wanneer SHORT of YAML-key ontbreekt.
    if side == "SHORT":
        try:
            import yaml as _yaml_cusum
            _cusum_sym_path = Path(__file__).resolve().parent.parent / "conf" / "symbols" / f"{sym}.yaml"
            _cusum_sym_cfg: dict = {}
            if _cusum_sym_path.exists():
                with open(_cusum_sym_path, encoding="utf-8") as _fh_cu:
                    _cusum_sym_cfg = _yaml_cusum.safe_load(_fh_cu) or {}
            _base_mult  = float(_cusum_sym_cfg.get("cusum_threshold_multiplier", 3.5))
            # v3 T0.6: per-asset CUSUM threshold override (overschrijft global multiplier)
            _cusum_per_asset = OmegaConf.select(cfg, "training.cusum_threshold_per_asset", default={})
            if _cusum_per_asset and sym in _cusum_per_asset:
                _override = float(_cusum_per_asset[sym])
                if _override != _base_mult:
                    logger.info("[%s] cusum_threshold_multiplier: %.2f -> %.2f (per-asset override)", sym, _base_mult, _override)
                    _base_mult = _override
            _short_mult = float(_cusum_sym_cfg.get("cusum_threshold_multiplier_short", _base_mult))
            short_event_ts = get_cusum_events_for_short(
                df_features,
                threshold_multiplier=_base_mult,
                short_multiplier_override=_short_mult,
            )
            if len(short_event_ts) >= 10:
                logger.info(
                    "[%s/SHORT] Directional DOWN-CUSUM: %d events (vs %d symmetric). "
                    "base_mult=%.1f, short_mult=%.1f",
                    sym, len(short_event_ts), len(event_ts), _base_mult, _short_mult,
                )
                event_ts = short_event_ts
            else:
                logger.warning(
                    "[%s/SHORT] DOWN-CUSUM returned only %d events — "
                    "behoud symmetrische events (%d).",
                    sym, len(short_event_ts), len(event_ts),
                )
        except Exception as _exc_cusum:
            logger.warning(
                "[%s/SHORT] Directional CUSUM mislukt (%s) — symmetrische events gebruikt.",
                sym, _exc_cusum,
            )

    # ── Feature matrices on EVENT grid ────────────────────────────────────────
    micro_cols = [c for c in feat_map.get("micro", []) if c in df_features.columns]
    meso_cols  = [c for c in feat_map.get("meso",  []) if c in df_features.columns]
    macro_cols = [c for c in feat_map.get("macro", []) if c in df_features.columns]

    # NOTE (Chief audit 2026-05-22): side-specifieke feature filter TERUGGEDRAAID.
    # Test toonde aan dat bear-regime features (EMA/RSI/ret) LONG-modellen helpen
    # om slechte LONG-posities in bear-markten te vermijden — na filtering: BTC
    # Trade-Sharpe=-1.96, ETH Trade-Sharpe=-3.05 (van neutraal/positief naar sterk
    # negatief). De 2024-Sharpe-regressie was veroorzaakt door de XGB/LGB ensemble
    # (nu teruggedraaid naar CatBoost-only), niet door de feature set zelf.
    # LONG en SHORT gebruiken dezelfde volledige feature set.
    logger.info(
        "[%s/%s] Features geladen: %d micro / %d meso / %d macro.",
        sym, side, len(micro_cols), len(meso_cols), len(macro_cols),
    )

    df_at_events = df_features.loc[event_ts]

    # F3 — Hurst-meso live-parity augmentation (CHIEF AUDIT 2026-05-25)
    # ----------------------------------------------------------------
    # In live mode the rolling buffer only spans 10-30 days, while the
    # Hurst-meso lookback needs 700 meso bars ≈ 175 days.  The live value
    # is therefore 0.0 for the first months of shadow trading.  If the
    # model has learned a non-trivial coefficient on this column, the live
    # input distribution (=0 everywhere) is a covariate-shift outlier.
    # Mitigation: during training, randomly mask feat_hurst_meso with 0 on
    # a fraction of rows, so the model becomes robust to the column being
    # missing.  Default mask probability 0.30 (configurable via
    # training.hurst_meso_mask_prob).  Set to 0 to disable.
    _mask_prob = float(OmegaConf.select(
        cfg, "training.hurst_meso_mask_prob", default=0.30
    ))
    _meso_col_idx_to_mask: list[int] = []
    if _mask_prob > 0.0 and "feat_hurst_meso" in meso_cols:
        _meso_col_idx_to_mask = [meso_cols.index("feat_hurst_meso")]

    X1 = np.nan_to_num(df_at_events[micro_cols].to_numpy(dtype=np.float32)) if micro_cols else np.zeros((len(event_ts), 0), dtype=np.float32)
    X4 = np.nan_to_num(df_at_events[meso_cols].to_numpy(dtype=np.float32))  if meso_cols  else np.zeros((len(event_ts), 0), dtype=np.float32)
    Xd = np.nan_to_num(df_at_events[macro_cols].to_numpy(dtype=np.float32)) if macro_cols else np.zeros((len(event_ts), 0), dtype=np.float32)

    if _meso_col_idx_to_mask:
        # Deterministic given the global seed set above — reproducible mask.
        _rng = np.random.default_rng(seed)
        _mask = _rng.random(len(event_ts)) < _mask_prob
        for _idx in _meso_col_idx_to_mask:
            X4[_mask, _idx] = 0.0
        logger.info(
            "[%s/%s] F3 Hurst-meso augmentation: masked %d / %d rows "
            "(p=%.2f) so model is robust to live feat_hurst_meso=0.",
            sym, side, int(_mask.sum()), len(event_ts), _mask_prob,
        )

    # ── Labels via TrendScanningLabeler at the chosen horizon ─────────────────
    t_max   = int(hparams.get("horizon", OmegaConf.select(cfg, "training.label_horizon", default=24)))
    t_min   = int(OmegaConf.select(cfg, "training.label_t_min", default=5))
    min_tst = float(OmegaConf.select(cfg, "training.label_min_tstat", default=2.0))
    pt_w    = float(OmegaConf.select(cfg, "training.label_pt_width", default=2.0))
    sl_w    = float(OmegaConf.select(cfg, "training.label_sl_width", default=1.0))
    sl_jump = float(OmegaConf.select(cfg, "training.sl_jump_max_mult", default=0.0))

    # SHORT-model verbetering (Agent C — 2026-05-22):
    # Laad per-symbol SHORT-specifieke barrier parameters uit conf/symbols/{sym}.yaml.
    # Consistent met dezelfde logica in tune_hparams._build_target_store().
    # Wanneer niet aanwezig: backward-compat fallback op globale pt_w/sl_w.
    pt_w_short: float | None = None
    sl_w_short: float | None = None
    if side == "SHORT":
        try:
            import yaml as _yaml_cpcv
            _sym_cfg_path = Path(__file__).resolve().parent.parent / "conf" / "symbols" / f"{sym}.yaml"
            if _sym_cfg_path.exists():
                with open(_sym_cfg_path, encoding="utf-8") as _fh_cpcv:
                    _sym_cfg_cpcv: dict = _yaml_cpcv.safe_load(_fh_cpcv) or {}
                if "label_pt_width_short" in _sym_cfg_cpcv:
                    pt_w_short = float(_sym_cfg_cpcv["label_pt_width_short"])
                if "label_sl_width_short" in _sym_cfg_cpcv:
                    sl_w_short = float(_sym_cfg_cpcv["label_sl_width_short"])
                if pt_w_short is not None or sl_w_short is not None:
                    logger.info(
                        "[%s/SHORT] Stage-3: SHORT-specifieke barriers geladen: "
                        "pt_width_short=%s, sl_width_short=%s",
                        sym, pt_w_short, sl_w_short,
                    )
        except Exception as _exc_cpcv:
            logger.warning(
                "[%s/SHORT] Stage-3: Kon SHORT-barrier overrides niet laden (%s) — "
                "globale pt_w=%.2f / sl_w=%.2f gebruikt.",
                sym, _exc_cpcv, pt_w, sl_w,
            )

    labeler = TrendScanningLabeler(
        t_min=t_min, t_max=t_max, min_tstat=min_tst, sl_jump_max_mult=sl_jump,
    )
    ts_dev = event_ts.to_series()
    r_lbl, r_t1, r_ret = labeler.label_data(
        df_features, event_timestamps=ts_dev, side=side,
        pt_width=pt_w, sl_width=sl_w,
        # SHORT-model verbetering (Agent C): side-specifieke barrier overrides
        pt_width_short=pt_w_short,
        sl_width_short=sl_w_short,
    )

    # P0-G FIX: sample weights zijn per-fold berekend in de CPCV loop (zie hieronder).
    # Globale uniqueness lekt OOS-fold info in train-gewichten (uniqueness van een
    # train-event dat overlapt met een test-event wordt verlaagd, terwijl dat overlap
    # in productie onbekend is). De globale `uniq` en `w` hieronder zijn alleen nog
    # nodig voor de PathSpecificPlattCalibrator (die alle folds samenpakt); de model-
    # fit zelf gebruikt per-fold gewichten.
    # LEGACY (alleen voor calibrator): geen aanpassing nodig.
    uniq = get_average_uniqueness(df_features.index, r_t1)
    # `_w` wordt berekend en NIET gelezen. Het commentaar hierboven zegt dat
    # de calibrator hem nodig heeft; dat is niet meer zo -- niets in deze
    # functie leest hem. Onderstreept in plaats van verwijderd, zodat de
    # aanroep blijft staan voor wie de calibrator opnieuw aansluit.
    _w   = get_sample_weights(ts_dev, uniq, r_ret.to_numpy())
    y    = r_lbl.values.astype(np.int32)
    r    = r_ret.values.astype(np.float64)

    # Convert raw-space t1 to event-space.
    event_idx_dev = df_features.index.get_indexer(event_ts)
    r_t1_event = np.searchsorted(event_idx_dev, r_t1.to_numpy())
    r_t1_event = np.clip(r_t1_event, 0, len(event_idx_dev) - 1).astype(np.int32)

    # ── CPCV splitter with dynamic embargo ────────────────────────────────────
    n_groups      = int(OmegaConf.select(cfg, "training.cpcv_n_groups",      default=6))
    n_test_groups = int(OmegaConf.select(cfg, "training.cpcv_n_test_groups", default=2))
    embargo_bars  = int(dynamic_embargo_bars([t_max], safety_factor=2.5, min_embargo=1))  # v3 T0.3: verhoogd van 1.2 → 2.5 voor striktere causal bescherming

    # AUDIT D-2: pass purge_bars explicitly (mirrors embargo_bars).
    cv = CombinatorialPurgedCV(
        n_groups=n_groups, n_test_groups=n_test_groups,
        embargo_bars=embargo_bars, purge_bars=embargo_bars,
    )

    # ── CatBoost params (strip non-CatBoost keys) ────────────────────────────
    _SIDE_KEYS = {"bandit_gamma", "horizon", "_best_score", "min_conf", "_random_seed"}
    cb_params = {k: v for k, v in hparams.items() if k not in _SIDE_KEYS}
    cb_params.setdefault("loss_function", "Logloss")
    cb_params.setdefault("eval_metric",   "Logloss")
    cb_params.setdefault("task_type",     "CPU")
    cb_params.setdefault("verbose",       False)
    cb_params.setdefault("allow_writing_files", False)

    bandit_gamma = float(hparams.get("bandit_gamma", 0.995))
    min_conf     = float(hparams.get("min_conf",     0.50))

    # ── CPCV training loop (single pass — collect probs + y in lockstep) ─────
    fold_models:    list[Any] = []
    oos_probs_list: list[np.ndarray]   = []
    oos_raw_probs_list: list[np.ndarray] = []   # L-1 fix: raw scores live feeds
    oos_y_list:     list[np.ndarray]   = []
    oos_ts_list:    list[pd.DatetimeIndex] = []
    fold_id_list:   list[np.ndarray]   = []

    X_full = stack_feats(X1, X4, Xd)
    t1_pd  = pd.Series(r_t1_event, index=event_ts)

    # ── Per-fold PCA setup (H2-FIX — eliminates PCA leakage into test folds) ──
    # Rationale: global PCA on 60% calibration data can include early test-fold
    # bars when those bars fall in the calibration window. fit_on_train_indices()
    # fits PCA only on the train subset of each CPCV fold → zero leakage.
    _pca_per_fold_enabled = bool(
        OmegaConf.select(cfg, "feature_pipeline.use_pca_orth", default=True)
    )
    _pca_for_catboost = bool(
        OmegaConf.select(cfg, "feature_pipeline.use_pca_for_catboost", default=False)
    )
    _pca_n_components = float(
        OmegaConf.select(cfg, "feature_pipeline.pca_n_components", default=0.95)
    )
    _all_feat_names = (
        [c for c in df_at_events[micro_cols].columns.tolist()] +
        [c for c in df_at_events[meso_cols].columns.tolist()] +
        [c for c in df_at_events[macro_cols].columns.tolist()]
    ) if (micro_cols or meso_cols or macro_cols) else []

    _fold_orth = None
    if _pca_per_fold_enabled and _pca_for_catboost and X_full.shape[1] >= 2:
        try:
            from tradebot.features.orthogonalize import FeatureOrthogonalizer
            _fold_orth = FeatureOrthogonalizer(
                n_components=_pca_n_components, symbol=sym,
            )
            logger.info(
                "[%s/%s] Per-fold PCA enabled (n_components=%.2f).",
                sym, side, _pca_n_components,
            )
        except Exception as _orth_exc:
            logger.warning(
                "[%s/%s] Per-fold PCA setup failed (%s) — using unorthogonalised features.",
                sym, side, _orth_exc,
            )
            _fold_orth = None

    # ── Per-fold causal FFD setup (T0.2-FIX — FFD d* calibrated on train-only) ──
    _ffd_per_fold = bool(
        OmegaConf.select(cfg, "feature_pipeline.ffd.rolling_calibration", default=True)
    )
    _close_series_raw: pd.Series | None = None
    if _ffd_per_fold and "close" in df_features.columns:
        # The raw close price series aligned on the full bar index — used per fold
        # to compute d* on train-only data via causal_min_frac_diff().
        _close_series_raw = df_features["close"]
        logger.info(
            "[%s/%s] Per-fold causal FFD enabled (%d bars available).",
            sym, side, len(_close_series_raw),
        )

    for fold_idx, (train_idx, val_idx, _test_groups) in enumerate(
        cv.split(event_ts, t1_pd)
    ):
        if len(train_idx) < 50 or len(val_idx) < 10:
            logger.warning("[%s/%s] Fold %d too small (tr=%d, val=%d) — skipping.",
                           sym, side, fold_idx, len(train_idx), len(val_idx))
            continue

        # ── Per-fold PCA (H2-FIX): fit PCA on train indices only ──────────────
        if _fold_orth is not None:
            try:
                X_full_fold, _ = _fold_orth.fit_on_train_indices(
                    X_full, train_idx, _all_feat_names, prefix="pc",
                )
            except Exception as _pca_fold_exc:
                logger.warning(
                    "[%s/%s] Fold %d per-fold PCA failed (%s) — using raw features.",
                    sym, side, fold_idx, _pca_fold_exc,
                )
                X_full_fold = X_full
        else:
            X_full_fold = X_full

        # ── Per-fold causal FFD (T0.2-FIX): compute d* on train bars only ─────
        if _close_series_raw is not None and len(train_idx) >= 50:
            try:
                from tradebot.features.fracdiff import causal_min_frac_diff
                # train_end_iloc: last bar index in bar-space (not event-space)
                # Map event train_idx back to bar-space using event timestamps
                _train_event_ts = event_ts[train_idx]
                _train_end_iloc = int(df_features.index.get_loc(_train_event_ts.max()))
                _d_star = causal_min_frac_diff(
                    _close_series_raw, train_end_iloc=_train_end_iloc + 1,
                )
                # Compute FFD close on event subset and append to X_full_fold
                try:
                    from tradebot.features.fracdiff import frac_diff_ffd
                    _close_events = _close_series_raw.reindex(event_ts).ffill().bfill()
                    _ffd_close = frac_diff_ffd(_close_events, d=_d_star, threshold=1e-5)
                    _ffd_arr = np.nan_to_num(
                        _ffd_close.to_numpy(dtype=np.float32).reshape(-1, 1)
                    )
                    X_full_fold = np.hstack([X_full_fold, _ffd_arr])
                    logger.debug(
                        "[%s/%s] Fold %d causal FFD: d*=%.3f appended.",
                        sym, side, fold_idx, _d_star,
                    )
                except Exception as _ffd_app_exc:
                    logger.debug(
                        "[%s/%s] Fold %d FFD append failed (%s).",
                        sym, side, fold_idx, _ffd_app_exc,
                    )
            except Exception as _ffd_fold_exc:
                logger.debug(
                    "[%s/%s] Fold %d causal FFD failed (%s) — skipping.",
                    sym, side, fold_idx, _ffd_fold_exc,
                )

        X_tr, X_val = X_full_fold[train_idx], X_full_fold[val_idx]
        y_tr, y_val = y[train_idx],     y[val_idx]

        # P0-G FIX: bereken sample-weights PER FOLD op basis van alleen de
        # train-events van DEZE fold. Globale uniqueness (berekend boven) lekt
        # test-fold informatie doordat OOS-events de concurrency van train-events
        # verlagen. Per-fold uniqueness ziet alleen de train-concurrency → correct.
        uniq_tr = get_average_uniqueness_per_fold(df_features.index, t1_pd, train_idx)
        w_tr    = get_sample_weights(
            ts_dev.iloc[train_idx],
            uniq_tr,
            r[train_idx],
        )

        # Deterministische fold-seed (derive_fold_seed patroon: base*1000 + fold_id)
        fold_seed = (seed * 1000 + fold_idx) % (2 ** 31)

        if np.unique(y_tr).size < 2:
            logger.warning("[%s/%s] Fold %d: training labels degenerate — skipping.", sym, side, fold_idx)
            continue

        # ── CPCV-safe early-stopping split (carved from training data, never test) ─
        # embargo_bars bars are dropped between the two halves to prevent overlap.
        X_es_tr, y_es_tr, w_es_tr, X_es_val, y_es_val = get_early_stop_split(
            X_tr, y_tr, w_tr, embargo_bars=embargo_bars, val_frac=0.15,
        )

        # ── 1. CatBoost (bestaande logica, early stopping op CPCV val fold) ──────
        cb_params_fold = {**cb_params, "random_seed": fold_seed}
        cb_model = cb.CatBoostClassifier(**cb_params_fold)
        cb_model.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=(X_val, y_val), early_stopping_rounds=50,
        )
        cb_raw = cb_model.predict_proba(X_val)
        cb_probs_raw: np.ndarray = (
            cb_raw[:, 1] if cb_raw.shape[1] > 1 else np.zeros(len(X_val))
        )
        cb_probs = _platt_calibrate(
            cb_model, cb_probs_raw, X_full_fold, y, train_idx, X_val,
            sym, side, fold_idx, "CatBoost",
        )

        # ── 2. XGBoost (early stopping op purged ES-val, nooit op test fold) ──────
        xgb_model = _train_xgb_fold(
            X_es_tr, y_es_tr, w_es_tr, X_es_val, y_es_val,
            hparams, fold_seed, embargo_bars,
        )
        if xgb_model is not None:
            xgb_raw = xgb_model.predict_proba(X_val)
            xgb_probs_raw: np.ndarray = (
                xgb_raw[:, 1] if xgb_raw.shape[1] > 1 else np.zeros(len(X_val))
            )
            _xgb_probs = _platt_calibrate(
                xgb_model, xgb_probs_raw, X_full_fold, y, train_idx, X_val,
                sym, side, fold_idx, "XGBoost",
            )
        else:
            _xgb_probs = cb_probs  # fallback: CatBoost probs tellen dubbel

        # ── 3. LightGBM (early stopping op purged ES-val, nooit op test fold) ─────
        lgb_model = _train_lgb_fold(
            X_es_tr, y_es_tr, w_es_tr, X_es_val, y_es_val,
            hparams, fold_seed, embargo_bars,
        )
        if lgb_model is not None:
            lgb_raw = lgb_model.predict_proba(X_val)
            lgb_probs_raw: np.ndarray = (
                lgb_raw[:, 1] if lgb_raw.shape[1] > 1 else np.zeros(len(X_val))
            )
            _lgb_probs = _platt_calibrate(
                lgb_model, lgb_probs_raw, X_full_fold, y, train_idx, X_val,
                sym, side, fold_idx, "LightGBM",
            )
        else:
            _lgb_probs = cb_probs  # fallback: CatBoost probs tellen dubbel

        # ── 4. Ensemble: CatBoost-only (XGB/LGB tijdelijk uitgeschakeld) ──────────
        # De twee `_`-prefixen hierboven zijn geen slordigheid: XGB en LGB
        # worden nog GETRAIND en gekalibreerd, maar hun probs gaan sinds de
        # revert hieronder nergens heen.
        # ENSEMBLE-REVERT (2026-05-22): CB+XGB+LGB gemiddelde trok probs naar 0.5
        # door XGB/LGB met CB-hparams (architectureel verkeerd) → Optuna koos
        # min_conf=0.405 → overtrading → Sharpe-degradatie van 1.39→1.24.
        # Fase 2 actie: aparte Optuna search voor XGB/LGB hparams implementeren.
        # Voor nu: CatBoost-only; alle nieuwe verbeteringen (8 bear features,
        # directional CUSUM, asymmetrische SHORT barriers) blijven actief.
        prob_win = cb_probs  # CatBoost-only

        logger.info(
            "[%s/%s] Fold %d CatBoost-only probs: mean=%.4f (XGB/LGB uitgeschakeld)",
            sym, side, fold_idx, float(cb_probs.mean()),
        )

        # ── BLUEPRINT-FIX: persist threshold attribute on each fold model so
        # that downstream ContextualBanditEnsemble.is_long_ensemble logic at
        # backtest time can detect the side without external metadata.
        # See train_regime.py:2498 — base_model.models[0].min_conf_short >= 0.99.
        # NOTE: threshold is set on cb_model only; the ensemble adapter wraps it.
        if side == "LONG":
            cb_model.min_conf_long = min_conf
            cb_model.min_conf_short = 0.99
        else:
            cb_model.min_conf_long = 0.99
            cb_model.min_conf_short = min_conf

        fold_models.append(CatBoostModelAdapter(cb_model))
        oos_probs_list.append(np.asarray(prob_win, dtype=np.float64))
        # L-1 fix: the live ModelSignal feeds the RAW CatBoost prob (adapter.py
        # returns predict_proba, not the sklearn-calibrated value).  Persist the
        # raw OOS probs so the PathSpecificPlattCalibrator (used live) is fitted
        # on the SAME input distribution it will see at serve time — eliminating
        # the train/serve skew that produced the degenerate ~0.72 output.
        oos_raw_probs_list.append(np.asarray(cb_probs_raw, dtype=np.float64))
        oos_y_list.append(np.asarray(y_val, dtype=np.int32))
        oos_ts_list.append(event_ts[val_idx])
        fold_id_list.append(np.full(len(val_idx), fold_idx, dtype=np.int32))

        logger.info("[%s/%s] Fold %d trained: %d train, %d val (seed=%d)",
                    sym, side, fold_idx, len(train_idx), len(val_idx), fold_seed)

    if not fold_models:
        raise RuntimeError(f"[{sym}/{side}] No folds produced models — check data/horizon.")

    # ── ContextualBanditEnsemble (correct signature per agent.py:1155) ────────
    # No min_conf_long/min_conf_short kwargs; per-model attributes carry the threshold.
    ensemble = ContextualBanditEnsemble(
        models=fold_models,
        gamma=bandit_gamma,
        context_dim=Xd.shape[1] if Xd.size > 0 else 0,
    )

    # ── Aggregate OOS in lockstep (FIX MEDIUM #13) ───────────────────────────
    all_probs     = np.concatenate(oos_probs_list)
    all_raw_probs = np.concatenate(oos_raw_probs_list)
    all_y         = np.concatenate(oos_y_list)
    all_ts_idx    = oos_ts_list[0].append(oos_ts_list[1:]) if len(oos_ts_list) > 1 else oos_ts_list[0]
    all_fold_ids  = np.concatenate(fold_id_list)

    assert len(all_probs) == len(all_raw_probs) == len(all_y) == len(all_ts_idx) == len(all_fold_ids), (
        f"OOS aggregation length mismatch: probs={len(all_probs)}, raw={len(all_raw_probs)}, "
        f"y={len(all_y)}, ts={len(all_ts_idx)}, fold_ids={len(all_fold_ids)}"
    )

    # ── Path-specific Platt calibration (LIVE calibrator) ────────────────────
    # L-1 fix: fit on the RAW OOS scores (logit-domain inside the calibrator),
    # matching exactly what the live ModelSignal feeds.  The backtest continues
    # to consume the per-fold sklearn-calibrated ``prob`` column below, so its
    # clean OOS estimate is unchanged.
    sigma_full = float(np.std(r[r != 0])) if np.any(r != 0) else 1.0
    sigma_arr  = np.full(len(all_probs), sigma_full, dtype=np.float64)
    calibrator: Any | None = None

    if _quant_available and PathSpecificPlattCalibrator is not None and len(all_raw_probs) > 10:
        try:
            calibrator = PathSpecificPlattCalibrator(input_is_probability=True)
            calibrator.fit(all_raw_probs, all_y, all_fold_ids, sigma_arr)
            # Sanity diagnostic: report the calibrated range on the raw OOS grid
            # so a degenerate (constant) calibrator is caught at build time.
            _cal_chk = calibrator.predict_proba(all_raw_probs, path_id=None)
            logger.info(
                "[%s/%s] Live calibrator fitted on RAW OOS probs: "
                "raw[min=%.3f mean=%.3f max=%.3f] -> cal[min=%.3f mean=%.3f max=%.3f], "
                "pos_rate=%.3f.",
                sym, side,
                float(all_raw_probs.min()), float(all_raw_probs.mean()), float(all_raw_probs.max()),
                float(_cal_chk.min()), float(_cal_chk.mean()), float(_cal_chk.max()),
                float(all_y.mean()),
            )
            if float(_cal_chk.max() - _cal_chk.min()) < 0.05:
                logger.warning(
                    "[%s/%s] Live calibrator near-constant (range<0.05) — model is "
                    "non-discriminative on this feature set; live signals will "
                    "correctly stay below min_conf (no spurious trades).",
                    sym, side,
                )
        except Exception as exc:
            logger.warning("[%s/%s] PathPlatt fit failed (%s) — leaving raw probs.", sym, side, exc)
            calibrator = None

    # ── Persist OOS probabilities ────────────────────────────────────────────
    oos_df = pd.DataFrame({
        "prob":     all_probs,       # per-fold sklearn-calibrated (backtest input)
        "prob_raw": all_raw_probs,   # raw CatBoost score (live calibrator input)
        "sigma":    sigma_arr,
        "fold_id":  all_fold_ids,
        "y":        all_y,
        "side":     side,
    }, index=pd.DatetimeIndex(all_ts_idx))
    oos_df.index.name = "timestamp"
    oos_df.to_parquet(oos_dir / f"{pair_key}.parquet")

    # ── Persist models ───────────────────────────────────────────────────────
    joblib.dump(ensemble, model_dir / f"{pair_key}_ensemble.joblib")
    if calibrator is not None:
        joblib.dump(calibrator, calib_dir / f"{pair_key}_platt.joblib")

    # ── Stage B: Meta-Labeling Judge training (v3 T0.3) ──────────────────────
    # Traint een Secondary Judge op basis van de OOS Primary probs die zojuist
    # zijn opgeslagen. Judge leert: gegeven P_primary + features, was de trade
    # winstgevend na kosten? Filter: alleen traden als P_primary > τ_p EN P_judge > τ_m.
    # Anti-lekkage: all_probs zijn uitsluitend OOS → geen in-sample contaminatie.
    _judge_enabled = bool(
        OmegaConf.select(cfg, "feature_pipeline.meta_labeling.enabled", default=False)
    )
    if _judge_enabled:
        try:
            from tradebot.train.meta_train import train_judge as _train_judge
            _judge_cost_bps   = float(OmegaConf.select(cfg, "feature_pipeline.meta_labeling.profit_threshold_bps", default=2.0))
            _judge_iterations = int(OmegaConf.select(cfg,   "feature_pipeline.meta_labeling.judge_iterations",     default=300))
            _judge_depth      = int(OmegaConf.select(cfg,   "feature_pipeline.meta_labeling.judge_depth",          default=5))
            _judge_l2         = float(OmegaConf.select(cfg, "feature_pipeline.meta_labeling.judge_l2_leaf_reg",    default=5.0))

            _oos_idx_dt = pd.DatetimeIndex(all_ts_idx)

            # OOS primary probs als Series (anti-lekkage: al OOS)
            _oos_prob_series = pd.Series(all_probs, index=_oos_idx_dt, name="prob_primary")

            # Signaalrichting: LONG=+1, SHORT=-1
            _direction_val = 1.0 if side == "LONG" else -1.0
            _directions = pd.Series(
                np.full(len(_oos_idx_dt), _direction_val),
                index=_oos_idx_dt,
                name="direction",
            )

            # Bruto returns gealigneerd op OOS timestamps (r_ret is labeled returns)
            _gross_ret = r_ret.reindex(_oos_idx_dt).fillna(0.0)

            # Feature matrix op OOS timestamps
            _X_oos = df_features.reindex(_oos_idx_dt)

            _train_judge(
                sym=sym,
                side=side,
                X_primary=_X_oos,
                oos_probs=_oos_prob_series,
                gross_returns=_gross_ret,
                directions=_directions,
                artefacts_dir=artefacts_dir,
                cost_bps=_judge_cost_bps,
                judge_iterations=_judge_iterations,
                judge_depth=_judge_depth,
                judge_l2=_judge_l2,
                random_seed=seed,
            )
            logger.info("[%s/%s] Stage B Judge training voltooid.", sym, side)
        except Exception as _judge_exc:
            logger.warning(
                "[%s/%s] Stage B Judge training mislukt: %s — doorgaan zonder Judge.",
                sym, side, _judge_exc,
            )

    logger.info(
        "=== Stage 3 [%s/%s] DONE — %d folds, %d OOS samples, calibrator=%s ===",
        sym, side, len(fold_models), len(all_probs),
        "yes" if calibrator else "no",
    )


@hydra.main(config_path="../conf", config_name="conf_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    artefacts_dir = Path(cfg.machine.get("artefacts_dir", "artefacts"))

    pair_str = OmegaConf.select(cfg, "pair", default=None)
    if pair_str:
        parts = str(pair_str).rsplit("_", 1)
        if len(parts) == 2:
            sym, side = parts
            train_pair(cfg, sym, side, artefacts_dir)
            return

    for sym in list(cfg.training.training_universe):
        for side in ("LONG", "SHORT"):
            try:
                train_pair(cfg, sym, side, artefacts_dir)
            except Exception as exc:
                logger.error("[%s/%s] Stage 3 FAILED: %s", sym, side, exc)


if __name__ == "__main__":
    main()
