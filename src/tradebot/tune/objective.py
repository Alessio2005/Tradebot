"""objective.py — Optuna HPO objective for binary CPCV classification.

Extracted from train_regime.py lines 1631-2403.

This is the core HPO engine: given a trial's hyperparameter sample,
it runs a full CPCV loop, simulates non-overlapping trades per fold,
reconstructs CPCV return paths, and returns a deflated Sharpe ratio.

Architecture notes:
  • The objective does NOT train final models — only fold-level models
    for scoring. Final CPCV training happens in apps/train_cpcv.py (Stage 3).
  • PathSpecificPlattCalibrator: fit ONCE after the full CV loop on
    aggregated (uncal_scores, y_val, fold_ids) — not per-fold.
  • Deflated Sharpe penalty (López de Prado) accounts for multiple
    testing across Optuna trials.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, cast

import catboost as cb
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.calibration import CalibratedClassifierCV

# Phase 0: idem train/catboost.py - scikit-learn is gepind op >=1.6.
from sklearn.frozen import FrozenEstimator as _FrozenEstimator
from sklearn.metrics import log_loss

from tradebot.backtest._kernels import calc_non_overlapping_stats

# ---------------------------------------------------------------------------
# Canonical tradebot.* imports
# ---------------------------------------------------------------------------
from tradebot.cv.cpcv import CombinatorialPurgedCV, build_cpcv_return_paths
from tradebot.cv.uniqueness import (
    get_average_uniqueness_per_fold,
    get_sample_weights,
)
from tradebot.execution.spread import compute_annualised_sharpe
from tradebot.features.blocks import stack_feats

# ---------------------------------------------------------------------------
# Evaluation helpers — migrated to tradebot.backtest.evaluation (Wave 5)
# ---------------------------------------------------------------------------
from ..backtest.evaluation import (
    block_bootstrap_path_sharpes,
    count_git_commits,
    deflated_sharpe_penalty,
)
from ..features.orthogonalize import FeatureOrthogonalizer

# ---------------------------------------------------------------------------
# PathSpecificPlattCalibrator
# ---------------------------------------------------------------------------
# Phase 0: `..train.calibration` is een interne module. De except-tak zette de
# calibrator op `Any`, waarna de pad-specifieke Platt-kalibratie stilzwijgend
# werd overgeslagen en ruwe, ongekalibreerde CatBoost-kansen werden gebruikt.
from ..train.calibration import PathSpecificPlattCalibrator
from ..train.ensemble import ContextualBanditEnsemble  # noqa: F401 — kept for type hints

# ---------------------------------------------------------------------------
from ..utils.failfast import TradebotContractError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NUMBA-SAFE ARRAY HELPERS (Item 11 — type & contiguity safety)
# ---------------------------------------------------------------------------

def _f64c(arr: np.ndarray) -> np.ndarray:
    """Force float64 + C-contiguous layout for safe Numba ingest."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _i32c(arr: np.ndarray) -> np.ndarray:
    """Same as _f64c but for int32 index arrays (t1, event indices)."""
    return np.ascontiguousarray(arr, dtype=np.int32)


# =============================================================================
# 5. OPTUNA OBJECTIVE (BINARY)
# =============================================================================
def optuna_objective_binary(
    trial: optuna.Trial,
    cfg: DictConfig,
    X1: np.ndarray,
    X4: np.ndarray,
    Xd: np.ndarray,
    target_store: dict[int, Any],
    timestamps: pd.Series,
    cv: CombinatorialPurgedCV,
    side_name: str,
    spread: float = 0.0010,
    # SHORT-CONF-FIX (Agent C, 2026-05-22):
    # Per-asset bovengrens voor min_conf Optuna search.
    # LONG: 0.65 (ongewijzigd). SHORT: typisch 0.42 (zie conf/symbols/{sym}.yaml).
    # Reden: SHORT modellen hebben gemiddelde calibreerde prob ~0.20-0.30.
    # Bij min_conf=0.50 (midden van [0.35,0.65]) vuren 0 SHORT signalen.
    min_conf_search_high: float = 0.65,
    # H2-FIX: per-fold PCA parameters (optioneel; achterwaarts compatibel).
    # Wanneer aangeleverd: PCA wordt per CPCV-fold gefit op uitsluitend de
    # trainingsindices (fit_on_train_indices). Zonder deze parameters wordt
    # de bestaande geschaalde feature-matrix X1/X4/Xd direct gebruikt.
    X1_raw: np.ndarray | None = None,   # ongeschaalde ruwe micro-features (n_events_dev × p_micro)
    X4_raw: np.ndarray | None = None,   # ongeschaalde ruwe meso-features  (n_events_dev × p_meso)
    Xd_raw: np.ndarray | None = None,   # ongeschaalde ruwe macro-features (n_events_dev × p_macro)
    orth_micro: FeatureOrthogonalizer | None = None,
    orth_meso: FeatureOrthogonalizer | None = None,
    orth_macro: FeatureOrthogonalizer | None = None,
    cols_micro: list[str] | None = None,
    cols_meso:  list[str] | None = None,
    cols_macro: list[str] | None = None,
    # FFD-LEAKAGE-FIX (per-fold variant): optionele FeatureEngineer + ruwe
    # bar-dataframe voor per-fold FFD d-kalibratie.
    # Wanneer aangeleverd: voor elke CPCV-fold wordt de d-cache gewist en
    # opnieuw gevuld via fit_d_on_train_indices(df_bars, sub_tr_bar_idx).
    # De X-matrices bevatten op dat moment al de globaal-gecalibreerde FFD-
    # features (burn-in fix in _add_ffd_features), maar door de cache hier
    # te synchroniseren zijn eventuele aanvullende add_features()-aanroepen
    # (bijv. voor live-inferentie na training) gegarandeerd consistent met
    # de fold. Voor een VOLLEDIGE eliminatie van FFD-lekkage moeten de X-
    # matrices per fold worden her-berekend (zie TODO hieronder).
    ffd_engineer: Any | None = None,    # FeatureEngineer instantie (engine.ta_engineer)
    df_bars: Any | None = None,         # ruwe bar-DataFrame met 'close' kolom
) -> float:
    # ---------------------------------------------------------
    # 1. PARAMETER SPACE CONFIGURATION
    # ---------------------------------------------------------
    selected_h = trial.suggest_categorical('horizon', list(target_store.keys()))

    # SHORT-CONF-FIX: gebruik per-asset bovengrens (0.42 voor SHORT, 0.65 voor LONG)
    min_conf = trial.suggest_float('min_conf', 0.35, min_conf_search_high)

    # GAMMA-FIX: gamma=0.995 (default) geeft half-life ≈ 138 trades.
    # Bij 5 trades/dag duurt aanpassing aan een nieuw regime ~28 dagen — te traag
    # voor crypto. Optuna sweep over log-schaal: 0.970 (HL≈23) tot 0.9995 (HL≈1385).
    bandit_gamma = trial.suggest_float('bandit_gamma', 0.970, 0.9995, log=True)

    params = {
        'iterations': trial.suggest_int('iterations', 600, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 15.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
        # bandit_gamma is stored in the Optuna trial directly (not a CatBoost param)
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'task_type': 'CPU',
        'verbose': False,
        'allow_writing_files': False,
        'border_count': 128,
        # SB-CLASS-WEIGHTS-FIX (Item 1): we gebruiken Sequential Bootstrap (SB)
        # om de uniciteit van overlappende labels te corrigeren. SB past de
        # **steekproefverdeling** fysiek aan (unieke trades vaker getrokken),
        # waardoor de natuurlijke class imbalance verandert. Als CatBoost
        # daarna ook nog 'auto_class_weights=Balanced' toepast, balanceert hij
        # op de SB-distributie i.p.v. de WARE marktdistributie — SB en
        # CatBoost sturen elkaar dan het bos in.
        # Oplossing: bereken class_weights handmatig op de ORIGINELE y_tr
        # vóór SB en geef ze als statische dict mee per fold (zie verderop).
        # We laten 'auto_class_weights' hier weg om dubbeltelling te vermijden.
    }

    # Data uitpakken
    y, _w, returns, t1_indices = target_store[selected_h]
    t1_series = pd.Series(t1_indices, index=timestamps.index)


    # ---------------------------------------------------------
    # 2. CROSS-VALIDATION LOOP
    # ---------------------------------------------------------
    # fold_scores wordt NIET meer gebruikt als eindscore.
    # Het blijft bestaan voor de deflated-Sharpe penalty (historische variantie).
    fold_scores = []
    fold_loglosses = []

    # CPCV PAD-RECONSTRUCTIE: sla returns per test_groups tuple op.
    # Na de loop bouwen we doorlopende paden en berekenen Sharpe PER PAD.
    fold_returns_dict: dict[tuple, pd.Series] = {}

    total_trades_all_folds = 0
    total_wins = 0  # Accumulated across folds

    # ── BLUEPRINT-FIX (II.3): aggregeer ongekalibreerde scores per fold zodat
    # we ná de loop één PathSpecificPlattCalibrator kunnen fitten met
    # path-specifieke (A_p, B_p) i.p.v. de globale CalibratedClassifierCV.
    # De per-fold CalibratedClassifierCV blijft tijdens de loop actief om de
    # Optuna-Sharpe consistent te houden met live-runtime; de path-specific
    # calibrator wordt naast de trial-state opgeslagen voor downstream gebruik.
    agg_uncal_scores: list[np.ndarray] = []
    agg_y_val:        list[np.ndarray] = []
    agg_fold_ids:     list[np.ndarray] = []
    agg_sigma:        list[np.ndarray] = []

    for fold_idx, (train_idx, val_idx, test_groups) in enumerate(cv.split(cast(pd.DatetimeIndex, pd.DatetimeIndex(timestamps)), t1_series)):

        # --- A. Data Slicing (Met veilige Chronologische ES Split) ---
        n_train = len(train_idx)

        idx_tr = int(n_train * 0.80)
        raw_sub_tr_idx = train_idx[:idx_tr]
        sub_es_idx = train_idx[idx_tr:]

        if len(sub_es_idx) < 10:
            logger.warning(f"Fold {fold_idx}: Te weinig data voor ES. Skipping fold.")
            return -1000.0

        es_start_idx = sub_es_idx[0]

        safe_t1_train = np.clip(t1_indices[raw_sub_tr_idx], 0, len(timestamps) - 1)

        valid_mask = safe_t1_train < es_start_idx

        sub_tr_idx = raw_sub_tr_idx[valid_mask]
        sub_cal_idx = sub_es_idx

        # CPCV-CALIBRATIE-FIX + ITEM 10 OPTUNA-DECOUPLING:
        # De 20% sub_es_idx wordt chronologisch in DRIE delen gesplitst:
        #   • eerste 50% → early stopping (CatBoost ziet via eval_pool)
        #   • daarna 25% → HP-Optuna (decouples HP-search van val_idx)
        #   • laatste 25% → Platt scaling (model heeft deze NOOIT gezien)
        #
        # Item 10 leakage-detail:
        #   In de oude 70/30-opzet was sub_es_early[0:0.7] zowel ES-set
        #   als (impliciet) Optuna's tuning-doel — Optuna kon HPs kiezen
        #   die exact deze 70% bars goed early-stopten, waarna val_idx
        #   sluipenderwijs in-sample werd voor de HP-zoekruimte.
        #   Door een aparte HP-eval slice in te voegen (`sub_hp_idx`) en
        #   de log_loss daarop te scoren, ziet Optuna een set die noch
        #   ES, noch Platt, noch val_idx is. De search-space is dan
        #   eerlijk gedeflateerd t.o.v. de ware OOS-distributie.
        n_es_total  = len(sub_es_idx)
        n_es_early  = max(5, int(n_es_total * 0.50))
        n_hp        = max(5, int(n_es_total * 0.25))
        # Veilige fallback bij krappe sub_es: knabbel HP/Platt eerst af.
        if n_es_early + n_hp >= n_es_total - 5:
            n_hp        = max(5, (n_es_total - n_es_early) // 2)
        sub_es_early_idx = sub_es_idx[:n_es_early]
        sub_hp_idx       = sub_es_idx[n_es_early : n_es_early + n_hp]
        sub_platt_idx    = sub_es_idx[n_es_early + n_hp:]

        if len(sub_tr_idx) < 10:
            logger.warning(f"Fold {fold_idx}: Purge liet te weinig train data over. Skipping.")
            return -1000.0

        y_tr, y_es, _y_cal, y_val = y[sub_tr_idx], y[sub_es_idx], y[sub_cal_idx], y[val_idx]
        ret_val = returns[val_idx]

        # SB-CLASS-WEIGHTS-FIX (Item 1): bereken class_weights op de
        # ORIGINELE y_tr (vóór Sequential Bootstrap). SB herschikt zo
        # daar de natuurlijke class-imbalance verloren gaat. Door de
        # gewichten hier te bevriezen weet CatBoost altijd de WARE
        # marktverhouding terug, ongeacht hoe SB de samples heeft her-
        # geschud. Inverse-frequentie weighting (zelfde formule als
        # CatBoost's 'Balanced' op een onverstoorde dataset).
        _y_tr_pre_sb = np.asarray(y_tr).astype(np.int64)
        _n_pre = max(int(_y_tr_pre_sb.size), 1)
        _n_pos = int(np.sum(_y_tr_pre_sb == 1))
        _n_neg = int(np.sum(_y_tr_pre_sb == 0))
        if _n_pos > 0 and _n_neg > 0:
            _w0 = float(_n_pre) / (2.0 * float(_n_neg))
            _w1 = float(_n_pre) / (2.0 * float(_n_pos))
            fold_class_weights: dict[int, float] | None = {0: _w0, 1: _w1}
        else:
            # Single-class fold: Balanced kan niet rekenen, val terug op uniform.
            fold_class_weights = None

        t1_val = t1_indices[val_idx]
        t1_val_rel = np.searchsorted(val_idx, t1_val)
        t1_val_rel = np.clip(t1_val_rel, 0, len(y_val) - 1).astype(np.int32)

        # --- B. Dynamische Sample Weights per Fold ---
        fold_anchor_time = timestamps.iloc[sub_tr_idx[-1]]

        # ── PER-FOLD UNIQUENESS (Item 9) ────────────────────────────────
        # De globale `w` is berekend over de volledige dataset en bevat
        # daarmee concurrency-bijdragen van OOS test-events. We hercompu-
        # teren de uniqueness UITSLUITEND over de events van deze train-
        # fold (raw_sub_tr_idx ∪ sub_es_idx == train_idx). Het out-of-
        # sample pad (val_idx) ziet hierdoor nooit een gewicht dat
        # afhing van zijn eigen aanwezigheid.
        fold_uniq_train: np.ndarray = get_average_uniqueness_per_fold(
            cast(pd.DatetimeIndex, pd.DatetimeIndex(timestamps)),
            t1_series,
            train_idx,
        )
        # Mapping: train_idx == [raw_sub_tr_idx ; sub_es_idx]
        fold_uniq_raw_sub_tr: np.ndarray = fold_uniq_train[:idx_tr]
        fold_uniq_sub_tr:     np.ndarray = fold_uniq_raw_sub_tr[valid_mask]
        fold_uniq_sub_es:     np.ndarray = fold_uniq_train[idx_tr:]

        w_tr = get_sample_weights(
            timestamps.iloc[sub_tr_idx],
            fold_uniq_sub_tr,
            returns[sub_tr_idx],
            anchor_time=fold_anchor_time,
        )
        w_es = get_sample_weights(
            timestamps.iloc[sub_es_idx],
            fold_uniq_sub_es,
            returns[sub_es_idx],
            anchor_time=fold_anchor_time,
        )

        # --- C. Data Slicing (Features) ---
        # H2-FIX: per-fold PCA (geen global fit_transform lekkage).
        # Als ruwe (ongeschaalde) feature-matrices + orthogonalizers zijn meegestuurd,
        # wordt de PCA per fold gefit op UITSLUITEND de trainingsindices van die fold
        # (fit_on_train_indices). Zo lekt de variantiestructuur van toekomstige
        # test-bars nooit terug via de PCA-kalibratie.
        # Zonder raw-parameters (backwards-compatibel) valt de code terug op de
        # bestaande geschaalde matrices X1/X4/Xd.

        # FFD-LEAKAGE-FIX (per-fold variant): kalibreer d opnieuw op uitsluitend
        # de train-bars van deze fold. Dit synchroniseert de d-cache zodat
        # eventuele aanvullende add_features()-aanroepen fold-consistent zijn.
        # De huidige X-matrices zijn gebouwd met de burn-in gecalibreerde d
        # (zie _add_ffd_features). Voor volledige herberekening van de FFD
        # kolommen per fold: vervang hier de relevante kolom-slices in X1/Xd
        # nadat _add_ffd_features opnieuw is aangeroepen op het fold-subset.
        if ffd_engineer is not None and df_bars is not None and len(sub_tr_idx) >= 30:
            ffd_engineer.clear_d_cache()
            ffd_engineer.fit_d_on_train_indices(
                df_bars, sub_tr_idx, timeframe="micro"
            )

        # Micro (X1) — per-fold PCA
        if (orth_micro is not None
                and X1_raw is not None
                and X1_raw.shape[1] >= 2
                and len(sub_tr_idx) >= 2):
            _X1_fold, _ = orth_micro.fit_on_train_indices(
                X1_raw, sub_tr_idx, cols_micro or [], "micro_pc"
            )
            x1_tr_s = _X1_fold[sub_tr_idx]
            x1_es_s = _X1_fold[sub_es_idx]
            x1_val_s = _X1_fold[val_idx]
        else:
            x1_tr_s, x1_es_s, x1_val_s = X1[sub_tr_idx], X1[sub_es_idx], X1[val_idx]

        # Meso (X4) — per-fold PCA
        if (orth_meso is not None
                and X4_raw is not None
                and X4_raw.shape[1] >= 2
                and len(sub_tr_idx) >= 2
                and X4 is not None
                and len(X4) > 0):
            _X4_fold, _ = orth_meso.fit_on_train_indices(
                X4_raw, sub_tr_idx, cols_meso or [], "meso_pc"
            )
            x4_tr_s = _X4_fold[sub_tr_idx]
            x4_es_s = _X4_fold[sub_es_idx]
            x4_val_s = _X4_fold[val_idx]
        elif X4 is not None and len(X4) > 0:
            x4_tr_s, x4_es_s, x4_val_s = X4[sub_tr_idx], X4[sub_es_idx], X4[val_idx]
        else:
            x4_tr_s = x4_es_s = x4_val_s = None

        # Macro (Xd) — per-fold PCA
        if (orth_macro is not None
                and Xd_raw is not None
                and Xd_raw.shape[1] >= 2
                and len(sub_tr_idx) >= 2
                and Xd is not None
                and len(Xd) > 0):
            _Xd_fold, _ = orth_macro.fit_on_train_indices(
                Xd_raw, sub_tr_idx, cols_macro or [], "macro_pc"
            )
            xd_tr_s = _Xd_fold[sub_tr_idx]
            xd_es_s = _Xd_fold[sub_es_idx]
            xd_val_s = _Xd_fold[val_idx]
        elif Xd is not None and len(Xd) > 0:
            xd_tr_s, xd_es_s, xd_val_s = Xd[sub_tr_idx], Xd[sub_es_idx], Xd[val_idx]
        else:
            xd_tr_s = xd_es_s = xd_val_s = None

        X_tr_final = stack_feats(x1_tr_s, x4_tr_s, xd_tr_s)
        X_es_final = stack_feats(x1_es_s, x4_es_s, xd_es_s)
        X_val_final = stack_feats(x1_val_s, x4_val_s, xd_val_s)

        # AUDIT-FIX (K3 — Tree-vs-Linear Feature Path):
        #   PCA op niet-lineaire features (jump intensity, FFD-volume) vernietigt
        #   exact het signaal dat CatBoost-bomen probeerden te isoleren via niet-
        #   lineaire splits. Zet ``feature_pipeline.use_pca_for_catboost: False``
        #   in cfg om CatBoost de RUWE features te voeden terwijl de bandit
        #   (lineair RFF/UCB) wel de PCA-projectie krijgt voor numerieke
        #   stabiliteit van de Sherman-Morrison updates.
        #
        #   Default ``True`` = backward-compatibel (oude gedrag). De aanbeveling
        #   uit de De Prado audit is ``False`` voor crypto-pijplijnen.
        _use_pca_for_cat: bool = bool(
            OmegaConf.select(cfg, "feature_pipeline.use_pca_for_catboost", default=True)
        )
        if not _use_pca_for_cat:
            def _slice_or_none(mat: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
                if mat is None or mat.size == 0:
                    return None
                return mat[idx]
            x1_tr_raw_s  = _slice_or_none(X1_raw, sub_tr_idx)  if X1_raw is not None else x1_tr_s
            x4_tr_raw_s  = _slice_or_none(X4_raw, sub_tr_idx)  if X4_raw is not None else x4_tr_s
            xd_tr_raw_s  = _slice_or_none(Xd_raw, sub_tr_idx)  if Xd_raw is not None else xd_tr_s
            x1_es_raw_s  = _slice_or_none(X1_raw, sub_es_idx)  if X1_raw is not None else x1_es_s
            x4_es_raw_s  = _slice_or_none(X4_raw, sub_es_idx)  if X4_raw is not None else x4_es_s
            xd_es_raw_s  = _slice_or_none(Xd_raw, sub_es_idx)  if Xd_raw is not None else xd_es_s
            x1_val_raw_s = _slice_or_none(X1_raw, val_idx)     if X1_raw is not None else x1_val_s
            x4_val_raw_s = _slice_or_none(X4_raw, val_idx)     if X4_raw is not None else x4_val_s
            xd_val_raw_s = _slice_or_none(Xd_raw, val_idx)     if Xd_raw is not None else xd_val_s

            X_tr_final  = stack_feats(x1_tr_raw_s,  x4_tr_raw_s,  xd_tr_raw_s)
            X_es_final  = stack_feats(x1_es_raw_s,  x4_es_raw_s,  xd_es_raw_s)
            X_val_final = stack_feats(x1_val_raw_s, x4_val_raw_s, xd_val_raw_s)
            logger.debug(
                "[Fold %d] CatBoost krijgt RUWE features (use_pca_for_catboost=False). "
                "Bandit blijft op PCA-projectie voor SM-stabiliteit.",
                fold_idx,
            )

        # --- C.5. Uniqueness-weighted training (SB-REMOVAL — AUDIT-FIX Round 3) ---
        # Sequential Bootstrapping (SB) en sample_weight corrigeren BEIDE voor
        # overlappende Triple-Barrier events (AFML Ch. 4). Combinatie = dubbele
        # bestraffing: SB reduceert de steekproefkans van overlappende samples;
        # sample_weight reduceert hun loss-gradient. Gecombineerd effect: het model
        # weegt geïsoleerde trades extreem zwaar en negeert clustermarkten — het
        # omgekeerde van robuust leren.
        #
        # Fix (Round 3 Audit): verwijder de fysieke SB-resample-stap volledig.
        # w_tr (berekend in stap B via get_sample_weights + per-fold uniqueness)
        # handelt de uniciteitsrol volledig af via CatBoost's loss-gradiënt.
        # X_tr_final en y_tr blijven in chronologische volgorde — geen reshuffle.
        logger.debug(
            "[Fold %d] SB verwijderd (Round-3-fix): %d training-events, "
            "gemiddelde uniqueness-weight = %.4f",
            fold_idx, len(y_tr), float(np.mean(w_tr)) if len(w_tr) > 0 else 0.0,
        )

        # CPCV-CALIBRATIE-FIX + ITEM 10:
        # X_es_final is al in tijdsvolgorde (sub_es_idx is chronologisch).
        # • [0   :n_es_early]              → early stopping (eval_pool)
        # • [n_es_early : n_es_early+n_hp] → HP-Optuna (decoupled HP score)
        # • [n_es_early+n_hp :]            → Platt scaling
        X_es_early_final = X_es_final[:n_es_early]
        X_hp_final       = X_es_final[n_es_early : n_es_early + n_hp]
        X_platt_final    = X_es_final[n_es_early + n_hp:]

        y_es_early = y_es[:n_es_early]
        y_hp       = y_es[n_es_early : n_es_early + n_hp]
        y_platt    = y_es[n_es_early + n_hp:]

        w_es_early = w_es[:n_es_early]

        # --- D. Base Model Training (Met Early Stopping op ES-early subset) ---
        # SB-CLASS-WEIGHTS-FIX (Item 1): injecteer pre-SB class_weights als
        # statische dict in de fold-params. Geen 'auto_class_weights' toepassen.
        fold_params = dict(params)
        if fold_class_weights is not None:
            fold_params['class_weights'] = fold_class_weights
        model = cb.CatBoostClassifier(**fold_params)
        # DTYPE-FIX: cast exact hier naar float32 — CatBoost traint sneller
        # op float32 en voert de conversie intern toch uit. copy=False hergebruikt
        # het geheugen als de array al contiguous float32 is (no-op), anders
        # vindt hier de enige allocatie in de hot-loop plaats.
        train_pool = cb.Pool(X_tr_final.astype(np.float32, copy=False), y_tr, weight=w_tr)
        eval_pool  = cb.Pool(X_es_early_final.astype(np.float32, copy=False), y_es_early, weight=w_es_early)

        model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=50,
            use_best_model=True,
        )
        trial.set_user_attr("best_iteration", int(model.tree_count_ or 0))

        # --- E. Platt Kalibratie + Validatie Predictie ---
        # CPCV-CALIBRATIE-FIX: kalibreer het model op de aparte Platt-set
        # voordat de validatiekansen worden berekend. Zo zijn de kansen die
        # Optuna beoordeelt IDENTIEK aan de gecalibreerde productiekansen —
        # de hyperparameter-keuze is dan consistent met de inzetomgeving.
        try:
            platt_feasible = (
                len(y_platt) >= 10
                and np.unique(y_platt).size >= 2
            )
            if platt_feasible:
                _est = _FrozenEstimator(model) if _FrozenEstimator is not None else cast(Any, model)
                _calibrator = CalibratedClassifierCV(
                    estimator=_est, cv="prefit", method="sigmoid"
                )
                _calibrator.fit(X_platt_final, y_platt)
                _eval_model = _calibrator
            else:
                # Te weinig samples in de Platt-set: gebruik ongekalibreerd model.
                # Dit kan gebeuren bij kleine datasets; het is een acceptable
                # fallback omdat het model dan nog steeds consistent is over folds.
                _eval_model = model

            val_probs = _eval_model.predict_proba(X_val_final)
            prob_win: np.ndarray = (
                val_probs[:, 1] if val_probs.shape[1] > 1
                else np.zeros(len(val_probs), dtype=np.float64)
            )

            # ── BLUEPRINT-FIX (II.3): collect ONGEKALIBREERDE scores +
            # per-fold sigma voor de path-specific Platt-fit ná de loop.
            # We trekken hier opzettelijk uit ``model`` (raw CatBoost), niet
            # uit ``_eval_model`` (al gekalibreerd via prefit-CalibratedClassifierCV)
            # — Path-Platt verlangt de raw decision-function als input.
            _raw_proba = model.predict_proba(X_val_final)
            _raw_scores: np.ndarray = (
                np.asarray(_raw_proba[:, 1], dtype=np.float64)
                if hasattr(_raw_proba, "shape") and _raw_proba.shape[1] > 1
                else np.zeros(len(X_val_final), dtype=np.float64)
            )
            _y_val_arr: np.ndarray = np.asarray(y_val, dtype=np.float64).flatten()
            _fold_ids_arr: np.ndarray = np.full(
                _raw_scores.size, fill_value=int(fold_idx), dtype=np.int64
            )
            _sigma_scalar: float = float(np.std(np.asarray(ret_val))) if len(ret_val) > 1 else 1.0
            if not np.isfinite(_sigma_scalar) or _sigma_scalar <= 0.0:
                _sigma_scalar = 1.0
            _sigma_arr: np.ndarray = np.full(
                _raw_scores.size, fill_value=_sigma_scalar, dtype=np.float64
            )
            if _raw_scores.size == _y_val_arr.size:
                agg_uncal_scores.append(_raw_scores)
                agg_y_val.append(_y_val_arr)
                agg_fold_ids.append(_fold_ids_arr)
                agg_sigma.append(_sigma_arr)

            # ── ITEM 10: log_loss op HP-Optuna slice (niet val_idx) ─────
            # Decoupled score: Optuna's HP-zoekruimte wordt gedeflateerd
            # t.o.v. een set die noch in training, noch in early-stopping,
            # noch in Platt, noch in val_idx zit. Hierdoor vermindert
            # de implicit selection-bias op val_idx.
            ll: float
            if (
                len(y_hp) >= 5
                and X_hp_final.shape[0] == len(y_hp)
                and np.unique(y_hp).size >= 2
            ):
                hp_probs = _eval_model.predict_proba(X_hp_final)
                ll = float(log_loss(y_hp, hp_probs))
            else:
                # CHIEF AUDIT 2026-05-23 (P-2): GEEN val_idx-fallback meer.
                # Eerder werd hier ``log_loss(y_val, val_probs)`` gebruikt —
                # exact de leakage-bron die Item 10 probeerde te elimineren.
                # In plaats daarvan skippen we de fold met een straf-score
                # (-1000.0). Optuna ziet dit als een zeer slecht trial-fold;
                # MedianPruner snijdt zo'n run vroegtijdig af. Dit is veiliger
                # dan een gebiasde lage log_loss te belonen.
                logger.warning(
                    "HP-segment te klein (%d) — fold %d geskipped met straf-score.",
                    len(y_hp), fold_idx,
                )
                ll = 1000.0  # zeer hoge log_loss → fold telt als slecht
            fold_loglosses.append(ll)
        except Exception as e:
            # Phase 0: hier werd een gefaalde fold beloond met een VERZONNEN
            # log_loss van 10.0 en prob_win = zeros. Optuna optimaliseerde
            # vervolgens over een mengsel van echte en gefabriceerde
            # foldscores, en de trial telde gewoon mee in M. Dat maakt elke
            # DSR-correctie op die trials betekenisloos.
            raise TradebotContractError(
                f"Predictie faalde in fold {fold_idx}: {e}. Een gefaalde fold "
                f"krijgt geen straf-score meer: de trial wordt afgebroken."
            ) from e

        # --- F. NON-OVERLAPPING SIMULATION ---
        n_val = len(prob_win)
        zeros_arr = np.zeros(n_val, dtype=np.float64)
        zeros_int_arr = np.zeros(n_val, dtype=np.int32)

        spread_arr = np.full(n_val, spread, dtype=np.float64)

        rolling_thresh = np.full(n_val, min_conf, dtype=np.float64)
        dummy_high = np.full(n_val, 2.0, dtype=np.float64)

        if side_name == "LONG":
            probs_long, probs_short = prob_win.astype(np.float64), zeros_arr
            ret_long, ret_short = ret_val, zeros_arr
            t1_long, t1_short = t1_val_rel, zeros_int_arr
            thresh_long_arr, thresh_short_arr = rolling_thresh, dummy_high
        else:
            probs_long, probs_short = zeros_arr, prob_win.astype(np.float64)
            ret_long, ret_short = zeros_arr, ret_val
            t1_long, t1_short = zeros_int_arr, t1_val_rel
            thresh_long_arr, thresh_short_arr = dummy_high, rolling_thresh

        # NUMBA-FIX (Item 11): force float64/int32 + C-contiguous before kernel.
        active_returns, _, active_indices = calc_non_overlapping_stats(
            _f64c(probs_short), _f64c(probs_long),
            _f64c(ret_short), _f64c(ret_long),
            _i32c(t1_short), _i32c(t1_long),
            _f64c(spread_arr),
            _f64c(thresh_long_arr), _f64c(thresh_short_arr),
        )

        n_trades = len(active_returns)
        total_trades_all_folds += n_trades
        total_wins += int(np.sum(active_returns > 0))

        if n_trades < 1:
            # SMOOTH-PENALTY-FIX (Item 3): de oude harde stap (-0.5 * (1.0 - n/5))
            # gaf TPE geen gradiënt om langs te zoeken — Optuna's TPE bouwt een
            # GP over de zoekruimte en knikken creëren blinde vlekken. We
            # vervangen de stap door een gladde log-penalty die continu af-
            # neemt naarmate er meer (counterfactual) trades zouden zijn,
            # zodat de optimizer een richting voelt naar meer activiteit.
            #
            # penalty_score = -1.0 / log1p(1)  ≈ -1.443  bij n_trades = 0,
            # nadert 0 voor n → ∞, zonder stap.
            penalty_score = -1.0 / float(np.log1p(1.0 + max(n_trades, 0)))
            fold_scores.append(penalty_score)
            # Lege Series opslaan zodat de pad-reconstructie niet crasht
            fold_returns_dict[test_groups] = pd.Series(
                [], dtype=np.float64, name="returns"
            )
        else:
            trade_timestamps = timestamps.iloc[val_idx[active_indices]]
            annualized_score = compute_annualised_sharpe(
                active_returns, trade_timestamps.to_numpy()
            )
            fold_scores.append(annualized_score)

            # --- CPCV PAD-OPSLAG ---
            # Sla de gerealiseerde trade-returns op als tijdgeïndexeerde Serie.
            # build_cpcv_return_paths() plakt deze segmenten later aan elkaar
            # tot doorlopende backtest-paden.
            fold_returns_dict[test_groups] = pd.Series(
                active_returns,
                index=trade_timestamps,
                name="returns",
            )

        # Cleanup Memory
        del train_pool, eval_pool, model
        del X_tr_final, X_es_final, X_es_early_final, X_hp_final, X_platt_final, X_val_final
        del x1_tr_s, x1_es_s, x1_val_s
        del sub_tr_idx, sub_es_idx, sub_es_early_idx, sub_hp_idx, sub_platt_idx

        gc.collect()

    # ---------------------------------------------------------
    # BLUEPRINT-FIX (II.3) — PATH-SPECIFIC PLATT na CPCV-loop
    # ---------------------------------------------------------
    # Standaard Platt fit één globale (A, B) over alle CPCV-folds samen — en
    # middelt daarmee de unieke regime-blootstelling van elk pad uit. Hier
    # combineren we de aggregaten en fitten ``PathSpecificPlattCalibrator``
    # met een (A_p, B_p) per CPCV-pad.  Het fit-resultaat wordt aan de
    # trial-state gekoppeld zodat downstream code (final-fit, internal_backtest)
    # de path-specific calibratie kan ophalen via ``trial.user_attrs``.
    if len(agg_uncal_scores) > 0:
        try:
            _scores_all = np.concatenate(agg_uncal_scores).astype(np.float64)
            _y_all      = np.concatenate(agg_y_val).astype(np.float64)
            _paths_all  = np.concatenate(agg_fold_ids).astype(np.int64)
            _sigma_all  = np.concatenate(agg_sigma).astype(np.float64)
            if (
                _scores_all.size == _y_all.size == _paths_all.size == _sigma_all.size
                and _scores_all.size >= 10
                and np.unique(_y_all).size >= 2
            ):
                _path_platt = PathSpecificPlattCalibrator(
                    recency_lambda=1.0, vol_floor=1e-4, max_iter=50, tol=1e-7,
                ).fit_per_path(
                    scores=_scores_all,
                    labels=_y_all,
                    path_ids=_paths_all,
                    bar_volatility=_sigma_all,
                )
                # Stash op de trial — Pylance: Optuna's user_attrs accepteert
                # arbitrary objects via storage.set_trial_user_attr; pickle-veilig.
                trial.set_user_attr("path_platt_n_paths", len(_path_platt._params))
                trial.set_user_attr("path_platt_global_pos_rate", float(_path_platt._global.pos_rate))
                logger.info(
                    "PathSpecificPlattCalibrator gefit (n_samples=%d, paden=%d).",
                    _scores_all.size, len(_path_platt._params),
                )
            else:
                logger.warning(
                    "PathSpecificPlattCalibrator overgeslagen: "
                    "n=%d, classes=%d.",
                    _scores_all.size, int(np.unique(_y_all).size),
                )
        except Exception as _exc:
            # Phase 0: een gefaalde pad-specifieke Platt-fit liet de GLOBALE
            # CalibratedClassifierCV als enige kalibratielaag over. De trial
            # rapporteerde daarna kansen uit een ANDER kalibratiemodel dan de
            # config voorschreef, zonder dat dit in de trial-attributen
            # terechtkwam.
            raise TradebotContractError(
                f"PathSpecificPlattCalibrator-fit faalde: {_exc}. Er wordt niet "
                f"stilzwijgend teruggevallen op de globale calibratielaag."
            ) from _exc

    # ---------------------------------------------------------
    # 3. AGGREGATIE & SCORING (CPCV Path-Level + Multiple Testing Deflatie)
    # ---------------------------------------------------------
    # CORRECTE CPCV EVALUATIE (López de Prado, AFML ch. 12):
    # Reconstrueer S doorlopende backtest-paden en bereken Sharpe PER PAD.
    # De distributie over paden is de echte out-of-sample maatstaf.
    # Het gemiddelde van fold-Sharpes (oud) bestrafte onterecht modellen die
    # in één fold slecht presteren maar over het volledige pad robuust zijn.
    path_series_list = build_cpcv_return_paths(fold_returns_dict, cv.n_groups)

    if path_series_list:
        path_sharpes = [
            compute_annualised_sharpe(p.to_numpy(), p.index.to_numpy())
            for p in path_series_list
            if len(p) > 0
        ]
    else:
        # Fallback: als pad-reconstructie mislukt (te weinig data),
        # val terug op fold-scores om trial niet te vernietigen.
        path_sharpes = fold_scores

    # ─── BLOCK BOOTSTRAP OVER CPCV-PADEN (Item 3) ─────────────────────────
    # De paden delen onderliggende data → onderling gecorreleerd. Naïeve
    # std() onderschat de variantie. Stationary block-bootstrap (Politis-
    # Romano) levert een Sharpe-distributie met effectieve vrijheidsgraden.
    bs_stats = block_bootstrap_path_sharpes(
        [p for p in path_series_list if len(p) > 0],
        n_boot=int(cfg.training.get("bootstrap_n", 500)),
        avg_block_size=float(cfg.training.get("bootstrap_block_size", 20.0)),
    )

    if path_sharpes:
        avg_sharpe = float(np.mean(path_sharpes))
    else:
        avg_sharpe = 0.0

    # Gebruik bootstrap-std (corrigeert voor pad-correlatie) wanneer beschikbaar.
    boot_std = float(bs_stats.get("sharpe_std", 0.0))
    naive_std = float(np.std(path_sharpes)) if len(path_sharpes) > 1 else 0.0
    std_sharpe = boot_std if boot_std > 1e-9 else naive_std

    # 1. Basis Robuuste Sharpe — penaliseer hoge bootstrap-spreiding
    base_objective = avg_sharpe - (0.5 * std_sharpe)

    # 2. Haal historische trial-data op om de variantie in de search space te meten
    completed_trials = [
        t for t in trial.study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    # Haal het totaal geplande aantal trials uit de config (fallback naar 200)
    N_trials = int(cfg.training.get('optuna_trials', 200))

    # FIX 1: Bepaal de variantie UITSLUITEND op basis van de initiële random exploration fase.
    # Optuna's TPE gebruikt standaard ~10-25 startup trials. We pakken hier de eerste 25.
    n_startup_trials = 25

    # EARLY-TRIAL BIAS FIX (2026-05-24): The old code gated list collection
    # behind `len(completed_trials) > 5`, so the first five trials always
    # received hist_sharpes_list=[] → deflated_sharpe_penalty returns 0.0 for
    # len(hist_sharpes)<2 → those trials escaped the MT correction entirely.
    # Trial #4 with raw_sharpe=-0.22 could "win" with final=-0.69, while trial
    # #33 with raw_sharpe=2.97 got penalty=6.29 → final=-3.47 (buried).
    #
    # Fix A: always collect from ALL completed trials (remove the >5 gate).
    #   P1.4-FIX rationale still applies: use ALL trials, not [:n_startup_trials],
    #   so σ_SR reflects the full experimental budget.
    # Fix B: pad hist_sharpes_list to n_startup_trials synthetic zero-Sharpe
    #   observations.  Zeros are conservative (lower σ_SR when the true
    #   distribution is unknown) but guarantee deflated_sharpe_penalty always
    #   sees ≥ n_startup_trials entries — giving every trial the same baseline
    #   correction from the very first iteration onward.
    hist_sharpes_list: list[float] = [
        float(t.user_attrs.get("sharpe", 0.0))
        for t in completed_trials          # ALL trials, not [:n_startup_trials]
        if "sharpe" in t.user_attrs
    ]
    # Pad so early trials are never exempt from σ_SR estimation.
    hist_sharpes_list = (
        [0.0] * max(0, n_startup_trials - len(hist_sharpes_list))
        + hist_sharpes_list
    )

    # ─── PRIOR-EXPERIMENT AWARE DSR PENALTY (Item 1) ──────────────────────
    # Quant-firma's tellen ook eerdere experimenten mee — de research-
    # iteraties die aan deze code voorafgingen. Approximatie: git-commits.
    n_prior = count_git_commits(
        repo_path=Path(cfg.training.get("repo_path", "."))
            if cfg.training.get("repo_path") else None,
        fallback=int(cfg.training.get("prior_experiments_fallback", 0)),
    )
    # CHIEF AUDIT 2026-05-29 (DSR not too strict): cap_factor 5.0 → 1.0.
    # penalty = σ_SR·√(2·lnN) capped at cap_factor·σ_SR.  At the default cap of
    # 5·σ_SR the penalty could reach ~6+ (documented mt=6.29), large enough to
    # INVERT the trial ranking — a Sharpe-2.97 trial buried below a Sharpe-(-0.2)
    # early trial.  Capping at 1·σ_SR keeps a principled multiple-testing
    # deflation (one std of the cross-trial Sharpe distribution) while
    # guaranteeing the highest genuine Sharpe is never out-ranked by the
    # penalty.  Combined with the pad-to-25 fix this removes the early-trial /
    # first-trial selection bias the way the audit requires.
    multiple_testing_penalty = deflated_sharpe_penalty(
        hist_sharpes_list,
        n_optuna_trials=N_trials,
        n_prior_experiments=n_prior,
        cap_factor=1.0,
    )

    # 3. Deflated Objective
    final_objective = base_objective - multiple_testing_penalty

    # 4. Liquiditeits/Trade-Frequentie Penalty (Voorkomt curve-fitting op 3 gelukstrades)
    # SMOOTH-PENALTY-FIX (Item 3): vervang harde lineaire stap door een gladde
    # sigmoidale penalty zodat TPE een continue gradient ervaart richting meer
    # trades. Het oude (75 - n) * 0.2 produceerde een knik op exact n = 75 — TPE
    # snapte niet waarom de objective ineens 15 punten verloor toen trade 74
    # ontbrak. De sigmoid heeft een soft inflection rond n_target en verdwijnt
    # asymptotisch boven het target.
    n_target = 75.0
    sigmoid_steepness = 0.10  # 1/trades — controle over hoe scherp de overgang is
    # penalty_amplitude = max penalty when n_trades → 0
    penalty_amplitude = 15.0
    smooth_penalty = penalty_amplitude / (
        1.0 + float(np.exp(sigmoid_steepness * (float(total_trades_all_folds) - n_target)))
    )
    final_objective -= smooth_penalty

    # ─── PORTFOLIO-CAP-AWARENESS (Item 4) ────────────────────────────────────
    # Single-asset CPCV-paden zijn al non-overlapping op één kant. Maar bij
    # de live multi-asset run kunnen BTC/ETH/SOL op dezelfde bar gelijktijdig
    # entry-en — de PortfolioBacktester schaalt dan via gross/net cap of de
    # ``max_portfolio_leverage`` MTM-pas trades naar beneden.  Optuna ziet
    # daar niets van: het optimaliseert een pad dat de live laag stilzwijgend
    # zou trimmen.
    #
    # We voegen een penalty toe die proportioneel is met (i) verwachte
    # gelijktijdige bezettingstijd, geapproximeerd door de fractie bars met
    # een open trade in deze fold-pool, vermenigvuldigd met (ii) een
    # configureerbare coëfficiënt (``optuna_pf_shrink_penalty_coef``).
    # Default 0.0 = uit (backwards-compat); zet op bv. 0.5 wanneer in een
    # multi-asset omgeving wordt getuned. De penalty is convexe in
    # bezettingstijd: hoe drukker de pipeline, hoe meer Optuna verleid wordt
    # naar selectievere strategieën.
    pf_shrink_coef = float(cfg.training.get("optuna_pf_shrink_penalty_coef", 0.0))
    if pf_shrink_coef > 0.0:
        # bezettingstijd ≈ trades × avg_horizon / total_bars over alle folds.
        # We benaderen avg_horizon via de t_max suggestion uit dit trial.
        # CHIEF AUDIT 2026-05-23 (P-1): "horizon" is een label-horizon
        # (categorisch, in label-eenheden), niet een bar-horizon. Eerder
        # werd dit als bar-count gebruikt waardoor de penalty puur ruis was.
        # We proberen eerst ``trend_scan_t_max`` (echte bar-horizon van de
        # Trend Scanner) en vallen alleen terug op ``horizon`` indien beide
        # ontbreken — dan blijft de penalty een grove maar consistente proxy.
        # NB: pf_shrink_coef is alleen zinvol als ``trend_scan_t_max`` in de
        # trial-parameters zit; anders is de schatting indicatief.
        _t_max_suggest = int(
            trial.params.get(
                "trend_scan_t_max",
                trial.params.get("horizon", 24),
            )
        )
        # som-bars uit fold_returns_dict — niet exact, maar TPE geeft alleen
        # om monotonie.
        approx_total_bars = max(
            sum(len(v) for v in fold_returns_dict.values()) or 1, 1
        )
        approx_occupancy = float(total_trades_all_folds * _t_max_suggest) / float(
            max(approx_total_bars, 1)
        )
        # Convex in occupancy: kwadratisch zodat een verdubbeling 4× zo zwaar
        # bestraft wordt — past bij de werkelijkheid waarbij de portfolio-cap
        # super-lineair bindt.
        pf_shrink_penalty = pf_shrink_coef * (approx_occupancy ** 2)
        final_objective -= pf_shrink_penalty
        trial.set_user_attr("pf_occupancy", float(approx_occupancy))
        trial.set_user_attr("pf_shrink_penalty", float(pf_shrink_penalty))

    avg_ll = np.mean(fold_loglosses) if fold_loglosses else 10.0

    # 5. Opslaan van attributen voor de database en post-analyse
    trial.set_user_attr("sharpe", float(avg_sharpe))
    trial.set_user_attr("log_loss", float(avg_ll))
    trial.set_user_attr("trades", int(total_trades_all_folds))
    trial.set_user_attr("wins", int(total_wins))
    trial.set_user_attr("mt_penalty", float(multiple_testing_penalty))
    trial.set_user_attr("bootstrap_std", float(std_sharpe))
    trial.set_user_attr("bootstrap_lower_5", float(bs_stats.get("sharpe_lower_5", 0.0)))
    trial.set_user_attr("bootstrap_upper_95", float(bs_stats.get("sharpe_upper_95", 0.0)))
    trial.set_user_attr("effective_dof", float(bs_stats.get("effective_dof", 0.0)))
    trial.set_user_attr("n_prior_experiments", int(n_prior))

    logger.info(
        f"[{side_name}] Trial {trial.number:03d} | Deflated Obj: {final_objective:.4f} "
        f"(MT-Penalty: -{multiple_testing_penalty:.4f} @ N_eff={N_trials + n_prior}) | "
        f"Raw Sharpe: {avg_sharpe:.2f} (boot std={std_sharpe:.2f}, "
        f"5%={bs_stats.get('sharpe_lower_5', 0.0):.2f}) | "
        f"Trades: {total_trades_all_folds} | LL: {avg_ll:.3f}"
    )

    return final_objective


# ---------------------------------------------------------------------------
# TunableObjective — callable class adapter (tune/__init__.py public API)
# ---------------------------------------------------------------------------

class TunableObjective:
    """Callable Optuna objective wrapping ``optuna_objective_binary``.

    Constructs a partial application that fixes the heavy configuration
    (cfg, feature data, label stores) and returns a trial-scoped callable
    for Optuna's ``study.optimize()``.

    Usage
    -----
    obj = TunableObjective(cfg=cfg, sym=sym, side=side,
                           df_merged=df, feat_map=feat_map,
                           target_store=targets, horizon_keys=h_keys)
    study.optimize(obj, n_trials=cfg.tune.n_trials)
    """

    def __init__(
        self,
        cfg: Any,
        sym: str,
        side: str,
        df_merged: Any,
        feat_map: Any,
        target_store: Any,
        horizon_keys: Any,
        **kwargs: Any,
    ) -> None:
        self._cfg = cfg
        self._sym = sym
        self._side = side
        self._df_merged = df_merged
        self._feat_map = feat_map
        self._target_store = target_store
        self._horizon_keys = horizon_keys
        self._kwargs = kwargs

    def __call__(self, trial: Any) -> float:
        return optuna_objective_binary(
            trial=trial,
            cfg=self._cfg,
            sym=self._sym,
            side=self._side,
            df_merged=self._df_merged,
            feat_map=self._feat_map,
            target_store=self._target_store,
            horizon_keys=self._horizon_keys,
            **self._kwargs,
        )
