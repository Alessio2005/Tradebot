"""tune_hparams.py — DAG Stage 2: Feature artefacts → best Optuna hyperparams.

Entrypoint for ``dvc run`` / ``dvc repro tune_hparams``.

Canonical labeling pattern (mirrors train_regime.py lines 4194-4216):
  • TrendScanningLabeler is the SINGLE source of truth for direction + t1.
  • Sample-weights via get_average_uniqueness + get_sample_weights.
  • t1 converted to event-space via np.searchsorted (AFML Ch.4).

Determinism (HIGH-FIX):
  • Optuna sampler seed pinned via cfg.training.random_seed (default 42).
  • CatBoost random_seed pulled from same seed at trial level.
  • The combination of pruner + seeded TPESampler reproduces best-params
    exactly across re-runs (acceptance criterion §6.4).

Outputs:
  artefacts/hparams/{SYM}_{SIDE}.json
  artefacts/hparams/{SYM}_{SIDE}_study.db
  artefacts/hparams/{SYM}_{SIDE}_best_score.json
"""
from __future__ import annotations

import json
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tradebot.cv.cpcv import CombinatorialPurgedCV
from tradebot.cv.uniqueness import get_average_uniqueness, get_sample_weights
from tradebot.labeling.cusum import get_cusum_events, get_cusum_events_for_short
from tradebot.labeling.trend_scanning import TrendScanningLabeler
from tradebot.tune.objective import optuna_objective_binary
from tradebot.tune.pruning import create_median_pruner
from tradebot.utils.arrays import validate_or_die
from tradebot.schemas.events import EventSchema
from tradebot.schemas.features import FeatureBlockSchema

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# I/O helpers
# =============================================================================

def _load_artefacts(artefacts_dir: Path, sym: str):
    """Load Stage-1 outputs and validate against schemas."""
    feat_path   = artefacts_dir / "features" / f"{sym}.parquet"
    events_path = artefacts_dir / "events"   / f"{sym}.parquet"
    map_path    = artefacts_dir / f"feature_map_{sym}.json"

    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found for {sym}: {feat_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events not found for {sym}: {events_path}")

    df_features = pd.read_parquet(feat_path)
    df_events   = pd.read_parquet(events_path)
    feat_map    = json.loads(map_path.read_text()) if map_path.exists() else {}

    # ── Schema enforcement at stage boundary (HIGH-FIX #9) ───────────────────
    # Pandera "filter" mode in EventSchema strips extras gracefully.
    try:
        df_features = validate_or_die(df_features, FeatureBlockSchema, lazy=True, sample=10_000)
        df_events   = validate_or_die(df_events,   EventSchema,        lazy=True, sample=10_000)
    except Exception as exc:
        logger.warning("[%s] Stage-2 schema validation softened to warning: %s", sym, exc)

    return df_features, df_events, feat_map


# =============================================================================
# Target store (canonical pattern from train_regime.py)
# =============================================================================

def _build_target_store(
    df_features: pd.DataFrame,
    event_timestamps: pd.DatetimeIndex,
    side: str,
    horizons: List[int],
    cfg: DictConfig,
    sym: str = "",
) -> Dict[int, tuple]:
    """Mirror train_regime.py lines 4194-4216 — TrendScanningLabeler per horizon.

    Returns dict mapping horizon → (y_array, w_array, ret_array, t1_event_space).

    SHORT-model verbetering (Agent C — 2026-05-22):
    Wanneer side="SHORT" wordt label_pt_width_short en label_sl_width_short
    uit conf/symbols/{sym}.yaml geladen (als beschikbaar) en doorgegeven aan
    TrendScanningLabeler.label_data() als pt_width_short / sl_width_short.
    Dit geeft kortere PT barriers voor SHORT → hogere label=1 rate →
    betere class balance voor SHORT-model training.
    """
    t_min          = int(OmegaConf.select(cfg, "training.label_t_min",      default=5))
    min_tstat      = float(OmegaConf.select(cfg, "training.label_min_tstat", default=2.0))
    sl_jump_mult   = float(OmegaConf.select(cfg, "training.sl_jump_max_mult", default=0.0))
    pt_width       = float(OmegaConf.select(cfg, "training.label_pt_width", default=2.0))
    sl_width       = float(OmegaConf.select(cfg, "training.label_sl_width", default=1.0))

    # SHORT-model verbetering (Agent C — 2026-05-22):
    # Laad per-symbol SHORT-specifieke barrier parameters uit conf/symbols/{sym}.yaml.
    # Wanneer niet aanwezig: backward-compat fallback op globale pt_width/sl_width.
    pt_width_short: float | None = None
    sl_width_short: float | None = None
    if side == "SHORT" and sym:
        try:
            import yaml as _yaml
            _sym_path = Path(__file__).resolve().parent.parent / "conf" / "symbols" / f"{sym}.yaml"
            if _sym_path.exists():
                with open(_sym_path, encoding="utf-8") as _fh:
                    _sym_cfg: dict = _yaml.safe_load(_fh) or {}
                if "label_pt_width_short" in _sym_cfg:
                    pt_width_short = float(_sym_cfg["label_pt_width_short"])
                if "label_sl_width_short" in _sym_cfg:
                    sl_width_short = float(_sym_cfg["label_sl_width_short"])
                if pt_width_short is not None or sl_width_short is not None:
                    logger.info(
                        "[%s/SHORT] SHORT-specifieke barriers geladen: "
                        "pt_width_short=%s, sl_width_short=%s",
                        sym, pt_width_short, sl_width_short,
                    )
        except Exception as _exc:
            logger.warning(
                "[%s/SHORT] Kon SHORT-barriere overrides niet laden (%s) — "
                "globale pt_width=%.2f / sl_width=%.2f gebruikt.",
                sym, _exc, pt_width, sl_width,
            )

    ts_dev = event_timestamps.to_series()
    event_idx_dev = df_features.index.get_indexer(event_timestamps)

    target_store: Dict[int, tuple] = {}

    for h in horizons:
        try:
            labeler = TrendScanningLabeler(
                t_min=t_min, t_max=int(h), min_tstat=min_tstat,
                sl_jump_max_mult=sl_jump_mult,
            )
            r_lbl, r_t1, r_ret = labeler.label_data(
                df_features, event_timestamps=ts_dev, side=side,
                pt_width=pt_width, sl_width=sl_width,
                # SHORT-model verbetering (Agent C — 2026-05-22):
                # Geef side-specifieke barrier overrides door wanneer beschikbaar.
                # None = backward-compat (geen gedragsverandering voor LONG of
                # SHORT zonder YAML-override).
                pt_width_short=pt_width_short,
                sl_width_short=sl_width_short,
            )
            uniq    = get_average_uniqueness(df_features.index, r_t1)
            w_full  = get_sample_weights(ts_dev, uniq, r_ret.to_numpy())

            # Raw-space t1 → event-space for downstream CPCV (AFML Ch.4).
            r_t1_event_space = np.searchsorted(event_idx_dev, r_t1.to_numpy())
            r_t1_event_space = np.clip(r_t1_event_space, 0, len(event_idx_dev) - 1).astype(np.int32)

            target_store[int(h)] = (r_lbl.values, w_full, r_ret.values, r_t1_event_space)
            logger.info("[h=%d] Target built: %d events, win-rate=%.1f%%",
                        h, len(r_lbl), float(r_lbl.mean()) * 100)
        except Exception as exc:
            logger.warning("[h=%d] Target build failed: %s", h, exc)

    if not target_store:
        raise ValueError(f"No valid horizons produced targets for side={side}.")
    return target_store


# =============================================================================
# Per-pair tune
# =============================================================================

def tune_pair(
    cfg: DictConfig,
    sym: str,
    side: str,
    artefacts_dir: Path,
) -> Dict[str, Any]:
    """Run Optuna study for one (symbol, side) pair."""
    seed = int(OmegaConf.select(cfg, "training.random_seed", default=42))
    random.seed(seed); np.random.seed(seed)

    hparam_dir = artefacts_dir / "hparams"
    hparam_dir.mkdir(parents=True, exist_ok=True)

    pair_key    = f"{sym}_{side}"
    study_db    = hparam_dir / f"{pair_key}_study.db"
    result_path = hparam_dir / f"{pair_key}.json"
    score_path  = hparam_dir / f"{pair_key}_best_score.json"

    logger.info("=== Stage 2 [%s/%s] START (seed=%d) ===", sym, side, seed)

    df_features, df_events, feat_map = _load_artefacts(artefacts_dir, sym)
    event_ts: pd.DatetimeIndex = df_events.index

    # ── Directional DOWN-CUSUM voor SHORT tuning (SHORT-CUSUM-FIX) ───────────
    # Zelfde logica als train_cpcv.py: gebruik lagere DOWN-drempel voor SHORT events
    # zodat Stage 2 (Optuna) en Stage 3 (CPCV training) dezelfde event grid zien.
    if side == "SHORT":
        try:
            import yaml as _yaml_cusum_tune
            _cusum_tune_path = Path(__file__).resolve().parent.parent / "conf" / "symbols" / f"{sym}.yaml"
            _cusum_tune_cfg: dict = {}
            if _cusum_tune_path.exists():
                with open(_cusum_tune_path, encoding="utf-8") as _fh_ct:
                    _cusum_tune_cfg = _yaml_cusum_tune.safe_load(_fh_ct) or {}
            _base_mult_t  = float(_cusum_tune_cfg.get("cusum_threshold_multiplier", 3.5))
            _short_mult_t = float(_cusum_tune_cfg.get("cusum_threshold_multiplier_short", _base_mult_t))
            short_event_ts_tune = get_cusum_events_for_short(
                df_features,
                threshold_multiplier=_base_mult_t,
                short_multiplier_override=_short_mult_t,
            )
            if len(short_event_ts_tune) >= 10:
                logger.info(
                    "[%s/SHORT] Stage-2 DOWN-CUSUM: %d events (vs %d symmetric). "
                    "base=%.1f, short=%.1f",
                    sym, len(short_event_ts_tune), len(event_ts), _base_mult_t, _short_mult_t,
                )
                event_ts = short_event_ts_tune
            else:
                logger.warning(
                    "[%s/SHORT] Stage-2 DOWN-CUSUM: slechts %d events — symmetrisch behouden.",
                    sym, len(short_event_ts_tune),
                )
        except Exception as _exc_ct:
            logger.warning(
                "[%s/SHORT] Stage-2 directional CUSUM mislukt (%s) — symmetrisch gebruikt.",
                sym, _exc_ct,
            )

    # ── Feature matrices (per blueprint §3.3 — float32 on disk, f64 in kernels) ─
    micro_cols = [c for c in feat_map.get("micro", []) if c in df_features.columns]
    meso_cols  = [c for c in feat_map.get("meso",  []) if c in df_features.columns]
    macro_cols = [c for c in feat_map.get("macro", []) if c in df_features.columns]

    # IMPORTANT: feature matrices are over EVENT timestamps (df_events.index),
    # NOT over df_features.index. The CPCV operates on events.
    df_at_events = df_features.loc[event_ts]
    X1 = np.nan_to_num(df_at_events[micro_cols].to_numpy(dtype=np.float32)) if micro_cols else np.zeros((len(event_ts), 0), dtype=np.float32)
    X4 = np.nan_to_num(df_at_events[meso_cols].to_numpy(dtype=np.float32))  if meso_cols  else np.zeros((len(event_ts), 0), dtype=np.float32)
    Xd = np.nan_to_num(df_at_events[macro_cols].to_numpy(dtype=np.float32)) if macro_cols else np.zeros((len(event_ts), 0), dtype=np.float32)

    # ── Target store (canonical TrendScanning pattern) ────────────────────────
    # AANBEVELING (Agent C — SHORT model verbetering, 2026-05-22):
    # _build_target_store() geeft dezelfde label_pt_width/sl_width aan LONG en SHORT.
    # Voor SHORT moet dit asymmetrisch zijn (bear-trends bereiken een kleiner PT
    # sneller → hogere label=1 rate → betere class balance → model leert SHORT winst).
    # Implementeer side-specific parameter-loading via:
    #   if side == "SHORT":
    #       pt_width = float(sym_cfg.get("label_pt_width_short", pt_width))
    #       sl_width = float(sym_cfg.get("label_sl_width_short", sl_width))
    # Lees sym_cfg via conf/symbols/{sym}.yaml (aanwezig: label_pt_width_short,
    # label_sl_width_short zijn al toegevoegd aan alle symbol YAML bestanden).
    horizons = list(OmegaConf.select(cfg, "training.label_horizons", default=[12, 24, 48]))
    target_store = _build_target_store(df_features, event_ts, side, horizons, cfg, sym=sym)

    # ── CPCV splitter with dynamic embargo ────────────────────────────────────
    n_groups       = int(OmegaConf.select(cfg, "training.cpcv_n_groups",      default=6))
    n_test_groups  = int(OmegaConf.select(cfg, "training.cpcv_n_test_groups", default=2))
    t_max_dyn      = max(horizons)
    embargo_bars   = int(np.ceil(t_max_dyn * 1.2)) + 1   # blueprint §II.4 safety-factor

    # AUDIT D-2: bar-space purge required. Use embargo_bars as the purge
    # floor — both must be ≥ t_max + 1 to keep CPCV leakage-free.
    cv = CombinatorialPurgedCV(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        embargo_bars=embargo_bars,
        purge_bars=embargo_bars,
    )

    # ── Optuna study with deterministic TPE sampler ──────────────────────────
    storage    = f"sqlite:///{study_db}?journal_mode=WAL&timeout=60"
    study_name = f"{pair_key}_v1"
    n_trials   = int(OmegaConf.select(cfg, "training.optuna_trials", default=100))
    timeout    = OmegaConf.select(cfg, "training.optuna_timeout_seconds", default=None)

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner  = create_median_pruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    spread = float(OmegaConf.select(cfg, "training.spread", default=0.0010))

    # SHORT-CONF-FIX (Agent C, 2026-05-22 — nu geïmplementeerd):
    # SHORT modellen hebben gecalibreerde probs ≈ 0.20-0.30. Met de standaard
    # search range [0.35, 0.65] kiest Optuna min_conf > 0.42 → 0 actieve signalen
    # (AVAXUSDT_SHORT 2026: max_prob=0.378 < min_conf=0.50 → geen trades).
    # Per-asset bovengrens geladen uit conf/symbols/{sym}.yaml: min_conf_short_max.
    if side == "SHORT":
        try:
            import yaml as _yaml
            _sym_path = Path(__file__).resolve().parent.parent / "conf" / "symbols" / f"{sym}.yaml"
            _sym_cfg_tune: dict = {}
            if _sym_path.exists():
                with open(_sym_path, encoding="utf-8") as _fh:
                    _sym_cfg_tune = _yaml.safe_load(_fh) or {}
            min_conf_high = float(_sym_cfg_tune.get("min_conf_short_max", 0.42))
        except Exception as _exc:
            logger.warning("[%s/SHORT] min_conf_short_max laden mislukt (%s) — default 0.42", sym, _exc)
            min_conf_high = 0.42
        logger.info("[%s/SHORT] min_conf search range: [0.35, %.2f]", sym, min_conf_high)
    else:
        min_conf_high = 0.65  # LONG: ongewijzigd

    # ── OPTUNA EARLY-TRIAL BIAS (documented 2026-05-24) ─────────────────────
    # The deflated_sharpe_penalty in optuna_objective_binary grows with the
    # number of completed trials. Early trials (0-4) run before hist_sharpes_list
    # is populated, so they face ZERO MT-penalty. This biases the study toward
    # whichever random configuration happened to run first.
    #
    # Known consequence for ETH LONG (2026-05-24):
    #   Trial #4 (min_conf=0.583, raw_sharpe=-0.220, mt=0.000) beat
    #   Trial #33 (min_conf=0.374, raw_sharpe=2.974, mt=6.287)
    #   because trial #4 ran before any penalty accumulated.
    #
    # ATTEMPTED FIX: using trial#33 params in Stage 3 → Stage 4 result was
    #   WORSE (Sharpe=2.53, MaxDD=12.6% vs baseline 3.40/6.3%). Root cause:
    #   min_conf=0.374 passes 86% of events in Stage 4's full backtest
    #   (vs ~5-10% in CPCV folds due to non-overlapping filter). The Optuna
    #   score for trial #33 was misleadingly high because CPCV test folds
    #   only expose a fraction of events per fold.
    #
    # LESSON: do NOT override Optuna best_params with higher-raw-sharpe trials
    #   when the min_conf is very different. The Optuna selection (trial #4,
    #   min_conf=0.583) is the correct production threshold.
    #
    # PROPER FIX: modify deflated_sharpe_penalty() to initialize
    #   hist_sharpes_list with N_startup=25 synthetic zero-sharpe entries
    #   so ALL trials face a consistent baseline MT-penalty. See spawn task
    #   "Fix Optuna MT-penalty early-trial bias in objective.py".
    # ──────────────────────────────────────────────────────────────────────────

    objective_fn = partial(
        optuna_objective_binary,
        cfg=cfg,
        X1=X1, X4=X4, Xd=Xd,
        target_store=target_store,
        timestamps=pd.Series(event_ts, index=event_ts),
        cv=cv,
        side_name=side,
        spread=spread,
        min_conf_search_high=min_conf_high,
    )

    completed_already = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    trials_remaining  = max(n_trials - completed_already, 0)

    if trials_remaining > 0:
        study.optimize(
            objective_fn,
            n_trials=trials_remaining,
            timeout=float(timeout) if timeout is not None else None,
            gc_after_trial=True,
        )

    best_params = dict(study.best_params)
    best_score  = float(study.best_value)

    # Pin random_seed into the persisted hparams so Stage 3 inherits determinism.
    best_params["_random_seed"] = seed

    result_path.write_text(json.dumps({**best_params, "_best_score": best_score}, indent=2))
    score_path.write_text(json.dumps({"sharpe_deflated": best_score}, indent=2))

    logger.info(
        "=== Stage 2 [%s/%s] DONE — score=%.4f, %d/%d trials ===",
        sym, side, best_score, completed_already + trials_remaining, n_trials,
    )
    return best_params


@hydra.main(config_path="../conf", config_name="conf_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    artefacts_dir = Path(cfg.machine.get("artefacts_dir", "artefacts"))

    pair_str = OmegaConf.select(cfg, "pair", default=None)
    if pair_str:
        parts = str(pair_str).rsplit("_", 1)
        if len(parts) == 2:
            sym, side = parts
            tune_pair(cfg, sym, side, artefacts_dir)
            return

    for sym in list(cfg.training.training_universe):
        for side in ("LONG", "SHORT"):
            try:
                tune_pair(cfg, sym, side, artefacts_dir)
            except Exception as exc:
                logger.error("[%s/%s] Stage 2 FAILED: %s", sym, side, exc)


if __name__ == "__main__":
    main()
