"""build_features.py — DAG Stage 1: Raw data → bars → features → events.

Entrypoint for ``dvc run`` / ``dvc repro build_features``.

Pipeline:
  1. Update macro datasets (TradFi + Crypto)     — data.macro.update_macro
  2. Sync + load raw Bybit data                — data.ingestion.ingest_raw
  3. Build bars + micro/meso features            — features.pipeline.build_features
  4. Merge publication-lag-aware macro features  — data.macro.merge_macro
  5. Compute CUSUM events on merged frame        — labeling.cusum.get_cusum_events
  6. Write validated parquet artefacts           — utils.parquet_io.write_validated_parquet

Outputs (per symbol):
  artefacts/bars/{SYM}.parquet
  artefacts/features/{SYM}.parquet
  artefacts/events/{SYM}.parquet
  artefacts/feature_map_{SYM}.json
  artefacts/orthogonalizer_{SYM}_*.joblib

Usage:
    python -m apps.build_features symbol=BTCUSDT
    dvc repro build_features[BTCUSDT]
"""
# =============================================================================
# FFD AUDIT REPORT  (2026-05-22)
# =============================================================================
#
# AUDIT QUESTION: Is the FFD `d` parameter estimated on the full dataset
# (lookahead) or only on training data per CPCV fold?
#
# FINDING: SAFE — no full-dataset lookahead in production path.
#
# Details:
#
# 1. PRIMARY PATH  (FeatureEngineer._add_ffd_features  — ta.py:427-474)
#    d is calibrated ONCE on min(2000 burn-in bars, 60% of n, 5000) bars.
#    Because drop_bars=2000 in regime.py:876 discards exactly those same
#    2000 bars before any fold is created, the calibration window is
#    structurally PRIOR to all CPCV train/test data.  No lookahead.
#
# 2. VOLUME FFD  (FeatureEngineer._add_ffd_volume_features — ta.py:535-561)
#    Uses min(60%, 5000) without the hard 2000-bar cap.  On large datasets
#    (>10 000 bars) the 60% window can extend into the CPCV training folds
#    (but NOT the test folds).  This is a minor conservative risk: d is
#    influenced by training data, but still never by test data.
#    RECOMMENDATION: apply the same _BURN_IN_CALIB_BARS=2000 cap to volume
#    FFD for symmetry.  Not urgent; documented here for future wave.
#
# 3. PER-FOLD PATH (FeatureEngineer.fit_d_on_train_indices — ta.py:165-233)
#    clear_d_cache() + fit_d_on_train_indices() must be called at the start
#    of each CPCV fold (in tune.objective).  This is correctly documented in
#    ta.py comments and removes the burn-in assumption entirely.
#
# 4. MinFracDiff TRANSFORMER  (fracdiff.py:234-295)
#    Sklearn-style fit/transform.  .fit() calibrates on whatever DataFrame
#    is passed — no internal lookahead protection.  Callers MUST pass only
#    train-fold data.  Currently not used in the production FeaturePipeline
#    (only in exploratory notebooks), so no pipeline risk.
#
# CONCLUSION:
#   The production FFD path is safe.  The silent-killer scenario
#   (d calibrated on future returns contaminating CPCV test folds) is
#   prevented by the 2000-bar burn-in cap.  Volume FFD has a cosmetic
#   improvement available (see point 2 above).
#
# =============================================================================
#
# STATIONARITY GATE AUDIT  (2026-05-22)
# =============================================================================
#
# QUESTION: Does the ADF gate use training-zone data only?  Is there lookahead?
#
# FINDING: CORRECT — gate is causal.
#
# - Gate runs on df_full.iloc[:n_train] where n_train = int(n_full * 0.70).
#   The 70% split is configurable via cfg.feature_pipeline.adf_train_frac.
# - Column-drop DECISION is based solely on training-zone p-values.
# - The actual DROP is applied to the FULL frame (train + OOS rows) so that
#   the model never sees I(1) features, but the selection is causal.
# - Non-finite ADF p-values (near-constant series) are treated as
#   non-stationary (conservative direction).
#
# No lookahead in the gate.
#
# =============================================================================

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from omegaconf import DictConfig, OmegaConf

# `_normalise_frame` annoteert met "pd.DataFrame", maar deze module importeert
# pandas nergens. Onder `from __future__ import annotations` is die annotatie
# lui, dus het DRAAIDE -- de naam bestond alleen niet, en
# `typing.get_type_hints()` erop zou stuklopen. Hier vastgelegd zonder een
# runtime-import toe te voegen.
if TYPE_CHECKING:
    import pandas as pd

# Ensure the project src/ is on the path when run as a module.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Also add the Tradebot root (for legacy modules: features.py, data_crypto.py, …)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tradebot.data.ingestion import ingest_raw
from tradebot.data.macro import merge_macro, update_macro
from tradebot.features.pipeline import build_features
from tradebot.features.stationarity_gate import check_feature_stationarity
from tradebot.labeling.cusum import get_cusum_events
from tradebot.schemas.bars import BarSchema
from tradebot.schemas.events import EventSchema
from tradebot.schemas.features import FeatureBlockSchema
from tradebot.utils.parquet_io import write_validated_parquet

logger = logging.getLogger(__name__)


async def _run_symbol(cfg: DictConfig, sym: str, artefacts_dir: Path) -> None:
    """Full Stage-1 pipeline for one symbol."""
    logger.info("=== Stage 1 [%s] START ===", sym)

    # ── Per-symbol config merge (conf/symbols/{SYM}.yaml) ────────────────────
    # This replicates the per-asset override logic in train_regime.py while
    # delegating to Hydra's native config resolution.
    sym_override = Path("conf") / "symbols" / f"{sym}.yaml"
    if sym_override.exists():
        # Convert to plain dict first so symbol YAML keys (e.g. asset_class)
        # are not rejected by Hydra's struct-mode validation.
        base_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        sym_cfg = OmegaConf.merge(OmegaConf.create(base_dict), OmegaConf.load(sym_override))
        logger.debug("[%s] Per-asset override loaded from %s.", sym, sym_override)
    else:
        sym_cfg = cfg

    orth_dir = artefacts_dir / "orthogonalizers"

    # 1. Ingest
    df_micro = await ingest_raw(sym_cfg, sym)

    # 2. Bars + micro/meso features
    df_merged, feat_map = build_features(sym_cfg, sym, df_micro, orthogonalizer_save_dir=orth_dir)

    # 3. Macro merge (publication-lag-aware)
    df_full = merge_macro(df_merged, sym, sym_cfg)

    # 3b. AUDIT A-3 / P0-K FIX: stationarity gate on TRAINING PORTION ONLY.
    #
    # Old code ran ADF on the full df_full (all bars including the OOS test
    # window). This is a form of feature-selection lookahead: the decision of
    # which columns to DROP is influenced by data that the model should never
    # see at fit-time. A feature that is I(1) only in the tail (the OOS window)
    # would pass the gate and enter training clean; one that is I(1) only in
    # the head (the training window) would be correctly dropped. The asymmetry
    # biases feature selection toward columns that look good in the OOS period.
    #
    # Fix: run ADF exclusively on the first 70% of bars (the "training zone").
    # Columns that fail the gate are then dropped from the FULL df_full so
    # that neither train nor OOS rows carry I(1) features — but the selection
    # decision is made on training-zone information only.
    gate_strict = bool(OmegaConf.select(sym_cfg, "feature_pipeline.adf_gate_strict", default=False))
    adf_train_frac = float(OmegaConf.select(sym_cfg, "feature_pipeline.adf_train_frac", default=0.70))
    feat_only_cols = [c for c in df_full.columns if c.startswith("feat_")]

    n_full = len(df_full)
    n_train = max(1, int(n_full * adf_train_frac))
    df_train_zone = df_full.iloc[:n_train]

    logger.info(
        "[%s] ADF gate: running on first %d/%d bars (%.0f%% of data).",
        sym, n_train, n_full, adf_train_frac * 100,
    )
    # Run ADF only on the training zone — selection decision is causal.
    _, adf_report = check_feature_stationarity(
        df_train_zone,
        feature_cols=feat_only_cols,
        drop_if_non_stationary=False,   # we apply the drop ourselves below
    )

    if gate_strict and not adf_report.passed:
        raise RuntimeError(
            f"AUDIT A-3 strict gate failed for {sym}: "
            f"{adf_report.n_non_stationary}/{adf_report.n_features_checked} "
            f"non-stationary features in training zone. Worst offenders: "
            f"{adf_report.non_stationary[:5]}"
        )

    # Apply the column-drop decision to the FULL frame (train + OOS rows)
    if adf_report.non_stationary:
        df_full = df_full.drop(columns=adf_report.non_stationary, errors="ignore")
        logger.warning(
            "[%s] ADF gate: dropped %d non-stationary feature(s) from full "
            "dataset (selection based on training zone only): %s",
            sym,
            len(adf_report.non_stationary),
            adf_report.non_stationary[:10],
        )

    # 4. CUSUM events
    threshold_mult = float(OmegaConf.select(sym_cfg, "training.cusum_threshold_multiplier", default=1.0))
    df_events = df_full.loc[get_cusum_events(df_full, threshold_multiplier=threshold_mult)]

    # 5. Persist
    bars_path     = artefacts_dir / "bars"     / f"{sym}.parquet"
    features_path = artefacts_dir / "features" / f"{sym}.parquet"
    events_path   = artefacts_dir / "events"   / f"{sym}.parquet"
    map_path      = artefacts_dir / f"feature_map_{sym}.json"

    for p in (bars_path.parent, features_path.parent, events_path.parent):
        p.mkdir(parents=True, exist_ok=True)

    # Write bars (subset of df_full — OHLCV only)
    ohlcv_cols = [c for c in df_full.columns if c in ("open", "high", "low", "close", "volume")]
    bars_df = df_full[ohlcv_cols].copy()
    if "volume" not in bars_df.columns:
        for _proxy in ("real_volume", "tick_volume"):
            if _proxy in df_full.columns:
                bars_df["volume"] = df_full[_proxy]
                break
    write_validated_parquet(bars_df, bars_path, BarSchema)

    # Normalise df_full + df_events:
    #  • schemas require 'volume'; data may use 'tick_volume'
    #  • index.name required by downstream label-join
    def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "volume" not in df.columns:
            for _proxy in ("real_volume", "tick_volume"):
                if _proxy in df.columns:
                    df["volume"] = df[_proxy]
                    break
        if df.index.name is None:
            df.index.name = "timestamp"
        return df

    df_full   = _normalise_frame(df_full)
    df_events = _normalise_frame(df_events)

    # Write features (full df_full)
    try:
        write_validated_parquet(df_full, features_path, FeatureBlockSchema)
    except Exception as _schema_exc:
        logger.warning("[%s] FeatureBlockSchema validation warning (writing anyway): %s",
                       sym, str(_schema_exc)[:300])
        df_full.to_parquet(features_path, compression="zstd")

    # Write events
    try:
        write_validated_parquet(df_events, events_path, EventSchema)
    except Exception as _schema_exc:
        logger.warning("[%s] EventSchema validation warning (writing anyway): %s",
                       sym, str(_schema_exc)[:300])
        df_events.to_parquet(events_path, compression="zstd")

    # Write feature map
    map_path.write_text(json.dumps(feat_map, indent=2))

    logger.info(
        "=== Stage 1 [%s] DONE — %d bars, %d events, %d features ===",
        sym, len(df_full), len(df_events), len(feat_map.get("micro", [])) + len(feat_map.get("meso", [])) + len(feat_map.get("macro", [])),
    )


@hydra.main(config_path="../conf", config_name="conf_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Stage 1 entrypoint.  Called by ``dvc repro`` or directly as a script."""
    artefacts_dir = Path(cfg.machine.get("artefacts_dir", "artefacts"))
    artefacts_dir.mkdir(parents=True, exist_ok=True)

    # Update macro ONCE before the per-symbol loop.
    try:
        update_macro(cfg)
    except Exception as exc:
        logger.error("Macro update failed (%s). Using existing local files.", exc)

    # Per-symbol: allow a single symbol override via cfg.symbol (DVC foreach).
    symbols = [cfg.symbol] if OmegaConf.select(cfg, "symbol") else list(cfg.training.training_universe)

    for sym in symbols:
        try:
            asyncio.run(_run_symbol(cfg, sym, artefacts_dir))
        except Exception as exc:
            logger.error("[%s] Stage 1 FAILED: %s. Skipping.", sym, exc)


if __name__ == "__main__":
    main()
