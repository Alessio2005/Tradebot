# src/tradebot/live/judge_gate.py
"""Meta-labeling JudgeGate — secondary CatBoost filter on primary signals.

Shared between paper_trade_runner (historical replay) and live_paper_trader
(real-time shadow mode).  Mirrors the feature construction used in
train/meta_train.py:build_judge_features():

    judge_x = [raw_features[:n_feat-5], prob_primary, hour_sin, hour_cos, dow_sin, dow_cos]

Judge models are stored in artefacts/judge_models/{SYM}_{SIDE}_judge.cbm.
Feature counts vary by symbol (128 or 131) due to NaN-filtering in CPCV.
The gate uses the model's own n_feat to slice the raw feature vector.

Thresholds:
    LONG  tau = 0.40  (P_primary × P_judge > 0.40)
    SHORT tau = 0.75  (P_primary × P_judge > 0.75)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["JudgeGate", "build_judge_gates"]

_JUDGE_TAU_LONG: float = 0.40
# B-1 FIX (2026-05-27): SHORT tau lowered from 0.75 → 0.40.
# With tau=0.75, a SHORT is only passable when cal_prob × p_judge > 0.75.
# Because the Platt calibrator caps SHORT cal_prob at ~0.73–0.75, the product
# cal_prob × p_judge ≤ 0.73 × 1.0 = 0.73 < 0.75 → mathematically impossible.
# No SHORT trade would ever execute regardless of feature quality.
# Lowering to 0.40 (same as LONG) restores SHORT signal flow while still
# requiring p_judge > 0.40 / cal_prob ≈ 0.55 — a non-trivial quality bar.
# The deferred calibration fix (LINK SHORT Platt bias) should raise SHORT
# cal_prob above 0.75 in a future retrain; at that point tau can be re-raised.
_JUDGE_TAU_SHORT: float = 0.40


class JudgeGate:
    """Secondary CatBoost meta-labeling filter.

    Parameters
    ----------
    sym : Trading symbol (e.g. "ETHUSDT").
    side : "LONG" or "SHORT".
    judge_dir : Directory containing ``{sym}_{side}_judge.cbm`` files.
    artefacts_dir : Root artefacts directory.  Used to load the training
        feature parquet column order so ``passes()`` can reindex live
        features to the exact layout seen during judge training.
        (P0-3 FIX 2026-05-27: positional slice breaks when live
        FeaturePipeline produces a different number of columns than the
        training parquet, e.g. when spot fallback omits fundingRate/volume.)
    """

    def __init__(
        self,
        sym: str,
        side: str,
        judge_dir: Path,
        artefacts_dir: Path | None = None,
    ) -> None:
        from catboost import CatBoostClassifier

        path = judge_dir / f"{sym}_{side}_judge.cbm"
        if not path.exists():
            raise FileNotFoundError(f"JudgeGate: model not found: {path}")
        self._model = CatBoostClassifier()
        self._model.load_model(str(path))
        self._n_feat = len(self._model.feature_names_)
        self._tau = _JUDGE_TAU_LONG if side == "LONG" else _JUDGE_TAU_SHORT
        self._sym = sym
        self._side = side

        # P0-3 FIX: load training column order from artefacts/features/{sym}.parquet
        # so passes() reindexes live features to match training layout exactly.
        # This is needed because:
        #   • The judge model was trained on ALL columns of the training parquet
        #     (116 feat_* + 10 raw including fundingRate, volume) — total 126 raw.
        #   • The live FeaturePipeline may produce fewer columns (e.g. no fundingRate
        #     when SPOT_FALLBACK=1).  Positional [:n_feat-5] then slices wrong values.
        self._train_columns: list | None = None
        if artefacts_dir is not None:
            feat_parquet = artefacts_dir / "features" / f"{sym}.parquet"
            if feat_parquet.exists():
                import pyarrow.parquet as _pq
                self._train_columns = list(_pq.read_schema(str(feat_parquet)).names)

        logger.info(
            "JudgeGate [%s/%s]: loaded — %d features, tau=%.2f, "
            "train_cols=%s",
            sym, side, self._n_feat, self._tau,
            len(self._train_columns) if self._train_columns else "unknown",
        )

    def passes(
        self,
        row_df: pd.DataFrame,
        cal_prob: float,
        ts: pd.Timestamp,
    ) -> tuple[bool, float]:
        """Return (passes_gate, p_combined).

        Parameters
        ----------
        row_df  : Feature row from FeatureUpdater / feature parquet.
        cal_prob: Platt-calibrated primary signal probability.
        ts      : Bar timestamp (used for hour/DOW cyclic features).
        """
        h = ts.hour
        dow = ts.dayofweek

        # P0-3 FIX (2026-05-27): reindex live features to training column layout
        # so the positional slice [:n_feat-5] gets the correct features.
        # Missing columns (e.g. fundingRate on spot fallback) are filled with 0.
        if self._train_columns is not None:
            n_raw = self._n_feat - 5
            train_raw_cols = self._train_columns[:n_raw]
            if isinstance(row_df, pd.DataFrame):
                aligned = row_df.reindex(columns=train_raw_cols, fill_value=0.0)
                x_raw = aligned.values.flatten().astype(np.float32)
            else:
                # Series: reindex by label
                aligned = row_df.reindex(index=train_raw_cols, fill_value=0.0)
                x_raw = aligned.values.astype(np.float32)
        else:
            # Fallback: positional slice (may fail if column count differs)
            x_raw = row_df.values.flatten()[: self._n_feat - 5].astype(np.float32)

        time_feats = np.array(
            [
                np.sin(2 * np.pi * h / 24),
                np.cos(2 * np.pi * h / 24),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
            ],
            dtype=np.float32,
        )
        judge_x = np.concatenate([x_raw, [float(cal_prob)], time_feats]).reshape(1, -1)
        judge_x = np.nan_to_num(judge_x, nan=0.0, posinf=0.0, neginf=0.0)
        p_judge = float(self._model.predict_proba(judge_x)[0][1])
        p_combined = cal_prob * p_judge
        return p_combined > self._tau, p_combined


def build_judge_gates(
    symbols: list[str],
    judge_dir: Path,
    artefacts_dir: Path | None = None,
) -> dict[str, dict[str, JudgeGate]]:
    """Load all judge models for the given symbols.

    Returns a nested dict: {symbol: {"LONG": gate, "SHORT": gate}}.
    Missing models are skipped with a warning (gate absent → all signals pass).

    Parameters
    ----------
    artefacts_dir : Passed to JudgeGate for loading the training column layout
        (P0-3 fix — prevents CatBoostError when live columns differ from training).
    """
    gates: dict[str, dict[str, JudgeGate]] = {}
    for sym in symbols:
        sym_gates: dict[str, JudgeGate] = {}
        for side in ("LONG", "SHORT"):
            try:
                sym_gates[side] = JudgeGate(sym, side, judge_dir, artefacts_dir)
            except FileNotFoundError as exc:
                logger.warning("JudgeGate: skipping %s/%s — %s", sym, side, exc)
        if sym_gates:
            gates[sym] = sym_gates
    return gates
