# src/tradebot/live/model_signal.py
"""Live CatBoost + Platt signal wrapper — per-symbol, per-side.

Connects the trained CPCV ContextualBanditEnsemble to the live AlphaSignal
protocol. On every bar-close:

  1. Step the incremental CUSUM state machine (cheap O(1)).
  2. Call ensemble.update_live_scalers() every bar (keeps bandit weights warm).
  3. When CUSUM fires: extract feat_map features, call predict_greybox_strategy,
     apply Platt calibration, gate by min_conf and EMA-200 regime.
  4. Return a SignalResult for the bar (signal=0 if no trade).

Feature parity:
  The CPCV models were trained on raw feat_* columns from
  artefacts/features/{SYM}.parquet — no external PCA/scaling is applied
  before the CatBoost model (CatBoost handles normalisation internally).
  Live features are extracted from the FeatureUpdater's output DataFrame
  using the same feat_map stored in artefacts/feature_map_{SYM}.json.

Warm-start:
  On the first predict() call the CUSUM accumulator and EMA-200 are
  bootstrapped by replaying the last WARMUP_BARS rows of the feature parquet.
  This avoids the first-bar cold-start bias (s_pos/s_neg both zero, which
  biases the first event towards bars immediately after deployment).

EMA-200 regime filter (SHORT only):
  When close > EMA(close, 200) at CUSUM event time, any SHORT signal is
  suppressed.  This mirrors the regime filter in backtest_portfolio.py
  (_apply_short_regime_filter) and prevents bull-regime false shorts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from ..alpha.base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["ModelSignal", "ModelSignalConfig"]

# How many bars from the feature parquet to replay at warm-start.
# Must cover the longest ATR lookback (14 bars) + some CUSUM history.
_WARMUP_BARS = 200


class ModelSignalConfig:
    """Configuration for one ModelSignal instance.

    Parameters
    ----------
    symbol :
        Trading pair (e.g. ``"ETHUSDT"``).
    side :
        ``"LONG"`` or ``"SHORT"``.
    artefacts_dir :
        Root path to the artefacts directory
        (contains models/, calibrators/, feature_map_*.json, hparams/, features/).
    cusum_threshold_multiplier :
        ATR multiplier for the CUSUM filter.  Loaded from conf_config.yaml
        ``training.cusum_threshold_per_asset``.
    ema_regime_span :
        EMA span (bars) for the bull-regime filter applied to SHORT signals.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        artefacts_dir: str | Path = "artefacts",
        cusum_threshold_multiplier: float = 2.0,
        ema_regime_span: int = 200,
    ) -> None:
        self.symbol = symbol
        self.side = side.upper()
        self.artefacts_dir = Path(artefacts_dir)
        self.cusum_threshold_multiplier = cusum_threshold_multiplier
        self.ema_regime_span = ema_regime_span


class ModelSignal:
    """Live alpha signal backed by a CPCV bandit ensemble + Platt calibrator.

    Implements the AlphaSignal protocol.

    Parameters
    ----------
    config :
        Signal configuration.
    """

    def __init__(self, config: ModelSignalConfig) -> None:
        self._cfg = config
        sym = config.symbol
        side = config.side
        art = config.artefacts_dir

        # ── Load ensemble model ───────────────────────────────────────────────
        model_path = art / "models" / f"{sym}_{side}_ensemble.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Ensemble model not found: {model_path}. "
                "Run Stage 3 (train_cpcv) first."
            )
        self._ensemble = joblib.load(model_path)
        logger.info("ModelSignal [%s/%s]: loaded ensemble from %s", sym, side, model_path.name)

        # ── Load Platt calibrator ─────────────────────────────────────────────
        cal_path = art / "calibrators" / f"{sym}_{side}_platt.joblib"
        if not cal_path.exists():
            raise FileNotFoundError(
                f"Platt calibrator not found: {cal_path}. "
                "Run Stage 3 (train_cpcv) first."
            )
        self._calibrator = joblib.load(cal_path)
        logger.info("ModelSignal [%s/%s]: loaded calibrator from %s", sym, side, cal_path.name)

        # ── Load feature map ──────────────────────────────────────────────────
        fmap_path = art / f"feature_map_{sym}.json"
        if not fmap_path.exists():
            raise FileNotFoundError(
                f"Feature map not found: {fmap_path}. "
                "Run Stage 1 (build_features) first."
            )
        with open(fmap_path, encoding="utf-8") as fh:
            self._feat_map: dict[str, list[str]] = json.load(fh)
        self._micro_cols: list[str] = self._feat_map.get("micro", [])
        self._meso_cols:  list[str] = self._feat_map.get("meso",  [])
        self._macro_cols: list[str] = self._feat_map.get("macro", [])
        logger.info(
            "ModelSignal [%s/%s]: feature_map loaded — %d micro / %d meso / %d macro",
            sym, side, len(self._micro_cols), len(self._meso_cols), len(self._macro_cols),
        )

        # ── Load min_conf from hparams ────────────────────────────────────────
        hparam_path = art / "hparams" / f"{sym}_{side}.json"
        if hparam_path.exists():
            with open(hparam_path, encoding="utf-8") as fh:
                hparams = json.load(fh)
            self._min_conf: float = float(hparams.get("min_conf", 0.50))
            self._horizon_bars: int = int(hparams.get("horizon", 30))
        else:
            logger.warning(
                "ModelSignal [%s/%s]: hparams not found at %s — using min_conf=0.50.",
                sym, side, hparam_path,
            )
            self._min_conf = 0.50
            self._horizon_bars = 30
        logger.info(
            "ModelSignal [%s/%s]: min_conf=%.4f, horizon=%d bars",
            sym, side, self._min_conf, self._horizon_bars,
        )

        # ── Incremental CUSUM state ───────────────────────────────────────────
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._prev_close: Optional[float] = None
        self._prev_atr14: float = 1e-5

        # ── EMA-200 state for bull-regime filter ──────────────────────────────
        alpha_ema = 2.0 / (config.ema_regime_span + 1.0)
        self._ema_alpha: float = alpha_ema
        self._ema_close: Optional[float] = None  # None until first bar

        # ── Internal state ────────────────────────────────────────────────────
        self._is_warmed_up: bool = False
        self._last_ts: Optional[pd.Timestamp] = None

        # AlphaSignal protocol identity
        self.signal_id: str = f"{sym}_{side}_cpcv_catboost"

    # ------------------------------------------------------------------
    # AlphaSignal protocol
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Warm-start CUSUM state and EMA-200.

        Preference order (CHIEF AUDIT 2026-05-25, P1-2):
          1. Caller-supplied ``df`` if it contains a usable 'close' series —
             this is the LIVE rolling buffer pre-populated by the engine and
             reflects current market prices.
          2. Pre-computed feature parquet at artefacts/features/{SYM}.parquet
             — used to be the only source, but it's frozen at the training
             cutoff (potentially weeks/months old) and produced biased CUSUM
             thresholds vs current vol regime.

        Falling back from (1)→(2) is allowed only when ``df`` is empty.
        """
        art = self._cfg.artefacts_dir
        sym = self._cfg.symbol

        live_buffer_usable = (
            df is not None
            and not df.empty
            and "close" in df.columns
            and df["close"].notna().sum() >= 50  # need enough bars for ATR/EMA seed
        )

        if live_buffer_usable:
            logger.info(
                "ModelSignal [%s/%s]: warm-starting from LIVE buffer "
                "(%d bars, last close=%.4f).",
                sym, self._cfg.side, len(df), float(df["close"].iloc[-1]),
            )
            self._replay_history(df.tail(_WARMUP_BARS))
            self._is_warmed_up = True
            return

        # Fallback: stale features parquet.  Logged as WARNING so operators
        # spot the case where the live buffer was never pre-populated.
        feat_path = art / "features" / f"{sym}.parquet"
        if feat_path.exists():
            feat_df = pd.read_parquet(feat_path)
            warm_df = feat_df.tail(_WARMUP_BARS)
            logger.warning(
                "ModelSignal [%s/%s]: live buffer empty — falling back to "
                "STALE features parquet (%d bars, last close=%.4f). "
                "CUSUM threshold may not reflect current vol regime.",
                sym, self._cfg.side, len(warm_df),
                float(warm_df["close"].iloc[-1]) if "close" in warm_df.columns and not warm_df.empty else 0.0,
            )
            self._replay_history(warm_df)
        else:
            logger.error(
                "ModelSignal [%s/%s]: no live buffer AND no parquet at %s — "
                "CUSUM starts COLD.",
                sym, self._cfg.side, feat_path,
            )

        self._is_warmed_up = True

    def predict(
        self,
        df: pd.DataFrame,
        bar_ts: Optional[pd.Timestamp] = None,
    ) -> SignalResult:
        """Run CUSUM step and optionally generate a trade signal for this bar.

        Parameters
        ----------
        df :
            Single-row DataFrame from FeatureUpdater (last bar's features).
            Must contain 'close' column and ideally all feat_* columns.
            If feat_* columns are absent, the signal is suppressed.
        bar_ts :
            Bar close timestamp (P1-5 fix).  Used as SignalResult.timestamp
            for audit/TCA alignment.

        Returns
        -------
        SignalResult with signal=0 if no CUSUM event or prob < min_conf.
        SignalResult with signal=+1 (LONG) or -1 (SHORT) if signal fires.
        """
        # P1-5: prefer bar timestamp (parity with backtest) over wall-clock.
        # Fallback chain: caller-supplied bar_ts → df.index[-1] → now().
        if bar_ts is None and isinstance(df.index, pd.DatetimeIndex) and len(df.index):
            bar_ts = df.index[-1]
        now = bar_ts if bar_ts is not None else pd.Timestamp.now(tz="UTC")
        sym = self._cfg.symbol

        # ── Extract current close and ATR from feature row ────────────────────
        close = self._extract_scalar(df, "close", default=None)
        if close is None:
            logger.debug("ModelSignal [%s/%s]: no close in feature row — skipping.", sym, self._cfg.side)
            return self._flat_result(now)

        atr14 = self._extract_scalar(df, "feat_vol_gk", default=None)
        if atr14 is None:
            atr14 = self._extract_scalar(df, "feat_atr", default=self._prev_atr14)
        atr14 = float(max(atr14, 1e-8))

        # ── Lazy warm-start (if fit() was not called explicitly) ──────────────
        if not self._is_warmed_up:
            self._replay_history(df)
            self._is_warmed_up = True

        # ── Update EMA-200 ────────────────────────────────────────────────────
        if self._ema_close is None:
            self._ema_close = float(close)
        else:
            self._ema_close = self._ema_alpha * float(close) + (1.0 - self._ema_alpha) * self._ema_close

        # ── Step incremental CUSUM ─────────────────────────────────────────────
        # atr_causal = previous bar's ATR (causal fix: threshold[t] = ATR[t-1])
        atr_causal = float(self._prev_atr14)
        threshold = max(atr_causal * self._cfg.cusum_threshold_multiplier, 1e-8)

        cusum_fired = False
        if self._prev_close is not None:
            diff = float(close) - float(self._prev_close)
            self._s_pos = max(0.0, self._s_pos + diff)
            self._s_neg = min(0.0, self._s_neg + diff)

            if self._s_pos >= threshold or self._s_neg <= -threshold:
                cusum_fired = True
                self._s_pos = 0.0
                self._s_neg = 0.0

        # Advance state for next bar
        self._prev_close = float(close)
        self._prev_atr14 = atr14

        # ── Update bandit live scalers every bar ──────────────────────────────
        # This keeps the Thompson sampling weights warm regardless of CUSUM.
        x_micro, x_meso, x_macro = self._extract_features(df)
        self._ensemble.update_live_scalers(x_micro, x_meso, x_macro)

        # ── Skip prediction if CUSUM did not fire ────────────────────────────
        if not cusum_fired:
            return self._flat_result(now)

        # ── Regime filter for SHORT ───────────────────────────────────────────
        if self._cfg.side == "SHORT":
            bull_regime = (
                self._ema_close is not None and float(close) > self._ema_close
            )
            if bull_regime:
                logger.debug(
                    "ModelSignal [%s/SHORT]: CUSUM fired but price>EMA200 "
                    "(close=%.4f, ema=%.4f) — regime filter suppressed.",
                    sym, close, self._ema_close,
                )
                return self._flat_result(now)

        # ── Predict ───────────────────────────────────────────────────────────
        if x_micro is None or (hasattr(x_micro, '__len__') and len(x_micro) == 0):
            logger.warning(
                "ModelSignal [%s/%s]: CUSUM fired but no feat_* features in row "
                "— FeatureUpdater may not have pipeline_cfg set. Signal suppressed.",
                sym, self._cfg.side,
            )
            return self._flat_result(now)

        raw_result = self._ensemble.predict_greybox_strategy(
            x_micro, x_meso, x_macro
        )

        raw_prob = float(raw_result.get("prob_win", 0.0))

        # ── Platt calibration ─────────────────────────────────────────────────
        calibrated_probs = self._calibrator.predict_proba(
            np.array([raw_prob], dtype=np.float64),
            path_id=None,
        )
        cal_prob = float(calibrated_probs[0])

        # ── Confidence gate ───────────────────────────────────────────────────
        if cal_prob < self._min_conf:
            logger.debug(
                "ModelSignal [%s/%s]: CUSUM fired, cal_prob=%.4f < min_conf=%.4f — no trade.",
                sym, self._cfg.side, cal_prob, self._min_conf,
            )
            return self._flat_result(now)

        # ── Emit signal ───────────────────────────────────────────────────────
        signal_direction = 1.0 if self._cfg.side == "LONG" else -1.0
        logger.info(
            "ModelSignal [%s/%s]: SIGNAL FIRED — raw_prob=%.4f cal_prob=%.4f min_conf=%.4f",
            sym, self._cfg.side, raw_prob, cal_prob, self._min_conf,
        )
        return SignalResult(
            symbol=sym,
            timestamp=now,
            signal=signal_direction,
            confidence=cal_prob,
            horizon_bars=self._horizon_bars,
            signal_id=self.signal_id,
        )

    def predict_on_event(
        self,
        df: pd.DataFrame,
        bar_ts: Optional[pd.Timestamp] = None,
    ) -> SignalResult:
        """AFML-correct prediction: CUSUM already fired externally.

        Called by the live engine when the external CUSUMFilter fires and
        FeaturePipeline has computed FRESH features for this event.  Does NOT
        step the internal CUSUM accumulator — that is the responsibility of
        CUSUMFilter.  All other state (EMA-200, prev_close, prev_atr14) is
        advanced normally.

        Parameters
        ----------
        df :
            Fresh single-row feature DataFrame computed AT this CUSUM event.
        bar_ts :
            Timestamp of the bar that triggered the CUSUM event.  Used as
            SignalResult.timestamp so audit trails and TCA align with the
            backtest convention (signal-at-bar-close, not wall-clock).
            P1-5 fix (CHIEF AUDIT 2026-05-25).  Falls back to
            ``pd.Timestamp.now(tz=UTC)`` only if not provided.
        """
        now = bar_ts if bar_ts is not None else pd.Timestamp.now(tz="UTC")
        sym = self._cfg.symbol

        close = self._extract_scalar(df, "close", default=None)
        if close is None:
            return self._flat_result(now)

        atr14 = self._extract_scalar(df, "feat_vol_gk", default=None)
        if atr14 is None:
            atr14 = self._extract_scalar(df, "feat_atr", default=self._prev_atr14)
        atr14 = float(max(atr14, 1e-8))

        if not self._is_warmed_up:
            self._replay_history(df)
            self._is_warmed_up = True

        # Update EMA-200 (same as predict())
        if self._ema_close is None:
            self._ema_close = float(close)
        else:
            self._ema_close = (
                self._ema_alpha * float(close)
                + (1.0 - self._ema_alpha) * self._ema_close
            )

        # Advance internal state (no CUSUM step — external CUSUMFilter already fired)
        self._prev_close = float(close)
        self._prev_atr14 = atr14

        # Update bandit scalers every event
        x_micro, x_meso, x_macro = self._extract_features(df)
        self._ensemble.update_live_scalers(x_micro, x_meso, x_macro)

        # Regime filter for SHORT
        if self._cfg.side == "SHORT":
            if self._ema_close is not None and float(close) > self._ema_close:
                logger.debug(
                    "ModelSignal [%s/SHORT]: event received but price>EMA200 "
                    "(close=%.4f ema=%.4f) — regime filter suppressed.",
                    sym, close, self._ema_close,
                )
                return self._flat_result(now)

        if x_micro is None or (hasattr(x_micro, "__len__") and len(x_micro) == 0):
            logger.warning(
                "ModelSignal [%s/%s]: event but no feat_* features — suppressed.",
                sym, self._cfg.side,
            )
            return self._flat_result(now)

        raw_result = self._ensemble.predict_greybox_strategy(x_micro, x_meso, x_macro)

        raw_prob = float(raw_result.get("prob_win", 0.0))

        cal_prob = float(
            self._calibrator.predict_proba(
                np.array([raw_prob], dtype=np.float64), path_id=None
            )[0]
        )

        if cal_prob < self._min_conf:
            return self._flat_result(now)

        direction = 1.0 if self._cfg.side == "LONG" else -1.0
        logger.info(
            "ModelSignal [%s/%s]: SIGNAL FIRED (event-driven) — "
            "raw_prob=%.4f cal_prob=%.4f min_conf=%.4f",
            sym, self._cfg.side, raw_prob, cal_prob, self._min_conf,
        )
        return SignalResult(
            symbol=sym,
            timestamp=now,
            signal=direction,
            confidence=cal_prob,
            horizon_bars=self._horizon_bars,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> List[str]:
        """Return all feature column names consumed by this signal."""
        return self._micro_cols + self._meso_cols + self._macro_cols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_scalar(
        self,
        df: pd.DataFrame,
        col: str,
        default: Optional[float],
    ) -> Optional[float]:
        """Extract a single scalar value from the last row of df."""
        if col in df.columns:
            val = df[col].iloc[-1]
            if pd.notna(val):
                return float(val)
        return default

    def _extract_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract micro/meso/macro feature arrays from the last row of df."""
        row = df.iloc[-1]

        def _cols_to_array(cols: list[str]) -> np.ndarray:
            available = [c for c in cols if c in df.columns]
            if not available:
                return np.zeros((1, 0), dtype=np.float32)
            vals = row[available].to_numpy(dtype=np.float32)
            np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            return vals.reshape(1, -1)

        x_micro = _cols_to_array(self._micro_cols)
        x_meso  = _cols_to_array(self._meso_cols)
        x_macro = _cols_to_array(self._macro_cols)
        return x_micro, x_meso, x_macro

    def _replay_history(self, df: pd.DataFrame) -> None:
        """Replay historical bars to warm-start CUSUM state and EMA-200.

        Runs through each bar in chronological order, stepping the CUSUM
        accumulator and EMA without triggering predictions.
        """
        if df.empty or "close" not in df.columns:
            logger.warning(
                "ModelSignal [%s/%s]: _replay_history: no close column — CUSUM stays cold.",
                self._cfg.symbol, self._cfg.side,
            )
            return

        closes = df["close"].to_numpy(dtype=np.float64)

        # Use feat_vol_gk if available, else fallback to TR-ATR
        if "feat_vol_gk" in df.columns:
            atrs = df["feat_vol_gk"].ffill().fillna(1e-5).to_numpy(dtype=np.float64)
        else:
            high = df.get("high", df["close"]).to_numpy(dtype=np.float64)
            low  = df.get("low",  df["close"]).to_numpy(dtype=np.float64)
            prev_c = np.roll(closes, 1)
            tr = np.maximum(high - low, np.abs(high - prev_c))
            atrs = (
                pd.Series(tr).rolling(14, min_periods=1).mean().fillna(1e-5).to_numpy(dtype=np.float64)
            )

        mult = self._cfg.cusum_threshold_multiplier
        s_pos, s_neg = 0.0, 0.0
        ema_close = float(closes[0])
        alpha = self._ema_alpha

        prev_close = closes[0]
        prev_atr   = atrs[0]

        for i in range(1, len(closes)):
            c = float(closes[i])
            # EMA update
            ema_close = alpha * c + (1.0 - alpha) * ema_close
            # CUSUM step (atr_causal = atr[i-1])
            threshold = max(float(prev_atr) * mult, 1e-8)
            diff = c - float(prev_close)
            s_pos = max(0.0, s_pos + diff)
            s_neg = min(0.0, s_neg + diff)
            if s_pos >= threshold or s_neg <= -threshold:
                s_pos = 0.0
                s_neg = 0.0
            prev_close = c
            prev_atr   = float(atrs[i])

        # Commit warm-started state
        self._s_pos = s_pos
        self._s_neg = s_neg
        self._prev_close = float(prev_close)
        self._prev_atr14 = float(prev_atr)
        self._ema_close  = float(ema_close)
        logger.debug(
            "ModelSignal [%s/%s]: warm-start complete — s_pos=%.6f s_neg=%.6f "
            "ema200=%.4f prev_close=%.4f",
            self._cfg.symbol, self._cfg.side,
            self._s_pos, self._s_neg, self._ema_close, self._prev_close,
        )

    def _flat_result(self, ts: pd.Timestamp) -> SignalResult:
        """Return a neutral (no-trade) SignalResult."""
        return SignalResult(
            symbol=self._cfg.symbol,
            timestamp=ts,
            signal=0.0,
            confidence=0.0,
            horizon_bars=self._horizon_bars,
            signal_id=self.signal_id,
        )
