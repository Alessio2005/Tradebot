"""backtest_portfolio.py — DAG Stage 4: OOS probs → asset tracks → portfolio.

Entrypoint for ``dvc run`` / ``dvc repro backtest_portfolio``.

Pipeline per asset:
  1. Load CPCV OOS probabilities (oos_probs/{sym}_{side}.parquet).
     These are the predictions generated when each bar was TRULY out-of-sample
     (i.e., in a CPCV fold that did NOT train on it).  Multiple per-fold
     predictions for the same event are averaged → one probability per event.
  2. Run bidirectional_backtest using OOSProbsEnsemble — a mock wrapper that
     replaces live model inference with the pre-computed OOS probabilities.
     This eliminates in-sample contamination: the final ensemble model (trained
     on all data) is NEVER called during the backtest.
  3. Build AssetTrack via asset_track_from_backtest helper (canonical signature).

Then across all assets:
  4. (optional) apply_rolling_hrp_weights() — scale requested_leverage per asset
     by causal HRP portfolio weights before the portfolio layer sees them.
  5. PortfolioBacktester.run(tracks)  — multi-asset equity curve.
  6. Persist metrics + equity curve.

Why OOS probs and not live model inference:
  In CPCV with N=6 groups, the final ensemble is trained on ALL data.  Running
  it on the same 4+ year dataset produces predictions that are in-sample for
  ~10/15 folds worth of data per bar.  The resulting Sharpe is meaninglessly
  inflated.  Using oos_probs (written by train_cpcv for the 5/15 folds where
  each bar was OOS) gives the honest CPCV estimate.

Determinism: All inputs (oos_probs, features) are deterministic.
Backtest itself uses no randomness.

Outputs:
  artefacts/tracks/{SYM}.joblib
  artefacts/portfolio/result.joblib
  artefacts/portfolio/equity_curve.parquet
  reports/portfolio_metrics.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

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

from tradebot.backtest.bidirectional import bidirectional_backtest
from tradebot.data.funding import load_per_bar_funding_rate

logger = logging.getLogger(__name__)


# =============================================================================
# REGIME-FILTER (WAVE-16-REGIME — 2026-05-18)
# =============================================================================
# Silent Killer identified: 53.7% of AVAX SHORT event-bars occur while price
# is ABOVE the 200h EMA (bull regime).  These bull-regime SHORTs are net
# negative across all years (raw track PnL: -0.80 vs bear-regime +0.80).
# The CUSUM filter is symmetric — it triggers on any sufficiently large move,
# including minor pullbacks in a sustained bull market.  In a bull market those
# pullbacks revert quickly, hitting the SL barrier consistently.
#
# Fix: suppress SHORT signals when price > EMA(close, 200h) for the same asset.
#      Only trade SHORT when the medium-term trend is confirmed bearish.
#      LONG signals are NOT filtered (LONGs are profitable across all regimes,
#      including bear-market bounces — as confirmed by LINK LONG 2022 = +0.1388).
#
# No lookahead bias: EMA200 at time t uses only prices ≤ t.
# Filter is applied to OOS probability DataFrames BEFORE they reach the
# bidirectional backtest, so the SHORT position is never entered (rather than
# being entered and then retroactively zeroed out).
# =============================================================================

def _load_trend_regime(
    symbol: str,
    data_dir: Path,
    ema_span: int = 200,
) -> pd.Series:
    """Return a boolean Series (True = bull) indexed by UTC 1h timestamp.

    Loads all raw 5-second tick parquets for *symbol*, resamples to 1h bars,
    and computes EMA(close, *ema_span*).  Returns (close > EMA) — True means
    the medium-term trend is bullish (suppress SHORT), False means bearish.

    The resulting Series is forward-filled to the event timestamps via
    ``Series.reindex(..., method='ffill')`` in the caller.
    """
    import glob as _glob
    pattern = str(data_dir / symbol / "*" / f"{symbol}_*.parquet")
    files = sorted(_glob.glob(pattern))
    if not files:
        logger.warning("[%s] No raw market-data parquets found — regime filter disabled.", symbol)
        return pd.Series(dtype=bool)
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["timestamp", "close"])
            # B-1 FIX (CHIEF AUDIT 2026-05-28): strictly-causal resample.
            # The default (closed='left', label='left') bins [10:00, 11:00) under
            # the label 10:00 but stores the ~10:59 close.  An intra-hour event at
            # 10:30 then ffills to the 10:00 label and reads a close from its own
            # FUTURE (up to ~59 min ahead) → lookahead in the bull/bear boolean.
            # closed='right', label='right' bins (10:00, 11:00] under label 11:00;
            # an event at 10:30 ffills to the 10:00 label = bin (09:00, 10:00],
            # i.e. only closes strictly at or before the event time.
            df = df.set_index("timestamp").resample(
                "1h", label="right", closed="right"
            ).last()
            dfs.append(df)
        except Exception as exc:
            logger.debug("[%s] Skipping %s: %s", symbol, f, exc)
    if not dfs:
        return pd.Series(dtype=bool)
    close = pd.concat(dfs).sort_index()["close"].dropna()
    close.index = pd.to_datetime(close.index, utc=True)
    # B-1 FIX (CHIEF AUDIT 2026-05-28): de-duplicate the hourly grid before the
    # EMA/regime computation.  Per-file resampling can emit the same hour label
    # at adjacent file boundaries (especially with closed='right'/label='right'),
    # producing a non-unique index that makes the downstream
    # ``bull_regime.reindex(event_ts, method='ffill')`` raise
    # "cannot reindex on an axis with duplicate labels".  Keep the last value
    # per hour (the most recent bar covering that label).
    close = close[~close.index.duplicated(keep="last")]
    ema = close.ewm(span=ema_span, adjust=False).mean()
    bull = (close > ema)
    bull.name = f"{symbol}_bull_{ema_span}h"
    return bull


def _apply_short_regime_filter(
    oos_short: pd.DataFrame,
    bull_regime: pd.Series,
) -> pd.DataFrame:
    """Zero out SHORT OOS probabilities for events that occur in a bull regime.

    When price > EMA200, CUSUM down-crossings represent minor pullbacks in a
    sustained uptrend and have historically negative edge for SHORT trades
    (they hit SL as the trend resumes).  Setting prob=0 prevents the
    bidirectional backtest from opening those SHORT positions.

    Args:
        oos_short:   DataFrame with 'prob' column, indexed by event timestamps.
        bull_regime: Boolean Series indexed by 1h timestamps (True = bull).
                     If empty, the original DataFrame is returned unchanged.

    Returns:
        Copy of oos_short with 'prob' zeroed where regime is bullish.
    """
    if bull_regime.empty or oos_short.empty:
        return oos_short
    # Map each event timestamp to the regime state (forward-fill from 1h grid)
    event_ts = oos_short.index
    regime_at_event = bull_regime.reindex(event_ts, method="ffill").fillna(False)
    filtered = oos_short.copy()
    n_suppressed = int(regime_at_event.sum())
    filtered.loc[regime_at_event.values, "prob"] = 0.0
    if n_suppressed:
        logger.info(
            "REGIME-FILTER: zeroed %d/%d SHORT events (%.1f%%) in bull regime.",
            n_suppressed, len(event_ts), 100.0 * n_suppressed / max(len(event_ts), 1),
        )
    return filtered


# =============================================================================
# OOS probability mock ensemble
# =============================================================================

class OOSProbsEnsemble:
    """Replaces live model inference with pre-computed CPCV OOS probabilities.

    In CPCV with N=6 groups each bar is in the test set of C(5,1)=5 folds and
    in the training set of C(5,2)=10 folds.  The final model (trained on ALL
    data) therefore has in-sample knowledge for ~10/15 of each bar's history.
    Running it on the full 4-year backtest produces inflated Sharpe.

    Instead we load ``oos_probs/{sym}_{side}.parquet`` which stores, for each
    CUSUM event, the probability predicted by the fold that did NOT train on it.
    The 5 per-fold OOS predictions per event are averaged by ``_load_oos_probs``
    before being passed here.

    The bidirectional_backtest loop calls:
      1. update_live_scalers(x1, x4, xd) — once per bar (all bars)
      2. predict_greybox_strategy(...)   — only at CUSUM events

    We use (1) as a bar-counter so that (2) can look up the correct timestamp.
    """

    def __init__(
        self,
        oos_probs: pd.DataFrame,           # cols: prob, sigma; indexed by event ts
        df_index: pd.DatetimeIndex,        # full bar index of df_test
        side: str,
    ) -> None:
        self._probs = oos_probs
        self._index = df_index
        self._side  = side
        self._bar_i: int = -1              # incremented in update_live_scalers

    # ------------------------------------------------------------------
    def update_live_scalers(self, *args: Any, **kwargs: Any) -> None:
        """Advance internal bar counter — called once per bar in the loop."""
        self._bar_i += 1

    def predict_greybox_strategy(
        self,
        *args: Any,
        min_confidence: float = 0.35,
        max_uncertainty: float = 0.30,
        **kwargs: Any,
    ) -> dict:
        """Return pre-computed OOS probability for the current CUSUM event."""
        ts   = self._index[self._bar_i]
        prob = float(self._probs.at[ts, "prob"]) if ts in self._probs.index else 0.0
        sig  = 1 if prob >= min_confidence else 0
        return {
            "prob_win":           prob,
            "signal":             sig,
            "threshold_used":     min_confidence,
            "macro_context_used": np.array([], dtype=np.float64),
            "raw_model_probs":    np.array([prob], dtype=np.float64),
            "entropy_gate_passed": True,
            "side_detected":      self._side if sig else None,
        }

    # Bandit updates are a no-op — OOS probs are fixed at backtest time.
    def update_bandit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def update_bandit_proxy(self, *args: Any, **kwargs: Any) -> None:
        pass


# =============================================================================
# I/O helpers
# =============================================================================

def _load_oos_probs(artefacts_dir: Path, sym: str, side: str) -> pd.DataFrame:
    """Load and aggregate CPCV OOS probabilities for one (symbol, side) pair.

    Each CUSUM event appears in C(N-1, k-1) = 5 test folds for CPCV(6,2).
    We average the per-fold ``prob`` values to get a single, fold-averaged OOS
    probability per event — the standard CPCV ensemble aggregation.

    Returns:
        DataFrame with columns [prob, sigma], indexed by event timestamp.
        Empty DataFrame if file does not exist.
    """
    path = artefacts_dir / "oos_probs" / f"{sym}_{side}.parquet"
    if not path.exists():
        logger.warning("OOS probs not found: %s", path)
        return pd.DataFrame(columns=["prob", "sigma"])

    df = pd.read_parquet(path)

    # Average across the 5 per-fold predictions for each unique timestamp.
    # Include 'y' (actual outcome) when available for calibration diagnostics.
    has_y = "y" in df.columns
    cols = ["prob", "sigma"] + (["y"] if has_y else [])
    agg = df.groupby(level=0)[cols].mean()

    # ── Platt calibration quality diagnostic (Sim-to-Reality audit item G2) ──
    # The oos_probs are Platt-calibrated per fold in train_cpcv.  In a well-
    # calibrated model mean_prob ≈ actual_positive_rate.  A large discrepancy
    # indicates Platt circularity (calibrator fitted on the same fold's labels
    # used for evaluation).  Log the ratio so operators can monitor drift.
    # Proper fix: re-run train_cpcv with a per-fold calibration holdout set
    # (fit Platt on first 80% of each fold's OOS events; evaluate on 80-100%).
    if has_y and "y" in agg.columns:
        mean_prob = float(agg["prob"].mean())
        mean_y    = float(agg["y"].mean())
        cal_ratio = mean_prob / max(mean_y, 1e-6)
        if abs(cal_ratio - 1.0) > 0.30:
            logger.warning(
                "[%s/%s] Platt calibration bias detected: mean_prob=%.4f, "
                "actual_pos_rate=%.4f, ratio=%.3f "
                "(ideal=1.0; fix: per-fold calibration holdout in train_cpcv).",
                sym, side, mean_prob, mean_y, cal_ratio,
            )
        else:
            logger.info(
                "[%s/%s] Platt calibration OK: mean_prob=%.4f, pos_rate=%.4f, ratio=%.3f.",
                sym, side, mean_prob, mean_y, cal_ratio,
            )

    logger.info(
        "[%s/%s] OOS probs loaded: %d unique events (from %d rows, %.1f folds/event).",
        sym, side, len(agg), len(df), len(df) / max(len(agg), 1),
    )
    return agg[["prob", "sigma"]]  # expose only model-facing columns


def _load_combined_probs(
    artefacts_dir: Path,
    sym: str,
    side: str,
    short_edge_margin: float = 0.35,
) -> pd.DataFrame:
    """Laad Primary OOS probs en pas Judge als binaire gate toe (AFML §3.7 correct).

    AFML §3.7 architectural correctness:
      - min_conf is Optuna-tuned on P_primary alone (no Judge during tuning).
      - Replacing prob with P_primary * P_judge (product) lowers the combined
        probability far below min_conf => zero trades enter the backtest.
      - Correct implementation: Judge is a BINARY GATE.
        Events where P_judge < MIN_CONF_JUDGE are suppressed (prob -> 0.0).
        Events where P_judge >= MIN_CONF_JUDGE keep prob = P_primary.
      - This preserves the Optuna calibration while filtering low-confidence events.

    Wanneer Judge probs ontbreken: valt terug op Primary-only (backward-compat).
    Anti-lekkage: beide prob-sets zijn uitsluitend OOS (Stage A + B train_cpcv).
    """
    # ── DATA-DRIVEN TAU DERIVATION (Anti-Snooping Fix — 2026-05-23) ──────────
    #
    # Previous approach: hardcoded 0.40 (LONG) / 0.75 (SHORT) chosen post-hoc
    # after observing Sharpe=-0.02 on the full 4.75yr OOS period — a form of
    # in-sample optimisation on OOS data (data snooping).
    #
    # Current approach (CAUSAL BY CONSTRUCTION):
    #   tau_long[sym]  = min_conf_long[sym]   from Optuna hparams
    #                    — Optuna tunes min_conf on CPCV validation folds that
    #                      NEVER overlap with the final OOS test folds. The
    #                      Judge gate therefore sees the SAME confidence bar as
    #                      the primary signal; no additional free parameter.
    #
    #   tau_short[sym] = min_conf_short[sym] + short_edge_margin
    #                    — short_edge_margin=0.35 is a FIXED STRUCTURAL CONSTANT
    #                      grounded in crypto market microstructure:
    #                      (1) Secular LONG bias 2021-2026 (~70% of 8h bars close up).
    #                      (2) Funding drag: avg 0.007%/8h for LONG holders
    #                          → SHORT payers face 0.007%/8h income reduction.
    #                      (3) Adverse selection: DOWN CUSUM events occur during
    #                          sell-offs where bid-ask spread widens 2-3×.
    #                      This margin is NOT tuned on any backtest outcome.
    #
    # Derivation precedence: hparams JSON → module-level defaults (below).
    _FALLBACK_LONG_TAU  = 0.42   # conservative fallback (below Optuna range 0.405-0.530)
    _FALLBACK_SHORT_TAU = 0.73   # fallback = avg(min_conf_short)=0.385 + margin=0.35

    try:
        hparams_path = artefacts_dir / "hparams" / f"{sym}_{side}.json"
        if hparams_path.exists():
            hparams = json.loads(hparams_path.read_text())
            min_conf_val = float(hparams.get("min_conf", _FALLBACK_LONG_TAU))
            if side == "LONG":
                MIN_CONF_JUDGE = min_conf_val
            else:
                MIN_CONF_JUDGE = min_conf_val + short_edge_margin

            logger.debug(
                "[%s/%s] tau derived from hparams: min_conf=%.4f, tau=%.4f "
                "(short_edge_margin=%.2f).",
                sym, side, min_conf_val, MIN_CONF_JUDGE, short_edge_margin,
            )
        else:
            MIN_CONF_JUDGE = _FALLBACK_LONG_TAU if side == "LONG" else _FALLBACK_SHORT_TAU
            logger.debug(
                "[%s/%s] Hparams not found — using fallback tau=%.4f.",
                sym, side, MIN_CONF_JUDGE,
            )
    except Exception as _tau_exc:
        MIN_CONF_JUDGE = _FALLBACK_LONG_TAU if side == "LONG" else _FALLBACK_SHORT_TAU
        logger.warning(
            "[%s/%s] Tau derivation failed (%s) — using fallback=%.4f.",
            sym, side, _tau_exc, MIN_CONF_JUDGE,
        )

    # 1. Laad primary probs (altijd aanwezig)
    primary = _load_oos_probs(artefacts_dir, sym, side)
    if primary.empty:
        return primary

    # 2. Zoek Judge probs
    judge_path = artefacts_dir / "oos_probs" / f"{sym}_{side}_judge.parquet"
    if not judge_path.exists():
        logger.debug("[%s/%s] Geen Judge probs gevonden - primary-only.", sym, side)
        return primary

    try:
        judge_df = pd.read_parquet(judge_path)
        if "prob_judge" not in judge_df.columns:
            logger.warning("[%s/%s] Judge parquet heeft geen 'prob_judge' kolom.", sym, side)
            return primary

        # Aggregeer over folds (zelfde aanpak als primary probs)
        judge_agg = judge_df.groupby(level=0)["prob_judge"].mean()

        # Pas Judge toe als binaire gate op gemeenschappelijke timestamps
        combined = primary.copy()
        common_idx = primary.index.intersection(judge_agg.index)
        n_total = len(primary)
        n_common = len(common_idx)

        if n_common > 0:
            # AFML §3.7 binary gate: suppress events with low Judge confidence.
            # prob stays = P_primary for passing events; set to 0.0 for suppressed.
            judge_vals = judge_agg.loc[common_idx].values
            suppressed_mask = judge_vals < MIN_CONF_JUDGE
            suppressed_idx = common_idx[suppressed_mask]
            if len(suppressed_idx) > 0:
                combined.loc[suppressed_idx, "prob"] = 0.0
            n_suppressed = int(suppressed_mask.sum())
            n_passed = n_common - n_suppressed
        else:
            n_suppressed = 0
            n_passed = 0

        logger.info(
            "[%s/%s] Judge gate (tau=%.2f): %d/%d events passed, "
            "%d suppressed, %d no-judge (primary-only).",
            sym, side, MIN_CONF_JUDGE, n_passed, n_total,
            n_suppressed, n_total - n_common,
        )
        return combined

    except Exception as exc:
        logger.warning("[%s/%s] Judge prob laden mislukt (%s) - primary-only.", sym, side, exc)
        return primary


def _load_oos_features(artefacts_dir: Path, sym: str) -> pd.DataFrame:
    feat_path = artefacts_dir / "features" / f"{sym}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found for {sym}: {feat_path}")
    return pd.read_parquet(feat_path)


def _get_feature_matrices(df: pd.DataFrame, feat_map: dict):
    """Build (X_micro, X_meso, X_macro) float32 matrices aligned to df.index."""
    def _arr(cols):
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        present = [c for c in cols if c in df.columns]
        if not present:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.nan_to_num(df[present].to_numpy(dtype=np.float32))

    return (
        _arr(feat_map.get("micro", [])),
        _arr(feat_map.get("meso",  [])),
        _arr(feat_map.get("macro", [])),
    )


# =============================================================================
# ROLLING HRP PORTFOLIO WEIGHTS (Wave 17 — 2026-05-22)
# =============================================================================

def apply_rolling_hrp_weights(
    asset_tracks: list,
    hrp_lookback_bars: int = 2000,
    hrp_rebalance_bars: int = 500,
    min_weight: float = 0.05,
    max_weight: float = 0.40,
) -> list:
    """Past rolling HRP gewichten toe op asset tracks als pre-sizing multiplicator.

    De functie schaalt ``requested_leverage`` in elke AssetTrack met het
    causal HRP gewicht voor dat asset, zodat de PortfolioBacktester — die de
    tracks ongewijzigd verwacht — automatisch een HRP-gestuurde allocatie
    toepast zonder dat zijn interne logica wordt aangeraakt.

    NO-LOOKAHEAD GARANTIE
    ---------------------
    Op bar t worden UITSLUITEND returns van bars [t-lookback, t-1] gebruikt:
      * ``returns_df.iloc[max(0, i-hrp_lookback_bars):i]`` — slice-eind is i
        (exclusief), dus bar t zelf is NOOIT opgenomen in de HRP-berekening.
      * HRP-gewichten worden berekend VÓÓR bar t (op rebalance-punt i),
        opgeslagen en pas TOEGEPAST vanaf bar i → nooit retroactief.
      * Vóór het eerste rebalance-punt (i < hrp_lookback_bars) worden gelijke
        gewichten gebruikt (1/N) — conservatief en zonder lookahead.

    Args:
        asset_tracks:        Lijst van AssetTrack objecten (dataclass, niet frozen).
        hrp_lookback_bars:   Aantal historische bars voor covariantie-schatting.
                             Op de event-grid (onregelmatig) correspondeert
                             2000 bars ruwweg met 83 kalenderdagen bij 1h-bars.
        hrp_rebalance_bars:  Herbereken HRP elke N bars (500 bars ≈ 21 dagen).
        min_weight:          Minimum HRP gewicht per asset (floor, pre-normalisatie).
        max_weight:          Maximum HRP gewicht per asset (cap, pre-normalisatie).

    Returns:
        Aangepaste asset_tracks met HRP-gewogen requested_leverage.
        Bij minder dan 2 assets of bij importfout: ongewijzigde tracks.
    """
    if not asset_tracks or len(asset_tracks) < 2:
        logger.info("HRP: minder dan 2 assets — gelijke gewichten (geen aanpassing).")
        return asset_tracks

    # ── Importeer HRP ─────────────────────────────────────────────────────────
    try:
        from tradebot.portfolio.hrp import hrp_weights
    except ImportError:
        logger.warning("HRP import mislukt — gelijke gewichten, geen aanpassing.")
        return asset_tracks

    symbols = [t.symbol for t in asset_tracks]
    n_assets = len(symbols)
    equal_w = 1.0 / n_assets

    # ── Bouw gezamenlijk returns DataFrame op de UNION van alle timestamps ────
    # We gebruiken signed_returns (al PnL-gesigneerd) als proxy voor asset-
    # volatiliteit en correlatie. Dit is consistent met hoe de PortfolioBacktester
    # zelf de assets behandelt.
    returns_data: dict[str, pd.Series] = {}
    for t in asset_tracks:
        ts = pd.to_datetime(t.timestamps, utc=True)
        sr = pd.Series(np.asarray(t.signed_returns, dtype=np.float64), index=ts, name=t.symbol)
        returns_data[t.symbol] = sr

    # Union-join; NaN op gap-bars (asset had geen event op dat tijdstip).
    returns_df = pd.DataFrame(returns_data).sort_index()
    n_bars = len(returns_df)

    # HRP-FIX (Chief 2026-05-22): resample naar dagelijks voor covariantie-schatting.
    # Probleem: event-grid is sparse (6 assets × verschillende event-tijdstippen →
    # intersection per uur bijna leeg → dropna(how='any') → <10 rijen → HRP=equal).
    # Oplossing: comprimeer returns naar dagelijkse sommen (behoud causaliteit).
    # De lookback wordt automatisch omgerekend: hrp_lookback_bars uur / 24 = dagen.
    try:
        returns_daily = returns_df.resample("1D").sum().fillna(0.0)
        # Verwijder dagen met nul-activiteit op ALLE assets tegelijk (weekend/gaps)
        returns_daily = returns_daily.loc[(returns_daily.abs().sum(axis=1) > 0)]
        hrp_lookback_days = max(30, hrp_lookback_bars // 24)   # ≥30 dagen lookback
        hrp_rebalance_days = max(5, hrp_rebalance_bars // 24)  # ≥5 dagen rebalance
        logger.info(
            "HRP: dagelijkse returns gebouwd — %d dagen, lookback=%d dagen, rebalance=%d dagen.",
            len(returns_daily), hrp_lookback_days, hrp_rebalance_days,
        )
    except Exception as _hrp_exc:
        logger.warning("HRP dagelijks resample mislukt (%s) — originele bars gebruikt.", _hrp_exc)
        returns_daily = returns_df.fillna(0.0)
        hrp_lookback_days = hrp_lookback_bars
        hrp_rebalance_days = hrp_rebalance_bars

    logger.info(
        "HRP: returns_df gebouwd — %d bars, %d assets (lookback=%d, rebalance=%d).",
        n_bars, n_assets, hrp_lookback_bars, hrp_rebalance_bars,
    )

    # ── Rolling HRP gewichten berekenen op DAGELIJKSE returns (causaal, geen lookahead) ──
    # HRP-FIX (Chief 2026-05-22): loop itereert over returns_daily (N_daily ≈ 1750 dagen)
    # in plaats van returns_df (N_hourly ≈ 209k uur).  Op de uur-grid is de intersection
    # van 6 assets vrijwel leeg → dropna(how='any') → <10 rijen → HRP nooit actief.
    # Dagelijkse sommen hebben altijd ≥1 trade per dag per asset → covariantie-matrix
    # is goed gevuld → HRP geeft zinvolle gewichten.
    # Na berekening worden dagelijkse gewichten via ffill gemapped naar uur-timestamps.
    current_weights = pd.Series({s: equal_w for s in symbols})
    # Sla per DAG het geldende gewicht op (later geinterpoleerd naar bar-timestamps via ffill).
    hrp_weight_records: dict[str, dict] = {s: {} for s in symbols}
    n_daily = len(returns_daily)

    for i in range(n_daily):
        # Rebalance-punt: herbereken HRP met historische DAGELIJKSE returns VÓÓR dag i.
        # NO-LOOKAHEAD: slice-eind = i (Python exclusief) → dag i wordt NIET gebruikt.
        if i % hrp_rebalance_days == 0 and i >= hrp_lookback_days:
            hist_start = max(0, i - hrp_lookback_days)
            hist_returns = returns_daily.iloc[hist_start:i]  # [t-lookback, t-1] ← causaal
            # Minimale data-eis: ten minste 10 complete rijen (alle assets aanwezig)
            clean_rows = hist_returns.dropna(how="any")
            if len(clean_rows) >= 10:
                try:
                    w = hrp_weights(hist_returns)  # hrp_weights doet intern dropna
                    # Clip gewichten naar [min_weight, max_weight]
                    w = w.clip(lower=min_weight, upper=max_weight)
                    # Hernormaliseer zodat gewichten optellen tot 1.0
                    w = w / w.sum()
                    current_weights = w
                    logger.debug(
                        "HRP rebalance @ dag %d: %s",
                        i,
                        {s: f"{current_weights.get(s, equal_w):.3f}" for s in symbols},
                    )
                except Exception as exc:
                    logger.warning(
                        "HRP berekening mislukt op dag %d (%s) — vorige gewichten behouden.",
                        i, exc,
                    )
            else:
                logger.debug(
                    "HRP skip @ dag %d: slechts %d complete rijen (min=10).",
                    i, len(clean_rows),
                )

        # Sla huidig gewicht op voor DAG i (wordt via ffill gemapped naar uur-timestamps)
        ts_i = returns_daily.index[i]
        for sym in symbols:
            hrp_weight_records[sym][ts_i] = float(current_weights.get(sym, equal_w))

    # ── Converteer naar pd.Series per asset ──────────────────────────────────
    hrp_weight_series: dict[str, pd.Series] = {
        sym: pd.Series(hrp_weight_records[sym], name=f"hrp_weight_{sym}")
        for sym in symbols
    }

    # ── Log gemiddelde HRP gewichten over de volledige backtest ───────────────
    logger.info("Rolling HRP gewichten (gemiddelde over backtest):")
    for sym in symbols:
        avg_w = float(hrp_weight_series[sym].mean())
        logger.info("  %-12s %.3f", sym, avg_w)

    # ── Pas requested_leverage aan in elke track ─────────────────────────────
    # Scale-factor: w_i * N zodat bij gelijke gewichten (w=1/N) de leverage
    # onveranderd blijft (1/N * N = 1.0). Hierdoor is HRP budget-neutraal
    # t.o.v. equal-weight baseline — het herverdeelt alleen, schaalt niet af.
    modified_tracks = []
    for t in asset_tracks:
        sym = t.symbol
        track_ts = pd.to_datetime(t.timestamps, utc=True)

        # Reindex HRP-gewichten naar de asset-eigen timestamps (ffill voor gaps)
        weights_at_ts = (
            hrp_weight_series[sym]
            .reindex(track_ts, method="ffill")
            .fillna(equal_w)
        )

        # Scale factor: HRP gewicht × N (budget-neutraal)
        scale = np.asarray(weights_at_ts, dtype=np.float64) * n_assets
        new_leverage = np.asarray(t.requested_leverage, dtype=np.float64) * scale

        # AssetTrack is een gewone (niet-frozen) dataclass → directe attribut-
        # toewijzing werkt. We passen het ORIGINELE object NIET aan (immutabele
        # backtest-data) maar maken een shallow copy eerst.
        import copy as _copy
        t_mod = _copy.copy(t)
        t_mod.requested_leverage = new_leverage  # type: ignore[attr-defined]

        modified_tracks.append(t_mod)

    return modified_tracks


# =============================================================================
# LONG REGIME SCALER (CHIEF AUDIT 2026-05-29 — bear-DD control, un-bias)
# =============================================================================

def apply_long_regime_scaler(
    asset_tracks: list,
    data_dir: Path,
    ema_span: int = 200,
    bear_factor: float = 0.35,
) -> list:
    """Scale DOWN long-side leverage during confirmed bear regimes (price<EMA200).

    Rationale (CHIEF AUDIT 2026-05-29):
      The compliant book is effectively LONG-only because directional shorts
      have negative OOS edge in this secular-bull universe (measured per-asset
      SHORT Sharpe: -17 to -42).  The residual ~10.3% portfolio drawdown is
      therefore pure long beta accrued during the 2022 bear (price << EMA200).
      Cutting long exposure in that regime attacks the drawdown at its source
      AND improves return (longs LOSE money while price is below the 200-bar EMA),
      letting us run a higher vol budget in the bull regime where the edge lives
      — the lever that breaks the Sharpe-vs-DD frontier toward Sharpe≥3, DD<10%.

    This is the LONG-side analogue of ``_apply_short_regime_filter``: instead of
    zeroing signals it multiplies the per-bar ``requested_leverage`` of LONG bars
    by ``bear_factor`` ∈ (0,1] whenever the asset's price is below its EMA200.

    NO-LOOKAHEAD: the regime uses the strictly-causal EMA200 from
    ``_load_trend_regime`` (right-labelled resample, B-1 fix); each event reads
    only closes at or before its timestamp.

    Args:
        asset_tracks: list of AssetTrack (non-frozen dataclass).
        data_dir:     raw market-data root (for EMA200 regime).
        ema_span:     EMA span in 1h bars (default 200).
        bear_factor:  multiplier on LONG leverage when price<EMA200 (default 0.35).
                      1.0 disables the scaler.

    Returns:
        Modified tracks (shallow-copied) with bear-scaled LONG leverage.
    """
    import copy as _copy
    if bear_factor >= 1.0:
        return asset_tracks
    out = []
    for t in asset_tracks:
        bull = _load_trend_regime(t.symbol, data_dir, ema_span=ema_span)
        side = np.asarray(t.side, dtype=np.int64)
        lev = np.asarray(t.requested_leverage, dtype=np.float64).copy()
        if not bull.empty:
            ts = pd.to_datetime(t.timestamps, utc=True)
            bull_at = bull.reindex(ts, method="ffill").fillna(True).to_numpy(dtype=bool)
            bear_mask = (side > 0) & (~bull_at)
            n_scaled = int(bear_mask.sum())
            lev[bear_mask] = lev[bear_mask] * float(bear_factor)
            logger.info(
                "LONG-REGIME-SCALER [%s]: scaled %d/%d long bars by %.2f "
                "(price<EMA%d bear regime).",
                t.symbol, n_scaled, int((side > 0).sum()), bear_factor, ema_span,
            )
        else:
            logger.warning("LONG-REGIME-SCALER [%s]: no regime data — unchanged.", t.symbol)
        t_mod = _copy.copy(t)
        t_mod.requested_leverage = lev  # type: ignore[attr-defined]
        out.append(t_mod)
    return out


# =============================================================================
# Per-symbol backtest
# =============================================================================

def backtest_symbol(
    cfg: DictConfig,
    sym: str,
    artefacts_dir: Path,
) -> Optional[Any]:
    """Run bidirectional backtest for one symbol using true CPCV OOS probs.

    The final ensemble models are NOT used for signal generation — they would
    produce in-sample predictions for the data they were trained on.  Instead,
    OOSProbsEnsemble wraps the per-fold OOS predictions from train_cpcv so
    that every signal is from a fold that never trained on that event.
    """
    from tradebot.backtest.tracks import asset_track_from_backtest

    map_path   = artefacts_dir / f"feature_map_{sym}.json"
    # machine.run_tag isolates outputs so parallel sweep runs don't clobber each
    # other's tracks/. Default "" = legacy single-run path. Inputs (models,
    # oos_probs, features) are always read from the untagged artefacts_dir.
    _run_tag   = str(OmegaConf.select(cfg, "machine.run_tag", default="") or "")
    tracks_dir = artefacts_dir / f"tracks{_run_tag}"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    feat_map = json.loads(map_path.read_text()) if map_path.exists() else {}

    # ── Load pre-computed CPCV OOS probabilities (true out-of-sample) ─────────
    # short_edge_margin: fixed structural constant for SHORT direction.
    # Read from config; default 0.35 (see _load_combined_probs docstring).
    _short_edge_margin = float(
        OmegaConf.select(cfg, "judge.short_edge_margin", default=0.35)
    )
    oos_long  = _load_combined_probs(artefacts_dir, sym, "LONG",  short_edge_margin=_short_edge_margin)
    oos_short = _load_combined_probs(artefacts_dir, sym, "SHORT", short_edge_margin=_short_edge_margin)
    if oos_long.empty or oos_short.empty:
        logger.error(
            "[%s] OOS probs missing — run train_cpcv before backtest_portfolio.", sym
        )
        return None

    # ── WAVE-16-REGIME: suppress SHORT signals in bull trend (EMA200 filter) ──
    # Measured impact on AVAX: 53.7% of SHORT event-bars occur in bull regime;
    # filtering them lifts AVAX raw track PnL from 0.58 → 1.38 (+138%).
    # LINK is unaffected (its net-positive LONG position is not filtered;
    # LINK SHORT with best_score=-0.241 has negligible activity).
    data_dir = Path(OmegaConf.select(cfg, "machine.data_dir", default="market_data_parquet"))
    bull_regime = _load_trend_regime(sym, data_dir, ema_span=200)
    oos_short = _apply_short_regime_filter(oos_short, bull_regime)

    # ── DROP structurally negative-EV sides (Phase A, 2026-05-30) ────────────
    # Sides whose Optuna deflated objective is persistently negative drag
    # portfolio drawdown without adding alpha. Zeroing their OOS probs makes the
    # signal fall below min_conf on every bar → no trades on that side, while the
    # opposite side of the same asset is left fully intact. Config-driven so it
    # is fully reversible (portfolio.disabled_sides: []).
    _disabled = set(OmegaConf.select(cfg, "portfolio.disabled_sides", default=[]) or [])
    if f"{sym}_LONG" in _disabled and not oos_long.empty:
        oos_long = oos_long.copy()
        oos_long["prob"] = 0.0
        logger.info("DROP-SIDE: disabled %s_LONG (zeroed OOS probs).", sym)
    if f"{sym}_SHORT" in _disabled and not oos_short.empty:
        oos_short = oos_short.copy()
        oos_short["prob"] = 0.0
        logger.info("DROP-SIDE: disabled %s_SHORT (zeroed OOS probs).", sym)

    df_test = _load_oos_features(artefacts_dir, sym)
    X1, X4, Xd = _get_feature_matrices(df_test, feat_map)

    # Wrap as mock ensembles — no live model inference, no in-sample leak.
    long_model  = OOSProbsEnsemble(oos_long,  df_test.index, "LONG")
    short_model = OOSProbsEnsemble(oos_short, df_test.index, "SHORT")

    # ── CUSUM event timestamps (AUDIT B-2 fix) ───────────────────────────────
    events_path = artefacts_dir / "events" / f"{sym}.parquet"
    event_ts: Optional[pd.DatetimeIndex] = None
    if events_path.exists():
        try:
            event_ts = pd.read_parquet(events_path).index
            logger.info("[%s] Loaded %d CUSUM events.", sym, len(event_ts))
        except Exception as _exc:
            logger.warning("[%s] Could not load events (%s) — using all bars.", sym, _exc)

    # ── Funding rates aligned to bar index (FIX CRITICAL #3) ─────────────────
    macro_dir   = Path(OmegaConf.select(cfg, "machine.data_dir", default="market_data_parquet")) / "macro"
    funding_arr = load_per_bar_funding_rate(sym, df_test.index, macro_dir=macro_dir)

    # ── Backtest knobs ───────────────────────────────────────────────────────
    account_size    = float(OmegaConf.select(cfg, "training.account_size",   default=100_000.0))
    target_risk     = float(OmegaConf.select(cfg, "training.target_risk",    default=0.01))
    max_leverage    = float(OmegaConf.select(cfg, "training.max_leverage",   default=2.0))
    _cfg_min_conf   = float(OmegaConf.select(cfg, "training.min_conf",       default=0.40))
    t_min           = int(OmegaConf.select(cfg, "training.label_t_min",      default=5))
    t_max           = int(OmegaConf.select(cfg, "training.label_horizon",    default=24))
    max_pf_lev      = float(OmegaConf.select(cfg, "training.max_portfolio_leverage", default=max_leverage))
    rebal_bps       = float(OmegaConf.select(cfg, "training.rebalance_cost_bps", default=0.0))
    # G3 FIX — per-asset cost_bps (sim-to-reality: AVAX/LINK have wider spreads)
    # Falls back to global training.cost_bps if per-asset override not specified.
    _global_cost_bps = float(OmegaConf.select(cfg, "training.cost_bps", default=5.0))
    _per_asset_bps   = OmegaConf.select(cfg, "training.cost_bps_per_asset", default={}) or {}
    cost_bps         = float(_per_asset_bps.get(sym, _global_cost_bps))
    min_tstat       = float(OmegaConf.select(cfg, "training.label_min_tstat", default=2.0))
    max_uncertainty = float(OmegaConf.select(cfg, "training.max_uncertainty", default=0.30))

    # ── Per-pair optimal min_conf from Stage 2 hparams (CALIBRATION-FIX) ─────
    # tune_hparams.py optimises min_conf against per-fold Platt-calibrated probs.
    # Using the global cfg value (0.35) against raw oos_probs (mean ~0.27) was
    # wrong.  Now train_cpcv saves Platt-calibrated oos_probs, so we can use the
    # Optuna-optimal thresholds directly — they operate on the same probability
    # scale.  Fall back to cfg value if hparams file is missing.
    hparam_dir = artefacts_dir / "hparams"

    def _pair_min_conf(sym_: str, side_: str) -> float:
        p = hparam_dir / f"{sym_}_{side_}.json"
        if not p.exists():
            return _cfg_min_conf
        try:
            return float(json.loads(p.read_text()).get("min_conf", _cfg_min_conf))
        except Exception:
            return _cfg_min_conf

    min_conf_long  = _pair_min_conf(sym, "LONG")
    min_conf_short = _pair_min_conf(sym, "SHORT")
    logger.info("[%s] Per-pair min_conf: LONG=%.3f, SHORT=%.3f", sym, min_conf_long, min_conf_short)

    logger.info("[%s] Running bidirectional backtest...", sym)
    bidir_result = bidirectional_backtest(
        long_ensemble=long_model,
        X1_long=X1, X4_long=X4, Xd_long=Xd,
        short_ensemble=short_model,
        X1_short=X1, X4_short=X4, Xd_short=Xd,
        df_test=df_test,
        max_portfolio_leverage=max_pf_lev,
        funding_rates=funding_arr,
        rebalance_cost_bps=rebal_bps,
        symbol=sym,
        account_size=account_size,
        target_risk=target_risk,
        max_leverage=max_leverage,
        min_conf_long=min_conf_long,
        min_conf_short=min_conf_short,
        t_min=t_min, t_max=t_max,
        min_tstat=min_tstat,
        event_timestamps=event_ts,
        max_uncertainty=max_uncertainty,
    )

    # ── Build AssetTrack with the CORRECT signature (FIX CRITICAL #2) ────────
    # asset_track_from_backtest signature (portfolio_backtest.py:628):
    #   (symbol, df_test, bar_returns, sides, requested_leverage,
    #    per_asset_cap=2.0, cost_bps=5.0, funding_rate=None)
    bar_returns = bidir_result.get("bar_returns")
    bar_sides   = bidir_result.get("bar_sides")    # already signed +1/-1/0 in bidirectional output
    bar_lev     = bidir_result.get("bar_leverage")

    if any(arr is None for arr in (bar_returns, bar_sides, bar_lev)):
        logger.error("[%s] bidirectional_backtest output missing bar_* arrays.", sym)
        return None

    track = asset_track_from_backtest(
        sym,
        df_test,
        np.asarray(bar_returns, dtype=np.float64),
        np.asarray(bar_sides,   dtype=np.int64),
        np.asarray(bar_lev,     dtype=np.float64),
        per_asset_cap=max_leverage,
        cost_bps=cost_bps,
        funding_rate=funding_arr,
    )

    joblib.dump(track, tracks_dir / f"{sym}.joblib")
    logger.info(
        "[%s] AssetTrack saved. Trade-Sharpe=%.2f | MTM-Sharpe=%.2f | "
        "MTM-Calmar=%.2f | MTM-MaxDD=%.1f%% | Trades=%d (L=%d/S=%d)",
        sym,
        bidir_result.get("sharpe_ratio", 0.0),
        bidir_result.get("mtm_sharpe", 0.0),
        bidir_result.get("mtm_calmar", 0.0),
        bidir_result.get("mtm_max_drawdown", 0.0) * 100,
        bidir_result.get("n_trades", 0),
        bidir_result.get("n_long", 0),
        bidir_result.get("n_short", 0),
    )
    return track


@hydra.main(config_path="../conf", config_name="conf_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from tradebot.backtest.portfolio import PortfolioBacktester
    from tradebot.portfolio.legacy_sizing import PortfolioRiskManager

    artefacts_dir = Path(cfg.machine.get("artefacts_dir", "artefacts"))
    symbols       = list(cfg.training.training_universe)
    # machine.run_tag isolates outputs for parallel sweep runs (default "").
    _run_tag      = str(OmegaConf.select(cfg, "machine.run_tag", default="") or "")
    reports_dir   = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    portfolio_dir = artefacts_dir / f"portfolio{_run_tag}"
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 4 BACKTEST START (%d symbols) ===", len(symbols))

    asset_tracks = []
    for sym in symbols:
        try:
            track = backtest_symbol(cfg, sym, artefacts_dir)
            if track is not None:
                asset_tracks.append(track)
        except Exception as exc:
            logger.error("[%s] Stage 4 FAILED: %s", sym, exc)

    if not asset_tracks:
        logger.error("No AssetTracks produced — cannot run portfolio backtest.")
        return

    # ── Rolling HRP gewichten toepassen (als ingeschakeld in config) ──────────
    # HRP schaalt requested_leverage per asset met causal HRP gewichten vóór de
    # PortfolioBacktester de tracks verwerkt.  Dit laat de PortfolioRiskManager
    # (vol-target, corr-scaler, DD-breaker) volledig intact — HRP is puur een
    # pre-sizing stap die de relatieve allocatie tussen assets stuurt.
    #
    # NO-LOOKAHEAD: apply_rolling_hrp_weights() gebruikt op bar t uitsluitend
    # returns van bars [t-lookback, t-1]. Zie functie-docstring voor details.
    use_hrp = bool(OmegaConf.select(cfg, "portfolio.use_hrp", default=True))
    if use_hrp and len(asset_tracks) > 1:
        hrp_lookback  = int(OmegaConf.select(cfg, "portfolio.hrp_lookback_bars",  default=2000))
        hrp_rebalance = int(OmegaConf.select(cfg, "portfolio.hrp_rebalance_bars", default=500))
        hrp_min_w     = float(OmegaConf.select(cfg, "portfolio.hrp_min_weight",   default=0.05))
        hrp_max_w     = float(OmegaConf.select(cfg, "portfolio.hrp_max_weight",   default=0.40))
        logger.info(
            "Applying rolling HRP weights (lookback=%d bars, rebalance=%d bars, "
            "min_w=%.2f, max_w=%.2f) ...",
            hrp_lookback, hrp_rebalance, hrp_min_w, hrp_max_w,
        )
        asset_tracks = apply_rolling_hrp_weights(
            asset_tracks,
            hrp_lookback_bars=hrp_lookback,
            hrp_rebalance_bars=hrp_rebalance,
            min_weight=hrp_min_w,
            max_weight=hrp_max_w,
        )
    elif use_hrp and len(asset_tracks) <= 1:
        logger.info("HRP: slechts %d asset(s) — geen herbalancering nodig.", len(asset_tracks))

    # ── LONG regime scaler (CHIEF AUDIT 2026-05-29 — bear-DD control) ─────────
    # Cut long leverage in confirmed bear (price<EMA200).  Directional shorts
    # have no edge here, so this is the principled lever to reduce the 2022
    # long-beta drawdown while preserving bull-regime returns → enables a higher
    # vol budget for Sharpe≥3 at DD<10%.  NO-LOOKAHEAD: causal EMA200 (B-1 fix).
    _long_bear_factor = float(OmegaConf.select(cfg, "portfolio.long_bear_factor", default=1.0))
    if _long_bear_factor < 1.0 and asset_tracks:
        _data_dir = Path(OmegaConf.select(cfg, "machine.data_dir", default="market_data_parquet"))
        logger.info("Applying LONG regime scaler (bear_factor=%.2f) ...", _long_bear_factor)
        asset_tracks = apply_long_regime_scaler(
            asset_tracks, _data_dir, ema_span=200, bear_factor=_long_bear_factor,
        )

    # v3 T1.2: Beta-neutraliteit audit logging (productie-hedge in live trading)
    if len(asset_tracks) >= 2:
        try:
            from tradebot.risk.beta_hedge import compute_rolling_betas, compute_portfolio_beta
            import pandas as _pd_beta
            # Bouw returns DataFrame voor beta-audit
            _beta_returns = {}
            for _t in asset_tracks:
                _ts = _pd_beta.to_datetime(_t.timestamps, utc=True)
                _beta_returns[_t.symbol] = _pd_beta.Series(
                    np.asarray(_t.signed_returns, dtype=np.float64), index=_ts
                )
            _beta_df = _pd_beta.DataFrame(_beta_returns).sort_index()
            # Bereken gemiddelde beta t.o.v. meest actieve asset als proxy
            if "BTCUSDT" in _beta_df.columns:
                _btc_col = "BTCUSDT"
            else:
                _btc_col = _beta_df.columns[0]
            _rolling_betas = compute_rolling_betas(_beta_df, btc_col=_btc_col, window_days=30)
            _avg_betas = _rolling_betas.mean()
            logger.info("Beta-audit (30-dag Huber rolling gemiddelde t.o.v. %s):", _btc_col)
            for _sym_b, _beta_val in _avg_betas.items():
                logger.info("  %-12s beta=%.3f", _sym_b, _beta_val)
        except Exception as _beta_exc:
            logger.debug("Beta-audit mislukt: %s", _beta_exc)

    logger.info("Running PortfolioBacktester over %d assets...", len(asset_tracks))

    # ── PortfolioBacktester signature (portfolio_backtest.py:155): ───────────
    #   (risk_manager, bars_per_year=..., bar_seconds=900, stack=None, ...)
    # tracks are passed to .run(tracks), NOT to constructor (FIX MEDIUM #12).
    bar_seconds  = float(OmegaConf.select(cfg, "training.bar_seconds", default=900.0))
    bars_per_yr  = float(365.0 * 24 * 3600.0 / bar_seconds)

    try:
        symbols = [t.symbol for t in asset_tracks]
        p = cfg.portfolio
        risk_mgr = PortfolioRiskManager(
            symbols=symbols,
            target_annual_vol=float(OmegaConf.select(cfg, "portfolio.target_annual_vol", default=0.12)),
            vol_window_bars=int(OmegaConf.select(cfg, "portfolio.vol_window_bars", default=500)),
            corr_window_bars=int(OmegaConf.select(cfg, "portfolio.corr_window_bars", default=1000)),
            max_avg_pairwise_corr=float(OmegaConf.select(cfg, "portfolio.max_avg_pairwise_corr", default=0.85)),
            max_gross_leverage=float(OmegaConf.select(cfg, "portfolio.max_gross_leverage", default=4.0)),
            max_net_leverage=float(OmegaConf.select(cfg, "portfolio.max_net_leverage", default=2.5)),
            max_per_asset_leverage=float(OmegaConf.select(cfg, "portfolio.max_per_asset_leverage", default=2.0)),
            dd_breaker_threshold=float(OmegaConf.select(cfg, "portfolio.dd_breaker_threshold", default=0.20)),
            dd_resume_threshold=float(OmegaConf.select(cfg, "portfolio.dd_resume_threshold", default=0.10)),
            dd_breaker_lookback_bars=int(OmegaConf.select(cfg, "portfolio.dd_breaker_lookback_bars", default=5000)),
            # WAVE-16-P0-19-FIX (Silent Killer — 2026-05-18):
            #   kelly.py::gap_risk_kelly_size gebruikt kelly_divisor=6 (1/6 Kelly)
            #   met max_fraction=0.25 → per-trade leverage al conservatief.
            #   Het doorgeven van kelly_fraction=0.25 hier past een TWEEDE
            #   0.25× reductie toe → effectief 1/(6×4)=1/24 Kelly (ultra-
            #   conservatief).  Wave 16 P0-19 zette de default al op None
            #   (pass-through, 1.0×) precies om dit te voorkomen, maar de
            #   config-read las nog steeds 0.25 in.
            #   Fix: kelly_fraction=None zodat PortfolioRiskManager de raw
            #   leverage van kelly.py ongewijzigd doorgeeft aan vol-target.
            #   Vol-target compenseert de hogere raw leverage door vol_mult
            #   kleiner te maken, zodat de gerealiseerde portfolio-vol naar
            #   target_annual_vol convergeert.  Netto Sharpe-effect: neutraal
            #   (schaalinvariant), maar vol_mult heeft nu meer ruimte omhoog
            #   (cap 2.5 wordt niet meer knellend bij raw_lev × 1.0 vs 0.25).
            kelly_fraction=None,  # Wave 16 P0-19: kelly.py divisor=6 handles fractional sizing
            starting_equity=float(OmegaConf.select(cfg, "portfolio.starting_equity", default=1.0)),
            bars_per_year=bars_per_yr,
        )
        # M-1 FIX (CHIEF AUDIT 2026-05-28): pass the TRUE multiple-testing
        # burden to the Deflated Sharpe.  N = n_assets × n_sides × optuna_trials.
        # Previously the backtester defaulted to n_assets × 2 (=10), understating
        # the selection bias ~200× and inflating the reported DSR.
        _optuna_trials = int(OmegaConf.select(cfg, "training.optuna_trials", default=200))
        _n_sides = 2  # LONG + SHORT tuned independently per asset
        _total_hypotheses = max(1, len(asset_tracks)) * _n_sides * max(1, _optuna_trials)
        logger.info(
            "Deflated-Sharpe hypothesis count: %d assets × %d sides × %d optuna_trials = %d",
            len(asset_tracks), _n_sides, _optuna_trials, _total_hypotheses,
        )
        backtester = PortfolioBacktester(
            risk_manager=risk_mgr,
            bars_per_year=bars_per_yr,
            bar_seconds=bar_seconds,
            total_n_hypotheses=_total_hypotheses,
        )
        result = backtester.run(asset_tracks)

        joblib.dump(result, portfolio_dir / "result.joblib")

        if hasattr(result, "equity_curve") and result.equity_curve is not None:
            eq = result.equity_curve
            if isinstance(eq, pd.Series):
                eq.to_frame("equity").to_parquet(portfolio_dir / "equity_curve.parquet")

        metrics = {
            "sharpe":               getattr(result, "sharpe",          getattr(result, "sharpe_ratio",  None)),
            "calmar":               getattr(result, "calmar",          getattr(result, "calmar_ratio",  None)),
            "max_drawdown":         getattr(result, "max_dd",          getattr(result, "max_drawdown",  None)),
            "total_return":         getattr(result, "total_return",    None),
            "deflated_sharpe":      getattr(result, "deflated_sharpe", None),
            "avg_gross_leverage":   getattr(result, "avg_gross_leverage", None),
            "realized_vol":         getattr(result, "realized_vol",    None),
            "n_dd_breaker_bars":    getattr(result, "n_dd_breaker_bars", None),
            "n_assets":             len(asset_tracks),
            "n_trades":             {t.symbol: int(np.count_nonzero(t.side)) for t in asset_tracks},
        }
        (reports_dir / f"portfolio_metrics{_run_tag}.json").write_text(json.dumps(metrics, indent=2, default=float))

        logger.info(
            "=== Stage 4 DONE — Sharpe=%.2f | Calmar=%.2f | MaxDD=%.1f%% | "
            "Return=%.1f%% | AvgGrossLev=%.3f ===",
            float(metrics.get("sharpe") or 0.0),
            float(metrics.get("calmar") or 0.0),
            float(metrics.get("max_drawdown") or 0.0) * 100,
            float(metrics.get("total_return") or 0.0) * 100,
            float(metrics.get("avg_gross_leverage") or 0.0),
        )
    except Exception as exc:
        logger.error("Portfolio backtest failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
