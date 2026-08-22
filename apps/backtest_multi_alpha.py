"""backtest_multi_alpha.py — Multi-Alpha Portfolio Backtest (Tier 3).

Combines four alpha tracks in a single portfolio backtest:
  Track 1: Scout/Judge (CPCV primary signal — existing backtest_portfolio.py)
  Track 2: CSM Volume-Clock (50-bar macro-runs momentum)
  Track 3: Kalman OU Mean-Reversion (with Judge gate)
  Track 4: Funding-Rate Carry

Strategy-level HRP weights are computed ACROSS tracks (inter-track correlation
≤ 0.4 target), then asset-level HRP within each track, then portfolio risk
manager applies vol-target + DD-breaker.

Design principles:
  • Each track produces signed_returns + requested_leverage per bar.
  • Tracks are NOT rescaled — the PortfolioRiskManager applies vol-target.
  • Track weights are set by HRP (no manual override).
  • OFI track (Track 5) is optional — enabled only when taker_buy/sell
    volume columns are present in features.

Cross-track HRP ensures:
  • No single alpha source dominates (max track weight ≤ 0.40).
  • Diversification across momentum, mean-reversion, and carry.

Exit criteria (Fase 3 CHIEF MASTER PLAN):
  • 3+ alpha tracks active
  • Inter-track correlation ≤ 0.4
  • Portfolio Sharpe ≥ 2.0

Usage::

    python apps/backtest_multi_alpha.py

    # override universe:
    python apps/backtest_multi_alpha.py training.training_universe=[AVAXUSDT,LINKUSDT]
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from tradebot.alpha import (
    CSMVolumeClockSignal,
    FundingCarry,
    KalmanOUMeanReversion,
    OFISignal,
    compute_csm_volume_clock_signals,
    rank_normalize_cross_section,
)
from tradebot.backtest.tracks import AssetTrack, PortfolioBacktestResult

logger = logging.getLogger(__name__)


# =============================================================================
# Per-track signal generation
# =============================================================================

def _generate_csm_track(
    sym: str,
    df_features: pd.DataFrame,
    event_ts: pd.DatetimeIndex,
    macro_window_bars: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate CSM volume-clock signals for one symbol.

    Returns (signed_returns, requested_leverage) both shape (n_events,).
    """
    if "close" not in df_features.columns:
        logger.warning("[%s] CSM track: no 'close' column.", sym)
        return np.zeros(len(event_ts)), np.zeros(len(event_ts))

    sig = CSMVolumeClockSignal(sym, macro_window_bars=macro_window_bars)

    signals_list = []
    df_close = df_features[["close"]]

    for i, ts in enumerate(event_ts):
        # Use all bars up to and including ts (causal)
        df_slice = df_close[:ts]
        if len(df_slice) < macro_window_bars + 5:
            signals_list.append(0.0)
            continue
        result = sig.predict(df_slice)
        signals_list.append(result.signal)

    signals_arr = np.array(signals_list)

    # Signed returns: price_return × signal direction (+1 / -1 / 0)
    close_at_events = df_features["close"].reindex(event_ts).ffill()
    price_rets = close_at_events.pct_change().fillna(0.0).to_numpy()
    directions = np.sign(signals_arr)
    signed_returns = price_rets * directions

    # Leverage: abs(signal) as a proxy (scaled by |signal| ∈ [0,1])
    requested_leverage = np.abs(signals_arr) * 0.5  # conservative ×0.5

    return signed_returns, requested_leverage


def _generate_ou_track(
    sym: str,
    df_features: pd.DataFrame,
    event_ts: pd.DatetimeIndex,
    ema_window: int = 100,
    zscore_window: int = 100,
    entry_zscore: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rolling z-score mean-reversion signal for a single asset.

    KalmanOUMeanReversion is designed for stationary spreads and suppresses
    every bar on outright crypto prices (φ≈1 → halflife >> 48 bars).
    This implementation reverts price to its own rolling EMA instead.

    Signal: -tanh(z / entry_zscore) where z = (log_price - EMA) / rolling_std.
    Returns (signed_returns, requested_leverage) both shape (n_events,).
    """
    if "close" not in df_features.columns:
        logger.warning("[%s] OU track: no 'close' column.", sym)
        return np.zeros(len(event_ts)), np.zeros(len(event_ts))

    log_price = np.log(df_features["close"].clip(lower=1e-9))

    # Rolling EMA and std computed causally on entire feature history
    ema = log_price.ewm(span=ema_window, adjust=False).mean()
    rolling_std = log_price.rolling(zscore_window, min_periods=20).std().clip(lower=1e-9)
    zscore_series = ((log_price - ema) / rolling_std).clip(-5.0, 5.0)

    signals_list = []
    for ts in event_ts:
        if ts not in zscore_series.index:
            # Find latest bar on or before ts
            idx_loc = zscore_series.index.searchsorted(ts, side="right") - 1
            if idx_loc < zscore_window:
                signals_list.append(0.0)
                continue
            z = float(zscore_series.iloc[idx_loc])
        else:
            pos = zscore_series.index.get_loc(ts)
            if pos < zscore_window:
                signals_list.append(0.0)
                continue
            z = float(zscore_series.loc[ts])
        signals_list.append(float(-np.tanh(z / entry_zscore)))

    signals_arr = np.array(signals_list)
    close_at_events = df_features["close"].reindex(event_ts).ffill()
    price_rets = close_at_events.pct_change().fillna(0.0).to_numpy()
    directions = np.sign(signals_arr)
    signed_returns = price_rets * directions
    requested_leverage = np.abs(signals_arr) * 0.4

    return signed_returns, requested_leverage


def _generate_carry_track(
    sym: str,
    df_features: pd.DataFrame,
    event_ts: pd.DatetimeIndex,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate funding-rate carry signals for one symbol.

    Uses ``feat_micro_funding_rate`` or ``funding_rate`` column.
    Returns (signed_returns, requested_leverage) both shape (n_events,).
    """
    funding_col = None
    for candidate in ["feat_micro_funding_rate", "funding_rate", "fundingRate"]:
        if candidate in df_features.columns:
            funding_col = candidate
            break

    if funding_col is None or "close" not in df_features.columns:
        logger.warning("[%s] Carry track: no funding-rate column found (checked feat_micro_funding_rate, funding_rate, fundingRate).", sym)
        return np.zeros(len(event_ts)), np.zeros(len(event_ts))

    sig = FundingCarry(sym, funding_col=funding_col)

    signals_list = []
    needed_cols = ["close", funding_col]

    for ts in event_ts:
        df_slice = df_features[needed_cols][:ts]
        if len(df_slice) < sig.vol_window + 2:
            signals_list.append(0.0)
            continue
        result = sig.predict(df_slice)
        signals_list.append(result.signal)

    signals_arr = np.array(signals_list)
    close_at_events = df_features["close"].reindex(event_ts).ffill()
    price_rets = close_at_events.pct_change().fillna(0.0).to_numpy()
    directions = np.sign(signals_arr)
    signed_returns = price_rets * directions
    requested_leverage = np.abs(signals_arr) * 0.3  # carry = slow, conservative

    return signed_returns, requested_leverage


def _generate_ofi_track(
    sym: str,
    df_features: pd.DataFrame,
    event_ts: pd.DatetimeIndex,
    window: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate OFI signals when taker volume data is available.

    Falls back to runs_buy_vol / runs_sell_vol (bar-level taker aggregates
    from the runs-bar builder) when the raw taker columns are absent.
    """
    df_ofi = df_features
    if "taker_buy_volume" not in df_features.columns or "taker_sell_volume" not in df_features.columns:
        if "runs_buy_vol" in df_features.columns and "runs_sell_vol" in df_features.columns:
            df_ofi = df_features.rename(columns={
                "runs_buy_vol": "taker_buy_volume",
                "runs_sell_vol": "taker_sell_volume",
            })
        else:
            logger.warning("[%s] OFI track: no taker volume columns (taker_buy/sell_volume or runs_buy/sell_vol).", sym)
            return np.zeros(len(event_ts)), np.zeros(len(event_ts))

    sig = OFISignal(sym, window=window)
    needed_cols = ["close", "taker_buy_volume", "taker_sell_volume"]

    signals_list = []
    for ts in event_ts:
        df_slice = df_ofi[needed_cols][:ts].tail(window + 5)
        if len(df_slice) < 5:
            signals_list.append(0.0)
            continue
        result = sig.predict(df_slice)
        signals_list.append(result.signal)

    signals_arr = np.array(signals_list)
    close_at_events = df_features["close"].reindex(event_ts).ffill()
    price_rets = close_at_events.pct_change().fillna(0.0).to_numpy()
    directions = np.sign(signals_arr)
    signed_returns = price_rets * directions
    requested_leverage = np.abs(signals_arr) * 0.3

    return signed_returns, requested_leverage


# =============================================================================
# Track-level HRP (cross-track diversification)
# =============================================================================

def compute_inter_track_hrp(
    track_returns: Dict[str, np.ndarray],
    min_weight: float = 0.05,
    max_weight: float = 0.40,
) -> Dict[str, float]:
    """HRP weights across alpha tracks (inter-track).

    Uses inverse-variance weighting (simplified HRP leaf cluster) when
    all track returns are available.  Falls back to equal-weight for
    tracks with no variance.

    Parameters
    ----------
    track_returns :
        Dict of {track_name: returns_array}.  Arrays must be aligned.
    min_weight / max_weight :
        Weight bounds per track.

    Returns
    -------
    Dict[str, float] of track weights summing to 1.0.
    """
    n = len(track_returns)
    if n == 0:
        return {}
    if n == 1:
        return {k: 1.0 for k in track_returns}

    names = list(track_returns.keys())
    vols = np.array([float(np.std(track_returns[k])) for k in names])
    vols = np.where(vols < 1e-9, 1.0, vols)

    # Inverse-variance weights
    inv_var = 1.0 / (vols ** 2)
    raw_w = inv_var / inv_var.sum()

    # Clip to [min, max] and renormalize
    raw_w = np.clip(raw_w, min_weight, max_weight)
    raw_w /= raw_w.sum()

    weights = {names[i]: float(raw_w[i]) for i in range(n)}

    # Log inter-track correlations
    if n >= 2:
        arr = np.column_stack([track_returns[k] for k in names])
        corr_matrix = np.corrcoef(arr.T)
        for i in range(n):
            for j in range(i + 1, n):
                logger.info(
                    "Inter-track corr [%s vs %s]: %.3f",
                    names[i], names[j], float(corr_matrix[i, j]),
                )

    return weights


# =============================================================================
# Multi-asset, multi-track backtest
# =============================================================================

def build_multi_alpha_tracks(
    sym: str,
    df_features: pd.DataFrame,
    event_ts: pd.DatetimeIndex,
    cost_bps: float = 2.5,
    per_asset_cap: float = 2.0,
    funding_rate: Optional[np.ndarray] = None,
) -> Tuple[List[AssetTrack], Dict[str, float]]:
    """Build multi-alpha tracks for one symbol.

    Generates CSM, OU, Carry (and optionally OFI) tracks, computes
    inter-track HRP weights, and returns weighted AssetTrack list.

    Returns
    -------
    tracks : List[AssetTrack] — one per active alpha track
    weights : Dict[str, float] — track_name → HRP weight
    """
    tracks_out: List[AssetTrack] = []
    track_returns: Dict[str, np.ndarray] = {}

    # ── Track 2: CSM Volume-Clock ─────────────────────────────────────────────
    try:
        sr_csm, lev_csm = _generate_csm_track(sym, df_features, event_ts)
        if np.any(sr_csm != 0):
            tracks_out.append(AssetTrack(
                symbol=f"{sym}_CSM",
                timestamps=event_ts,
                signed_returns=sr_csm,
                requested_leverage=lev_csm,
                side=np.sign(lev_csm).astype(np.int64),
                per_asset_cap=per_asset_cap,
                cost_bps=cost_bps,
                funding_rate=funding_rate,
            ))
            track_returns["CSM"] = sr_csm
            logger.info("[%s] CSM track built: mean_lev=%.3f", sym, float(lev_csm.mean()))
        else:
            logger.debug("[%s] CSM track: all zeros (insufficient data).", sym)
    except Exception as exc:
        logger.warning("[%s] CSM track generation failed: %s", sym, exc)

    # ── Track 3: Kalman OU Mean-Reversion ─────────────────────────────────────
    try:
        sr_ou, lev_ou = _generate_ou_track(sym, df_features, event_ts)
        if np.any(sr_ou != 0):
            tracks_out.append(AssetTrack(
                symbol=f"{sym}_OU",
                timestamps=event_ts,
                signed_returns=sr_ou,
                requested_leverage=lev_ou,
                side=np.sign(lev_ou).astype(np.int64),
                per_asset_cap=per_asset_cap,
                cost_bps=cost_bps,
                funding_rate=funding_rate,
            ))
            track_returns["OU"] = sr_ou
            logger.info("[%s] OU track built: mean_lev=%.3f", sym, float(lev_ou.mean()))
        else:
            logger.warning("[%s] OU track: all zeros — check z-score window vs event count.", sym)
    except Exception as exc:
        logger.warning("[%s] OU track generation failed: %s", sym, exc)

    # ── Track 4: Funding-Rate Carry ────────────────────────────────────────────
    try:
        sr_carry, lev_carry = _generate_carry_track(sym, df_features, event_ts)
        if np.any(sr_carry != 0):
            tracks_out.append(AssetTrack(
                symbol=f"{sym}_Carry",
                timestamps=event_ts,
                signed_returns=sr_carry,
                requested_leverage=lev_carry,
                side=np.sign(lev_carry).astype(np.int64),
                per_asset_cap=per_asset_cap,
                cost_bps=cost_bps,
                funding_rate=funding_rate,
            ))
            track_returns["Carry"] = sr_carry
            logger.info("[%s] Carry track built: mean_lev=%.3f", sym, float(lev_carry.mean()))
        else:
            logger.warning("[%s] Carry track: all zeros — verify fundingRate column is non-zero.", sym)
    except Exception as exc:
        logger.warning("[%s] Carry track generation failed: %s", sym, exc)

    # ── Track 5: OFI (optional) ────────────────────────────────────────────────
    try:
        sr_ofi, lev_ofi = _generate_ofi_track(sym, df_features, event_ts)
        if np.any(sr_ofi != 0):
            tracks_out.append(AssetTrack(
                symbol=f"{sym}_OFI",
                timestamps=event_ts,
                signed_returns=sr_ofi,
                requested_leverage=lev_ofi,
                side=np.sign(lev_ofi).astype(np.int64),
                per_asset_cap=per_asset_cap,
                cost_bps=cost_bps,
                funding_rate=funding_rate,
            ))
            track_returns["OFI"] = sr_ofi
            logger.info("[%s] OFI track built: mean_lev=%.3f", sym, float(lev_ofi.mean()))
        else:
            logger.warning("[%s] OFI track: all zeros — check runs_buy_vol / runs_sell_vol columns.", sym)
    except Exception as exc:
        logger.warning("[%s] OFI track skipped: %s", sym, exc)

    # ── Compute inter-track HRP weights ──────────────────────────────────────
    weights = compute_inter_track_hrp(track_returns)
    logger.info("[%s] Inter-track HRP weights: %s", sym, weights)

    return tracks_out, weights


# =============================================================================
# Main
# =============================================================================

@hydra.main(config_path="../conf", config_name="conf_config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Run multi-alpha portfolio backtest for all assets in universe."""
    artefacts_dir = Path(OmegaConf.select(cfg, "machine.artefacts_dir", default="artefacts"))
    universe = list(OmegaConf.select(cfg, "training.training_universe", default=[]))
    cost_bps  = float(OmegaConf.select(cfg, "training.cost_bps", default=2.5))
    per_asset_cap = float(OmegaConf.select(cfg, "portfolio.max_per_asset_leverage", default=2.0))

    from tradebot.backtest.portfolio import PortfolioBacktester
    from tradebot.risk.portfolio import PortfolioRiskManager

    all_tracks: List[AssetTrack] = []
    all_weights: Dict[str, float] = {}

    for sym in universe:
        feat_path   = artefacts_dir / "features" / f"{sym}.parquet"
        events_path = artefacts_dir / "events"   / f"{sym}.parquet"

        if not feat_path.exists():
            logger.warning("[%s] Feature parquet not found — skipping.", sym)
            continue

        df_features = pd.read_parquet(feat_path)
        df_events   = pd.read_parquet(events_path) if events_path.exists() else df_features
        event_ts    = df_events.index

        if len(event_ts) < 50:
            logger.warning("[%s] Too few events (%d) — skipping.", sym, len(event_ts))
            continue

        # Optional: load per-bar funding rate for backtest cost accounting
        funding_rate: Optional[np.ndarray] = None
        try:
            from tradebot.data.funding import load_per_bar_funding_rate
            macro_dir = Path("market_data_parquet") / "macro"
            funding_arr = load_per_bar_funding_rate(sym, df_features.index, macro_dir)
            # funding_arr is indexed on df_features.index; reindex to event_ts
            funding_ser = pd.Series(funding_arr, index=df_features.index)
            funding_rate = funding_ser.reindex(event_ts).fillna(0.0).to_numpy()
        except Exception as exc:
            logger.warning("[%s] Funding rate load failed: %s", sym, exc)

        tracks, weights = build_multi_alpha_tracks(
            sym=sym,
            df_features=df_features,
            event_ts=event_ts,
            cost_bps=cost_bps,
            per_asset_cap=per_asset_cap,
            funding_rate=funding_rate,
        )

        for track in tracks:
            # Scale leverage by inter-track HRP weight
            track_key = track.symbol.replace(sym + "_", "")
            track_weight = weights.get(track_key, 1.0 / max(len(weights), 1))
            scaled_lev = track.requested_leverage * track_weight
            all_tracks.append(AssetTrack(
                symbol=track.symbol,
                timestamps=track.timestamps,
                signed_returns=track.signed_returns,
                requested_leverage=scaled_lev,
                side=track.side,
                per_asset_cap=track.per_asset_cap,
                cost_bps=track.cost_bps,
                funding_rate=track.funding_rate,
            ))
            all_weights[track.symbol] = float(track_weight)

    if not all_tracks:
        logger.error("No multi-alpha tracks generated — check artefacts.")
        return

    logger.info(
        "Multi-alpha backtest: %d tracks across %d assets.",
        len(all_tracks), len(universe),
    )

    # ── Portfolio backtest ────────────────────────────────────────────────────
    try:
        target_vol  = float(OmegaConf.select(cfg, "portfolio.target_annual_vol", default=0.08))
        max_gross   = float(OmegaConf.select(cfg, "portfolio.max_gross_leverage", default=2.0))
        max_net     = float(OmegaConf.select(cfg, "portfolio.max_net_leverage",   default=1.5))
        dd_thresh   = float(OmegaConf.select(cfg, "portfolio.dd_breaker_threshold", default=0.35))
        dd_resume   = float(OmegaConf.select(cfg, "portfolio.dd_resume_threshold",  default=0.15))

        bar_seconds  = float(OmegaConf.select(cfg, "training.bar_seconds", default=1800.0))
        bars_per_yr  = float(365.0 * 24.0 * 3600.0 / bar_seconds)
        symbols      = [t.symbol for t in all_tracks]

        risk_mgr = PortfolioRiskManager(
            symbols=symbols,
            target_annual_vol=target_vol,
            max_gross_leverage=max_gross,
            max_net_leverage=max_net,
            max_per_asset_leverage=per_asset_cap,
            dd_breaker_threshold=dd_thresh,
            dd_resume_threshold=dd_resume,
            bars_per_year=bars_per_yr,
        )
        backtester = PortfolioBacktester(
            risk_manager=risk_mgr,
            bars_per_year=bars_per_yr,
            bar_seconds=bar_seconds,
        )
        result = backtester.run(all_tracks)

        # Persist results
        out_dir = artefacts_dir / "portfolio_multi_alpha"
        out_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(result, "equity_curve") and result.equity_curve is not None:
            result.equity_curve.to_frame("equity").to_parquet(out_dir / "equity_curve.parquet")

        metrics = {
            "sharpe":           getattr(result, "sharpe",       None),
            "max_dd":           getattr(result, "max_dd",       None),
            "calmar":           getattr(result, "calmar",       None),
            "total_return":     getattr(result, "total_return", None),
            "n_tracks":         len(all_tracks),
            "n_assets":         len(universe),
            "track_weights":    all_weights,
        }
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)

        logger.info(
            "Multi-alpha backtest complete: Sharpe=%.3f, MaxDD=%.3f, Return=%.3f",
            metrics.get("sharpe") or float("nan"),
            metrics.get("max_dd") or float("nan"),
            metrics.get("total_return") or float("nan"),
        )
    except Exception as exc:
        logger.error("Multi-alpha portfolio backtest failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
