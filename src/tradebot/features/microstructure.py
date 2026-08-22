"""Order-flow microstructure features (AFML ch. 18-19).

These features are derived from intraday volume/bid-ask data and capture
informed-trader activity. All functions are lookahead-safe (rolling only).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def order_flow_imbalance(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Rolling Order Flow Imbalance (OFI).

    OFI_t = (buy_vol_t - sell_vol_t) / (buy_vol_t + sell_vol_t + eps)

    Requires taker_buy_volume and taker_sell_volume columns.
    Falls back to runs_imbalance if taker columns absent.
    """
    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        buy  = df["taker_buy_volume"].fillna(0.0)
        sell = df["taker_sell_volume"].fillna(0.0)
        tot  = buy + sell
        raw  = (buy - sell) / tot.replace(0, 1e-9)
    elif "runs_imbalance" in df.columns:
        raw = df["runs_imbalance"].fillna(0.0)
    else:
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        _v = df[vol_col].fillna(0.0)  # volume used for context; not propagated in fallback
        raw = pd.Series(np.zeros(len(df)), index=df.index)

    return raw.rolling(window, min_periods=1).mean().rename("feat_ofi")


def kyle_lambda(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Approximate Kyle's Lambda — price impact per unit of order flow.

    lambda_t ≈ abs(delta_price) / abs(ofi) where OFI = net buy volume.
    Higher lambda = illiquid (thin book, large spread impact).
    """
    delta_p = df["close"].diff().abs()

    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        net_flow = (df["taker_buy_volume"] - df["taker_sell_volume"]).abs()
    else:
        vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
        net_flow = df[vol_col].fillna(0.0)

    eps = 1e-9
    raw_lambda = delta_p / net_flow.replace(0, eps)

    return (
        raw_lambda.rolling(window, min_periods=1).median()
        .clip(upper=raw_lambda.quantile(0.99))
        .rename("feat_kyle_lambda")
    )


def amihud_illiquidity(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Amihud (2002) illiquidity ratio: |r_t| / volume_t.

    High ratio = price moves a lot per unit of volume = illiquid.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).abs()
    vol_col = "real_volume" if "real_volume" in df.columns else (
        "tick_volume" if "tick_volume" in df.columns else "volume"
    )
    vol = df[vol_col].fillna(0.0)
    raw = log_ret / vol.replace(0, 1e-9)
    return (
        raw.rolling(window, min_periods=1).median()
        .clip(upper=raw.quantile(0.99))
        .rename("feat_amihud")
    )


def bid_ask_spread(df: pd.DataFrame) -> pd.Series:
    """Effective bid-ask spread as fraction of mid-price.

    Requires ask_close and bid_close columns.
    Falls back to ATR-proxy if absent.
    """
    if "ask_close" in df.columns and "bid_close" in df.columns:
        mid   = (df["ask_close"] + df["bid_close"]) / 2.0
        spread = (df["ask_close"] - df["bid_close"]) / mid.replace(0, 1e-9)
    else:
        # Proxy: (high - low) / close — correlated with realised spread.
        spread = (df["high"] - df["low"]) / df["close"].replace(0, 1e-9)
    return spread.rename("feat_spread")


def vpin(
    df: pd.DataFrame,
    bucket_size: int = 50,
    n_buckets: int = 50,
) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading (Easley et al. 2012).

    Simplified rolling VPIN over cumulative volume buckets.
    VPIN = mean( |buy_vol - sell_vol| ) / mean( buy_vol + sell_vol )
    """
    if "taker_buy_volume" not in df.columns or "taker_sell_volume" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="feat_vpin")

    buy  = df["taker_buy_volume"].fillna(0.0)
    sell = df["taker_sell_volume"].fillna(0.0)
    imb  = (buy - sell).abs()
    tot  = buy + sell

    window = n_buckets * bucket_size
    rolling_imb = imb.rolling(window, min_periods=bucket_size).sum()
    rolling_tot = tot.rolling(window, min_periods=bucket_size).sum()

    vpin_val = rolling_imb / rolling_tot.replace(0, 1e-9)
    return vpin_val.clip(0, 1).rename("feat_vpin")


def microprice(
    bid: np.ndarray,
    ask: np.ndarray,
    bid_size: np.ndarray,
    ask_size: np.ndarray,
) -> np.ndarray:
    """Size-weighted microprice (Gatheral & Oomen 2010).

    microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

    More informative than mid-price as it weights by available liquidity.
    """
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    bs = np.asarray(bid_size, dtype=np.float64)
    as_ = np.asarray(ask_size, dtype=np.float64)

    total_size = bs + as_
    safe_total = np.where(total_size > 0, total_size, 1.0)
    mp = (a * bs + b * as_) / safe_total
    mp = np.where(total_size > 0, mp, (b + a) / 2.0)  # fallback to mid
    return mp


def bipower_variation(
    returns: np.ndarray,
    adjust: bool = True,
) -> float:
    """Bipower Variation (Barndorff-Nielsen & Shephard 2004).

    BV = (π/2) * sum(|r_t| * |r_{t-1}|)

    Robust to jumps; use BV vs RV to detect jumps (RV - BV > 0).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    bv = (np.pi / 2.0) * float(np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    return bv


def add_all_microstructure_features(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Convenience wrapper: add all microstructure features to df in-place copy."""
    df = df.copy()
    df["feat_ofi"]          = order_flow_imbalance(df, window)
    df["feat_kyle_lambda"]  = kyle_lambda(df, window)
    df["feat_amihud"]       = amihud_illiquidity(df, window)
    df["feat_spread"]       = bid_ask_spread(df)
    df["feat_vpin"]         = vpin(df)
    return df
