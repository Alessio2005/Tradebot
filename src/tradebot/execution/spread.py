"""execution/spread.py — Spread detection + Annualised Sharpe.

Extracted from train_regime.py lines 1473-1587.
Bit-identical to the monolith.  Public API:

  detect_spread(df, configured_spread)          → scalar float
  compute_dynamic_spread_arr(df, ...)           → np.ndarray[float64]
  compute_annualised_sharpe(trade_returns, ts)  → float

Strangler-fig: train_regime.py should import from here and delete its own
definitions (_detect_spread, _compute_dynamic_spread_arr,
_compute_annualised_sharpe) after the equivalence test passes.

Design notes:
  • Square-root market-impact model (Almgren-Chriss / Kyle-λ style):
        spread ≈ 2·taker_fee + k·σ_bar·sqrt(Q/depth_bar)
  • detect_spread returns a SCALAR (median over sample) so that downstream
    consumers expecting a scalar (TrendScanningLabeler, Triple Barrier) keep
    working.  Use compute_dynamic_spread_arr for per-bar resolution.
  • Sharpe computed daily (irregular trades → daily buckets), annualised
    with sqrt(365.25).  This is the SINGLE canonical implementation shared
    by both Optuna folds AND internal_backtest.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

if TYPE_CHECKING:
    from tradebot.execution.fees import FeeSchedule

logger = logging.getLogger(__name__)


# =============================================================================
# SPREAD DETECTION
# =============================================================================

def detect_spread(df: pd.DataFrame, configured_spread: float) -> float:
    """Detect bid/ask presence and return a realistic scalar round-trip spread.

    For Dukascopy/Forex (bid/ask OHLC present): transaction costs are baked
    into the barrier levels — returns 0.0 to avoid double-counting.

    For Bybit/Crypto (mid-price only): computes a square-root
    market-impact proxy that scales with volatility and inversely with
    order-book depth (approximated by per-bar taker volume).

        spread ≈ 2·taker_fee + k·σ_bar·sqrt(Q/depth_bar)

    Returns the **median** over the sample so that downstream scalar
    consumers (TrendScanningLabeler, Triple Barrier) keep working.
    For per-bar resolution use ``compute_dynamic_spread_arr``.

    Parameters
    ----------
    df               : Bar DataFrame (must have 'close'; optionally
                       'bid_close', 'ask_close', 'feat_vol_gk',
                       'taker_buy_volume', 'taker_sell_volume', 'tick_volume').
    configured_spread: Fallback scalar spread (e.g. from Hydra config).

    Returns
    -------
    float
        Scalar effective round-trip spread in price-relative terms.
    """
    if "bid_close" in df.columns and "ask_close" in df.columns:
        return 0.0  # costs baked into bid/ask barrier execution

    spread_arr = compute_dynamic_spread_arr(df, fallback=configured_spread)
    if spread_arr.size == 0:
        return float(configured_spread)

    median_spread = float(np.nanmedian(spread_arr))
    if not np.isfinite(median_spread) or median_spread <= 0.0:
        return float(configured_spread)
    return median_spread


def compute_dynamic_spread_arr(
    df: pd.DataFrame,
    fallback: float = 0.0010,
    taker_fee_round_trip: float = 0.0010,
    impact_k: float = 0.5,
    ref_trade_size: float = 1.0,
    ref_trade_notional_usd: float | None = None,
    fee_schedule: FeeSchedule | None = None,
) -> np.ndarray:
    """Per-bar square-root impact spread for mid-price data.

    Returns ``np.ndarray[float64]`` of length ``len(df)`` with the
    estimated effective round-trip cost per bar.

    Model:
        σ_bar      = feat_vol_gk (fallback: |ret| rolling-std).
        depth_bar  = taker_buy_volume + taker_sell_volume (or tick_volume).
        spread_bar = taker_fee_round_trip + impact_k·σ_bar·sqrt(Q/depth_bar)

    Bounded: floor = pure fee, cap = 5×fallback.

    Parameters
    ----------
    df                   : Bar DataFrame.
    fallback             : Fallback scalar spread used for cap computation.
    taker_fee_round_trip : Bybit taker fee × 2 (round-trip default 10 bps).
    impact_k             : Kyle-λ style impact constant (default 0.5).
    ref_trade_size       : Trade size Q in BASE ASSET units (e.g., 1.0 BTC).
                           Deprecated for cross-asset use — prefer
                           ref_trade_notional_usd which is asset-agnostic.
    ref_trade_notional_usd : CHIEF AUDIT 2026-05-23 (P3.1) — Trade size in
                           USDT notional.  When provided, overrides
                           ref_trade_size and uses the median close price to
                           derive a base-asset-equivalent Q, making the impact
                           formula dimensionally consistent across BTC/ETH/SOL.
                           Recommended default: 10_000.0 ($10K round lot).
    """
    # CHIEF AUDIT 2026-05-23 (H6): wanneer een live FeeSchedule (VIP1-3, etc.)
    # is meegegeven, override de hardcoded VIP0-default met de live round-trip
    # taker rate (one-way taker_bps * 2 → decimal). Voorkomt over-modellering
    # van fees voor real-world accounts die niet op VIP0 zitten.
    if fee_schedule is not None:
        taker_fee_round_trip = float(fee_schedule.taker_bps) * 2.0 * 1e-4

    n = len(df)
    if n == 0:
        return np.array([], dtype=np.float64)

    # ── sigma proxy ──────────────────────────────────────────────────────────
    if "feat_vol_gk" in df.columns:
        sigma = df["feat_vol_gk"].astype(np.float64).to_numpy()
    else:
        ret = df["close"].astype(np.float64).pct_change().fillna(0.0)
        sigma = (
            ret.rolling(20, min_periods=5).std().bfill().fillna(0.0).to_numpy()
        )

    # ── depth proxy ──────────────────────────────────────────────────────────
    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        depth = (
            df["taker_buy_volume"].astype(np.float64).to_numpy()
            + df["taker_sell_volume"].astype(np.float64).to_numpy()
        )
    elif "tick_volume" in df.columns:
        depth = df["tick_volume"].astype(np.float64).to_numpy()
    else:
        depth = np.ones(n, dtype=np.float64)

    safe_depth = np.where(depth > 1e-9, depth, 1e-9)

    # CHIEF AUDIT 2026-05-23 (P3.1 + P3.3): Cross-asset ref_trade_size fix.
    # Determine effective Q (trade size in same units as depth):
    if ref_trade_notional_usd is not None and "close" in df.columns:
        # Convert notional USD to base-asset equivalent using median close.
        # This makes impact dimensionally consistent across all assets:
        #   Q_btc = $10K / $65K = 0.154 BTC, Q_sol = $10K / $150 = 66.7 SOL.
        med_price = max(float(np.nanmedian(df["close"].astype(np.float64))), 1e-9)
        effective_ref = ref_trade_notional_usd / med_price
    else:
        effective_ref = ref_trade_size  # legacy: base-asset Q, not cross-asset safe

    # Depth unit heuristic guard: warn if volume looks like USDT not base asset.
    # When depth >> close × expected_base_volume, the ratio Q/depth is distorted.
    if "close" in df.columns and len(df) > 0:
        med_close = max(float(np.nanmedian(df["close"].astype(np.float64))), 1e-9)
        med_depth = float(np.nanmedian(safe_depth))
        if med_depth > med_close * 1e6:  # depth looks like USDT × 1M+ units
            logger.warning(
                "compute_dynamic_spread_arr: depth median (%.2e) >> "
                "close median (%.2e) — volume column may be in USDT instead of "
                "base asset.  Impact ratio Q/depth will be distorted by ~price.  "
                "Pass volume in base-asset units or use ref_trade_notional_usd.",
                med_depth, med_close,
            )

    impact = impact_k * sigma * np.sqrt(effective_ref / safe_depth)

    spread_arr = taker_fee_round_trip + np.nan_to_num(
        impact, nan=0.0, posinf=0.0, neginf=0.0
    )

    cap = 5.0 * max(fallback, taker_fee_round_trip)
    spread_arr = np.clip(spread_arr, taker_fee_round_trip, cap)
    return spread_arr.astype(np.float64, copy=False)


# =============================================================================
# ANNUALISED SHARPE (CANONICAL)
# =============================================================================

def compute_annualised_sharpe(
    trade_returns: np.ndarray,
    timestamps: np.ndarray,
) -> float:
    """Compute annualised Sharpe from irregular trade returns.

    Resamples trades into daily buckets (days without trades → 0 return)
    and annualises with sqrt(365.25).

    This is the SINGLE canonical implementation shared by Optuna fold
    scoring AND internal_backtest — eliminates the dual-definition bug.

    Parameters
    ----------
    trade_returns : float64 array of per-trade net returns.
    timestamps    : array-like of trade entry/exit timestamps (DatetimeIndex-
                    compatible).

    Returns
    -------
    float
        Annualised Sharpe ratio (0.0 if fewer than 2 unique days or std≈0).
    """
    if len(trade_returns) == 0:
        return 0.0

    ts = pd.Series(trade_returns, index=pd.DatetimeIndex(timestamps))
    dti = cast(pd.DatetimeIndex, ts.index)
    first_day = dti.floor("D")[0]
    last_day  = dti.ceil("D")[-1]
    full_range = pd.date_range(start=first_day, end=last_day, freq="D")

    # Zero-padding is intentional: in live trading, non-trade days contribute
    # 0% to the equity curve. fill_value=0.0 correctly models idle capital and
    # produces the calendar-time Sharpe that matches live performance.
    # Removing zero-padding would inflate Sharpe by 1/√occupancy ≈ 1.58× at
    # 40% occupancy, making the backtest MORE optimistic than live — the
    # opposite of what is desired.
    daily = ts.resample("D").sum().reindex(full_range, fill_value=0.0)
    avg = daily.mean()
    std = daily.std()

    if pd.isna(std) or std < 1e-9:
        return 0.0

    return float((avg / std) * np.sqrt(365.25))


# ── Backward-compat aliases (snake_case private names from monolith) ──────────
_detect_spread              = detect_spread
_compute_dynamic_spread_arr = compute_dynamic_spread_arr
_compute_annualised_sharpe  = compute_annualised_sharpe


# =============================================================================
# CORWIN-SCHULTZ HIGH-LOW SPREAD ESTIMATOR (Phase 5)
# =============================================================================
def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Effectieve relatieve bid-ask spread uit high/low — Corwin & Schultz (2012).

    Phase 5. De gecertificeerde store bevat GEEN bid/ask (docs/DATA_REGISTER.md
    §6), en `assumed_half_spread_bps` in `conf/execution/fees.yaml` is een
    aanname die zichzelf voorlopig noemt. Deze estimator is het alternatief dat
    wél op de data steunt.

    Het idee: de high-low range van één bar bevat zowel de echte
    prijsvolatiliteit als de spread, maar volatiliteit schaalt met de LENGTE van
    het venster en de spread niet. Twee opeenvolgende eendaagse ranges versus de
    tweedaagse range identificeren de twee componenten daarom apart::

        beta  = E[ (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2 ]
        gamma = ( ln( max(H_t,H_{t+1}) / min(L_t,L_{t+1}) ) )^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
                - sqrt( gamma / (3 - 2*sqrt(2)) )
        S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

    `S` is de RELATIEVE ROUNDTRIP-spread; de half-spread is `S/2`.

    Negatieve schattingen worden op nul gezet en NIET weggegooid. Corwin &
    Schultz §II.B laten zien dat een negatieve alpha ontstaat wanneer de echte
    spread klein is ten opzichte van de ruis in de range; het gemiddelde over
    een venster is dan zuiverder mét die nullen dan zonder. Ze verwijderen zou
    de schatting systematisch naar boven trekken.

    Returns
    -------
    pd.Series
        Relatieve roundtrip-spread per bar, op de index van `high`. De eerste
        bar is NaN: de estimator heeft twee opeenvolgende bars nodig en is
        daarmee causaal zodra hij op `t` wordt gelezen voor een beslissing over
        `t+1`.
    """
    require(
        bool(high.index.equals(low.index)),
        "corwin_schultz_spread: high en low staan niet op dezelfde tijdas.",
        DataContractError,
        n_high=len(high), n_low=len(low),
    )
    h = high.astype("float64")
    lo = low.astype("float64")
    require(
        bool((h.dropna() > 0).all()) and bool((lo.dropna() > 0).all()),
        "corwin_schultz_spread: niet-positieve prijzen.",
        DataContractError,
    )

    hl = np.log(h / lo) ** 2
    beta = hl + hl.shift(1)

    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([lo, lo.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rename("cs_spread")
