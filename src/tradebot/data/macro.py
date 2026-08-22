"""macro.py — Macro data update pipeline and publication-lag-aware merge.

Extracted from train_regime.py:
  • update_macro() ← lines 3495-3522   (macro update before training loop)
  • merge_macro()  ← lines 3614-3759   (per-symbol macro merge with lag fix)

PUBLICATION-LAG-FIX (Item 7 — preserved from train_regime.py):
  Daily macro statistics are NOT published at 00:00 UTC.  CPI/NFP/FOMC etc.
  arrive with a delay.  A raw backward merge_asof would leak future (not-yet-
  published) values into intraday crypto bars on the publication day.

  Fix: shift macro timestamps +12h BEFORE merge_asof, except for columns
  whose names match the ``realtime_pattern`` (yields, ticker prices that are
  continuously known).  Per-asset and cross-asset crypto macro get the same
  12h lag for consistency.

  Clock-drift tolerance (Issue #7): ``macro_tolerance=2d`` / ``rt_tolerance=1d``
  prevents a stale macro feed (gap, downtime) from being silently forward-filled
  far beyond the source's update frequency.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import cast

import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_PUBLICATION_LAG: pd.Timedelta = pd.Timedelta(hours=12)
_MACRO_TOLERANCE: pd.Timedelta = pd.Timedelta(days=2)
_RT_TOLERANCE:    pd.Timedelta = pd.Timedelta(days=1)
_REALTIME_RE = re.compile(r"(?i)yield|real[_-]?time|rt_|live_|tick_")


# =============================================================================
# MACRO UPDATE (TradFi + Crypto)
# =============================================================================

def update_macro(cfg: DictConfig) -> None:
    """Refresh TradFi and Crypto macro data files from upstream sources.

    Calls each fetcher's ``run_pipeline()`` method; on failure logs an error
    and continues — callers use existing local files as fallback.

    Parameters
    ----------
    cfg:
        Hydra DictConfig.  Uses ``cfg.machine.data_dir`` (default
        ``"market_data_parquet"``) and ``cfg.training.training_universe``.
    """
    from .crypto_macro import CryptoMacroFetcher, MultiCryptoMacroFetcher
    from .tradfi_macro import MacroDataFetcher

    macro_dir: str = cfg.machine.get("data_dir", "market_data_parquet") + "/macro"
    symbols: list  = list(cfg.training.training_universe)

    logger.info("Updating macro datasets (TradFi + Crypto)...")

    # ── TradFi: Yields, Credit-spread, DXY, DVOL, Fear & Greed ───────────────
    try:
        MacroDataFetcher(data_dir=macro_dir).run_pipeline()
        logger.info("TradFi macro updated.")
    except Exception as exc:
        logger.error("TradFi macro update failed: %s. Using existing files.", exc)

    # ── Crypto macro: per-asset funding + cross-asset market features ─────────
    crypto_syms: list[str] = [
        s for s in symbols
        if isinstance(s, str)
        and s.upper().endswith(("USDT", "BUSD", "USDC", "FDUSD"))
    ]

    try:
        if crypto_syms:
            MultiCryptoMacroFetcher(
                data_dir=macro_dir,
                symbols=crypto_syms,
            ).run_pipeline()
        else:
            CryptoMacroFetcher(data_dir=macro_dir).run_pipeline()
        logger.info("Crypto macro updated (%d symbols).", len(crypto_syms))
    except Exception as exc:
        logger.error("Crypto macro update failed: %s. Using existing files.", exc)


# =============================================================================
# MACRO MERGE (publication-lag-aware, per-symbol)
# =============================================================================

def _load_macro_df(path: Path, sym: str) -> pd.DataFrame | None:
    """Load a parquet file and normalise its DatetimeIndex.  Returns None if missing."""
    if not path.exists():
        logger.warning("[%s] Macro file not found: %s", sym, path)
        return None
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.set_index("timestamp", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _merge_asof_lagged(
    df_bars: pd.DataFrame,
    df_macro: pd.DataFrame,
    sym: str,
    lag: pd.Timedelta = _PUBLICATION_LAG,
    tolerance: pd.Timedelta = _MACRO_TOLERANCE,
    rt_tolerance: pd.Timedelta = _RT_TOLERANCE,
) -> pd.DataFrame:
    """Apply publication-lag to macro timestamps, then merge_asof onto bars.

    Realtime columns (yield, live prices) are exempt from the lag but still
    subject to a 1-day clock-drift tolerance.

    CHIEF AUDIT 2026-05-23 (FIX 4 / lag-stacking note):
    ``data/tradfi_macro.py`` ALREADY applies a per-indicator publication-lag
    via a Timedelta index shift (e.g. CPI = +12h, FOMC = +18h) before saving
    the parquet.  When this function is called on those TradFi features the
    additional ``_PUBLICATION_LAG`` (12h) below stacks on top, so the total
    effective lag is per-indicator + 12h.  This is intentionally conservative
    (over-lagging is causal, under-lagging is a leak) but if you want the
    minimal lag you can pass ``lag=Timedelta(0)`` to this helper for
    upstream-already-lagged macro frames.
    """
    realtime_cols: list[str] = [
        c for c in df_macro.columns if _REALTIME_RE.search(str(c))
    ]
    lagged_cols: list[str] = [c for c in df_macro.columns if c not in realtime_cols]

    df_sorted = df_bars.sort_index()

    if lagged_cols:
        df_lag = df_macro[lagged_cols].copy()
        df_lag.index = cast(pd.DatetimeIndex, df_lag.index) + lag
        df_lag = df_lag.sort_index()
        df_sorted = pd.merge_asof(
            df_sorted, df_lag,
            left_index=True, right_index=True,
            direction="backward", tolerance=tolerance,
        )

    if realtime_cols:
        df_rt = df_macro[realtime_cols].sort_index()
        df_sorted = pd.merge_asof(
            df_sorted, df_rt,
            left_index=True, right_index=True,
            direction="backward", tolerance=rt_tolerance,
        )

    logger.info(
        "[%s] Macro merged — lag=%s on %d cols, realtime exempt=%d cols.",
        sym, lag, len(lagged_cols), len(realtime_cols),
    )
    return df_sorted


def merge_macro(
    df_bars: pd.DataFrame,
    sym: str,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Merge per-asset and cross-asset macro features onto bars.

    Parameters
    ----------
    df_bars:
        DataFrame of bars + micro/meso features from FeaturePipeline.
    sym:
        Trading-pair symbol, e.g. "BTCUSDT".
    cfg:
        Hydra DictConfig.  Uses ``cfg.machine.data_dir``.

    Returns
    -------
    pd.DataFrame
        ``df_bars`` with macro columns appended, NaN-filled, deduplicated.
    """
    macro_dir = Path(cfg.machine.get("data_dir", "market_data_parquet")) / "macro"
    is_crypto  = any(t in sym.upper() for t in ("USDT", "BTC", "ETH"))

    # ── Per-asset macro ────────────────────────────────────────────────────────
    if is_crypto:
        per_asset = macro_dir / f"macro_crypto_{sym}.parquet"
        macro_path = per_asset if per_asset.exists() else macro_dir / "macro_crypto_daily.parquet"
        macro_market_path: Path | None = macro_dir / "macro_crypto_market.parquet"
    else:
        macro_path = macro_dir / "macro_tradfi_daily.parquet"
        macro_market_path = None

    df_merged = df_bars.copy()
    df_macro = _load_macro_df(macro_path, sym)

    if df_macro is not None:
        df_merged = _merge_asof_lagged(df_merged, df_macro, sym)
    else:
        logger.warning("[%s] Continuing without per-asset macro.", sym)

    # ── Cross-asset market-wide crypto macro (BTC-dominance, funding dispersion) ─
    if is_crypto and macro_market_path is not None:
        df_market = _load_macro_df(macro_market_path, sym)
        if df_market is not None:
            try:
                df_merged = _merge_asof_lagged(
                    df_merged, df_market, sym,
                    lag=_PUBLICATION_LAG, tolerance=_MACRO_TOLERANCE,
                )
                logger.info("[%s] Cross-asset market macro merged (%d cols).", sym, df_market.shape[1])
            except Exception as exc:
                logger.warning("[%s] Could not merge market macro: %s", sym, exc)

    # ── Clean-up: inf → NaN → ffill → 0, dedup index ─────────────────────────
    import numpy as np
    df_merged = df_merged.replace([np.inf, -np.inf], float("nan"))

    # CHIEF AUDIT 2026-05-23 (FIX 3 / silent macro outage):
    # The blanket ``.ffill().fillna(0.0)`` collapses three distinct states
    # (genuine macro reading, stale ffilled value, missing/zero-filled bar)
    # into a single numeric column, so the model learns "macro=0 → neutral"
    # and triggers that same neutral pathway whenever the upstream feed has
    # an outage in live trading.  Emit an explicit availability flag BEFORE
    # the ffill so any downstream code can mask features when the feed is
    # actually down.  The flag is computed on the bar-level macro columns
    # only (those starting with ``feat_macro``), so per-symbol micro/meso
    # features in ``df_bars`` are NOT counted.
    macro_cols = [c for c in df_merged.columns if c.startswith("feat_macro")]
    if macro_cols:
        df_merged["feat_macro_available"] = (
            df_merged[macro_cols].notna().any(axis=1).astype("int8")
        )
    else:
        df_merged["feat_macro_available"] = np.int8(0)

    df_merged = df_merged.ffill().fillna(0.0)

    if not df_merged.index.is_unique:
        df_merged.index = df_merged.index + pd.to_timedelta(
            df_merged.groupby(level=0).cumcount(), unit="ns"
        )
    if not df_merged.index.is_unique:
        df_merged = df_merged[~df_merged.index.duplicated(keep="first")]

    return df_merged
