# src/tradebot/live/feature_updater.py
"""Incremental feature calculation per new bar.

Maintains a rolling window of raw 1-minute bars per symbol and runs
FeaturePipeline.transform() every REFRESH_EVERY_N_BARS bars to produce
features identical to those used in the backtest.

Parity design (CHIEF AUDIT 2026-05-25 — docstring corrected)
────────────────────────────────────────────────────────────
Training (backtest):  5-second bars from market_data_parquet → FeaturePipeline
                      → micro runs bars (~538 5s-bars per micro bar ≈ 44.8 min).

Live:                 5-second bars from Bybit publicTrade (linear perp)
                      → SAME FeaturePipeline → SAME micro runs bars.

Both paths call the SAME build_features() function (features/pipeline.py)
with the SAME merged Hydra config (conf/config.yaml ⊕ conf/conf_config.yaml,
P0-1 fix).  ZERO bar-resolution difference; CVD distribution matches because
both consume trade-tape-derived bars with taker_buy_volume populated.

CUSUM correctness
─────────────────
FeaturePipeline runs every N bars (default N=45).  Between refreshes the
LAST computed feature row is returned, but with the CURRENT bar's close
and GK-vol proxy injected so ModelSignal's incremental CUSUM always steps
on the correct live price.

Design constraint (R-1): only information from bars ≤ t is used for bar t.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FeatureUpdaterConfig", "FeatureUpdater"]


class FeatureUpdaterConfig:
    """Configuration for the incremental feature updater.

    Parameters
    ----------
    window_bars :
        Number of 1-minute bars to keep in the rolling buffer.
        At 45 bars/micro-bar, 15 000 bars ≈ 10.4 days ≈ 333 micro bars —
        sufficient burn-in for ATR/RSI/GK-vol (need ~200) and fracdiff
        (need ~300).  Hurst-meso (window=700 meso bars ≈ 175 days) will
        remain 0.0 during shadow trading; this is a known and accepted
        limitation.
    refresh_every_n_bars :
        Run FeaturePipeline every N incoming bars.  Set to 45 for 1m bars
        (≈ one micro bar per refresh).  Set to 1 to recompute every bar
        (expensive; for testing only).  Default 1 preserves backward
        compatibility when feature_pipeline_cfg is None.
    feature_columns :
        If set (and feature_pipeline_cfg is None), return only these
        columns from the buffer (legacy pass-through).
    use_feature_store :
        Append each refresh's features to the FeatureStore on disk.
    feature_store_root :
        Path to the FeatureStore root directory.
    feature_pipeline_cfg :
        Hydra DictConfig forwarded to ``features.pipeline.build_features``.
        When None, falls back to raw OHLCV pass-through (legacy mode).
    expected_feature_names :
        Feature column names expected from the training pipeline.  When
        provided, a drift check fires on the first pipeline refresh.
    force_refresh_window_overrides :
        Per-symbol override for the bar window passed to FeaturePipeline
        during ``force_refresh()`` (CUSUM events).  Key is the symbol
        (e.g. ``"AVAXUSDT"``), value is either an int (capped window) or
        ``None`` to use the full buffer.

        Background: illiquid symbols (AVAX: ~0.00078 runs-bars/5s-bar) only
        produce ~7.8 runs bars from the default 10 000-bar force_refresh
        window — far below the 100-bar minimum.  Setting the override to
        ``None`` passes all 180 000 buffer bars; the Numba kernel
        ``numba_runs_bars`` processes this in seconds (vs minutes for liquid
        symbols), so there is no asyncio-blocking concern for AVAX.
    """

    def __init__(
        self,
        window_bars: int = 600,
        refresh_every_n_bars: int = 1,
        feature_columns: Optional[List[str]] = None,
        use_feature_store: bool = False,
        feature_store_root: Optional[str] = None,
        feature_pipeline_cfg=None,
        expected_feature_names: Optional[List[str]] = None,
        force_refresh_window_overrides: Optional[Dict[str, Optional[int]]] = None,
    ) -> None:
        self.window_bars = window_bars
        self.refresh_every_n_bars = max(1, int(refresh_every_n_bars))
        self.feature_columns = feature_columns
        self.use_feature_store = use_feature_store
        self.feature_store_root = feature_store_root
        self.feature_pipeline_cfg = feature_pipeline_cfg
        self.expected_feature_names = expected_feature_names
        self.force_refresh_window_overrides: Dict[str, Optional[int]] = (
            force_refresh_window_overrides or {}
        )


class FeatureUpdater:
    """Stateful per-symbol rolling-window feature computer.

    Parameters
    ----------
    config :
        Updater configuration.
    """

    # Maximum bars passed to FeaturePipeline on a force_refresh().
    # 180K-bar full-buffer conversion blocked the asyncio loop for 80+ min.
    # 10K bars (= ~14h of 5s data) is ample burn-in for all features except
    # Hurst-meso (already accepted as missing during shadow period).
    _FORCE_REFRESH_WINDOW: int = 10_000

    def __init__(self, config: FeatureUpdaterConfig) -> None:
        self._cfg = config
        # Store bars as (timestamp, dict) tuples instead of pd.Series objects.
        # pd.DataFrame(list_of_dicts) is ~10x faster than from list_of_Series.
        self._buffers: Dict[str, List[tuple]] = defaultdict(list)
        self._store: Optional[object] = None

        # Bar counter per symbol — controls when FeaturePipeline fires.
        self._bar_counts: Dict[str, int] = defaultdict(int)

        # Cached last-computed feature row per symbol.  Returned between
        # refreshes with the current bar's close/vol injected (see update()).
        self._feature_cache: Dict[str, Optional[pd.DataFrame]] = {}

        # P0-H: track parity check state per symbol (fired once per symbol).
        self._parity_checked: Dict[str, bool] = {}

        # F4 — optional drift monitor (lazy: attached by caller after init).
        # See live_paper_trader.py for wiring.
        self._drift_monitor: Optional[object] = None

        # B-1 (2026-05-27): track usable bars produced by the last
        # force_refresh() per symbol.  Exposed via get_last_usable_bars()
        # so the engine can detect partial burn-in and bypass JudgeGate.
        self._last_usable_bars: Dict[str, int] = {}

        if config.use_feature_store and config.feature_store_root:
            from ..featurestore.store import FeatureStore
            self._store = FeatureStore(config.feature_store_root)

    # ------------------------------------------------------------------
    # F4 — drift monitor injection
    # ------------------------------------------------------------------

    def attach_drift_monitor(self, monitor: object) -> None:
        """Attach a LiveDriftMonitor instance.

        After attachment, every successful FeaturePipeline refresh will
        invoke ``monitor.observe(symbol, feature_row)`` so the monitor
        compares live distributions against the training reference.
        """
        self._drift_monitor = monitor

    # ------------------------------------------------------------------
    # Pre-population (warm-start from historical 1m bars)
    # ------------------------------------------------------------------

    def pre_populate(self, symbol: str, history_df: pd.DataFrame) -> None:
        """Pre-fill the rolling buffer with historical 1-minute OHLCV bars.

        Must be called BEFORE the live feed starts.  Bars are injected
        directly without triggering FeaturePipeline (too expensive for
        thousands of bars).  The pipeline runs on the FIRST live bar,
        at which point the buffer already contains the full historical
        context — giving exactly the same burn-in as the backtest.

        Parameters
        ----------
        symbol :
            Trading pair, e.g. ``"ETHUSDT"``.
        history_df :
            DataFrame with columns ``open, high, low, close, volume`` and
            a UTC-aware DatetimeIndex, sorted ascending.  Rows in excess
            of ``window_bars`` are trimmed (oldest removed first).
        """
        required = {"open", "high", "low", "close"}
        if not required.issubset(history_df.columns):
            logger.warning(
                "FeatureUpdater.pre_populate [%s]: missing OHLCV columns "
                "(%s) — buffer stays empty.",
                symbol, required - set(history_df.columns),
            )
            return

        buf = self._buffers[symbol]
        buf.clear()

        trim_df = history_df.tail(self._cfg.window_bars)

        # ── Vectorized path — avoids iterrows() ──────────────────────────────
        # iterrows() on 1.5M rows creates 1.5M pd.Series objects → massive GC
        # pressure and 10-30 min runtime for large buffers.  to_dict("records")
        # builds Python dicts directly in C: ~50-100x faster.
        #
        # Step 1: normalise column names (real_volume → volume).
        if "real_volume" in trim_df.columns and "volume" not in trim_df.columns:
            trim_df = trim_df.rename(columns={"real_volume": "volume"})
        if "volume" not in trim_df.columns:
            trim_df = trim_df.assign(volume=0.0)

        # Step 2: select only the columns the buffer needs.
        keep_cols = [
            c for c in (
                "open", "high", "low", "close", "volume",
                "taker_buy_volume", "taker_sell_volume", "tick_volume",
            )
            if c in trim_df.columns
        ]
        slim = trim_df[keep_cols].copy()

        # Step 3: fill NaN with 0.0 for numeric columns (avoids float("nan")
        # propagation into the Numba HRB kernel which expects finite floats).
        slim = slim.fillna(0.0)

        # Step 4: vectorised conversion — produces list[dict] in C.
        timestamps = slim.index.tolist()
        records: list[dict] = slim.to_dict("records")

        buf.extend(zip(timestamps, records))

        # Estimate bar resolution from the buffer's median timestamp delta.
        # Reports both bar count and days — the previous fixed "1m-bars"
        # label produced 125-day claims for a 10.4-day 5s buffer.
        if len(buf) >= 2:
            ts_arr = pd.to_datetime([b[0] for b in buf[-100:]], utc=True)
            deltas_s = ts_arr.to_series().diff().dt.total_seconds().dropna()
            bar_seconds = float(deltas_s.median()) if not deltas_s.empty else 60.0
        else:
            bar_seconds = 60.0
        days = len(buf) * bar_seconds / 86400.0
        logger.info(
            "FeatureUpdater [%s]: pre-populated buffer with %d bars "
            "(median bar = %.0fs ⇒ %.1f days of history).",
            symbol, len(buf), bar_seconds, days,
        )

    # ------------------------------------------------------------------
    # Per-bar update
    # ------------------------------------------------------------------

    def update(self, symbol: str, bar: pd.Series) -> Optional[pd.DataFrame]:
        """Append ``bar`` to the buffer and return a feature row.

        FeaturePipeline is invoked only every ``refresh_every_n_bars``
        bars.  Between refreshes the CACHED last feature row is returned
        with the current bar's close (and a GK-vol proxy) patched in,
        so that ModelSignal's CUSUM always steps on the correct price.

        Parameters
        ----------
        symbol :
            Trading pair.
        bar :
            Single OHLCV bar as pd.Series with a UTC-aware Timestamp name.

        Returns
        -------
        pd.DataFrame with one row, or None when the buffer is too short.
        """
        buf = self._buffers[symbol]
        bar_ts = bar.name
        bar_dict = bar.to_dict()
        buf.append((bar_ts, bar_dict))
        if len(buf) > self._cfg.window_bars:
            buf.pop(0)

        if len(buf) < 2:
            return None

        self._bar_counts[symbol] += 1

        # ── Decide whether to re-run FeaturePipeline ────────────────────────
        should_refresh = (
            self._cfg.feature_pipeline_cfg is not None
            and self._bar_counts[symbol] % self._cfg.refresh_every_n_bars == 0
        )

        if should_refresh:
            df = self._buf_to_df(buf)
            features = self._compute_features(df, symbol=symbol)
            if features is not None and not features.empty:
                self._feature_cache[symbol] = features.iloc[[-1]].copy()
                if self._store is not None:
                    try:
                        self._store.append(features, symbol)
                    except Exception as exc:
                        logger.warning("FeatureUpdater: store append failed: %s", exc)

        elif self._cfg.feature_pipeline_cfg is None:
            # Legacy pass-through: compute every bar (original behaviour).
            df = self._buf_to_df(buf)
            features = self._compute_features(df, symbol=symbol)
            if features is not None and not features.empty:
                return features.iloc[[-1]]
            return None

        # ── Return cached feature row with live close injected ───────────────
        cached = self._feature_cache.get(symbol)
        if cached is None:
            return None

        result = cached.copy()

        # Inject current bar's price so CUSUM uses the live close, not the
        # stale close from the last FeaturePipeline run.
        close_val = float(bar_dict.get("close", 0.0))
        high_val  = float(bar_dict.get("high",  close_val))
        low_val   = float(bar_dict.get("low",   close_val))

        for col, val in (
            ("close",  close_val),
            ("open",   float(bar_dict.get("open",  close_val))),
            ("high",   high_val),
            ("low",    low_val),
            ("volume", float(bar_dict.get("volume", 0.0))),
        ):
            if col in result.columns:
                result[col] = val

        # Update GK-vol proxy so CUSUM threshold tracks current volatility.
        # GK estimator: 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)²  ≈ (H−L)/C for speed.
        if close_val > 0 and "feat_vol_gk" in result.columns:
            gk_proxy = (high_val - low_val) / close_val
            if gk_proxy > 0:
                result["feat_vol_gk"] = gk_proxy

        return result

    # ------------------------------------------------------------------
    # AFML event-driven refresh
    # ------------------------------------------------------------------

    def force_refresh(self, symbol: str) -> Optional[pd.DataFrame]:
        """Run FeaturePipeline NOW regardless of the bar-count schedule.

        Called by the engine immediately after CUSUMFilter fires so that
        ModelSignal.predict_on_event() receives features computed AT the
        CUSUM event time — not from up to 44.8 minutes ago.

        This is the AFML-correct behaviour: features are computed once per
        runs bar (i.e. once per CUSUM event), not on a fixed clock schedule.

        Thread-safe: takes a list-snapshot of the buffer before processing so
        that concurrent ``update()`` calls from the asyncio event loop cannot
        corrupt the slice while force_refresh runs in a thread executor
        (see ``force_refresh_async``).

        Returns the last-row feature DataFrame, or None if the buffer is
        too short or FeaturePipeline fails.
        """
        # Thread-safe snapshot — list() copies the reference list atomically
        # (GIL-protected).  Concurrent update() appends to the original list
        # and do not affect buf after this point.
        buf = list(self._buffers[symbol])
        if len(buf) < 2:
            logger.debug("FeatureUpdater.force_refresh [%s]: buffer too short (%d).", symbol, len(buf))
            return None

        # Per-symbol window override:
        #   None  → use the full buffer (recommended for all symbols post W-2
        #           fix; ensures enough runs bars regardless of density).
        #   int N → slice to the last N bars (legacy latency cap; removed by
        #           W-2 because force_refresh now runs in a thread executor and
        #           no longer blocks the asyncio event loop).
        if symbol in self._cfg.force_refresh_window_overrides:
            override = self._cfg.force_refresh_window_overrides[symbol]
            if override is None:
                buf_slice = buf          # use full buffer
            else:
                buf_slice = buf[-override:] if len(buf) > override else buf
        else:
            window = self._FORCE_REFRESH_WINDOW
            buf_slice = buf[-window:] if len(buf) > window else buf
        df = self._buf_to_df(buf_slice)

        features = self._compute_features(df, symbol=symbol)
        if features is not None and not features.empty:
            self._feature_cache[symbol] = features.iloc[[-1]].copy()
            # B-1: record usable bars for JudgeGate burn-in bypass in engine.
            self._last_usable_bars[symbol] = len(features)
            logger.debug(
                "FeatureUpdater.force_refresh [%s]: pipeline ran on %d input bars "
                "→ %d feat rows, %d feat cols.",
                symbol, len(buf_slice), len(features), len(features.columns),
            )
            return features.iloc[[-1]]
        logger.warning(
            "FeatureUpdater.force_refresh [%s]: pipeline returned empty "
            "(%d input bars). Check min_usable_bars vs runs-bar density.",
            symbol, len(buf_slice),
        )
        return None

    def get_last_usable_bars(self, symbol: str) -> int:
        """Return usable bars from the last force_refresh() for this symbol.

        Used by the engine to detect partial burn-in and bypass JudgeGate
        (B-1 fix, 2026-05-27).  Returns 0 if no force_refresh has been called
        for this symbol yet.
        """
        return self._last_usable_bars.get(symbol, 0)

    async def force_refresh_async(self, symbol: str) -> Optional[pd.DataFrame]:
        """Async wrapper: runs force_refresh() in a thread-pool executor.

        W-2 fix (2026-05-27): force_refresh on a 1.5M-bar buffer can take
        3–10 s for liquid symbols (ETH/SOL produce ~3 000 HRB bars via the
        Numba kernel).  Running it synchronously in the asyncio event loop
        blocked bar processing for that duration, causing the CB feed-timeout
        to fire on slow CUSUM events.

        This wrapper offloads the CPU-bound pipeline to the default thread
        pool so the event loop stays responsive.  Thread safety is guaranteed
        by force_refresh() taking a list-snapshot of the buffer at entry.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.force_refresh, symbol)

    # ------------------------------------------------------------------
    # Buffer → DataFrame (fast path)
    # ------------------------------------------------------------------

    @staticmethod
    def _buf_to_df(buf: list) -> pd.DataFrame:
        """Convert the (ts, dict) buffer to a DatetimeIndex DataFrame.

        Using a list-of-dicts code path is ~10x faster than pd.DataFrame
        from a list of pd.Series objects — avoids per-Series alignment cost.
        """
        if not buf:
            return pd.DataFrame()
        timestamps, records = zip(*buf)
        df = pd.DataFrame(list(records))
        df.index = pd.DatetimeIndex(
            pd.to_datetime(list(timestamps), utc=True)
        )
        return df

    # ------------------------------------------------------------------
    # Feature computation (FeaturePipeline or pass-through)
    # ------------------------------------------------------------------

    def _compute_features(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> Optional[pd.DataFrame]:
        """Run FeaturePipeline on the current buffer DataFrame.

        Identical to the training code path (build_features → FeaturePipeline
        .transform → df_merged).  The last row of df_merged corresponds to
        the most recently completed micro bar and is the feature vector the
        model was trained to consume.

        Falls back to pass-through when ``feature_pipeline_cfg`` is None.
        """
        if self._cfg.feature_pipeline_cfg is not None:
            try:
                from ..features.pipeline import build_features
                feat_df, _ = build_features(
                    self._cfg.feature_pipeline_cfg,
                    symbol,
                    df,
                    orthogonalizer_save_dir=None,  # never refit in live mode
                )
            except Exception as exc:
                # CHIEF AUDIT 2026-05-25: the old "return cached" fallback
                # masked a config-merge bug that produced 0 signals for 14
                # days of paper trading.  Distinguish two cases:
                #   (a) burn-in (cache is None, first refresh fails because
                #       a tier dropped all bars) → log warning, return None.
                #   (b) we HAD a cache and the pipeline now fails →
                #       raise to halt the run rather than silently trading on
                #       stale features.
                cached = self._feature_cache.get(symbol)
                if cached is None:
                    logger.warning(
                        "FeatureUpdater [%s]: FeaturePipeline failed on first "
                        "refresh (%s) — returning None (burn-in).",
                        symbol, exc,
                    )
                    return None
                logger.critical(
                    "FeatureUpdater [%s]: FeaturePipeline failed AFTER having "
                    "valid features (%s). Refusing to return stale cache — "
                    "halting to prevent silent staleness.",
                    symbol, exc,
                )
                raise RuntimeError(
                    f"FeatureUpdater [{symbol}]: FeaturePipeline failure after "
                    f"warm-up — see logs."
                ) from exc

            # Feature health check — NaN / Inf / zero-variance detection.
            # Runs on every pipeline refresh (every ~45 min at 5s bar cadence).
            # High NaN rates indicate FeaturePipeline burn-in issues or a
            # data quality problem in the rolling buffer.
            try:
                from ..monitoring.feature_health import check_feature_health
                feat_cols = [c for c in feat_df.columns if c.startswith("feat_")]
                if feat_cols:
                    health = check_feature_health(
                        feat_df[feat_cols].tail(10),
                        nan_threshold=0.20,  # allow up to 20% NaN during burn-in
                    )
                    if health.has_critical:
                        logger.warning(
                            "FeatureUpdater [%s]: %d feature health issues "
                            "(NaN/Inf). First 5: %s",
                            symbol, health.n_issues,
                            [str(i) for i in health.issues[:5]],
                        )
            except Exception as hc_exc:
                logger.debug(
                    "FeatureUpdater [%s]: health check skipped: %s", symbol, hc_exc
                )

            # P0-H parity check: log column drift on first refresh.
            if (
                symbol not in self._parity_checked
                and self._cfg.expected_feature_names is not None
            ):
                live_cols     = frozenset(feat_df.columns)
                expected_cols = frozenset(self._cfg.expected_feature_names)
                missing = expected_cols - live_cols
                extra   = live_cols - expected_cols
                if missing or extra:
                    logger.warning(
                        "FeatureUpdater [%s] FEATURE DRIFT — "
                        "missing: %s | extra: %s",
                        symbol,
                        sorted(missing)[:10],
                        sorted(extra)[:10],
                    )
                else:
                    logger.info(
                        "FeatureUpdater [%s]: parity check PASS (%d features).",
                        symbol, len(live_cols),
                    )
                self._parity_checked[symbol] = True

            # F4 — feed every successful refresh into the live drift monitor.
            if self._drift_monitor is not None:
                try:
                    self._drift_monitor.observe(symbol, feat_df.tail(1))
                except Exception as dm_exc:
                    logger.debug(
                        "FeatureUpdater [%s]: drift monitor observe failed: %s",
                        symbol, dm_exc,
                    )

            return feat_df

        # ── Legacy pass-through ───────────────────────────────────────────
        cols = self._cfg.feature_columns
        if cols is not None:
            available = [c for c in cols if c in df.columns]
            return df[available] if available else df
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_buffer(self, symbol: str) -> pd.DataFrame:
        """Return the current rolling buffer as a DataFrame (for diagnostics)."""
        buf = self._buffers.get(symbol, [])
        return self._buf_to_df(buf) if buf else pd.DataFrame()

    def bars_since_last_refresh(self, symbol: str) -> int:
        """Number of bars since the last FeaturePipeline refresh for this symbol."""
        n = self._cfg.refresh_every_n_bars
        count = self._bar_counts.get(symbol, 0)
        return count % n if n > 1 else 0
