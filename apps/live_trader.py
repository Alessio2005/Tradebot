"""apps/live_trader.py — live engine entry point with full model wiring.

Usage:
    python apps/live_trader.py [+live.mode=paper] [+live.symbols=[ETHUSDT,...]]

Mode:
    paper (default): replays Bybit WS bars without placing real orders.
    live            : routes orders to the Bybit V5 REST API.
                      Requires BYBIT_API_KEY + BYBIT_API_SECRET in env.

Model wiring:
    For each symbol in live.symbols, a ModelSignal (LONG + SHORT) is loaded
    from artefacts/models/{SYM}_{SIDE}_ensemble.joblib.  Both sides run
    concurrently; the PortfolioController reconciles opposing signals via HRP.

Feature parity (EXACT — zero resolution difference):
    Training:   5s raw bars from market_data_parquet/*.parquet
                → FeaturePipeline.transform()
                → micro runs bars (~538 5s-bars per bar, 44.8 min avg)
                → 111 features (40 micro / 32 meso / 39 macro)

    Live:       5s bars from Bybit publicTrade aggregation (bar_seconds=5)
                → SAME FeaturePipeline.transform()
                → SAME micro runs bars (~538 5s-bars per bar)
                → SAME 111 features including CVD (taker_buy_volume included)

    Why publicTrade and not kline?
      Bybit kline WS minimum interval is 1m and carries no taker-side split.
      publicTrade gives every individual trade; we aggregate client-side into
      5s tumbling windows — the same process used to produce the training data.

    FeatureUpdater runs FeaturePipeline every 538 incoming 5s-bars (≈ one
    micro runs bar).  Between refreshes the last feature row is returned with
    the current bar's close and GK-vol proxy injected so CUSUM always steps
    on the live price.

    At startup the rolling buffer is pre-populated directly from the local
    5s parquets (no resampling) — 180 000 bars ≈ 10.4 days.  CVD columns
    (taker_buy_volume, taker_sell_volume) are passed through so burn-in
    features are computed identically to training.

    Known limitation: Hurst-meso (window=700 meso bars ≈ 175 days) will be
    0.0 during the 14-day shadow period.  Accepted and logged at startup.

Shadow mode (required before live):
    Run with +live.mode=paper for >=14 days.  Verify equity curve, trade
    frequency, and Sharpe match OOS expectations before live.mode=live.
    See docs/model_risk_policy.md §2.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from tradebot.live.engine import LiveEngine, LiveEngineConfig
from tradebot.live.feature_updater import FeatureUpdater, FeatureUpdaterConfig
from tradebot.live.feed import Feed, FeedConfig
from tradebot.live.model_signal import ModelSignal, ModelSignalConfig
from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_ARTEFACTS = _ROOT / "artefacts"
_RAW_DATA = _ROOT / "market_data_parquet"

# Pre-populate each symbol's rolling buffer with this many 5s-bars.
# 180 000 5s-bars = 10.4 days × 24h × 60min × 12 bars/min ≈ 333 micro bars.
# Sufficient burn-in: ATR/RSI/GK (~200), fracdiff (~300), entropy (~150).
# Hurst-meso (window=700 meso bars ≈ 175 days) will be 0 during shadow
# trading — logged and accepted.
_WARMUP_BARS: int = 180_000

# Per-asset CUSUM threshold multipliers (mirror conf_config.yaml)
_CUSUM_MULTIPLIERS: dict[str, float] = {
    "ETHUSDT":  2.5,
    "SOLUSDT":  2.5,
    "AVAXUSDT": 2.2,
    "LINKUSDT": 2.0,
    "DOTUSDT":  2.0,
    "BTCUSDT":  1.5,
}


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------

def _check_credentials(mode: str) -> None:
    """Abort early if live mode is requested without Bybit API keys.

    BUG-7 fix: without this check the engine starts, the first order
    attempt raises an exception deep inside the router, and the engine
    continues running in a half-broken state (no orders placed, no halt).
    """
    if mode != "live":
        return
    api_key    = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    if not api_key or not api_secret:
        raise RuntimeError(
            "live.mode=live requires BYBIT_API_KEY and BYBIT_API_SECRET "
            "environment variables.  Set them before starting the engine."
        )
    logger.info("live_trader: Bybit API credentials present ✓")


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def _build_signals(symbols: list[str], sides: list[str] = ("LONG", "SHORT")) -> list[ModelSignal]:
    """Instantiate one ModelSignal per (symbol, side) pair.

    Skips pairs for which the ensemble model or calibrator artefacts are
    missing (warning logged) so that the engine can start even if some
    models have not been trained yet.
    """
    signals: list[ModelSignal] = []
    for sym in symbols:
        cusum_mult = _CUSUM_MULTIPLIERS.get(sym, 2.0)
        for side in sides:
            try:
                cfg = ModelSignalConfig(
                    symbol=sym,
                    side=side,
                    artefacts_dir=_ARTEFACTS,
                    cusum_threshold_multiplier=cusum_mult,
                    ema_regime_span=200,
                )
                sig = ModelSignal(cfg)
                signals.append(sig)
                logger.info("live_trader: loaded ModelSignal [%s/%s]", sym, side)
            except FileNotFoundError as exc:
                logger.warning("live_trader: skipping [%s/%s] — %s", sym, side, exc)
    return signals


def _load_feature_names(symbols: list[str]) -> dict[str, list[str]]:
    """Load expected feature names per symbol from feature_map JSON files.

    Used by FeatureUpdater for P0-H parity checks at startup.
    """
    feature_names: dict[str, list[str]] = {}
    for sym in symbols:
        fmap_path = _ARTEFACTS / f"feature_map_{sym}.json"
        if fmap_path.exists():
            with open(fmap_path, encoding="utf-8") as fh:
                fm = json.load(fh)
            combined = fm.get("micro", []) + fm.get("meso", []) + fm.get("macro", [])
            feature_names[sym] = combined
    return feature_names


# ---------------------------------------------------------------------------
# Historical buffer pre-population
# ---------------------------------------------------------------------------

def _load_5s_history(sym: str, warmup_bars: int = _WARMUP_BARS) -> pd.DataFrame | None:
    """Load raw 5s-bar parquets DIRECTLY — NO resampling.

    Returns the last ``warmup_bars`` rows from the local
    ``market_data_parquet/{sym}/**/*.parquet`` files at their native 5s
    resolution with ALL columns intact (including taker_buy_volume and
    taker_sell_volume needed for CVD features).

    This is IDENTICAL to what FeaturePipeline consumed during training.
    No conversion, no resolution change — exact same data format.

    Returns None if no local raw data is found for this symbol.
    """
    sym_dir = _RAW_DATA / sym
    if not sym_dir.exists():
        logger.warning(
            "live_trader [%s]: raw data dir %s not found — buffer stays cold.",
            sym, sym_dir,
        )
        return None

    all_files = sorted(sym_dir.glob("**/*.parquet"))
    if not all_files:
        logger.warning("live_trader [%s]: no raw parquet files found.", sym)
        return None

    # 180 000 5s-bars ≈ 10.4 days.  Each monthly parquet ≈ 512 000 rows.
    # Two monthly files cover 10.4 days comfortably.
    n_files_needed = max(2, warmup_bars // 450_000 + 2)
    recent_files = all_files[-n_files_needed:]

    dfs: list[pd.DataFrame] = []
    for f in recent_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as exc:
            logger.warning(
                "live_trader [%s]: failed to load %s: %s", sym, f.name, exc
            )

    if not dfs:
        return None

    raw = pd.concat(dfs, ignore_index=True)

    # ── Ensure UTC-aware DatetimeIndex ───────────────────────────────────────
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp")
    else:
        raw.index = pd.to_datetime(raw.index, utc=True)

    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    # ── Normalise volume column name: real_volume → volume ───────────────────
    # The parquets store base-asset volume as real_volume; FeaturePipeline
    # expects the column to be named "volume".
    if "real_volume" in raw.columns and "volume" not in raw.columns:
        raw = raw.rename(columns={"real_volume": "volume"})
    if "volume" not in raw.columns:
        raw["volume"] = 0.0

    # ── Keep the last warmup_bars rows ───────────────────────────────────────
    raw = raw.tail(warmup_bars)

    logger.info(
        "live_trader [%s]: loaded %d raw 5s-bars from %d file(s) "
        "(%.1f days of history, up to %s). "
        "CVD columns present: taker_buy=%s taker_sell=%s",
        sym, len(raw), len(recent_files),
        len(raw) / (24.0 * 60.0 * 12.0),
        raw.index[-1].strftime("%Y-%m-%d %H:%M UTC") if len(raw) > 0 else "N/A",
        "taker_buy_volume" in raw.columns,
        "taker_sell_volume" in raw.columns,
    )
    return raw


def _prepopulate_buffers(
    fu: FeatureUpdater,
    symbols: list[str],
    warmup_bars: int = _WARMUP_BARS,
) -> None:
    """Pre-fill each symbol's FeatureUpdater buffer with raw 5s-bar history.

    Loads the 5s parquets DIRECTLY (no resampling) so the buffer contains
    the same bar format as the live @aggTrade feed.  FeaturePipeline sees
    identical data in warm-start and live operation — zero distribution shift.
    """
    for sym in symbols:
        history = _load_5s_history(sym, warmup_bars)
        if history is None or history.empty:
            logger.warning(
                "live_trader [%s]: skipping buffer pre-population "
                "(no raw data available).", sym,
            )
            continue
        fu.pre_populate(sym, history)


# ---------------------------------------------------------------------------
# Engine assembly
# ---------------------------------------------------------------------------

def _build_engine(cfg: DictConfig) -> tuple[LiveEngine, Feed]:
    live_cfg = cfg.get("live", {})
    symbols: list[str] = list(live_cfg.get("symbols", ["ETHUSDT"]))
    mode: str = str(live_cfg.get("mode", "paper"))
    # bar_seconds=5 is fixed — matches training data exactly.
    # Bybit kline WS min interval is 1m; the live feed uses publicTrade
    # and aggregates client-side.  DO NOT change this value.
    bar_seconds: int = 5
    equity: float = float(live_cfg.get("initial_equity", 200_000.0))

    # BUG-7: Validate Bybit credentials before constructing anything.
    _check_credentials(mode)

    # ── Feed — 5s bars via publicTrade aggregation ────────────────────────────
    # In live mode: connects to Bybit v5/public/linear publicTrade.{sym} and
    # aggregates trades into 5s tumbling windows (matching training data).
    # In paper mode: replays historical bars from feed._bars dict.
    # shadow mode: real Bybit WS feed + PaperOMS (no real orders)
    # paper mode:  replay historical bars (feed stays empty without historical_bars)
    feed_cfg = FeedConfig(
        symbols=symbols,
        bar_seconds=bar_seconds,
        paper_mode=(mode not in ("live", "shadow")),
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    feed = Feed(config=feed_cfg, queue=queue)

    # ── Model signals (one LONG + one SHORT per symbol) ───────────────────────
    signals = _build_signals(symbols)
    if not signals:
        logger.warning(
            "live_trader: NO model signals loaded. Engine will run but "
            "generate no trades. Check artefacts/models/ directory."
        )

    # ── Signal runner. STAGE D (C6): de runner heeft geen `min_confidence`
    #    meer -- audit paragraaf 14 sluit modelvertrouwen uit als parameter die
    #    de positiegrootte bepaalt (no-go 8). Elk signaal poortwacht zichzelf in
    #    de alphalaag, waar de backtest dezelfde code draait.
    sr_cfg = SignalRunnerConfig(
        use_combiner=False,   # One model per side; no IC combiner needed
    )
    sr = SignalRunner(signals=signals, config=sr_cfg)

    # ── Feature names for P0-H parity check ───────────────────────────────────
    feat_names_by_sym = _load_feature_names(symbols)
    # Flatten to the union of all symbols' expected feature names.
    # FeatureUpdater's parity check fires per-symbol on the first bar.
    all_expected_feats = sorted(
        {n for names in feat_names_by_sym.values() for n in names}
    )
    if all_expected_feats:
        logger.info(
            "live_trader: feature parity check armed for %d features across %s.",
            len(all_expected_feats), list(feat_names_by_sym.keys()),
        )

    # ── FeatureUpdater config — exact backtest/live parity ────────────────────
    # window_bars=180_000: holds 10.4 days of 5s-bars = ~333 micro bars.
    #   Sufficient burn-in: ATR/RSI/GK (~200), fracdiff (~300), entropy (~150).
    #   Hurst-meso (window=700 meso bars ≈ 175 days) will be zero during shadow
    #   trading — logged and accepted.
    # refresh_every_n_bars=538: FeaturePipeline fires every 538 incoming 5s-bars.
    #   538 5s-bars = 44.8 min = one micro runs-bar (exact training rate).
    #   Between refreshes the CACHED feature row is returned with the current
    #   bar's close/GK-vol injected so CUSUM always steps on the live price.
    #   CVD features (feat_cvd_*, feat_rvd_z) are computed correctly because
    #   taker_buy_volume is now included in every bar_series.
    fu_cfg = FeatureUpdaterConfig(
        window_bars=180_000,
        refresh_every_n_bars=538,
        feature_pipeline_cfg=cfg,
        use_feature_store=False,
        expected_feature_names=all_expected_feats if all_expected_feats else None,
    )

    # ── Engine ────────────────────────────────────────────────────────────────
    audit_path = str(live_cfg.get("audit_log", "artefacts/audit/audit.jsonl"))
    engine_cfg = LiveEngineConfig(
        symbols=symbols,
        initial_equity=equity,
        mode=mode,
        interval="5s",             # informational — actual resolution in FeedConfig
        audit_log_path=audit_path,
        fu_config=fu_cfg,
    )
    engine = LiveEngine(config=engine_cfg, signal_runner=sr, feed=feed)
    engine._queue = queue
    feed._queue = queue

    # ── Pre-populate FeatureUpdater buffers ──────────────────────────────────
    # Inject the last _WARMUP_BARS 1m-bars (resampled from 5s parquets) so that
    # FeaturePipeline has full burn-in on bar 1 of live trading.  Same 1m
    # resolution as the live feed — zero distribution shift.
    # ModelSignal CUSUM/EMA is bootstrapped separately below via sig.fit().
    _prepopulate_buffers(engine._fu, symbols, warmup_bars=_WARMUP_BARS)

    # ── Warm-start each signal's CUSUM/EMA from the LIVE buffer (P1-2 fix) ──
    # The buffer was just populated with 10.4 days of current 5s-bars.  Using
    # that to seed CUSUM is strictly preferable to the stale features parquet
    # (which can be weeks old and reflect a different vol regime).
    for sig in signals:
        try:
            warm_df = engine._fu.get_buffer(sig._cfg.symbol)
            sig.fit(warm_df)
        except Exception as exc:
            logger.warning("live_trader: fit() failed for %s: %s", sig.signal_id, exc)

    logger.info(
        "live_trader: engine ready — mode=%s symbols=%s equity=%.0f "
        "signals=%d (LONG+SHORT pairs) feed=aggTrade/5s feature_pipeline=ACTIVE "
        "window=%d_bars refresh_every=%d_bars",
        mode, symbols, equity, len(signals),
        fu_cfg.window_bars, fu_cfg.refresh_every_n_bars,
    )
    return engine, feed


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    engine, _ = _build_engine(cfg)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
