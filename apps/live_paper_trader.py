"""apps/live_paper_trader.py — Live shadow trader for 14-day MRM validation.

Connects to Bybit WebSocket (real publicTrade feed), computes features via
FeaturePipeline, filters signals through JudgeGate, and fills orders via
PaperOMS at real market prices.  NO real orders are placed.

Writes artefacts/paper_trade/state.json + equity_curve.jsonl + audit.jsonl
so the Streamlit dashboard at http://localhost:8501 updates in real time.

Usage:
    python apps/live_paper_trader.py

Requirements:
    - artefacts/models/{SYM}_{SIDE}_ensemble.joblib     (ModelSignal)
    - artefacts/judge_models/{SYM}_{SIDE}_judge.cbm     (JudgeGate)
    - market_data_parquet/{SYM}/**/*.parquet             (buffer warm-start)
    - conf/config.yaml + conf/env/dev.yaml               (FeaturePipeline)

Shadow trading period: >= 14 days (MRM policy §2).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_ARTEFACTS = _ROOT / "artefacts"
_JUDGE_DIR = _ARTEFACTS / "judge_models"
_STATE_DIR = _ARTEFACTS / "paper_trade"
_CONF_DIR = str(_ROOT / "conf")

logger = logging.getLogger("live_paper_trader")

# ── Shadow-mode symbols and equity ────────────────────────────────────────────
_SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
_INITIAL_EQUITY = 200_000.0
_AUDIT_PATH = str(_STATE_DIR / "audit.jsonl")

_CUSUM_MULTIPLIERS = {
    "ETHUSDT": 2.5,
    "SOLUSDT": 2.5,
    "AVAXUSDT": 2.2,
    "LINKUSDT": 2.0,
    "DOTUSDT": 2.0,
}

# ── W-2 (2026-05-27): Per-symbol warmup targets ───────────────────────────────
# Root cause of burn-in bias (Run 1/Run 2): force_refresh window capped at
# 500 k bars → insufficient HRB runs bars for ETH/SOL.  Full-burn-in requires:
#   ETH : ≥ 1.36M raw bars  (0.0022 density × 3 000 target HRB)
#   SOL : ≥ 1.11M raw bars  (0.0027 density × 3 000 target HRB)
#   AVAX: ≥  625k raw bars  (0.0048 density × 3 000 target HRB)
#   LINK: ≥  625k raw bars  (0.0048 density × 3 000 target HRB)
#   DOT : ≥  476k raw bars  (0.0063 density × 3 000 target HRB)
#
# Each target gives ≥ 1 000 usable HRB bars after the 2 000-bar burn-in window,
# which is well above the JudgeGate ACTIVE threshold (500 usable).  No bypass
# of JudgeGate is needed once the buffer is pre-warmed at startup.
#
# Memory: per-symbol buffer stored as list[(ts, dict)].  Dict ~360 bytes/entry:
#   ETH 1.5M × 360 B = 540 MB
#   SOL 1.2M × 360 B = 432 MB
#   AVAX 700k × 360 B = 252 MB
#   LINK 700k × 360 B = 252 MB
#   DOT  500k × 360 B = 180 MB
#   Total ≈ 1.66 GB  (acceptable on any 8+ GB trading machine)
_WARMUP_BARS_PER_SYMBOL: dict[str, int] = {
    "ETHUSDT":  1_500_000,   # → ≈ 3 300 HRB → 1 300 usable ✅
    "SOLUSDT":  1_200_000,   # → ≈ 3 240 HRB → 1 240 usable ✅
    "AVAXUSDT":   700_000,   # → ≈ 3 360 HRB → 1 360 usable ✅
    "LINKUSDT":   700_000,   # → ≈ 3 360 HRB → 1 360 usable ✅
    "DOTUSDT":    500_000,   # → ≈ 3 150 HRB → 1 150 usable ✅
}
# Global window = max of per-symbol targets (FeatureUpdater rolls the buffer
# at this limit; smaller symbols simply never fill it to the max).
_WARMUP_BARS = max(_WARMUP_BARS_PER_SYMBOL.values())  # 1_500_000


def _ensure_dirs() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    (_ARTEFACTS / "audit").mkdir(parents=True, exist_ok=True)

    # Set up logging after dirs exist so FileHandler can write
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(_STATE_DIR / "live_paper.log"), mode="a"),
        ],
    )


def _clear_state_files() -> None:
    """Clear previous run's state files for a clean start."""
    for fname in ("audit.jsonl", "equity_curve.jsonl", "state.json"):
        p = _STATE_DIR / fname
        if p.exists():
            p.unlink()
    logger.info("Cleared previous state files.")


def _build_engine_shadow():
    """Build LiveEngine in shadow mode with JudgeGate and state writes."""
    import json

    # Load Hydra config without @hydra.main decorator
    from hydra.core.global_hydra import GlobalHydra

    from tradebot.live.circuit_breaker import CircuitBreakerConfig
    from tradebot.live.engine import LiveEngine, LiveEngineConfig
    from tradebot.live.execution_controller import ExecutionControllerConfig
    from tradebot.live.feature_updater import FeatureUpdaterConfig
    from tradebot.live.feed import Feed, FeedConfig
    from tradebot.live.model_signal import ModelSignal, ModelSignalConfig
    from tradebot.live.portfolio_controller import PortfolioControllerConfig  # CHIEF-3
    from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig
    from tradebot.schemas.config import RiskConfig, load_config
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=_CONF_DIR, version_base=None)
    cfg = compose(
        config_name="config",
        overrides=[
            "env=prod",   # use prod CUSUM thresholds
            "live.mode=shadow",
            f"live.symbols={list(_SYMBOLS)}",
            f"live.initial_equity={_INITIAL_EQUITY}",
            f"live.audit_log={_AUDIT_PATH}",
        ],
    )
    GlobalHydra.instance().clear()

    # ── P0-1 FIX (CHIEF AUDIT 2026-05-25) ────────────────────────────────────
    # conf/config.yaml does NOT contain the feature_pipeline block — that lives
    # in conf/conf_config.yaml (used by training stages).  Without it, every
    # CUSUM event silently returns cached features and the bot never trades.
    # Merge conf_config.yaml on top of the live cfg so FeaturePipeline finds
    # cfg.feature_pipeline.use_runs_bars / target_micro_bars / use_pca_orth.
    _train_cfg_path = _ROOT / "conf" / "conf_config.yaml"
    if not _train_cfg_path.exists():
        raise FileNotFoundError(
            f"P0-1 fix: training cfg not found at {_train_cfg_path} — "
            "FeaturePipeline will crash live. Cannot proceed."
        )
    train_cfg = OmegaConf.load(str(_train_cfg_path))
    # OmegaConf.merge respects struct mode of `cfg`; conf_config.yaml has
    # its own `defaults` key meant only for its own composition.  Strip it
    # and turn off struct mode briefly so we can pull in extra blocks
    # (feature_pipeline, machine, …) that the live config has never seen.
    if isinstance(train_cfg, type(cfg)) and "defaults" in train_cfg:
        train_cfg = OmegaConf.create(
            {k: v for k, v in train_cfg.items() if k != "defaults"}
        )
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, train_cfg)
    OmegaConf.set_struct(cfg, True)

    # Hard assert — must NEVER boot with a missing feature_pipeline block.
    # The previous silent fallback (cached features) burned 14 days of paper
    # trading in 2026-05-25 with 0 signals fired.
    fp_keys_required = ("use_runs_bars", "target_micro_bars", "use_pca_orth")
    for k in fp_keys_required:
        val = OmegaConf.select(cfg, f"feature_pipeline.{k}", default=None)
        if val is None:
            raise RuntimeError(
                f"P0-1: cfg.feature_pipeline.{k} missing after merge — "
                "live FeaturePipeline will crash. Check conf_config.yaml."
            )
    logger.info(
        "P0-1 OK: feature_pipeline merged "
        "(use_runs_bars=%s, target_micro_bars=%s, use_pca_orth=%s).",
        cfg.feature_pipeline.use_runs_bars,
        cfg.feature_pipeline.target_micro_bars,
        cfg.feature_pipeline.use_pca_orth,
    )

    # ── Feed (real Bybit publicTrade WebSocket) ────────────────────────────────
    queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
    feed_cfg = FeedConfig(
        symbols=_SYMBOLS,
        bar_seconds=5,
        paper_mode=False,   # REAL WebSocket — shadow mode
        replay_delay_s=0.0,
    )
    feed = Feed(config=feed_cfg, queue=queue)

    # ── ModelSignals (one LONG + one SHORT per symbol) ────────────────────────
    # NOTE: sig.fit() is intentionally NOT called here — it's called AFTER
    # _prepopulate_buffers() so the warm-start uses LIVE buffer prices
    # instead of stale features-parquet prices (P1-2 fix).
    signals = []
    for sym in _SYMBOLS:
        cusum_mult = _CUSUM_MULTIPLIERS.get(sym, 2.0)
        for side in ("LONG", "SHORT"):
            try:
                sig_cfg = ModelSignalConfig(
                    symbol=sym,
                    side=side,
                    artefacts_dir=_ARTEFACTS,
                    cusum_threshold_multiplier=cusum_mult,
                    ema_regime_span=200,
                )
                sig = ModelSignal(sig_cfg)
                signals.append(sig)
                logger.info("ModelSignal loaded: %s/%s", sym, side)
            except FileNotFoundError as exc:
                logger.warning("Skipping %s/%s: %s", sym, side, exc)

    sr = SignalRunner(
        signals=signals,
        # use_combiner=True with min_direction_delta=0.02:
        #   When both LONG (signal=+1) and SHORT (signal=-1) fire for the same
        #   symbol (e.g. Platt-calibrator OOD regime where both cal_probs ≈0.72),
        #   the disambiguation logic in SignalRunner.predict_on_event() fires:
        #     delta = |conf_LONG - conf_SHORT|
        #     delta < 0.02  → FLAT (ambiguous bar, no position change)
        #     delta >= 0.02 → winner elected by higher confidence
        #   ICWeightedCombiner is only applied when all raw results agree on
        #   direction (multiple independent alphas for the same side).
        #
        #   The initial portfolio is opened via forced _pending_rebalance
        #   injection below (see "Force initial equal-weight LONG allocation").
        #   Subsequent signal-gated rebalances only fire when direction is clear.
        config=SignalRunnerConfig(
            # STAGE D (C6): de runner had hier `min_confidence=0.0` staan om
            # dubbele filtering te vermijden -- ModelSignal poortwacht al op
            # zijn eigen Optuna-drempel (`_min_conf`, 0,35-0,41 per pair/side
            # uit artefacts/hparams/{SYM}_{SIDE}.json). Die parameter bestaat
            # niet meer: audit paragraaf 14 sluit modelvertrouwen uit als
            # bepaler van de positiegrootte (no-go 8). De poort van ModelSignal
            # blijft, en dat was hier feitelijk al de enige.
            use_combiner=True,
            min_direction_delta=0.02,
        ),
    )

    # ── FeatureUpdater (runs FeaturePipeline every 538 5s-bars ≈ 45 min) ─────
    # Load expected feature names for parity check
    feat_names: list[str] = []
    for sym in _SYMBOLS:
        fmap = _ARTEFACTS / f"feature_map_{sym}.json"
        if fmap.exists():
            with open(fmap) as fh:
                fm = json.load(fh)
            feat_names = fm.get("micro", []) + fm.get("meso", []) + fm.get("macro", [])
            break   # same columns across symbols

    # W-2 (2026-05-27): full-buffer force_refresh — no window cap.
    #
    # Previous W-1 setup: force_refresh_window_overrides = {sym: 500_000} capped
    # force_refresh at 500k bars.  For ETH (density 0.0022 runs/raw bar) this
    # produced only ~1 100 HRB bars — below the 2 000-bar burn-in → 100 usable
    # (floor) → JudgeGate BYPASS, Platt calibrators in OOD regime → both
    # LONG+SHORT firing at ~0.72 → combiner always returned LONG → buy-and-hold.
    #
    # Fix: set force_refresh_window_overrides = {sym: None} for ALL symbols so
    # force_refresh always processes the full pre-warmed buffer:
    #
    #   Symbol   | Buffer bars | HRB bars | usable | JudgeGate
    #   ---------+-------------+----------+--------+----------
    #   ETHUSDT  | 1 500 000   |  ~3 300  | ~1 300 | ACTIVE ✓
    #   SOLUSDT  | 1 200 000   |  ~3 240  | ~1 240 | ACTIVE ✓
    #   AVAXUSDT |   700 000   |  ~3 360  | ~1 360 | ACTIVE ✓
    #   LINKUSDT |   700 000   |  ~3 360  | ~1 360 | ACTIVE ✓
    #   DOTUSDT  |   500 000   |  ~3 150  | ~1 150 | ACTIVE ✓
    #
    # Latency: force_refresh now runs in force_refresh_async() via thread executor
    # (W-2 engine.py change) so the 3–10 s CPU time does not block the asyncio
    # event loop.  CUSUM events arrive every ~45 min per symbol; 10 s processing
    # is well within the 120 s CB feed-timeout.
    fu_cfg = FeatureUpdaterConfig(
        window_bars=_WARMUP_BARS,   # 1_500_000 — largest per-symbol target
        refresh_every_n_bars=538,   # one micro bar ≈ 44.8 min
        feature_pipeline_cfg=cfg,
        use_feature_store=False,
        expected_feature_names=feat_names if feat_names else None,
        # None = use full buffer for every symbol — no latency cap.
        force_refresh_window_overrides={sym: None for sym in _SYMBOLS},
    )

    # ── Engine config ─────────────────────────────────────────────────────────
    # STAGE D, C1 -- de drempels komen uit de soevereine policy, niet uit code.
    # Hier stonden `max_drawdown_pct=0.08`, `max_daily_loss_pct=0.03` en
    # `max_position_age_h=48` als losse getallen. Ze waren gelijk aan
    # `conf/risk/default.yaml` en niets hield ze gelijk: een wijziging in de
    # policy liet deze app stilzwijgend op de oude limieten doorhandelen. Dat is
    # exact de divergentie die `CircuitBreakerConfig.from_risk_config` in
    # `live/engine.py` opheft; deze app deelt hem nu.
    risk_policy = load_config(_ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
    cb_cfg = CircuitBreakerConfig.from_risk_config(
        risk_policy,
        # Live-eigen, zonder tegenhanger in de policy: 120s gap before CB trips
        # (P0-3: allow spot WS reconnect delay; engine.run() seeds last_feed_ts
        # at start so clock ticks from t=0).
        feed_timeout_sec=120,
    )
    # CHIEF-1 (2026-05-28) — INERTIA FILTER active.
    #   max_weight_change=0.25  : turnover cap per rebalance (existing)
    #   min_notional_per_trade  : drop any rebalance < $1 000 absolute
    #   min_weight_change       : drop any rebalance < 2 % weight delta
    # Stops the $234/$304 DOT churn observed in Run 4b that was eating alpha
    # via half-spread + fees on near-zero informational content.
    ec_cfg = ExecutionControllerConfig(
        risk=risk_policy,
        max_weight_change=0.25,
        min_notional_per_trade=1_000.0,
        min_weight_change=0.02,
    )
    # CHIEF-3 (2026-05-28) - Signal-conditional posterior tilt.
    #   signal_tilt_strength=0.30 -> 30 % of the final weight is driven by
    #   signal-confidence edge; 70 % stays pure risk-parity. Without this the
    #   allocator ignored expected_returns and every asset got equal $.
    #
    # PHASE 7/8 STAGE C-1 - allocator gewijzigd van "hrp" naar "erc". HRP is
    # Research Track (audit paragraaf 13.1) en technisch geblokkeerd voor
    # productie (fase-6 no-go 15); dit is het LIVE paper-trading-pad. ERC valt
    # bij een mislukte solve terug op inverse volatility, de baseline waartegen
    # HRP zich nog moet bewijzen. De tilt-overlay werkt ongewijzigd: hij zit in
    # PortfolioController, niet in de allocator.
    pc_cfg = PortfolioControllerConfig(
        method="erc",
        min_history_bars=60,
        signal_tilt_strength=0.30,
    )

    engine_cfg = LiveEngineConfig(
        symbols=_SYMBOLS,
        initial_equity=_INITIAL_EQUITY,
        mode="shadow",
        interval="5s",
        audit_log_path=_AUDIT_PATH,
        cb_config=cb_cfg,
        ec_config=ec_cfg,
        pc_config=pc_cfg,  # CHIEF-3: signal-tilted HRP
        fu_config=fu_cfg,
        metrics_port=8000,
        judge_dir=_JUDGE_DIR,
        state_out_dir=_STATE_DIR,
        state_write_every_n=50,
    )
    # Wire AFML-correct external CUSUM multipliers
    engine_cfg.cusum_multipliers = _CUSUM_MULTIPLIERS

    engine = LiveEngine(config=engine_cfg, signal_runner=sr, feed=feed)
    engine._queue = queue
    feed._queue = queue

    # ── F4 — attach live drift monitor ───────────────────────────────────────
    # Reference is the tail of artefacts/features/{SYM}.parquet (training set);
    # current is the last 100 live feature rows.  PSI computed every refresh
    # (~45 min at 5s cadence).  Output to drift_report.jsonl for offline review.
    try:
        from tradebot.monitoring.live_drift_monitor import (
            LiveDriftMonitor,
            LiveDriftMonitorConfig,
        )
        drift_cfg = LiveDriftMonitorConfig(
            artefacts_dir=_ARTEFACTS,
            ref_window=2000,
            current_window=100,
            check_every_n=1,
            drift_report_path=_STATE_DIR / "drift_report.jsonl",
            top_n_logged=10,
        )
        drift_monitor = LiveDriftMonitor(drift_cfg, _SYMBOLS)
        engine._fu.attach_drift_monitor(drift_monitor)
        logger.info("F4: LiveDriftMonitor attached for %d symbols.", len(_SYMBOLS))
    except Exception as exc:
        logger.warning("F4: LiveDriftMonitor wiring failed (%s) — continuing without drift checks.", exc)

    # ── Pre-populate FeatureUpdater buffers from local database (market_data_parquet) ──
    # W-2 (2026-05-27): loads per-symbol bar counts from _WARMUP_BARS_PER_SYMBOL
    # so every symbol reaches ≥ 3 000 HRB runs bars from the start.
    logger.info(
        "Pre-populating FeatureUpdater buffers from market_data_parquet "
        "(per-symbol targets: ETH=%d, SOL=%d, AVAX=%d, LINK=%d, DOT=%d)...",
        _WARMUP_BARS_PER_SYMBOL["ETHUSDT"],
        _WARMUP_BARS_PER_SYMBOL["SOLUSDT"],
        _WARMUP_BARS_PER_SYMBOL["AVAXUSDT"],
        _WARMUP_BARS_PER_SYMBOL["LINKUSDT"],
        _WARMUP_BARS_PER_SYMBOL["DOTUSDT"],
    )
    _prepopulate_buffers(engine._fu, _SYMBOLS)

    # ── P1-2 FIX: warm-start ModelSignal CUSUM/EMA from the live buffer ──────
    # _replay_history uses 'close' + (high, low) if present.  After
    # _prepopulate_buffers the engine's FeatureUpdater holds 1.5M 5s-bars (ETH)
    # with current market prices — the right basis for CUSUM thresholds.
    # Falls back internally to artefacts/features/{SYM}.parquet if buffer is
    # empty, with a WARNING.
    for sig in signals:
        sym = sig._cfg.symbol
        try:
            warm_df = engine._fu.get_buffer(sym)
            sig.fit(warm_df)
        except Exception as exc:
            logger.warning("fit() failed for %s/%s: %s", sym, sig._cfg.side, exc)

    # ── W-2 FIX: pre-warm feature cache for ALL symbols before feed starts ───
    # After _prepopulate_buffers the buffer holds the full per-symbol history.
    # Running force_refresh() NOW (synchronously, before asyncio starts) seeds
    # the feature cache with fully warmed features — no burn-in delay on the
    # first CUSUM event.  Timing: ~3–10 s per symbol (Numba kernel; acceptable
    # at startup).  The running process will trade immediately from bar 1.
    logger.info(
        "W-2: pre-warming feature cache for all symbols "
        "(may take 3–8 min: FeaturePipeline on full buffer per symbol)..."
    )
    for sym in _SYMBOLS:
        try:
            features = engine._fu.force_refresh(sym)
            if features is not None:
                usable = engine._fu.get_last_usable_bars(sym)
                logger.info(
                    "W-2 feature cache warm [%s]: %d usable HRB bars ✅",
                    sym, usable,
                )
            else:
                logger.warning(
                    "W-2 feature cache warm [%s]: force_refresh returned None — "
                    "buffer may still be empty.", sym,
                )
        except Exception as exc:
            logger.error("W-2 feature cache warm [%s] FAILED: %s", sym, exc)

    # ── CHIEF-2 (2026-05-28) — Seed PortfolioController._price_history ───────
    # so HRP runs on day 1 with the SAME historical closes that warmed the
    # ModelSignal CUSUM state.  Without this, the optimizer fell back to
    # equal-weight ($40 040 × 5) on the first rebalance — ignoring that ETH's
    # realised vol is ~⅓ of AVAX/SOL.  Now HRP sees the volatility structure
    # immediately and a *risk-balanced* portfolio opens.
    seed_closes: dict[str, list[float]] = {}
    for sym in _SYMBOLS:
        try:
            buf_df = engine._fu.get_buffer(sym)
            if buf_df is not None and not buf_df.empty and "close" in buf_df.columns:
                seed_closes[sym] = buf_df["close"].dropna().tolist()
        except Exception as exc:
            logger.warning("HRP seed [%s] failed: %s", sym, exc)
    if seed_closes:
        engine._pc.seed_price_history(seed_closes)
    else:
        logger.warning("CHIEF-2: no seed closes available — HRP will boot in equal-weight fallback.")

    # ── CHIEF-3 (2026-05-28) — Real-confidence initial allocation ────────────
    # W-2 / use_combiner=True fix:
    #   With use_combiner=True the portfolio only opens when a CUSUM event fires
    #   AND the direction is unambiguous (|conf_L - conf_S| >= min_direction_delta).
    #   During early run or in Platt-OOD regime (both cal_probs ≈0.72), direction
    #   is ambiguous → no signal → portfolio stays in cash.  We seed the engine
    #   with a real per-symbol signal so the first rebalance fires immediately
    #   AND so the PortfolioController's signal-tilt overlay (λ=0.30) sees
    #   meaningful cal_probs rather than placeholder 0.5s.
    #
    # Approach: query each ModelSignal's CURRENT live posterior via
    # predict_on_event() on the warm feature cache.  The winner per symbol (by
    # |conf − 0.5|) feeds the initial pending_rebalance.  When the model is
    # genuinely flat we fall back to a LONG prior (crypto convention).
    from tradebot.alpha.base import SignalResult as _SR

    _init_ts = pd.Timestamp.now(tz="UTC")
    per_symbol: dict[str, _SR] = {}
    for sig in signals:
        sym = sig._cfg.symbol
        side = sig._cfg.side
        try:
            # FeatureUpdater has no public accessor for the cached features yet
            # (they are written in force_refresh()); read the cache directly.
            features = engine._fu._feature_cache.get(sym)
            if features is None or features.empty:
                continue
            result = sig.predict_on_event(features, bar_ts=_init_ts) \
                if hasattr(sig, "predict_on_event") else sig.predict(features, bar_ts=_init_ts)
            if result is None or result.signal == 0:
                continue
            edge = abs(float(result.confidence) - 0.5)
            existing = per_symbol.get(sym)
            existing_edge = abs(float(existing.confidence) - 0.5) if existing else -1.0
            if edge > existing_edge:
                per_symbol[sym] = result
        except Exception as exc:
            logger.debug("Initial signal query [%s/%s] skipped: %s", sym, side, exc)

    n_real = 0
    for _sym in _SYMBOLS:
        if _sym in per_symbol:
            engine._pending_rebalance[_sym] = per_symbol[_sym]
            n_real += 1
        else:
            # Genuine FLAT — fall back to LONG prior with neutral confidence so
            # the symbol still gets a baseline HRP weight; signal-tilt overlay
            # contributes nothing for this leg (signed_edge = 0).
            engine._pending_rebalance[_sym] = _SR(
                symbol=_sym,
                timestamp=_init_ts,
                signal=1.0,
                confidence=0.5,
                horizon_bars=1,
                signal_id=f"{_sym}_initial_alloc",
            )
    logger.info(
        "Injected initial allocation: %d/%d real signals (with cal_prob), "
        "%d fallback LONG-priors.  Portfolio opens on first bar with prices "
        "via signal-tilted HRP.",
        n_real, len(_SYMBOLS), len(_SYMBOLS) - n_real,
    )

    # Fat-finger guard: allow large initial fills
    for sym in _SYMBOLS:
        engine._ec._last_order_qty[sym] = 1e9

    return engine


def _load_5s_history(sym: str, warmup_bars: int) -> pd.DataFrame | None:
    """Load last ``warmup_bars`` rows from market_data_parquet/{sym}/*.parquet.

    W-2 (2026-05-27): ``warmup_bars`` is now per-symbol (``_WARMUP_BARS_PER_SYMBOL``).
    Reads only as many monthly parquet files as needed to cover the target count,
    avoiding unnecessary I/O for symbols with high HRB density (DOT/LINK/AVAX).
    """
    sym_dir = _ROOT / "market_data_parquet" / sym
    if not sym_dir.exists():
        logger.warning("market_data_parquet/%s not found — buffer stays cold.", sym)
        return None
    all_files = sorted(sym_dir.glob("**/*.parquet"))
    if not all_files:
        logger.warning("No parquet files in market_data_parquet/%s.", sym)
        return None

    # Each monthly file holds ~450k 5s bars.  Load enough recent files to
    # guarantee we have at least warmup_bars rows after concatenation.
    rows_per_file = 450_000
    n_files_needed = max(2, warmup_bars // rows_per_file + 2)
    recent_files = all_files[-n_files_needed:]

    dfs = []
    for f in recent_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", f.name, exc)

    if not dfs:
        return None

    raw = pd.concat(dfs, ignore_index=True)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp")
    else:
        raw.index = pd.to_datetime(raw.index, utc=True)
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    if "real_volume" in raw.columns and "volume" not in raw.columns:
        raw = raw.rename(columns={"real_volume": "volume"})
    if "volume" not in raw.columns:
        raw["volume"] = 0.0

    loaded = raw.tail(warmup_bars)
    logger.info(
        "[%s] Loaded %d raw bars from %d parquet files (target %d bars)",
        sym, len(loaded), len(recent_files), warmup_bars,
    )
    return loaded


def _prepopulate_buffers(fu, symbols: list[str]) -> None:
    """Pre-fill FeatureUpdater buffers with per-symbol historical 5s bars.

    W-2 (2026-05-27): uses ``_WARMUP_BARS_PER_SYMBOL`` so each symbol loads
    exactly the number of raw bars needed to reach ≥ 3 000 HRB runs bars —
    ensuring full burn-in (≥ 1 000 usable bars) for ALL symbols before the
    live feed starts.  This eliminates the Platt-calibrator OOD bias that
    caused both LONG+SHORT to fire at ~0.72 simultaneously and locked the
    portfolio into buy-and-hold.
    """
    for sym in symbols:
        target_bars = _WARMUP_BARS_PER_SYMBOL.get(sym, _WARMUP_BARS)
        history = _load_5s_history(sym, target_bars)
        if history is not None and not history.empty:
            fu.pre_populate(sym, history)
            bar_seconds = 5.0   # publicTrade-derived 5s bars
            days = len(history) * bar_seconds / 86_400.0
            logger.info(
                "Buffer warm-start [%s]: %d bars loaded (%.1f days of 5s data)",
                sym, len(history), days,
            )
        else:
            logger.warning("No raw 5s data for %s — buffer stays cold.", sym)


def _start_dashboard() -> None:
    """Launch Streamlit dashboard in background if not already running."""
    import socket
    try:
        s = socket.create_connection(("localhost", 8501), timeout=1)
        s.close()
        logger.info("Streamlit dashboard already running at http://localhost:8501")
        return
    except OSError:
        pass
    monitor_path = Path(__file__).parent / "paper_monitor.py"
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(monitor_path),
         "--server.port=8501", "--server.headless=true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Streamlit dashboard started at http://localhost:8501")


def main() -> None:
    _ensure_dirs()

    # Ask user before clearing previous run (skip if stdin is not a TTY)
    if (_STATE_DIR / "audit.jsonl").exists() and sys.stdin.isatty():
        ans = input(
            "\nPrevious paper trade data found. Clear for fresh start? [y/N]: "
        ).strip().lower()
        if ans == "y":
            _clear_state_files()
    elif (_STATE_DIR / "audit.jsonl").exists():
        logger.info("Non-interactive mode — appending to existing state files.")

    _start_dashboard()

    logger.info(
        "=== LIVE PAPER TRADER starting ===\n"
        "  Symbols : %s\n"
        "  Equity  : $%s\n"
        "  Feed    : Bybit publicTrade WebSocket (5s bars)\n"
        "  Judge   : %s\n"
        "  Dashboard: http://localhost:8501",
        _SYMBOLS, f"{_INITIAL_EQUITY:,.0f}", _JUDGE_DIR,
    )

    engine = _build_engine_shadow()

    logger.info("Engine ready — connecting to Bybit WebSocket...")
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user — writing final state...")
        engine._write_state()
        logger.info("Live paper trader stopped.")


if __name__ == "__main__":
    main()
